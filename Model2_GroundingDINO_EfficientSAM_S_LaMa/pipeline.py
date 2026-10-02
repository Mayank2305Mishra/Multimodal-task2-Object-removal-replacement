from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"
GROUNDING_DIR = MODEL_DIR / "grounding_dino"
EFFICIENTSAM_DIR = MODEL_DIR / "efficient_sam"
LAMA_DIR = MODEL_DIR / "lama"

GROUNDING_WEIGHT = GROUNDING_DIR / "model.safetensors"
EFFICIENTSAM_ENCODER = EFFICIENTSAM_DIR / "efficientsam_s_encoder.onnx"
EFFICIENTSAM_DECODER = EFFICIENTSAM_DIR / "efficientsam_s_decoder.onnx"
LAMA_WEIGHT = LAMA_DIR / "lama_fp32.onnx"


@dataclass(frozen=True)
class Config:
    grounding_box_threshold: float = 0.30
    grounding_text_threshold: float = 0.25
    max_detections: int = 5
    box_expand_ratio: float = 0.03
    mask_clip_to_box: bool = True
    closing_kernel: int = 3
    dilation_kernel: int = 3
    lama_size: int = 512
    intra_op_threads: int = 0
    inter_op_threads: int = 0


def _ort_options(cfg: Config) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if cfg.intra_op_threads > 0:
        options.intra_op_num_threads = cfg.intra_op_threads
    if cfg.inter_op_threads > 0:
        options.inter_op_num_threads = cfg.inter_op_threads
    return options


class EfficientSAMSOnnx:
    """EfficientSAM-S split ONNX encoder/decoder adapter."""

    def __init__(self, encoder_path: Path, decoder_path: Path, cfg: Config):
        options = _ort_options(cfg)
        self.encoder = ort.InferenceSession(
            str(encoder_path), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), options, providers=["CPUExecutionProvider"]
        )

        self.encoder_input = self.encoder.get_inputs()[0].name
        self.decoder_inputs = {x.name for x in self.decoder.get_inputs()}
        required = {
            "image_embeddings",
            "batched_point_coords",
            "batched_point_labels",
            "orig_im_size",
        }
        missing = required - self.decoder_inputs
        if missing:
            raise RuntimeError(
                f"EfficientSAM-S decoder is missing inputs: {sorted(missing)}. "
                f"Found: {sorted(self.decoder_inputs)}"
            )

    @staticmethod
    def preprocess(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected RGB image with shape HxWx3")
        return image_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def encode(self, image_rgb: np.ndarray) -> np.ndarray:
        x = self.preprocess(image_rgb)
        embedding = self.encoder.run(None, {self.encoder_input: x})[0]
        return np.asarray(embedding, dtype=np.float32)

    def predict_box(
        self,
        image_embedding: np.ndarray,
        box_xyxy: np.ndarray,
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        points = np.array([[[[x1, y1], [x2, y2]]]], dtype=np.float32)
        labels = np.array([[[2.0, 3.0]]], dtype=np.float32)
        orig_im_size = np.asarray(original_size, dtype=np.int64)

        outputs = self.decoder.run(
            None,
            {
                "image_embeddings": image_embedding,
                "batched_point_coords": points,
                "batched_point_labels": labels,
                "orig_im_size": orig_im_size,
            },
        )

        if len(outputs) < 2:
            raise RuntimeError(f"Unexpected EfficientSAM decoder outputs: {len(outputs)}")

        predicted_logits = np.asarray(outputs[0])
        predicted_iou = np.asarray(outputs[1])

        masks = predicted_logits >= 0.0
        while masks.ndim > 3 and masks.shape[1] == 1:
            masks = masks[:, 0]
        while masks.ndim > 3 and masks.shape[1] == 1:
            masks = masks[:, 0]
        while masks.ndim > 2 and masks.shape[0] == 1:
            masks = masks[0]

        if masks.ndim == 3:
            candidate_scores = predicted_iou.reshape(-1)
            idx = int(np.argmax(candidate_scores)) if candidate_scores.size else 0
            if idx >= masks.shape[0]:
                idx = 0
            mask = masks[idx]
        elif masks.ndim == 2:
            mask = masks
        else:
            raise RuntimeError(
                f"Unexpected EfficientSAM-S mask shape: {predicted_logits.shape} -> {masks.shape}"
            )

        scores = predicted_iou.reshape(-1)
        score = float(scores[0]) if scores.size else 0.0
        mask = mask.astype(np.uint8)
        if mask.shape != original_size:
            mask = cv2.resize(
                mask,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask > 0, score


class LamaOnnx:
    """Fixed 512x512 LaMa ONNX adapter.

    The selected public export expects float32 RGB image and erase mask inputs
    with shapes [1,3,512,512] and [1,1,512,512]. Mask value 1 denotes the hole
    to inpaint and 0 denotes pixels to preserve. The output is RGB in [0,255].
    """

    def __init__(self, model_path: Path, cfg: Config):
        self.size = int(cfg.lama_size)
        if self.size != 512:
            raise ValueError("The bundled LaMa ONNX model is fixed at 512x512")
        self.session = ort.InferenceSession(
            str(model_path), _ort_options(cfg), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError("LaMa ONNX must expose image and mask inputs")

        names = {x.name for x in inputs}
        self.image_name = "image" if "image" in names else inputs[0].name
        self.mask_name = "mask" if "mask" in names else inputs[1].name
        self.output_name = self.session.get_outputs()[0].name

        image_meta = next(x for x in inputs if x.name == self.image_name)
        mask_meta = next(x for x in inputs if x.name == self.mask_name)
        if image_meta.type != "tensor(float)" or mask_meta.type != "tensor(float)":
            raise RuntimeError(
                f"Unexpected LaMa input types: image={image_meta.type}, mask={mask_meta.type}"
            )

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        size = self.size

        # The shipped model is fixed at 512x512. Keep the mask aligned with the
        # image by resizing both to the model canvas and then compositing the
        # generated pixels back onto the original resolution.
        image_512 = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
        mask_binary = (mask > 127).astype(np.uint8)
        mask_512 = cv2.resize(mask_binary, (size, size), interpolation=cv2.INTER_NEAREST)

        image_nchw = image_512.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        mask_nchw = mask_512[None, None].astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.image_name: image_nchw, self.mask_name: mask_nchw},
        )
        result = np.asarray(outputs[0])
        if result.ndim == 4:
            result = result[0]
        if result.ndim == 3 and result.shape[0] == 3:
            result = result.transpose(1, 2, 0)
        if result.shape != (size, size, 3):
            raise RuntimeError(f"Unexpected LaMa output shape: {result.shape}")

        result_512 = np.clip(result, 0, 255).astype(np.uint8)
        result_full = cv2.resize(result_512, (w, h), interpolation=cv2.INTER_CUBIC)

        # Preserve the original image everywhere outside the requested mask.
        output = np.where(mask_binary[..., None].astype(bool), result_full, image_rgb)
        return output.astype(np.uint8)


class Model2Pipeline:
    """CPU-only text-guided object removal pipeline.

    Grounding DINO Tiny -> EfficientSAM-S ONNX -> LaMa ONNX.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._check_weights()
        self._load_grounding()
        self.segmenter = EfficientSAMSOnnx(
            EFFICIENTSAM_ENCODER,
            EFFICIENTSAM_DECODER,
            self.config,
        )
        self.lama = LamaOnnx(LAMA_WEIGHT, self.config)

    def _check_weights(self) -> None:
        required = [
            GROUNDING_WEIGHT,
            GROUNDING_DIR / "config.json",
            GROUNDING_DIR / "preprocessor_config.json",
            GROUNDING_DIR / "tokenizer.json",
            EFFICIENTSAM_ENCODER,
            EFFICIENTSAM_DECODER,
            LAMA_WEIGHT,
        ]
        missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Missing Model 2 weights/files:\n"
                + "\n".join(f"  - {item}" for item in missing)
                + "\n\nRun: bash download_weights.sh"
            )

    def _load_grounding(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.grounding_processor = AutoProcessor.from_pretrained(
            str(GROUNDING_DIR), local_files_only=True
        )
        self.grounding_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                str(GROUNDING_DIR), local_files_only=True
            )
            .eval()
            .to("cpu")
        )

    @staticmethod
    def to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
        arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Image must have shape HxWx3")
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def clean_prompt(prompt: str) -> str:
        text = str(prompt).strip().lower()
        prefixes = (
            "remove the ",
            "remove ",
            "delete the ",
            "delete ",
            "erase the ",
            "erase ",
            "get rid of the ",
            "get rid of ",
        )
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        return text.strip(" .")

    @staticmethod
    def expand_box(box_xyxy: np.ndarray, width: int, height: int, ratio: float) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        dx = bw * ratio
        dy = bh * ratio
        return np.asarray(
            [
                max(0.0, x1 - dx),
                max(0.0, y1 - dy),
                min(float(width - 1), x2 + dx),
                min(float(height - 1), y2 + dy),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def box_mask(box_xyxy: np.ndarray, height: int, width: int) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        out = np.zeros((height, width), dtype=np.uint8)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 < x1 or y2 < y1:
            return out
        out[y1 : y2 + 1, x1 : x2 + 1] = 255
        return out

    @staticmethod
    def overlay_box(image: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        out = image.copy()
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 255), 3)
        return out

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        out = image.copy()
        m = mask > 0
        tint = np.zeros_like(out)
        tint[..., 0] = 255
        tint[..., 1] = 60
        tint[..., 2] = 60
        out[m] = (alpha * tint[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
        return out

    def ground(self, image_rgb: np.ndarray, prompt: str) -> dict[str, Any]:
        clean = self.clean_prompt(prompt)
        if not clean:
            raise ValueError("Object prompt is empty")

        pil = Image.fromarray(image_rgb)
        inputs = self.grounding_processor(
            images=pil,
            text=[[clean]],
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.config.grounding_box_threshold,
            text_threshold=self.config.grounding_text_threshold,
            target_sizes=[image_rgb.shape[:2]],
            text_labels=[[clean]],
        )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        text_labels = results.get("text_labels", results.get("labels"))
        if boxes is None or scores is None or len(boxes) == 0:
            raise RuntimeError(f'Grounding DINO found no object for "{clean}"')

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        order = np.argsort(-scores_np)[: self.config.max_detections]

        if text_labels is None:
            labels = [clean] * len(boxes_np)
        else:
            labels = [str(v) for v in text_labels]

        detections = []
        for idx in order:
            detections.append(
                {
                    "box": boxes_np[idx],
                    "score": float(scores_np[idx]),
                    "label": labels[idx] if idx < len(labels) else clean,
                }
            )

        best = detections[0]
        return {
            "prompt": prompt,
            "clean_prompt": clean,
            "detections": detections,
            "best_box": best["box"],
            "best_score": best["score"],
            "best_label": best["label"],
        }

    def segment(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> dict[str, Any]:
        h, w = image_rgb.shape[:2]
        box = self.expand_box(
            box_xyxy,
            width=w,
            height=h,
            ratio=self.config.box_expand_ratio,
        )

        t0 = time.perf_counter()
        embedding = self.segmenter.encode(image_rgb)
        encoder_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        mask_bool, score = self.segmenter.predict_box(
            embedding, box, original_size=(h, w)
        )
        decoder_ms = (time.perf_counter() - t1) * 1000.0

        mask = (mask_bool.astype(np.uint8) * 255)

        if self.config.mask_clip_to_box:
            allowed = self.box_mask(box, h, w)
            mask = cv2.bitwise_and(mask, allowed)

        if self.config.closing_kernel > 1:
            k = max(1, int(self.config.closing_kernel))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if self.config.dilation_kernel > 1:
            k = max(1, int(self.config.dilation_kernel))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel)
            if self.config.mask_clip_to_box:
                allowed = self.box_mask(box, h, w)
                mask = cv2.bitwise_and(mask, allowed)

        return {
            "mask": mask,
            "score": score,
            "box": box,
            "encoder_ms": encoder_ms,
            "decoder_ms": decoder_ms,
        }

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.lama.inpaint(image_rgb, mask)

    def remove(self, image: Image.Image | np.ndarray, prompt: str) -> dict[str, Any]:
        image_rgb = self.to_rgb(image)
        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        grounding = self.ground(image_rgb, prompt)
        grounding_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        segmentation = self.segment(image_rgb, grounding["best_box"])
        segmentation_total_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        result = self.inpaint(image_rgb, segmentation["mask"])
        lama_ms = (time.perf_counter() - t2) * 1000.0

        total_ms = (time.perf_counter() - total_t0) * 1000.0

        return {
            "original": image_rgb,
            "result": result,
            "grounding_overlay": self.overlay_box(image_rgb, segmentation["box"]),
            "mask_overlay": self.overlay_mask(image_rgb, segmentation["mask"]),
            "mask": segmentation["mask"],
            "box": segmentation["box"],
            "grounding": grounding,
            "segmentation": segmentation,
            "latency": {
                "grounding_ms": grounding_ms,
                "efficient_sam_encoder_ms": segmentation["encoder_ms"],
                "efficient_sam_decoder_ms": segmentation["decoder_ms"],
                "segmentation_total_ms": segmentation_total_ms,
                "lama_ms": lama_ms,
                "total_ms": total_ms,
            },
        }
