from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"
GROUNDING_DIR = MODEL_DIR / "grounding_dino"
EFFICIENTSAM_ENCODER = MODEL_DIR / "efficient_sam" / "encoder.onnx"
EFFICIENTSAM_DECODER = MODEL_DIR / "efficient_sam" / "decoder.onnx"
LAMA_CKPT = MODEL_DIR / "lama_fp32.onnx"


@dataclass
class PipelineConfig:
    grounding_box_threshold: float = 0.28
    grounding_text_threshold: float = 0.22
    max_detections: int = 5

    # EfficientSAM-S uses the image at its native resolution in the official
    # split-ONNX interface. The encoder itself handles the model transform.
    efficient_sam_max_side: int = 1024

    # Mask cleanup / removal coverage.
    close_kernel: int = 5
    dilation_kernel: int = 25
    dilation_iterations: int = 1
    min_component_area: int = 80

    # Give LaMa enough surrounding context.
    roi_padding: float = 0.90
    roi_min_padding: int = 48
    lama_size: int = 512
    roi_feather: int = 5

    intra_op_threads: int = 0
    inter_op_threads: int = 0


def _session_options(cfg: PipelineConfig):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if cfg.intra_op_threads > 0:
        opts.intra_op_num_threads = cfg.intra_op_threads
    if cfg.inter_op_threads > 0:
        opts.inter_op_num_threads = cfg.inter_op_threads
    return opts


def _first_existing(paths):
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


class EfficientSAMONNX:
    """Official EfficientSAM-S split-ONNX interface.

    Encoder input:
        batched_images

    Decoder inputs:
        image_embeddings
        batched_point_coords
        batched_point_labels
        orig_im_size

    Box prompting uses two corners with labels [2, 3].
    """

    def __init__(self, encoder_path: Path, decoder_path: Path, cfg: PipelineConfig):
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(
                "Missing EfficientSAM-S encoder/decoder. Run ./download_wts.sh first."
            )

        opts = _session_options(cfg)
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self.encoder_input = self.encoder.get_inputs()[0].name
        self.decoder_inputs = {x.name for x in self.decoder.get_inputs()}

        required = {
            "image_embeddings",
            "batched_point_coords",
            "batched_point_labels",
            "orig_im_size",
        }
        if not required.issubset(self.decoder_inputs):
            raise RuntimeError(
                "This is not the expected EfficientSAM-S split decoder.\n"
                f"Found: {sorted(self.decoder_inputs)}\n"
                f"Required: {sorted(required)}"
            )

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        # Official EfficientSAM ONNX example uses RGB float32 [0,1].
        return image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def encode(self, image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(image)
        return self.encoder.run(
            None,
            {self.encoder_input: tensor}
        )[0]

    def segment_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ):
        h, w = image.shape[:2]

        x1, y1, x2, y2 = map(float, box)
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, x1 + 1, w)
        y2 = np.clip(y2, y1 + 1, h)

        # Encoder and decoder operate in original-image coordinates.
        points = np.array(
            [[[[x1, y1], [x2, y2]]]],
            dtype=np.float32,
        )
        labels = np.array(
            [[[2.0, 3.0]]],
            dtype=np.float32,
        )

        t0 = time.perf_counter()
        embedding = self.encode(image)
        encoder_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        outputs = self.decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "batched_point_coords": points,
                "batched_point_labels": labels,
                "orig_im_size": np.array([h, w], dtype=np.int64),
            },
        )
        decoder_ms = (time.perf_counter() - t0) * 1000.0

        predicted_logits = np.asarray(outputs[0])
        predicted_iou = np.asarray(outputs[1])

        # Official output is [B, N, K, H, W].
        if predicted_logits.ndim != 5:
            raise RuntimeError(
                f"Unexpected EfficientSAM logits shape: {predicted_logits.shape}"
            )

        # Pick the highest predicted-IoU mask.
        iou_flat = predicted_iou.reshape(-1)
        best_idx = int(np.argmax(iou_flat))

        masks_flat = predicted_logits.reshape(
            -1,
            predicted_logits.shape[-2],
            predicted_logits.shape[-1],
        )
        logits = masks_flat[best_idx]

        mask = logits >= 0.0

        # Defensive resize in case the ONNX export does not apply orig_im_size.
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        return mask, float(iou_flat[best_idx]), encoder_ms, decoder_ms


class Model1Pipeline:
    """Grounding DINO-Tiny -> EfficientSAM-S -> LaMa FP32 ONNX."""

    def __init__(
        self,
        device: str = "cpu",
        config: Optional[PipelineConfig] = None,
    ):
        if str(device).lower() != "cpu":
            raise ValueError(
                "This Model 1 benchmark uses CPU ONNX inference. "
                "Use device='cpu'."
            )

        self.config = config or PipelineConfig()
        self.device = "cpu"

        self._check_weights()
        self._load_grounding()

        self.efficient_sam = EfficientSAMONNX(
            EFFICIENTSAM_ENCODER,
            EFFICIENTSAM_DECODER,
            self.config,
        )
        self._load_lama()

    def _check_weights(self):
        required = [
            GROUNDING_DIR / "config.json",
            GROUNDING_DIR / "model.safetensors",
            EFFICIENTSAM_ENCODER,
            EFFICIENTSAM_DECODER,
            LAMA_CKPT,
        ]
        missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\n\nRun: bash download_wts.sh"
            )

    def _load_grounding(self):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.grounding_processor = AutoProcessor.from_pretrained(
            str(GROUNDING_DIR),
            local_files_only=True,
        )
        self.grounding_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                str(GROUNDING_DIR),
                local_files_only=True,
            )
            .eval()
            .to("cpu")
        )

    def _load_lama(self):
        opts = _session_options(self.config)
        self.lama_session = ort.InferenceSession(
            str(LAMA_CKPT),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        names = {x.name for x in self.lama_session.get_inputs()}
        if not {"image", "mask"}.issubset(names):
            raise RuntimeError(
                f"Unexpected LaMa inputs: {sorted(names)}; expected image and mask."
            )

    @staticmethod
    def to_rgb(image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        else:
            image = np.asarray(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be HxWx3 RGB.")

        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def clean_prompt(prompt: str) -> str:
        p = str(prompt).strip().lower()

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
            if p.startswith(prefix):
                p = p[len(prefix):]
                break

        return p.strip(" .")

    def ground(self, image: np.ndarray, prompt: str):
        clean = self.clean_prompt(prompt)

        if not clean:
            raise ValueError("Prompt is empty.")

        inputs = self.grounding_processor(
            images=Image.fromarray(image),
            text=[[clean]],
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = self.grounding_model(**inputs)

        post_kwargs = {
            "target_sizes": [(image.shape[0], image.shape[1])],
            "text_threshold": self.config.grounding_text_threshold,
        }

        try:
            result = (
                self.grounding_processor
                .post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self.config.grounding_box_threshold,
                    **post_kwargs,
                )[0]
            )
        except TypeError:
            result = (
                self.grounding_processor
                .post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self.config.grounding_box_threshold,
                    **post_kwargs,
                )[0]
            )

        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("text_labels", result.get("labels"))

        if boxes is None or len(boxes) == 0:
            raise RuntimeError(
                f'Grounding DINO found no object for "{clean}". '
                "Try a more specific prompt."
            )

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        order = np.argsort(-scores)[: self.config.max_detections]

        detections = []
        for i in order:
            detections.append(
                {
                    "box": boxes[i].astype(np.float32),
                    "score": float(scores[i]),
                    "label": clean if labels is None else str(labels[i]),
                }
            )

        return detections

    @staticmethod
    def largest_component(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        if n <= 1:
            return binary.astype(bool)

        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        return (labels == best)

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)

        ck = max(1, int(self.config.close_kernel))
        if ck % 2 == 0:
            ck += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (ck, ck),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        mask = self.largest_component(mask)

        dk = max(1, int(self.config.dilation_kernel))
        if dk % 2 == 0:
            dk += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dk, dk),
        )

        mask = cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=max(1, int(self.config.dilation_iterations)),
        )

        return mask.astype(bool)

    def make_roi(self, image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]

        ys, xs = np.where(mask)

        if len(xs) == 0:
            raise RuntimeError("Final removal mask is empty.")

        x1, x2 = int(xs.min()), int(xs.max() + 1)
        y1, y2 = int(ys.min()), int(ys.max() + 1)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = max(
            self.config.roi_min_padding,
            int(bw * self.config.roi_padding),
        )
        pad_y = max(
            self.config.roi_min_padding,
            int(bh * self.config.roi_padding),
        )

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1, x2, y2)

def lama_inpaint(self, roi: np.ndarray, roi_mask: np.ndarray):
    """Run the Carve LaMa FP32 ONNX model using the model-card preprocessing.

    The official LaMa ONNX demo:
    - resizes image + mask to 512x512
    - converts image to CHW float32 [0,1]
    - converts mask to binary float32
    - runs `lama_fp32.onnx`
    - treats the model output as uint8 image data

    We keep that path exact here so the post-processing cannot silently
    turn a valid LaMa output back into the original image.
    """
    size = self.config.lama_size

    resized_image = cv2.resize(
        roi,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    resized_mask = cv2.resize(
        (roi_mask.astype(np.uint8) * 255),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Same numerical convention as the official LaMa demo:
    # image = CHW float32 in [0,1]
    image_input = (
        resized_image.astype(np.float32) / 255.0
    ).transpose(2, 0, 1)[None]

    # LaMa expects a binary mask: 1 = region to inpaint.
    mask_input = (
        (resized_mask > 0).astype(np.float32)
    )[None, None]

    outputs = self.lama_session.run(
        None,
        {
            "image": image_input,
            "mask": mask_input,
        },
    )

    raw = np.asarray(outputs[0])

    if raw.ndim != 4 or raw.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected LaMa output shape: {raw.shape}"
        )

    raw = raw[0].transpose(1, 2, 0)

    # The official demo converts its LaMa output directly to uint8.
    # Preserve that behavior instead of guessing output ranges.
    generated_512 = np.clip(raw, 0, 255).astype(np.uint8)

    generated = cv2.resize(
        generated_512,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    # Useful runtime diagnostics: these values tell us whether LaMa
    # actually changed anything in the masked area.
    return generated, generated_512, float(raw.min()), float(raw.max())

    def inpaint(self, image: np.ndarray, mask: np.ndarray):
        roi, roi_mask, roi_box = self.make_roi(image, mask)

        t0 = time.perf_counter()
        generated, generated_512, lama_min, lama_max = self.lama_inpaint(
            roi,
            roi_mask,
        )
        lama_ms = (time.perf_counter() - t0) * 1000.0

        # Composite ONLY inside the actual removal mask.
        # This prevents the generated ROI from overwriting unrelated pixels.
        alpha = (roi_mask > 0).astype(np.float32)

        feather = max(0, int(self.config.roi_feather))
        if feather > 0:
            sigma = max(0.5, feather / 3.0)
            alpha = cv2.GaussianBlur(
                alpha,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )

        # Keep alpha bounded and ensure pixels well inside the mask are
        # completely replaced by LaMa.
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        blended = (
            generated.astype(np.float32) * alpha
            + roi.astype(np.float32) * (1.0 - alpha)
        )

        result = image.copy()

        x1, y1, x2, y2 = roi_box

        result[y1:y2, x1:x2] = np.clip(
            blended,
            0,
            255,
        ).astype(np.uint8)

        # Compare LaMa output with the original only over the masked region.
        # If this is near zero, LaMa effectively returned the input and the
        # problem is inside model inference / preprocessing rather than
        # compositing.
        mask_bool = roi_mask.astype(bool)
        if np.any(mask_bool):
            mean_abs_change = float(
                np.mean(
                    np.abs(
                        generated.astype(np.float32)[mask_bool]
                        - roi.astype(np.float32)[mask_bool]
                    )
                )
            )
        else:
            mean_abs_change = 0.0

        return (
            result,
            generated,
            roi_mask,
            roi_box,
            lama_ms,
            {
                "raw_min": lama_min,
                "raw_max": lama_max,
                "mean_abs_change_in_mask": mean_abs_change,
                "mask_pixels": int(mask_bool.sum()),
            },
        )

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45,
    ):
        out = image.copy()
        m = mask.astype(bool)

        tint = np.zeros_like(out)
        tint[..., 0] = 255
        tint[..., 1] = 60
        tint[..., 2] = 60

        out[m] = (
            alpha * tint[m]
            + (1 - alpha) * out[m]
        ).astype(np.uint8)

        return out

    @staticmethod
    def overlay_box(
        image: np.ndarray,
        box: np.ndarray,
    ):
        out = image.copy()

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (60, 220, 255),
            3,
        )

        return out

    def remove(
        self,
        image: Image.Image | np.ndarray,
        prompt: str,
    ) -> dict[str, Any]:

        image = self.to_rgb(image)

        total_t = time.perf_counter()

        # 1. Text-conditioned detection.
        t0 = time.perf_counter()
        detections = self.ground(image, prompt)
        grounding_ms = (time.perf_counter() - t0) * 1000.0

        best = detections[0]
        box = best["box"]

        # 2. EfficientSAM-S segmentation.
        mask, sam_score, enc_ms, dec_ms = (
            self.efficient_sam.segment_box(
                image,
                box,
            )
        )

        raw_mask = mask.copy()

        # 3. Cleanup + controlled dilation.
        final_mask = self.refine_mask(mask)

        # 4. LaMa.
        (
            result,
            generated_roi,
            roi_mask,
            roi_box,
            lama_ms,
            lama_debug_stats,
        ) = self.inpaint(
            image,
            final_mask,
        )

        total_ms = (time.perf_counter() - total_t) * 1000.0

        # The raw LaMa panel shows what LaMa actually generated inside the ROI.
        lama_debug = image.copy()
        x1, y1, x2, y2 = roi_box
        lama_debug[y1:y2, x1:x2] = generated_roi

        return {
            "original": image,
            "result": result,
            "mask": (final_mask.astype(np.uint8) * 255),
            "raw_mask": (raw_mask.astype(np.uint8) * 255),
            "efficient_sam_mask_overlay": self.overlay_mask(
                image,
                final_mask,
            ),
            "grounding_overlay": self.overlay_box(
                image,
                box,
            ),
            "lama_raw_roi": lama_debug,
            "roi_mask": (roi_mask.astype(np.uint8) * 255),
            "lama_debug_stats": lama_debug_stats,
            "detections": detections,
            "selected_box": [float(x) for x in box],
            "grounding_score": float(best["score"]),
            "efficient_sam_iou": float(sam_score),
            "roi_box": roi_box,
            "latency": {
                "grounding_ms": grounding_ms,
                "efficient_sam_encoder_ms": enc_ms,
                "efficient_sam_decoder_ms": dec_ms,
                "lama_ms": lama_ms,
                "total_ms": total_ms,
            },
            "device": self.device,
        }
