from __future__ import annotations

import gc
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"
GROUNDING_DIR = MODEL_DIR / "grounding_dino"
EFFICIENTSAM_ENCODER = MODEL_DIR / "efficient_sam" / "encoder.onnx"
EFFICIENTSAM_DECODER = MODEL_DIR / "efficient_sam" / "decoder.onnx"
POWERPAINT_DIR = MODEL_DIR / "powerpaint_v2_1"
POWERPAINT_REPO = ROOT / "powerpaint"


@dataclass
class PipelineConfig:
    grounding_box_threshold: float = 0.28
    grounding_text_threshold: float = 0.22
    max_detections: int = 5
    close_kernel: int = 5
    dilation_kernel: int = 17
    dilation_iterations: int = 1
    min_component_area: int = 80
    roi_padding: float = 0.65
    roi_min_padding: int = 64
    pp_max_side: int = 768
    pp_steps: int = 30
    pp_guidance: float = 12.0
    pp_fitting_degree: float = 1.0
    seed: int = 42


def _session_options(cfg: PipelineConfig):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


class EfficientSAMONNX:
    def __init__(self, encoder_path: Path, decoder_path: Path, cfg: PipelineConfig):
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError("Missing EfficientSAM-S weights. Run ./download_wts.sh first.")
        opts = _session_options(cfg)
        self.encoder = ort.InferenceSession(str(encoder_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.encoder_input = self.encoder.get_inputs()[0].name
        required = {"image_embeddings", "batched_point_coords", "batched_point_labels", "orig_im_size"}
        found = {x.name for x in self.decoder.get_inputs()}
        if not required.issubset(found):
            raise RuntimeError(f"Wrong EfficientSAM decoder inputs. Found {sorted(found)}")

    def encode(self, image: np.ndarray):
        x = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.encoder.run(None, {self.encoder_input: x})[0]

    def segment_box(self, image: np.ndarray, box):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(float, box)
        x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
        x2, y2 = np.clip(x2, x1 + 1, w), np.clip(y2, y1 + 1, h)
        points = np.array([[[[x1, y1], [x2, y2]]]], dtype=np.float32)
        labels = np.array([[[2.0, 3.0]]], dtype=np.float32)
        t = time.perf_counter()
        embedding = self.encode(image)
        enc_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        out = self.decoder.run(None, {
            "image_embeddings": embedding,
            "batched_point_coords": points,
            "batched_point_labels": labels,
            "orig_im_size": np.array([h, w], dtype=np.int64),
        })
        dec_ms = (time.perf_counter() - t) * 1000
        logits = np.asarray(out[0]).reshape(-1, out[0].shape[-2], out[0].shape[-1])
        iou = np.asarray(out[1]).reshape(-1)
        idx = int(np.argmax(iou))
        mask = logits[idx] >= 0
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask, float(iou[idx]), enc_ms, dec_ms


class PowerPaintRemoval:
    """Official PowerPaint v2-1 BrushNet object-removal path.

    The official PowerPaint implementation treats object removal as a dedicated
    task and uses the learned P_ctxt/P_obj task tokens. This wrapper preserves
    that path rather than using a generic DiffusionPipeline call.
    """

    def __init__(self, checkpoint_dir: Path, repo_dir: Path, device: str):
        self.checkpoint_dir = checkpoint_dir
        self.repo_dir = repo_dir
        self.device = device
        if not repo_dir.exists():
            raise FileNotFoundError("PowerPaint source is missing. Run ./download_wts.sh")
        sys.path.insert(0, str(repo_dir))
        from diffusers import UniPCMultistepScheduler
        from transformers import CLIPTextModel
        from safetensors.torch import load_model
        from powerpaint.models.BrushNet_CA import BrushNetModel
        from powerpaint.models.unet_2d_condition import UNet2DConditionModel
        from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
        from powerpaint.utils.utils import TokenizerWrapper, add_tokens

        self._load_model = load_model
        self._UNet = UNet2DConditionModel
        self._BrushNet = BrushNetModel
        self._Pipeline = StableDiffusionPowerPaintBrushNetPipeline
        self._TokenizerWrapper = TokenizerWrapper
        self._add_tokens = add_tokens
        self._CLIPTextModel = CLIPTextModel
        self._UniPC = UniPCMultistepScheduler

        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        base = checkpoint_dir / "realisticVisionV60B1_v51VAE"
        brush = checkpoint_dir / "PowerPaint_Brushnet"
        if not (base / "unet").exists() or not brush.exists():
            raise FileNotFoundError(f"PowerPaint v2-1 checkpoint layout not found under {checkpoint_dir}")

        unet = self._UNet.from_pretrained(str(base), subfolder="unet", torch_dtype=dtype, local_files_only=True)
        text_encoder_brushnet = self._CLIPTextModel.from_pretrained(str(base), subfolder="text_encoder", torch_dtype=dtype, local_files_only=True)
        brushnet = self._BrushNet.from_unet(unet)

        self.pipe = self._Pipeline.from_pretrained(
            str(base), brushnet=brushnet, text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=dtype, low_cpu_mem_usage=False, safety_checker=None,
        )
        self.pipe.unet = self._UNet.from_pretrained(str(base), subfolder="unet", torch_dtype=dtype, local_files_only=True)
        self.pipe.tokenizer = self._TokenizerWrapper(
            from_pretrained=str(base), subfolder="tokenizer", revision=None,
            torch_type=dtype, local_files_only=True,
        )
        self._add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        self._load_model(self.pipe.brushnet, str(brush / "diffusion_pytorch_model.safetensors"))
        self.pipe.text_encoder_brushnet.load_state_dict(torch.load(str(brush / "pytorch_model.bin"), map_location="cpu"), strict=False)
        self.pipe.scheduler = self._UniPC.from_config(self.pipe.scheduler.config)

        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)

    @staticmethod
    def _task_prompts():
        # Matches official PowerPaint v2 object-removal prompt construction.
        prompt = "empty scene blur"
        negative = ""
        prompt_a = prompt + " P_ctxt"
        prompt_b = prompt + " P_ctxt"
        negative_a = negative + " P_obj"
        negative_b = negative + " P_obj"
        return prompt_a, prompt_b, negative_a, negative_b

    def __call__(self, image: Image.Image, mask: Image.Image, cfg: PipelineConfig):
        image = image.convert("RGB")
        mask = mask.convert("RGB")
        ow, oh = image.size
        scale = min(cfg.pp_max_side / max(ow, oh), 1.0)
        nw = max(64, int(round(ow * scale / 8) * 8))
        nh = max(64, int(round(oh * scale / 8) * 8))
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        mask = mask.resize((nw, nh), Image.Resampling.NEAREST)

        img_np = np.asarray(image).astype(np.uint8)
        mask_np = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
        # Official BrushNet path zeroes the masked region before inference.
        conditioned = img_np * (1.0 - mask_np[..., None])
        conditioned = Image.fromarray(conditioned.astype(np.uint8)).convert("RGB")

        prompt_a, prompt_b, neg_a, neg_b = self._task_prompts()
        seed = int(cfg.seed)
        if self.device == "cuda":
            generator = torch.Generator("cuda").manual_seed(seed)
        elif self.device == "mps":
            generator = torch.Generator(device="mps").manual_seed(seed)
        else:
            generator = torch.Generator("cpu").manual_seed(seed)

        t = time.perf_counter()
        result = self.pipe(
            promptA=prompt_a, promptB=prompt_b, promptU="empty scene blur",
            tradoff=cfg.pp_fitting_degree, tradoff_nag=cfg.pp_fitting_degree,
            image=conditioned, mask=mask,
            num_inference_steps=cfg.pp_steps,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            negative_promptA=neg_a, negative_promptB=neg_b, negative_promptU="",
            guidance_scale=cfg.pp_guidance, width=nw, height=nh,
        ).images[0]
        latency = (time.perf_counter() - t) * 1000

        result = result.resize((ow, oh), Image.Resampling.LANCZOS)
        # Only composite generated pixels where the actual mask exists.
        alpha = mask.resize((ow, oh), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=2))
        alpha_np = np.asarray(alpha).astype(np.float32) / 255.0
        src = np.asarray(image.resize((ow, oh), Image.Resampling.LANCZOS)).astype(np.float32)
        gen = np.asarray(result).astype(np.float32)
        out = gen * alpha_np[..., None] + src * (1.0 - alpha_np[..., None])
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), result, latency, (nw, nh)


class Model1Pipeline:
    """Grounding DINO-Tiny -> EfficientSAM-S -> PowerPaint v2-1 object removal."""

    def __init__(self, device: str = "auto", config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.device = self._select_device(device)
        self._check_weights()
        self._load_grounding()
        self.efficient_sam = EfficientSAMONNX(EFFICIENTSAM_ENCODER, EFFICIENTSAM_DECODER, self.config)
        self.powerpaint = None

    @staticmethod
    def _select_device(device):
        d = str(device).lower()
        if d == "auto":
            if torch.cuda.is_available(): return "cuda"
            if torch.backends.mps.is_available(): return "mps"
            return "cpu"
        if d == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        if d == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
        return d

    def _check_weights(self):
        required = [
            GROUNDING_DIR / "config.json",
            GROUNDING_DIR / "model.safetensors",
            EFFICIENTSAM_ENCODER,
            EFFICIENTSAM_DECODER,
            POWERPAINT_DIR / "realisticVisionV60B1_v51VAE" / "unet" / "config.json",
            POWERPAINT_DIR / "realisticVisionV60B1_v51VAE" / "unet" / "diffusion_pytorch_model.safetensors",
            POWERPAINT_DIR / "realisticVisionV60B1_v51VAE" / "vae" / "config.json",
            POWERPAINT_DIR / "realisticVisionV60B1_v51VAE" / "text_encoder" / "config.json",
            POWERPAINT_DIR / "realisticVisionV60B1_v51VAE" / "tokenizer" / "tokenizer_config.json",
            POWERPAINT_DIR / "PowerPaint_Brushnet" / "diffusion_pytorch_model.safetensors",
            POWERPAINT_DIR / "PowerPaint_Brushnet" / "pytorch_model.bin",
            POWERPAINT_REPO / "powerpaint" / "models" / "BrushNet_CA.py",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing model files:\n" + "\n".join(missing) + "\nRun ./download_wts.sh")

    def _load_grounding(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.ground_processor = AutoProcessor.from_pretrained(str(GROUNDING_DIR), local_files_only=True)
        self.ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(str(GROUNDING_DIR), local_files_only=True).to("cpu")
        self.ground_model.eval()

    def ground(self, image: Image.Image, prompt: str):
        import torch
        text = prompt.strip().lower().rstrip(".")
        if not text: raise ValueError("Enter an object prompt, e.g. 'bottle'.")
        if not text.endswith("."): text += "."
        inputs = self.ground_processor(images=image, text=text, return_tensors="pt")
        with torch.no_grad(): outputs = self.ground_model(**inputs)
        try:
            target_sizes = [image.size[::-1]]
            results = self.ground_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=self.config.grounding_box_threshold, text_threshold=self.config.grounding_text_threshold, target_sizes=target_sizes)[0]
        except TypeError:
            results = self.ground_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=self.config.grounding_box_threshold, text_threshold=self.config.grounding_text_threshold, target_sizes=target_sizes)[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("labels", [text] * len(boxes))
        order = np.argsort(-scores)[: self.config.max_detections]
        return [{"box": boxes[i].astype(float), "score": float(scores[i]), "label": str(labels[i])} for i in order]

    def _refine_mask(self, mask: np.ndarray):
        m = (mask.astype(np.uint8) * 255)
        k = self.config.close_kernel
        kernel = np.ones((k, k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = 1 + int(np.argmax(areas))
            if stats[idx, cv2.CC_STAT_AREA] >= self.config.min_component_area:
                m = np.where(labels == idx, 255, 0).astype(np.uint8)
        dk = self.config.dilation_kernel
        if dk > 1:
            m = cv2.dilate(m, np.ones((dk, dk), np.uint8), iterations=self.config.dilation_iterations)
        return m > 0

    def _roi(self, image: Image.Image, mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0: raise ValueError("Segmentation mask is empty")
        h, w = mask.shape
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        bw, bh = x2 - x1, y2 - y1
        pad = max(self.config.roi_min_padding, int(max(bw, bh) * self.config.roi_padding))
        rx1, ry1 = max(0, x1-pad), max(0, y1-pad)
        rx2, ry2 = min(w, x2+pad), min(h, y2+pad)
        crop = image.crop((rx1, ry1, rx2, ry2))
        crop_mask = Image.fromarray((mask[ry1:ry2, rx1:rx2].astype(np.uint8) * 255))
        return crop, crop_mask, (rx1, ry1, rx2, ry2)

    def _load_powerpaint(self):
        if self.powerpaint is None:
            self.powerpaint = PowerPaintRemoval(POWERPAINT_DIR, POWERPAINT_REPO, self.device)

    def _free_segmentation_models(self):
        # Release Grounding DINO before the diffusion model is loaded on 16 GB Macs.
        self.ground_model = None
        self.ground_processor = None
        gc.collect()
        if self.device == "mps": torch.mps.empty_cache()

    def remove(self, image: Image.Image, prompt: str):
        total_t = time.perf_counter()
        image = image.convert("RGB")
        t = time.perf_counter(); detections = self.ground(image, prompt); grounding_ms = (time.perf_counter()-t)*1000
        if not detections: raise RuntimeError(f"Grounding DINO found no object for '{prompt}'.")
        selected = detections[0]
        np_img = np.asarray(image)
        mask, sam_iou, sam_enc_ms, sam_dec_ms = self.efficient_sam.segment_box(np_img, selected["box"])
        mask = self._refine_mask(mask)
        crop, crop_mask, roi_box = self._roi(image, mask)
        self._load_powerpaint()
        result_crop, raw_crop, pp_ms, pp_size = self.powerpaint(crop, crop_mask, self.config)

        result = image.copy()
        result.paste(result_crop, (roi_box[0], roi_box[1]))
        total_ms = (time.perf_counter()-total_t)*1000

        mask_overlay = np_img.copy(); mask_overlay[mask] = (0.65*mask_overlay[mask] + 0.35*np.array([255,0,0])).astype(np.uint8)
        return {
            "original": image,
            "result": result,
            "mask": Image.fromarray((mask.astype(np.uint8)*255)),
            "mask_overlay": Image.fromarray(mask_overlay),
            "raw_powerpaint_roi": raw_crop,
            "roi_mask": crop_mask,
            "selected_box": selected["box"].tolist(),
            "grounding_score": selected["score"],
            "efficient_sam_iou": sam_iou,
            "roi_box": roi_box,
            "powerpaint_size": pp_size,
            "latency": {"grounding_ms": grounding_ms, "efficient_sam_encoder_ms": sam_enc_ms, "efficient_sam_decoder_ms": sam_dec_ms, "powerpaint_ms": pp_ms, "total_ms": total_ms},
            "device": self.device,
        }
