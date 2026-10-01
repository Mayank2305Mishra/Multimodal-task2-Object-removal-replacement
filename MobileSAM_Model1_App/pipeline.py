from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"
SAM_CKPT = MODEL_DIR / "mobile_sam.pt"
CLIP_CKPT = MODEL_DIR / "mobileclip_s0.pt"
LAMA_CKPT = MODEL_DIR / "lama_fp32.onnx"


@dataclass
class PipelineConfig:
    sam_points_per_side: int = 10
    sam_points_per_batch: int = 64
    sam_pred_iou_thresh: float = 0.84
    sam_stability_thresh: float = 0.88
    sam_min_mask_area: int = 100
    max_clip_candidates: int = 48
    min_area_ratio: float = 0.003
    max_area_ratio: float = 0.75
    clip_batch_size: int = 32
    refine_min_candidate_recall: float = 0.55
    refine_max_area_growth: float = 4.0
    refine_box_scale: float = 1.28
    refine_padding_min: int = 6
    lama_size: int = 512
    roi_padding: float = 0.35
    mask_dilation: int = 9
    roi_feather: int = 7


def choose_device(name: str = "auto") -> torch.device:
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Model1Pipeline:
    def __init__(self, device: str = "auto", config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.device = choose_device(device)
        # MobileSAM automatic mask generation is kept on CPU on MPS because
        # released MobileSAM builds can hit float64 conversion errors there.
        self.sam_device = torch.device("cpu") if self.device.type == "mps" else self.device
        self.clip_device = self.device
        self.sam_model = None
        self.sam_generator = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.lama_session = None
        self.lama_image_name = None
        self.lama_mask_name = None
        self.lama_output_name = None
        self.load_models()

    def check_weights(self):
        missing = [p.name for p in (SAM_CKPT, CLIP_CKPT, LAMA_CKPT) if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Missing local model weights in model_wts/:\n"
                + "\n".join(f"  - {x}" for x in missing)
                + "\n\nSee model_wts/README.md."
            )

    def load_models(self):
        self.check_weights()

        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

        # Compatibility with PyTorch versions where torch.load defaults to weights_only=True.
        original_load = torch.load
        def patched_load(*args, **kwargs):
            try:
                params = inspect.signature(original_load).parameters
                if "weights_only" in params and "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
            except Exception:
                pass
            return original_load(*args, **kwargs)
        torch.load = patched_load
        try:
            self.sam_model = sam_model_registry["vit_t"](checkpoint=str(SAM_CKPT))
        finally:
            torch.load = original_load

        self.sam_model.to(self.sam_device)
        self.sam_model.eval()
        self.sam_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=self.config.sam_points_per_side,
            points_per_batch=self.config.sam_points_per_batch,
            pred_iou_thresh=self.config.sam_pred_iou_thresh,
            stability_score_thresh=self.config.sam_stability_thresh,
            crop_n_layers=0,
            min_mask_region_area=self.config.sam_min_mask_area,
            output_mode="binary_mask",
        )

        import mobileclip
        from mobileclip.modules.common.mobileone import reparameterize_model
        self.clip_model, _, self.clip_preprocess = mobileclip.create_model_and_transforms(
            "mobileclip_s0", pretrained=str(CLIP_CKPT)
        )
        self.clip_tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
        self.clip_model.eval().to(self.clip_device)
        try:
            self.clip_model = reparameterize_model(self.clip_model)
            self.clip_model.eval().to(self.clip_device)
        except Exception:
            pass

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.lama_session = ort.InferenceSession(
            str(LAMA_CKPT), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.lama_session.get_inputs()
        named = {x.name: x for x in inputs}
        if "image" in named and "mask" in named:
            self.lama_image_name = "image"
            self.lama_mask_name = "mask"
        else:
            for x in inputs:
                shape = list(x.shape)
                if len(shape) != 4:
                    continue
                if shape[1] == 3 and self.lama_image_name is None:
                    self.lama_image_name = x.name
                elif shape[1] == 1 and self.lama_mask_name is None:
                    self.lama_mask_name = x.name
            if self.lama_image_name is None or self.lama_mask_name is None:
                raise RuntimeError(f"Could not identify LaMa image/mask inputs: {[x.name for x in inputs]}")
        self.lama_output_name = self.lama_session.get_outputs()[0].name

    @staticmethod
    def to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
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
        for prefix in (
            "remove the ", "remove ", "delete the ", "delete ",
            "erase the ", "erase ", "a photo of the ", "a photo of a ", "a photo of ",
        ):
            if p.startswith(prefix):
                return p[len(prefix):].strip()
        return p

    @staticmethod
    def mask_bbox(mask: np.ndarray) -> list[int] | None:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.45) -> np.ndarray:
        out = image.copy()
        m = np.asarray(mask, dtype=bool)
        tint = np.zeros_like(out)
        tint[..., 0] = 255
        tint[..., 1] = 60
        tint[..., 2] = 60
        out[m] = (alpha * tint[m] + (1 - alpha) * out[m]).astype(np.uint8)
        return out

    @staticmethod
    def crop_bbox(image: np.ndarray, box: list[int], pad=0.08) -> np.ndarray:
        H, W = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        px, py = int(bw*pad), int(bh*pad)
        x1, y1 = max(0, x1-px), max(0, y1-py)
        x2, y2 = min(W, x2+px), min(H, y2+py)
        return image[y1:y2, x1:x2]

    def masked_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        box = self.mask_bbox(mask)
        if box is None:
            return image
        H, W = image.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2-x1)*0.08), int((y2-y1)*0.08)
        x1p, y1p = max(0, x1-px), max(0, y1-py)
        x2p, y2p = min(W, x2+px), min(H, y2+py)
        crop = image[y1p:y2p, x1p:x2p].copy()
        m = np.asarray(mask, dtype=bool)[y1p:y2p, x1p:x2p]
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=7)
        crop[~m] = blurred[~m]
        return crop

    def generate_candidates(self, image):
        with torch.inference_mode():
            return self.sam_generator.generate(image)

    def filter_candidates(self, masks, shape):
        H, W = shape[:2]
        total = H * W
        out = []
        for m in masks:
            ratio = float(m["area"]) / max(1, total)
            if self.config.min_area_ratio <= ratio <= self.config.max_area_ratio:
                out.append(m)
        out.sort(
            key=lambda m: float(m.get("predicted_iou", 0.0)) * float(m.get("stability_score", 0.0)),
            reverse=True,
        )
        return out[: self.config.max_clip_candidates]

    def score_candidates(self, image, candidates, prompt):
        prompts = [
            f"a photo of a {prompt}",
            f"a photo containing a {prompt}",
            f"a close view of a {prompt}",
        ]
        tokens = self.clip_tokenizer(prompts).to(self.clip_device)
        with torch.inference_mode():
            text = self.clip_model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)

        box_imgs, masked_imgs = [], []
        for c in candidates:
            mask = np.asarray(c["segmentation"], dtype=bool)
            box = self.mask_bbox(mask)
            if box is None:
                x, y, w, h = c["bbox"]
                box = [int(x), int(y), int(x+w), int(y+h)]
            box_imgs.append(Image.fromarray(self.crop_bbox(image, box)))
            masked_imgs.append(Image.fromarray(self.masked_crop(image, mask)))

        def encode(images):
            chunks = []
            for start in range(0, len(images), self.config.clip_batch_size):
                batch = torch.stack([self.clip_preprocess(x) for x in images[start:start+self.config.clip_batch_size]]).to(self.clip_device)
                with torch.inference_mode():
                    feat = self.clip_model.encode_image(batch)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                chunks.append(feat.detach())
            return torch.cat(chunks, dim=0)

        box_feat = encode(box_imgs)
        masked_feat = encode(masked_imgs)
        return (
            0.72 * (box_feat @ text.T).mean(dim=1)
            + 0.28 * (masked_feat @ text.T).mean(dim=1)
        ).detach().float().cpu().numpy()

    def refine(self, image, candidate):
        H, W = image.shape[:2]
        initial = np.asarray(candidate["segmentation"], dtype=bool)
        ys, xs = np.where(initial)
        if len(xs) == 0:
            raise RuntimeError("Selected candidate is empty.")
        px, py = int(np.median(xs)), int(np.median(ys))
        x1 = int(candidate["bbox"][0])
        y1 = int(candidate["bbox"][1])
        x2 = int(candidate["bbox"][0] + candidate["bbox"][2])
        y2 = int(candidate["bbox"][1] + candidate["bbox"][3])
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        padx = max(self.config.refine_padding_min, int((self.config.refine_box_scale-1)*bw/2))
        pady = max(self.config.refine_padding_min, int((self.config.refine_box_scale-1)*bh/2))
        box = np.array([
            max(0, x1-padx), max(0, y1-pady),
            min(W-1, x2+padx), min(H-1, y2+pady)
        ], dtype=np.float32)
        predictor = self.sam_generator.predictor
        predictor.set_image(image)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[px, py]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                box=box,
                multimask_output=True,
            )
        initial_area = max(1, int(initial.sum()))
        ranked = []
        for idx, mask in enumerate(masks):
            mm = np.asarray(mask, dtype=bool)
            area = int(mm.sum())
            if not area:
                continue
            inter = int(np.logical_and(mm, initial).sum())
            union = int(np.logical_or(mm, initial).sum())
            recall = inter / initial_area
            iou = inter / max(1, union)
            growth = area / initial_area
            if recall < self.config.refine_min_candidate_recall or growth > self.config.refine_max_area_growth:
                continue
            score = 0.45*float(scores[idx]) + 0.35*recall + 0.25*iou - 0.10*max(0.0, growth-2.0)
            ranked.append((score, idx))
        if not ranked:
            return initial.astype(np.uint8)*255, 0.0
        _, best = max(ranked, key=lambda x: x[0])
        return masks[best].astype(np.uint8)*255, float(scores[best])

    def make_roi(self, image, mask):
        H, W = image.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask.")
        x1, x2 = int(xs.min()), int(xs.max()+1)
        y1, y2 = int(ys.min()), int(ys.max()+1)
        bw, bh = x2-x1, y2-y1
        padx = max(8, int(bw*self.config.roi_padding))
        pady = max(8, int(bh*self.config.roi_padding))
        x1, y1, x2, y2 = x1-padx, y1-pady, x2+padx, y2+pady
        side = max(x2-x1, y2-y1, 64)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        x1, y1 = int(round(cx-side/2)), int(round(cy-side/2))
        x2, y2 = x1+side, y1+side
        if x1 < 0: x2, x1 = x2-x1, 0
        if y1 < 0: y2, y1 = y2-y1, 0
        if x2 > W: x1, x2 = x1-(x2-W), W
        if y2 > H: y1, y2 = y1-(y2-H), H
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1,y1,x2,y2)

    def inpaint(self, image, mask):
        roi, roi_mask, roi_box = self.make_roi(image, mask)
        size = self.config.lama_size
        im = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
        mk = cv2.resize(roi_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        image_input = (im.astype(np.float32)/255.0).transpose(2,0,1)[None]
        mask_input = ((mk > 127).astype(np.float32))[None,None]
        out = self.lama_session.run(
            [self.lama_output_name],
            {self.lama_image_name:image_input, self.lama_mask_name:mask_input},
        )[0]
        out = np.asarray(out)[0].transpose(1,2,0)
        if float(out.min()) < -0.5 and float(out.max()) <= 1.5:
            out = (out + 1.0) * 0.5 * 255.0
        elif float(out.max()) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
        feather = max(1, int(self.config.roi_feather))
        if feather % 2 == 0: feather += 1
        alpha = cv2.GaussianBlur((roi_mask > 0).astype(np.float32), (feather, feather), 0)[...,None]
        blended = roi.astype(np.float32)*(1-alpha) + out.astype(np.float32)*alpha
        result = image.copy()
        x1,y1,x2,y2 = roi_box
        result[y1:y2,x1:x2] = np.clip(blended,0,255).astype(np.uint8)
        return result, roi_box

    def remove(self, image: Image.Image | np.ndarray, prompt: str) -> dict[str, Any]:
        image = self.to_rgb(image)
        clean = self.clean_prompt(prompt)
        if not clean:
            raise ValueError("Prompt is empty.")
        total_t = time.perf_counter()

        t = time.perf_counter()
        all_masks = self.generate_candidates(image)
        sam_proposal_ms = (time.perf_counter()-t)*1000
        candidates = self.filter_candidates(all_masks, image.shape)
        if not candidates:
            raise RuntimeError("No usable MobileSAM candidates.")

        t = time.perf_counter()
        clip_scores = self.score_candidates(image, candidates, clean)
        clip_ms = (time.perf_counter()-t)*1000

        H,W = image.shape[:2]
        combined = []
        for i,c in enumerate(candidates):
            ratio = float(c["area"])/max(1,H*W)
            sam_quality = float(c.get("predicted_iou",0))*float(c.get("stability_score",0))
            penalty = max(0.0, ratio-0.30)*0.30
            combined.append(float(clip_scores[i]) + 0.05*sam_quality - penalty)
        best_idx = int(np.argmax(combined))
        selected = candidates[best_idx]

        t = time.perf_counter()
        refined_mask, refine_score = self.refine(image, selected)
        sam_refine_ms = (time.perf_counter()-t)*1000

        k = max(1, int(self.config.mask_dilation))
        if k % 2 == 0: k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)

        t = time.perf_counter()
        result, roi_box = self.inpaint(image, refined_mask)
        lama_ms = (time.perf_counter()-t)*1000
        total_ms = (time.perf_counter()-total_t)*1000

        return {
            "original": image,
            "prompt": prompt,
            "mask": refined_mask,
            "overlay": self.overlay_mask(image, refined_mask),
            "result": result,
            "selected_candidate_mask": np.asarray(selected["segmentation"],dtype=np.uint8)*255,
            "selected_candidate": {
                "bbox": [float(v) for v in selected["bbox"]],
                "predicted_iou": float(selected.get("predicted_iou",0.0)),
                "stability_score": float(selected.get("stability_score",0.0)),
            },
            "proposal_count": len(all_masks),
            "clip_candidate_count": len(candidates),
            "clip_score": float(clip_scores[best_idx]),
            "combined_score": float(combined[best_idx]),
            "refine_score": float(refine_score),
            "roi_box": roi_box,
            "latency": {
                "mobile_sam_proposal_ms": sam_proposal_ms,
                "mobileclip_ms": clip_ms,
                "mobile_sam_refine_ms": sam_refine_ms,
                "lama_ms": lama_ms,
                "total_ms": total_ms,
            },
            "device": str(self.device),
            "sam_device": str(self.sam_device),
            "clip_device": str(self.clip_device),
        }
