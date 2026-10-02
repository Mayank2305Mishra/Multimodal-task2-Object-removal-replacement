from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "model_wts")).resolve()
DEVICE_NAME = os.getenv("DEVICE", "cpu").lower()
SAM_RESIZE = int(os.getenv("SAM_RESIZE", "640"))
MAX_CLIP_CANDIDATES = int(os.getenv("MAX_CLIP_CANDIDATES", "32"))
DILATE_KERNEL = int(os.getenv("DILATE_KERNEL", "15"))


def pick_weight(candidates: list[str], label: str) -> Path:
    for name in candidates:
        p = MODEL_DIR / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    tried = ", ".join(candidates)
    raise FileNotFoundError(
        f"Missing {label} checkpoint in {MODEL_DIR}. Tried: {tried}. "
        f"Put the weight file in ./model_wts or set MODEL_DIR."
    )


SAM_CKPT = pick_weight(
    ["mobile_sam.pt", "MobileSAM.pt", "mobile_sam_vit_t.pt"], "MobileSAM"
)
MOBILECLIP_CKPT = pick_weight(
    ["mobileclip_s0.pt", "MobileCLIP-S0.pt", "mobileclip_s0.pth"], "MobileCLIP-S0"
)
LAMA_CKPT = pick_weight(
    ["big-lama.pt", "big_lama.pt", "lama.pt"], "LaMa"
)


@dataclass
class ModelBundle:
    sam: Any
    mask_generator: Any
    clip_model: Any
    clip_tokenizer: Any
    clip_preprocess: Any
    lama: Any
    device: torch.device


def resolve_device() -> torch.device:
    if DEVICE_NAME == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if DEVICE_NAME == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_cpu() -> None:
    if DEVICE_NAME == "cpu":
        threads = max(1, min(os.cpu_count() or 1, 8))
        torch.set_num_threads(threads)


def normalize_prompt(prompt: str) -> str:
    p = re.sub(r"\s+", " ", prompt.strip().lower())
    fillers = (
        "remove the ", "remove ", "delete the ", "delete ",
        "erase the ", "erase ", "take out the ", "take out ",
        "get rid of the ", "get rid of ",
    )
    for filler in fillers:
        if p.startswith(filler):
            p = p[len(filler):].strip()
            break
    return p or prompt.strip()


def spatial_weight(bbox: list[float] | tuple[float, ...], img_w: int, img_h: int, prompt: str) -> float:
    x, y, w, h = bbox[:4]
    cx = (x + w / 2.0) / max(img_w, 1)
    cy = (y + h / 2.0) / max(img_h, 1)
    p = prompt.lower()
    score = 1.0
    if "left" in p:
        score *= 1.0 if cx <= 0.60 else 0.10
    if "right" in p:
        score *= 1.0 if cx >= 0.40 else 0.10
    if "center" in p or "middle" in p:
        score *= 1.0 if 0.30 <= cx <= 0.70 else 0.35
    if "top" in p:
        score *= 1.0 if cy <= 0.60 else 0.30
    if "bottom" in p:
        score *= 1.0 if cy >= 0.40 else 0.30
    return score


def mask_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    overlay = img.copy()
    m = mask > 0
    color = np.zeros_like(img)
    color[..., 0] = 255
    color[..., 1] = 80
    color[..., 2] = 80
    overlay[m] = (overlay[m] * (1.0 - alpha) + color[m] * alpha).astype(np.uint8)
    return overlay


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, k, iterations=1)


def load_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)


def _validate_runtime() -> None:
    # Import after dependency checks so the UI can provide a useful error if a clean env was not used.
    try:
        import torchvision  # noqa: F401
        import timm  # noqa: F401
        import mobile_sam  # noqa: F401
        import mobileclip  # noqa: F401
        import simple_lama_inpainting  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Model dependencies failed to import. Use the supplied requirements.txt in a fresh virtual environment. "
            f"Original error: {exc}"
        ) from exc


_validate_runtime()
setup_cpu()
DEVICE = resolve_device()

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry  # noqa: E402
import mobileclip  # noqa: E402
from mobileclip.modules.common.mobileone import reparameterize_model  # noqa: E402

os.environ["LAMA_MODEL"] = str(LAMA_CKPT)
from simple_lama_inpainting import SimpleLama  # noqa: E402


_LOAD_CACHE: ModelBundle | None = None
_LOAD_TIMES: dict[str, float] = {}


def load_models() -> ModelBundle:
    global _LOAD_CACHE
    if _LOAD_CACHE is not None:
        return _LOAD_CACHE

    t0 = time.perf_counter()
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_CKPT))
    sam.to(device=DEVICE)
    sam.eval()
    _LOAD_TIMES["MobileSAM load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        min_mask_region_area=100,
    )
    _LOAD_TIMES["SAM generator init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clip_model, _, clip_preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=str(MOBILECLIP_CKPT)
    )
    clip_tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    clip_model = clip_model.to(DEVICE).eval()
    try:
        clip_model = reparameterize_model(clip_model)
    except Exception:
        pass
    _LOAD_TIMES["MobileCLIP load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lama = SimpleLama(device=DEVICE)
    _LOAD_TIMES["LaMa load"] = time.perf_counter() - t0

    _LOAD_CACHE = ModelBundle(
        sam=sam,
        mask_generator=mask_generator,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        clip_preprocess=clip_preprocess,
        lama=lama,
        device=DEVICE,
    )
    return _LOAD_CACHE


@torch.inference_mode()
def run_text_object_removal(image: Image.Image | np.ndarray, prompt: str) -> dict[str, Any]:
    models = load_models()
    img = load_rgb(image)
    clean_prompt = normalize_prompt(prompt)
    orig_h, orig_w = img.shape[:2]

    scale = min(1.0, SAM_RESIZE / max(orig_h, orig_w))
    small_w = max(1, int(orig_w * scale))
    small_h = max(1, int(orig_h * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Stage 1: MobileSAM proposals.
    t0 = time.perf_counter()
    proposals = models.mask_generator.generate(small)
    sam_latency = time.perf_counter() - t0
    total_area = small_h * small_w

    candidates: list[dict[str, Any]] = []
    for m in proposals:
        area_ratio = float(m["area"]) / max(total_area, 1)
        if area_ratio < 0.003 or area_ratio > 0.75:
            continue
        sp = spatial_weight(m["bbox"], small_w, small_h, clean_prompt)
        if sp < 0.20:
            continue
        m = dict(m)
        m["spatial_weight"] = sp
        candidates.append(m)

    candidates.sort(key=lambda x: (x.get("stability_score", 0.0), x["area"]), reverse=True)
    candidates = candidates[:MAX_CLIP_CANDIDATES]
    if not candidates:
        raise RuntimeError("MobileSAM produced no usable candidate regions for this image/prompt.")

    # Stage 2: MobileCLIP scores each candidate against the text prompt.
    crops: list[torch.Tensor] = []
    valid_candidates: list[dict[str, Any]] = []
    for m in candidates:
        x, y, w, h = [int(v) for v in m["bbox"]]
        x2, y2 = min(small_w, x + max(w, 1)), min(small_h, y + max(h, 1))
        x, y = max(0, x), max(0, y)
        crop = small[y:y2, x:x2].copy()
        seg = m["segmentation"][y:y2, x:x2]
        crop[~seg] = 0
        crops.append(models.clip_preprocess(Image.fromarray(crop)))
        valid_candidates.append(m)

    t0 = time.perf_counter()
    image_batch = torch.stack(crops).to(models.device)
    text_tokens = models.clip_tokenizer([f"a photo of {clean_prompt}"]).to(models.device)
    image_features = models.clip_model.encode_image(image_batch)
    text_features = models.clip_model.encode_text(text_tokens)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarities = (100.0 * image_features @ text_features.T).squeeze(1).cpu().numpy()
    clip_latency = time.perf_counter() - t0

    final_scores = []
    for i, m in enumerate(valid_candidates):
        area_ratio = float(m["area"]) / max(total_area, 1)
        size_bonus = min(area_ratio, 0.30)
        final_scores.append(float(similarities[i]) * float(m["spatial_weight"]) + size_bonus)

    best_idx = int(np.argmax(final_scores))
    best = valid_candidates[best_idx]
    small_mask = best["segmentation"].astype(np.uint8) * 255
    full_mask = cv2.resize(small_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    clean_mask = dilate_mask(full_mask, DILATE_KERNEL)

    # Stage 3: LaMa inpainting.
    t0 = time.perf_counter()
    result = models.lama(Image.fromarray(img), Image.fromarray(clean_mask))
    lama_latency = time.perf_counter() - t0
    result_np = np.asarray(result.convert("RGB"), dtype=np.uint8)

    # MobileCLIP inspection image: masked crop of selected candidate.
    bx, by, bw, bh = [int(v) for v in best["bbox"]]
    bx2, by2 = min(small_w, bx + max(bw, 1)), min(small_h, by + max(bh, 1))
    bx, by = max(0, bx), max(0, by)
    selected_crop = small[by:by2, bx:bx2].copy()
    selected_crop[~best["segmentation"][by:by2, bx:bx2]] = 0
    selected_crop = cv2.resize(selected_crop, (max(1, selected_crop.shape[1] * 2), max(1, selected_crop.shape[0] * 2)), interpolation=cv2.INTER_NEAREST)

    total_latency = sam_latency + clip_latency + lama_latency
    return {
        "original": img,
        "sam_overlay": mask_overlay(img, clean_mask),
        "mask": clean_mask,
        "mobileclip_crop": selected_crop,
        "result": result_np,
        "timing": {
            "MobileSAM proposals (s)": sam_latency,
            "MobileCLIP scoring (s)": clip_latency,
            "LaMa inpainting (s)": lama_latency,
            "Total inference (s)": total_latency,
        },
        "metadata": {
            "prompt": prompt,
            "normalized_prompt": clean_prompt,
            "candidates": len(valid_candidates),
            "clip_similarity": float(similarities[best_idx]),
            "final_score": float(final_scores[best_idx]),
            "device": str(DEVICE),
        },
    }


def warmup() -> None:
    load_models()


def model_status() -> dict[str, str]:
    return {
        "device": str(DEVICE),
        "mobile_sam": str(SAM_CKPT),
        "mobileclip": str(MOBILECLIP_CKPT),
        "lama": str(LAMA_CKPT),
    }
