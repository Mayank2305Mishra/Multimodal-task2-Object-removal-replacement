from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parent
checks = [
ROOT / "model_wts/grounding_dino/model.safetensors",
ROOT / "model_wts/efficient_sam/encoder.onnx",
ROOT / "model_wts/efficient_sam/decoder.onnx",
ROOT / "model_wts/powerpaint_v2_1/realisticVisionV60B1_v51VAE/unet/config.json",
ROOT / "model_wts/powerpaint_v2_1/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model.safetensors",
ROOT / "model_wts/powerpaint_v2_1/realisticVisionV60B1_v51VAE/vae/config.json",
ROOT / "model_wts/powerpaint_v2_1/realisticVisionV60B1_v51VAE/text_encoder/config.json",
ROOT / "model_wts/powerpaint_v2_1/realisticVisionV60B1_v51VAE/tokenizer/tokenizer_config.json",
ROOT / "model_wts/powerpaint_v2_1/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
ROOT / "model_wts/powerpaint_v2_1/PowerPaint_Brushnet/pytorch_model.bin",
ROOT / "powerpaint/powerpaint/models/BrushNet_CA.py",
]
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
missing=[]
for p in checks:
    ok=p.is_file() or p.is_dir(); print("[OK]" if ok else "[MISSING]", p); missing += [] if ok else [p]
if missing: raise SystemExit("\nWeights/source incomplete. Run ./download_wts.sh again.")
print("Environment check passed. Safe to start Streamlit.")
