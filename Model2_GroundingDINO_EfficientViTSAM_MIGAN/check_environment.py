from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "model_wts"

modules = [
    "numpy",
    "PIL",
    "cv2",
    "onnxruntime",
    "torch",
    "transformers",
    "streamlit",
    "pandas",
]

print("Python environment")
for name in modules:
    print(f"  {'OK' if importlib.util.find_spec(name) else 'MISSING':7} {name}")

files = {
    "Grounding DINO/model.safetensors": MD / "grounding_dino" / "model.safetensors",
    "Grounding DINO/config.json": MD / "grounding_dino" / "config.json",
    "Grounding DINO/preprocessor_config.json": MD / "grounding_dino" / "preprocessor_config.json",
    "Grounding DINO/tokenizer.json": MD / "grounding_dino" / "tokenizer.json",
    "EfficientSAM-S encoder": MD / "efficient_sam" / "efficientsam_s_encoder.onnx",
    "EfficientSAM-S decoder": MD / "efficient_sam" / "efficientsam_s_decoder.onnx",
    "MI-GAN pipeline v2": MD / "migan" / "migan_pipeline_v2.onnx",
}

print("\nWeight files")
for label, path in files.items():
    if path.exists() and path.stat().st_size:
        print(f"  OK      {label:36} {path.stat().st_size / (1024**2):8.1f} MiB")
    else:
        print(f"  MISSING {label:36} {path}")

try:
    import onnxruntime as ort
    print("\nONNX Runtime providers:")
    print("  " + ", ".join(ort.get_available_providers()))
except Exception as exc:
    print(f"\nONNX Runtime check failed: {exc}")

print("\nUse CPUExecutionProvider for this package.")
