from pathlib import Path

ROOT = Path(__file__).resolve().parent
paths = [
    ROOT / "model_wts/grounding_dino/config.json",
    ROOT / "model_wts/grounding_dino/model.safetensors",
    ROOT / "model_wts/efficient_sam/encoder.onnx",
    ROOT / "model_wts/efficient_sam/decoder.onnx",
    ROOT / "model_wts/lama_fp32.onnx",
]

failed = False

for p in paths:
    ok = p.exists() and p.stat().st_size > 0
    print(("OK   " if ok else "MISS "), p)
    failed |= not ok

if failed:
    raise SystemExit("\nSome weights are missing. Run: bash download_wts.sh")

print("\nEnvironment weight check passed.")
