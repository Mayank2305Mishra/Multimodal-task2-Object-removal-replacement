from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parent
MD = ROOT / "model_wts"

mods = ["numpy", "PIL", "cv2", "onnxruntime", "torch", "torchvision", "transformers", "streamlit"]
print("Python:", sys.version)
for name in mods:
    try:
        m = importlib.import_module(name)
        print(f"{name}: OK ({getattr(m, '__version__', 'version n/a')})")
    except Exception as e:
        print(f"{name}: FAIL -> {type(e).__name__}: {e}")

required = [
    MD / "grounding_dino" / "model.safetensors",
    MD / "grounding_dino" / "config.json",
    MD / "grounding_dino" / "preprocessor_config.json",
    MD / "grounding_dino" / "tokenizer.json",
    MD / "efficient_sam" / "efficientsam_s_encoder.onnx",
    MD / "efficient_sam" / "efficientsam_s_decoder.onnx",
    MD / "lama" / "lama_fp32.onnx",
]
print("\nWeights:")
all_ok = True
for p in required:
    ok = p.exists() and p.stat().st_size > 0
    all_ok &= ok
    size = f"{p.stat().st_size / (1024**2):.1f} MB" if ok else "missing"
    print(("OK   " if ok else "MISS ") + f"{p.relative_to(ROOT)} [{size}]")

if all_ok:
    try:
        import onnxruntime as ort
        print("\nONNX Runtime providers:", ort.get_available_providers())
        for p in [
            MD / "efficient_sam" / "efficientsam_s_encoder.onnx",
            MD / "efficient_sam" / "efficientsam_s_decoder.onnx",
            MD / "lama" / "lama_fp32.onnx",
        ]:
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            print(f"ONNX CPU session OK: {p.name}")
            print("  inputs:", [(x.name, x.type, x.shape) for x in sess.get_inputs()])
            print("  outputs:", [(x.name, x.type, x.shape) for x in sess.get_outputs()])
    except Exception as e:
        all_ok = False
        print("ONNX session check failed:", type(e).__name__, e)

raise SystemExit(0 if all_ok else 1)
