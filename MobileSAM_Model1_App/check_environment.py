import sys
from pathlib import Path

print("Python:", sys.version)
modules = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("mobile_sam", "mobile_sam"),
    ("mobileclip", "mobileclip"),
    ("onnxruntime", "onnxruntime"),
    ("streamlit", "streamlit"),
    ("cv2", "cv2"),
    ("PIL", "PIL"),
]

for label, name in modules:
    try:
        mod = __import__(name)
        print(f"{label:14s}: OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"{label:14s}: ERROR {type(exc).__name__}: {exc}")

root = Path(__file__).resolve().parent / "model_wts"
for filename in ("mobile_sam.pt", "mobileclip_s0.pt", "lama_fp32.onnx"):
    p = root / filename
    status = p.exists() and p.is_file() and p.stat().st_size > 0
    print(f"weight {filename:18s}:", "FOUND" if status else "MISSING")

try:
    import torch
    print("MPS:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print("CUDA:", torch.cuda.is_available())
except Exception:
    pass
