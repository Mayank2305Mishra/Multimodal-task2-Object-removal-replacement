from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "model_wts"

print("Model 2 smoke test")

from pipeline import EFFICIENTSAM_DECODER, EFFICIENTSAM_ENCODER, MIGAN_WEIGHT  # noqa: E402
import onnxruntime as ort  # noqa: E402

for name, path in [
    ("EfficientSAM-S encoder", EFFICIENTSAM_ENCODER),
    ("EfficientSAM-S decoder", EFFICIENTSAM_DECODER),
    ("MI-GAN pipeline v2", MIGAN_WEIGHT),
]:
    if not path.exists():
        print(f"SKIP    {name}: missing {path}")
        continue
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"OK      {name}")
    print("        inputs:", [x.name for x in session.get_inputs()])
    print("        outputs:", [x.name for x in session.get_outputs()])

print("Smoke test finished.")
