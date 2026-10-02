from pathlib import Path
import numpy as np
from PIL import Image

from pipeline import Config, Model2Pipeline

root = Path(__file__).resolve().parent
print("Loading pipeline...")
pipeline = Model2Pipeline(Config())

# 512x512 synthetic image is only for plumbing validation; it is not a quality test.
img = np.zeros((512, 512, 3), dtype=np.uint8)
img[160:350, 180:350] = (180, 120, 60)

print("Smoke test only checks model/session loading and utility calls.")
print("LaMa model path:", pipeline.lama.session)
print("Example RGB image:", Image.fromarray(img).size)
print("OK")
