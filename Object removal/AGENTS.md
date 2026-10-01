# AGENTS.md — Text-Based Object Removal Pipeline

## Purpose

This project is a local-first Python + Streamlit object-removal application.
The user uploads an image, writes a natural-language instruction such as `remove the person on the left`, and the pipeline:

1. Generates candidate regions with **MobileSAM**.
2. Scores candidate regions against the text using **Apple MobileCLIP-S0**.
3. Dilates the selected mask and removes the object with **LaMa**.
4. Displays intermediate/final images and stage-level latency.

The model checkpoints are local and must live in `./model_wts`.

## Required model files

Expected filenames:

```text
model_wts/
├── mobile_sam.pt
├── mobileclip_s0.pt
└── big-lama.pt
```

The application does not download model weights at runtime. `pipeline.py` also accepts a few common alternate filenames, but the canonical names above should be used for reproducibility.

## Architecture

```text
Streamlit upload + prompt
          │
          ▼
      RGB image
          │
          ▼
 MobileSAM automatic masks
          │
          ├── spatial filtering
          └── candidate selection
          │
          ▼
 MobileCLIP-S0 text ↔ image scoring
          │
          ▼
      best candidate mask
          │
          ▼
       mask dilation
          │
          ▼
      LaMa inpainting
          │
          ▼
 final image + latency metrics
```

## Source-of-truth APIs

Use the official MobileSAM API:

```python
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam = sam_model_registry["vit_t"](checkpoint="./model_wts/mobile_sam.pt")
mask_generator = SamAutomaticMaskGenerator(sam)
```

MobileSAM documents `vit_t`, `SamAutomaticMaskGenerator`, and local checkpoint loading in its official repository.

Use the official MobileCLIP API:

```python
import mobileclip
model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0",
    pretrained="./model_wts/mobileclip_s0.pt",
)
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
```

For inference, normalize image and text embeddings and compute cosine similarity. The official MobileCLIP repository also documents reparameterizing MobileOne-based models for inference.

Use the local LaMa checkpoint through `LAMA_MODEL`:

```python
import os
os.environ["LAMA_MODEL"] = "./model_wts/big-lama.pt"
from simple_lama_inpainting import SimpleLama
lama = SimpleLama(device="cpu")
```

This prevents `simple-lama-inpainting` from downloading its default checkpoint.

## Installation rules

Use a **fresh virtual environment**. Prefer Python 3.11 for reproducibility.

Example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching PyTorch/TorchVision pair for the host first. For Linux CPU, the project requirements use:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install -r requirements.txt
```

Do not mix unrelated PyTorch, TorchVision, NumPy, Pillow, or TensorFlow binaries into this environment. A native-extension mismatch can surface as errors such as `numpy.dtype size changed`, `torchvision::nms does not exist`, or Pillow import failures.

## Running

From the project root:

```bash
streamlit run app.py
```

Open the local Streamlit URL, upload an image, enter a prompt, and click **Remove object**.

## Device behavior

The default device is CPU for predictable edge benchmarking.

Override with:

```bash
DEVICE=cpu streamlit run app.py
```

For a CUDA machine:

```bash
DEVICE=cuda streamlit run app.py
```

For Apple Silicon, MPS may be selected explicitly:

```bash
DEVICE=mps streamlit run app.py
```

CPU should remain the benchmark/default path unless a hardware-specific benchmark is explicitly requested.

## Latency accounting

The application reports three inference stages:

- `MobileSAM proposals`: automatic mask-generation time.
- `MobileCLIP scoring`: preprocessing + batch image encoding + text encoding + scoring.
- `LaMa inpainting`: selected-mask image restoration.
- `Total inference`: sum of the above three stages.

First-run model loading is intentionally excluded from the stage latency metrics because the models are cached in memory after their first load. The UI separately shows first-run wall-clock time.

When comparing machines/models, report whether the measurement is **cold-start** or **warm inference**.

## Prompt handling

The pipeline strips common action prefixes such as:

- `remove the`
- `remove`
- `delete the`
- `erase the`
- `get rid of the`

Spatial words (`left`, `right`, `top`, `bottom`, `center`) are used as lightweight candidate-region weighting. This is a heuristic, not a language model.

## Intermediate outputs

The Streamlit UI intentionally shows:

1. Original image.
2. MobileSAM selected-mask overlay.
3. MobileCLIP's best-matching masked crop.
4. LaMa final inpainted image.

MobileCLIP is a scoring/selection component, so its visual artifact is the selected candidate crop rather than a generated image.

## Performance knobs

Environment variables:

```text
MODEL_DIR=./model_wts
DEVICE=cpu
SAM_RESIZE=640
MAX_CLIP_CANDIDATES=32
DILATE_KERNEL=15
```

For faster CPU tests, reduce `SAM_RESIZE` or `MAX_CLIP_CANDIDATES`. For better region coverage, increase them at the cost of latency.

## Error-handling requirements

Do not silently fall back to downloading weights.

Missing checkpoints must raise a clear `FileNotFoundError` identifying:
- the expected directory,
- candidate filenames,
- and the specific model that is missing.

Dependency errors must be surfaced with the original exception because native package mismatches are environment problems, not model problems.

## Code-change rules

- Keep model paths relative to the repository root or configurable with `MODEL_DIR`.
- Do not hard-code `/kaggle/input/...` paths into the app.
- Do not add runtime checkpoint downloads.
- Keep inference deterministic where practical.
- Keep latency measurement around actual inference work.
- Avoid adding a second segmentation or inpainting model without updating this document and the latency table.
- Preserve the four-panel intermediate-output UI unless the user explicitly asks for a redesign.
