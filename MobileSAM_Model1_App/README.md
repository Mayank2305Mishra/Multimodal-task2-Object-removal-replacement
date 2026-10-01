# Model 1 — MobileSAM + MobileCLIP + LaMa

Complete local Streamlit app for the first object-removal setup.

## Architecture

```text
Image + text prompt
        ↓
MobileSAM automatic mask proposals
        ↓
MobileCLIP-S0 candidate scoring
        ↓
MobileSAM point + box refinement
        ↓
Mask dilation
        ↓
Local square ROI
        ↓
LaMa FP32 ONNX inpainting
        ↓
Removed image
```

The supplied COCO benchmark notebook and this app use the same model roles and the same default parameters.

## Files

```text
MobileSAM_Model1_App/
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
├── AGENTS.md
├── check_environment.py
├── run.sh
├── model_wts/
│   └── README.md
└── MobileSAM_MobileCLIP_LaMa_COCO10_EndToEnd.ipynb
```

## Weights

Add:

```text
model_wts/mobile_sam.pt
model_wts/mobileclip_s0.pt
model_wts/lama_fp32.onnx
```

The ZIP does not contain these large checkpoint files.

## Install on Apple Silicon

A fresh Python 3.10 or 3.11 environment is recommended for this model stack.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify

```bash
python check_environment.py
```

## Run

```bash
python -m streamlit run app.py
```

or:

```bash
./run.sh
```

## M4 runtime policy

When `auto` selects MPS:

- MobileSAM automatic proposals → CPU
- MobileCLIP-S0 → MPS
- LaMa ONNX → CPUExecutionProvider

This keeps the MobileSAM automatic-proposal stage away from the released MPS float64 issue encountered in this project.

## App features

- Upload JPG/PNG/WEBP image
- Free-form text object prompt
- MobileSAM proposal visualization
- MobileSAM refined mask
- LaMa removal output
- Per-stage and total latency
- Candidate count and MobileCLIP score
- Device diagnostics
- PNG downloads

## Benchmark notebook

`MobileSAM_MobileCLIP_LaMa_COCO10_EndToEnd.ipynb` benchmarks this same Model 1 on 10 COCO validation cases and reports target IoU, precision/recall, and per-stage latency.
