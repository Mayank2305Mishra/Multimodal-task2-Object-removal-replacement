# Model 2 — Grounding DINO Tiny + EfficientViT-SAM-L0 ONNX + MI-GAN

This is the corrected CPU/edge implementation. It does NOT import the PyTorch EfficientViT package, so it does not hit the `No module named 'triton'` error.

## Pipeline
```text
Text prompt
  ↓
Grounding DINO Tiny (CPU)
  ↓
Bounding box
  ↓
EfficientViT-SAM-L0 ONNX encoder + decoder (ONNX Runtime CPU)
  ↓
Precise mask
  ↓
MI-GAN 512 ONNX (ONNX Runtime CPU)
  ↓
Removed image
```

The official EfficientViT-SAM repository provides ONNX L0 encoder/decoder files and an ONNX inference path; this build follows that path. The L0 encoder is ~123 MB and decoder ~16.5 MB. The official docs report L0 as 45.7 COCO mAP, but their published 8.2 ms latency is Jetson AGX Orin TensorRT/fp16, not CPU. We benchmark the actual CPU pipeline ourselves.

## Required files
```text
model_wts/
├── grounding_dino/
│   ├── model.safetensors
│   ├── config.json
│   ├── preprocessor_config.json
│   └── remaining HF processor/tokenizer files
├── l0_encoder.onnx
├── l0_decoder.onnx
└── migan_pipeline_v2.onnx
```

## Install
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run
```bash
python -m streamlit run app.py
```

## No Triton
Do not install Triton on the Mac for this version. Triton is only being pulled by the PyTorch EfficientViT implementation; the ONNX deployment avoids that dependency.

## MI-GAN mask convention
The app internally uses 255 = object to remove. Before calling MI-GAN it is inverted to the MI-GAN convention 0 = erase and 255 = keep. The ONNX pipeline takes uint8 RGB image + uint8 grayscale mask and returns the composited result.
