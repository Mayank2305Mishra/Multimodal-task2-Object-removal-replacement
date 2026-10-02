# Model 1 — EfficientSAM-S + LaMa

## Pipeline

Image + text prompt
→ Grounding DINO-Tiny
→ EfficientSAM-S
→ mask cleanup + controlled dilation
→ local 512×512 LaMa
→ feathered composite
→ removed image

## Important fixes

This version replaces the previous MobileSAM stage with the official
EfficientSAM-S split ONNX encoder/decoder.

The EfficientSAM decoder is called with:

- `image_embeddings`
- `batched_point_coords`
- `batched_point_labels`
- `orig_im_size`

Box prompting uses the two box corners with labels `[2, 3]`.

The LaMa stage uses the fixed 512×512 `lama_fp32.onnx` contract:

- RGB float32 image in [0, 1]
- float32 binary mask in [0, 1]
- mask value 1 means the region to inpaint

The app exposes the raw LaMa ROI and the exact mask supplied to LaMa
so failures can be diagnosed without guessing.

## Setup

```bash
cd Model1_EfficientSAM_Updated

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

chmod +x download_wts.sh
./download_wts.sh

python check_environment.py

streamlit run app.py
```

Or:

```bash
./run.sh
```

## Expected weights

```text
model_wts/
├── grounding_dino/
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── efficient_sam/
│   ├── encoder.onnx
│   └── decoder.onnx
└── lama_fp32.onnx
```

No MobileSAM or MobileCLIP weights are required.

## If removal is weak

The sidebar exposes:

- Grounding box threshold
- Grounding text threshold
- Mask dilation
- ROI padding

For small objects, start around:

```text
box threshold: 0.25–0.30
text threshold: 0.20–0.25
mask dilation: 15
ROI padding: 0.50
```

Do not simply maximize dilation; excessive dilation can remove nearby structures.


## v2 LaMa fix

This version follows the preprocessing/postprocessing convention of the
Carve LaMa FP32 ONNX demo exactly and reports `mean_abs_change_in_mask`.
A value close to zero means LaMa returned almost the same content inside the
mask, which isolates the failure to LaMa inference/preprocessing.
