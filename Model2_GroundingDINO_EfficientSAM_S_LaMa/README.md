# Model 2 — Grounding DINO Tiny + EfficientSAM-S + Big-LaMa (ONNX)

CPU/edge-oriented local object-removal pipeline.

```text
image + text prompt
       |
       v
Grounding DINO Tiny
       |
       | bounding box
       v
EfficientSAM-S ONNX
       |
       | precise binary mask
       v
Big-LaMa fp32 ONNX
       |
       v
final image
```

## What changed

This version keeps the same Grounding DINO Tiny + EfficientSAM-S front end and replaces **MI-GAN** with **LaMa** for the inpainting stage.

The inpainting model used here is the **original LaMa `big-lama` weights exported to ONNX** as `lama_fp32.onnx`. This is the canonical `Carve/LaMa-ONNX` fp32 export, fixed at **512×512** and opset 17. Its model card explicitly recommends this fp32 export over the alternative `lama.onnx`; the selected public I/O contract uses float32 RGB image and float32 binary erase mask, with mask value `1` meaning the hole to fill and `0` meaning keep. The model returns RGB pixels in `[0,255]`. See the cited model card for the exact contract.

## Files

- `app.py` — Streamlit UI
- `pipeline.py` — complete local pipeline
- `requirements.txt` — Python dependencies
- `check_environment.py` — dependency, ONNX-session and weight diagnostics
- `smoke_test.py` — local plumbing check
- `download_weights.sh` — downloads all required public weights
- `run.sh` — creates `.venv`, installs dependencies, checks setup and launches Streamlit
- `COCO10_EfficientSAM_S_LaMa_Benchmark.ipynb` — 10-image benchmark using the same COCO manifest workflow

## Setup

```bash
cd Model2_GroundingDINO_EfficientSAM_S_LaMa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash download_weights.sh
python check_environment.py
python -m streamlit run app.py
```

Or:

```bash
./run.sh
```

## Local-only behavior

After `download_weights.sh` completes, inference loads Grounding DINO from `model_wts/grounding_dino`, EfficientSAM-S from `model_wts/efficient_sam`, and LaMa from `model_wts/lama`. No model is fetched during inference.

## LaMa preprocessing

Because the bundled LaMa ONNX export is fixed to 512×512, the pipeline resizes the RGB image and binary mask to 512×512, runs LaMa, resizes the generated image back to the original resolution, and composites generated pixels only inside the requested mask. Pixels outside the mask remain unchanged.

## Segmentation safeguards

The Grounding DINO box is expanded by a small configurable margin. The EfficientSAM-S mask can be clipped to that expanded box before inpainting, reducing spill into nearby regions. Small morphological closing/dilation can also be tuned.

## Benchmarking

For an apples-to-apples comparison with the earlier MI-GAN version, reuse the same `benchmark_data/coco10_manifest.json` and report:

- Grounding DINO latency
- EfficientSAM-S encoder latency
- EfficientSAM-S decoder latency
- segmentation total
- LaMa latency
- end-to-end latency
- visual grounding/mask/final outputs
- optional COCO GT box/mask IoU

Do not mix different COCO images/prompts or different warm-up policies when comparing runs.

## Important

Grounding and segmentation are separate from inpainting. A wrong Grounding DINO box can lead to a wrong mask even with a strong segmenter. The UI therefore keeps the grounding box and segmentation mask visible.


## LaMa weight provenance

`model_wts/lama/lama_fp32.onnx` is the canonical `Carve/LaMa-ONNX` export of the original PyTorch **big-lama** model. SHA-256: `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`. The public model card marks `lama_fp32.onnx` as the recommended export and `lama.onnx` as not recommended.
