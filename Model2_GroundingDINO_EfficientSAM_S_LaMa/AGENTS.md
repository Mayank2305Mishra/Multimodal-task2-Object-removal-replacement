# AGENTS.md — Model 2

## Scope

This project implements local CPU object removal with:

`Grounding DINO Tiny -> EfficientSAM-S ONNX -> MI-GAN ONNX`

## Hard requirements

- Keep inference local after model download.
- Do not add runtime Hugging Face model downloads.
- Use CPUExecutionProvider for ONNX Runtime.
- Do not introduce Triton or EfficientViT PyTorch dependencies.
- Keep intermediate Grounding DINO box and EfficientSAM-S mask visible in the UI.
- Keep stage-by-stage latency measurements.
- Do not silently replace EfficientSAM-S with EfficientViT-SAM L0/L2, MobileSAM, FastSAM, or SAM2.

## Weight locations

```text
model_wts/
  grounding_dino/model.safetensors
  grounding_dino/config.json
  grounding_dino/preprocessor_config.json
  grounding_dino/tokenizer.json
  efficient_sam/efficientsam_s_encoder.onnx
  efficient_sam/efficientsam_s_decoder.onnx
  migan/migan_pipeline_v2.onnx
```

## Segmentation contract

EfficientSAM-S ONNX uses an image encoder plus a point-prompt decoder. For a Grounding DINO box, feed `[x1,y1]` and `[x2,y2]` with point labels `2` and `3`.

## Validation

Before changing dependencies or model adapters:

```bash
python -m py_compile pipeline.py app.py check_environment.py
python check_environment.py
```

After installing weights, verify both ONNX sessions construct with CPUExecutionProvider before launching Streamlit.
