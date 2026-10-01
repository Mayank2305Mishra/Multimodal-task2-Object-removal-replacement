# AGENTS.md

## Scope

This folder contains Model 1: MobileSAM + MobileCLIP-S0 + LaMa.

## Pipeline

1. MobileSAM automatic mask proposals
2. Candidate filtering
3. MobileCLIP-S0 text/image scoring
4. Best candidate selection
5. MobileSAM point + box refinement
6. Mild mask dilation
7. Square local ROI
8. LaMa FP32 ONNX inpainting
9. Composite the ROI back into the original image

## Defaults

```text
sam_points_per_side = 10
sam_points_per_batch = 64
sam_pred_iou_thresh = 0.84
sam_stability_thresh = 0.88
sam_min_mask_area = 100
max_clip_candidates = 48
mask_dilation = 9
roi_padding = 0.35
LaMa = 512x512
```

## Debugging

Always inspect the selected MobileSAM candidate and refined mask before tuning LaMa. If the mask covers the wrong object/background, the inpainting stage is not the primary fault.

## Device

On MPS, MobileSAM proposal generation is intentionally on CPU. MobileCLIP uses MPS. LaMa uses CPU ONNX Runtime.

## Benchmark consistency

For comparisons with FastSAM and EdgeSAM, keep the image set, prompts, latency measurement method, and model parameters fixed unless the change is explicitly recorded.
