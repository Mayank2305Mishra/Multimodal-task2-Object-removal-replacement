# AGENTS.md — Model 2 CPU/ONNX

Architecture: `Grounding DINO Tiny -> EfficientViT-SAM-L0 ONNX -> MI-GAN ONNX`.

Never import `efficientvit.sam_model_zoo` in this project. Use the official ONNX encoder/decoder. Do not add Triton.

EfficientViT preprocessing follows the official ONNX reference: long-side resize to 512, ImageNet-style SAM pixel mean/std, and zero padding. The ONNX decoder encodes an XYXY box as two points with labels `[2,3]` and postprocesses masks back to the original image size.

MI-GAN ONNX expects uint8 NCHW image and mask. Its mask convention is 255=keep, 0=erase; invert the internal removal mask before inference.

Debug in order: grounding box -> segmentation mask -> inpainting output.
