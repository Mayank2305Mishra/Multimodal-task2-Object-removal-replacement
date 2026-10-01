# Model 2 required weights

Use these four weight groups only:

1. `grounding_dino/model.safetensors` plus the complete small config/processor/tokenizer files from:
   https://huggingface.co/IDEA-Research/grounding-dino-tiny
2. `l0_encoder.onnx` from:
   https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/onnx/l0_encoder.onnx?download=true
3. `l0_decoder.onnx` from:
   https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/onnx/l0_decoder.onnx?download=true
4. `migan_pipeline_v2.onnx` from:
   https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx?download=true

The old `efficientvit_sam_l0.pt` is NOT used by this build.
No Triton is required.
