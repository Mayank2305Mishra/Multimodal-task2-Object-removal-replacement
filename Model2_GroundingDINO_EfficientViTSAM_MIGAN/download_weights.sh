#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
W="$ROOT/model_wts"
GD="$W/grounding_dino"
ES="$W/efficient_sam"
MG="$W/migan"

mkdir -p "$GD" "$ES" "$MG"

# Current public Grounding DINO Tiny snapshot checked 2026-10-01.
GD_REV="a2bb814dd30d776dcf7e30523b00659f4f141c71"
GD_BASE="https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/$GD_REV"

curl -L --fail --retry 3 "$GD_BASE/model.safetensors" -o "$GD/model.safetensors"
curl -L --fail --retry 3 "$GD_BASE/config.json" -o "$GD/config.json"
curl -L --fail --retry 3 "$GD_BASE/preprocessor_config.json" -o "$GD/preprocessor_config.json"
curl -L --fail --retry 3 "$GD_BASE/added_tokens.json" -o "$GD/added_tokens.json"
curl -L --fail --retry 3 "$GD_BASE/special_tokens_map.json" -o "$GD/special_tokens_map.json"
curl -L --fail --retry 3 "$GD_BASE/tokenizer.json" -o "$GD/tokenizer.json"
curl -L --fail --retry 3 "$GD_BASE/tokenizer_config.json" -o "$GD/tokenizer_config.json"
curl -L --fail --retry 3 "$GD_BASE/vocab.txt" -o "$GD/vocab.txt"

# Official EfficientSAM-S ONNX split models.
ES_BASE="https://huggingface.co/yunyangx/EfficientSAM/resolve/main"
curl -L --fail --retry 3 "$ES_BASE/efficientsam_s_encoder.onnx" -o "$ES/efficientsam_s_encoder.onnx"
curl -L --fail --retry 3 "$ES_BASE/efficientsam_s_decoder.onnx" -o "$ES/efficientsam_s_decoder.onnx"

# MI-GAN pipeline v2 ONNX.
MG_BASE="https://huggingface.co/andraniksargsyan/migan/resolve/main"
curl -L --fail --retry 3 "$MG_BASE/migan_pipeline_v2.onnx" -o "$MG/migan_pipeline_v2.onnx"

echo
echo "Downloaded weights:"
find "$W" -type f -maxdepth 3 -print0 | xargs -0 -n1 ls -lh
echo
echo "Run: ./run.sh"
