#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
W="$ROOT/model_wts"
GD="$W/grounding_dino"
ES="$W/efficient_sam"
LA="$W/lama"

mkdir -p "$GD" "$ES" "$LA"

# Grounding DINO Tiny snapshot.
GD_REV="a2bb814dd30d776dcf7e30523b00659f4f141c71"
GD_BASE="https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/$GD_REV"
for f in model.safetensors config.json preprocessor_config.json added_tokens.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
  curl -L --fail --retry 3 "$GD_BASE/$f" -o "$GD/$f"
done

# EfficientSAM-S split ONNX models.
ES_BASE="https://huggingface.co/yunyangx/EfficientSAM/resolve/main"
curl -L --fail --retry 3 "$ES_BASE/efficientsam_s_encoder.onnx" -o "$ES/efficientsam_s_encoder.onnx"
curl -L --fail --retry 3 "$ES_BASE/efficientsam_s_decoder.onnx" -o "$ES/efficientsam_s_decoder.onnx"

# LaMa fp32 ONNX. Fixed 512x512 input.
LAMA_BASE="https://huggingface.co/Carve/LaMa-ONNX/resolve/main"
curl -L --fail --retry 3 "$LAMA_BASE/lama_fp32.onnx" -o "$LA/lama_fp32.onnx"

echo
echo "Downloaded weights:"
find "$W" -type f -maxdepth 3 -print0 | xargs -0 -n1 ls -lh

echo
echo "LaMa SHA256:"
ACTUAL_SHA=$(shasum -a 256 "$LA/lama_fp32.onnx" | awk '{print $1}')
echo "$ACTUAL_SHA"
EXPECTED_SHA="1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: LaMa checksum mismatch." >&2
  exit 1
fi
echo "LaMa checksum OK (canonical Carve/LaMa-ONNX lama_fp32.onnx)."
echo
echo "Run: ./run.sh"
