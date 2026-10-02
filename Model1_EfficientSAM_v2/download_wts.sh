#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${ROOT}/model_wts"
GROUNDING_DIR="${MODEL_DIR}/grounding_dino"
EFFICIENTSAM_DIR="${MODEL_DIR}/efficient_sam"

mkdir -p "${GROUNDING_DIR}" "${EFFICIENTSAM_DIR}"

download() {
    local url="$1"
    local out="$2"

    if [[ -s "${out}" ]]; then
        echo "[skip] ${out}"
        return
    fi

    echo "[download] ${url}"
    curl -fL --retry 3 --retry-delay 2 --progress-bar "${url}" -o "${out}"
}

echo "=== Grounding DINO-Tiny ==="
GD="https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/main"

download "${GD}/config.json" "${GROUNDING_DIR}/config.json"
download "${GD}/model.safetensors" "${GROUNDING_DIR}/model.safetensors"
download "${GD}/preprocessor_config.json" "${GROUNDING_DIR}/preprocessor_config.json"
download "${GD}/tokenizer.json" "${GROUNDING_DIR}/tokenizer.json"
download "${GD}/tokenizer_config.json" "${GROUNDING_DIR}/tokenizer_config.json"
download "${GD}/special_tokens_map.json" "${GROUNDING_DIR}/special_tokens_map.json"
download "${GD}/vocab.txt" "${GROUNDING_DIR}/vocab.txt"

echo "=== EfficientSAM-S split ONNX ==="
ES="https://huggingface.co/yunyangx/EfficientSAM/resolve/main"

download "${ES}/efficientsam_s_encoder.onnx" \
    "${EFFICIENTSAM_DIR}/encoder.onnx"

download "${ES}/efficientsam_s_decoder.onnx" \
    "${EFFICIENTSAM_DIR}/decoder.onnx"

echo "=== LaMa FP32 ONNX ==="
download \
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx" \
    "${MODEL_DIR}/lama_fp32.onnx"

echo
echo "=== Verifying ==="

required=(
    "${GROUNDING_DIR}/config.json"
    "${GROUNDING_DIR}/model.safetensors"
    "${EFFICIENTSAM_DIR}/encoder.onnx"
    "${EFFICIENTSAM_DIR}/decoder.onnx"
    "${MODEL_DIR}/lama_fp32.onnx"
)

for f in "${required[@]}"; do
    if [[ ! -s "${f}" ]]; then
        echo "[ERROR] Missing: ${f}" >&2
        exit 1
    fi

    printf '[OK] %s (%s MB)\n' \
        "${f}" \
        "$(du -m "${f}" | awk '{print $1}')"
done

echo
echo "All Model 1 weights are ready."
