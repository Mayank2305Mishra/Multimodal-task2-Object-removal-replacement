#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${ROOT}/model_wts"
GROUNDING_DIR="${MODEL_DIR}/grounding_dino"
EFFICIENTSAM_DIR="${MODEL_DIR}/efficient_sam"
POWERPAINT_DIR="${MODEL_DIR}/powerpaint_v2_1"
POWERPAINT_REPO="${ROOT}/powerpaint"

mkdir -p "${GROUNDING_DIR}" "${EFFICIENTSAM_DIR}" "${POWERPAINT_DIR}"
command -v python >/dev/null || { echo "[ERROR] python not found"; exit 1; }
command -v git >/dev/null || { echo "[ERROR] git not found"; exit 1; }

python -m pip install -q "huggingface_hub>=0.24"

python - "${ROOT}" <<'PYTHON'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(sys.argv[1]).resolve()
model = root / "model_wts"
gd = model / "grounding_dino"
es = model / "efficient_sam"
pp = model / "powerpaint_v2_1"

def download_file(repo, filename, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 0:
        print(f"[skip] {out}")
        return
    print(f"[download] {repo}:{filename}")
    cached = hf_hub_download(repo_id=repo, filename=filename)
    out.write_bytes(Path(cached).read_bytes())

for name in [
    "config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"
]:
    download_file("IDEA-Research/grounding-dino-tiny", name, gd / name)

download_file("yunyangx/EfficientSAM", "efficientsam_s_encoder.onnx", es / "encoder.onnx")
download_file("yunyangx/EfficientSAM", "efficientsam_s_decoder.onnx", es / "decoder.onnx")

marker = pp / "PowerPaint_Brushnet" / "diffusion_pytorch_model.safetensors"
if not marker.is_file() or marker.stat().st_size < 100_000_000:
    print("[download] JunhaoZhuang/PowerPaint-v2-1 (~9.5 GB)")
    snapshot_download(repo_id="JunhaoZhuang/PowerPaint-v2-1", local_dir=str(pp))
else:
    print(f"[skip] PowerPaint checkpoint already present: {marker}")
PYTHON

if [[ ! -f "${POWERPAINT_REPO}/powerpaint/models/BrushNet_CA.py" ]]; then
    rm -rf "${POWERPAINT_REPO}"
    echo "[clone] https://github.com/open-mmlab/PowerPaint.git"
    git clone --depth 1 https://github.com/open-mmlab/PowerPaint.git "${POWERPAINT_REPO}"
fi

echo
echo "=== FINAL WEIGHT CHECK ==="
required=(
  "${GROUNDING_DIR}/config.json"
  "${GROUNDING_DIR}/model.safetensors"
  "${EFFICIENTSAM_DIR}/encoder.onnx"
  "${EFFICIENTSAM_DIR}/decoder.onnx"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/unet/config.json"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model.safetensors"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/vae/config.json"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/vae/diffusion_pytorch_model.safetensors"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/text_encoder/config.json"
  "${POWERPAINT_DIR}/realisticVisionV60B1_v51VAE/tokenizer/tokenizer_config.json"
  "${POWERPAINT_DIR}/PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"
  "${POWERPAINT_DIR}/PowerPaint_Brushnet/pytorch_model.bin"
  "${POWERPAINT_REPO}/powerpaint/models/BrushNet_CA.py"
)
missing=0
for f in "${required[@]}"; do
  if [[ -s "${f}" ]] || [[ -d "${f}" ]]; then echo "[OK] ${f}"; else echo "[MISSING] ${f}"; missing=1; fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "[ERROR] One or more required files are missing. Do NOT start Streamlit yet."; exit 1
fi
echo
echo "All Model 1 weights and source files are ready."
