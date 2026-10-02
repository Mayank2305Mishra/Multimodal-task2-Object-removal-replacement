#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

chmod +x download_wts.sh
./download_wts.sh

python check_environment.py

python -m streamlit run app.py
