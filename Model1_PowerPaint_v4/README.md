# Model 1 — Grounding DINO-Tiny + EfficientSAM-S + PowerPaint v2-1

Grounding DINO-Tiny → EfficientSAM-S → mask cleanup → contextual ROI → PowerPaint v2-1 object removal → composite.

The official PowerPaint project explicitly supports object removal as a dedicated task. The v2-1 repository is about 9.52 GB and contains `PowerPaint_Brushnet` and `realisticVisionV60B1_v51VAE`.

## Install

```bash
cd ~/Desktop
unzip Model1_PowerPaint_v4.zip
cd Model1_PowerPaint_v4
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
chmod +x download_wts.sh
./download_wts.sh
python check_environment.py
streamlit run app.py
```

**Do not run Streamlit until the downloader ends with `All Model 1 weights and source files are ready.`**

If you already have v3, the downloader is resumable. Run the v4 `download_wts.sh`; it will keep valid existing files and complete missing files.
