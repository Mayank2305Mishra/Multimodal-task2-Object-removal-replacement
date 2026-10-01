from __future__ import annotations

import time

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Text Object Removal",
    page_icon="✂️",
    layout="wide",
)

st.title("Text-Based Object Removal")
st.caption("MobileSAM → MobileCLIP-S0 → LaMa · local weights · CPU-first")

try:
    from pipeline import load_models, model_status, run_text_object_removal
except Exception as exc:
    st.error("Pipeline dependencies could not be imported.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.subheader("Model configuration")
    status = model_status()
    st.code(status["device"])
    st.caption("All model checkpoints are loaded from ./model_wts")
    with st.expander("Checkpoint paths", expanded=False):
        st.write(status)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"],
)
prompt = st.text_input(
    "What should be removed?",
    placeholder="remove the person on the left",
)

run = st.button("Remove object", type="primary", use_container_width=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input")
    st.image(image, caption=f"{image.width} × {image.height}", use_container_width=True)

if run:
    if uploaded is None:
        st.warning("Upload an image first.")
        st.stop()
    if not prompt.strip():
        st.warning("Enter a removal prompt first.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")

    with st.status("Loading local models and running pipeline…", expanded=True) as status_box:
        try:
            load_t0 = time.perf_counter()
            result = run_text_object_removal(image, prompt)
            total_wall = time.perf_counter() - load_t0
            status_box.update(label="Completed", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label="Pipeline failed", state="error", expanded=True)
            st.exception(exc)
            st.stop()

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(result["original"], caption="Original", use_container_width=True)
    with c2:
        st.image(result["sam_overlay"], caption="MobileSAM — selected mask", use_container_width=True)
    with c3:
        st.image(result["mobileclip_crop"], caption="MobileCLIP — best matching region", use_container_width=True)
    with c4:
        st.image(result["result"], caption="LaMa — final removal", use_container_width=True)

    st.subheader("Latency")
    t = result["timing"]
    metric_cols = st.columns(4)
    for col, (label, value) in zip(metric_cols, t.items()):
        col.metric(label, f"{value:.3f} s")

    st.caption(f"UI wall-clock time: {total_wall:.3f} s · device: {result['metadata']['device']}")

    with st.expander("Pipeline details"):
        st.json(result["metadata"])
        st.write("Model load times (first run only):")
        # Accessing load cache indirectly is intentionally avoided; load_models() is cached.
        st.write("Model weights remain local; no checkpoint download is performed by the app.")
