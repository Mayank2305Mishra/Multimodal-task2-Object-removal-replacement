from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import Config, Model2Pipeline

st.set_page_config(
    page_title="Model 2 — Grounding DINO + EfficientSAM-S + MI-GAN",
    page_icon="🎯",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MD = ROOT / "model_wts"

st.title("Model 2 — Grounding DINO Tiny + EfficientSAM-S + MI-GAN")
st.caption("CPU-only local pipeline: text → grounding box → precise mask → object removal")


def weight_status() -> dict[str, bool]:
    required = {
        "Grounding DINO weight": MD / "grounding_dino" / "model.safetensors",
        "Grounding DINO config": MD / "grounding_dino" / "config.json",
        "Grounding DINO processor": MD / "grounding_dino" / "preprocessor_config.json",
        "Grounding DINO tokenizer": MD / "grounding_dino" / "tokenizer.json",
        "EfficientSAM-S encoder": MD / "efficient_sam" / "efficientsam_s_encoder.onnx",
        "EfficientSAM-S decoder": MD / "efficient_sam" / "efficientsam_s_decoder.onnx",
        "MI-GAN pipeline v2": MD / "migan" / "migan_pipeline_v2.onnx",
    }
    return {name: path.exists() and path.stat().st_size > 0 for name, path in required.items()}


@st.cache_resource(show_spinner=False)
def load_pipeline(box_threshold: float, text_threshold: float, max_detections: int, expand_ratio: float, close_k: int, dilate_k: int) -> Model2Pipeline:
    return Model2Pipeline(
        Config(
            grounding_box_threshold=box_threshold,
            grounding_text_threshold=text_threshold,
            max_detections=max_detections,
            box_expand_ratio=expand_ratio,
            closing_kernel=close_k,
            dilation_kernel=dilate_k,
            mask_clip_to_box=True,
        )
    )


with st.sidebar:
    st.header("Local weights")
    status = weight_status()
    for name, ok in status.items():
        st.write(("✅ " if ok else "❌ ") + name)

    st.divider()
    st.header("Grounding DINO")
    box_threshold = st.slider("Box threshold", 0.15, 0.60, 0.30, 0.01)
    text_threshold = st.slider("Text threshold", 0.10, 0.50, 0.25, 0.01)
    max_detections = st.slider("Max detections shown", 1, 10, 5, 1)

    st.divider()
    st.header("Mask cleanup")
    expand_ratio = st.slider("Box expansion", 0.00, 0.10, 0.03, 0.01)
    close_k = st.slider("Closing kernel", 1, 9, 3, 2)
    dilate_k = st.slider("Dilation kernel", 1, 9, 3, 2)
    st.caption("Mask is constrained to the expanded Grounding DINO box to reduce spill into nearby regions.")

if not all(status.values()):
    st.warning("Some local weights are missing. Run `bash download_weights.sh` from this folder, then reload.")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
prompt = st.text_input("Object to remove", placeholder="e.g. remove the person")
run = st.button("Run Model 2", type="primary", use_container_width=True)

if uploaded is not None:
    src = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1.65, 1])
    with left:
        st.image(src, caption="Original", use_container_width=True)
    with right:
        st.write("Prompt:", prompt or "—")
        st.write("Resolution:", f"{src.width} × {src.height}")

if run:
    if uploaded is None:
        st.error("Upload an image first.")
        st.stop()
    if not prompt.strip():
        st.error("Enter an object description.")
        st.stop()

    try:
        with st.spinner("Loading local models and running on CPU…"):
            pipeline = load_pipeline(
                box_threshold,
                text_threshold,
                max_detections,
                expand_ratio,
                close_k,
                dilate_k,
            )
            result = pipeline.remove(src, prompt)

        st.success("Object removal completed.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["grounding_overlay"], caption="1. Grounding DINO — box", use_container_width=True)
        with c2:
            st.image(result["mask_overlay"], caption="2. EfficientSAM-S — mask", use_container_width=True)
        with c3:
            st.image(result["result"], caption="3. MI-GAN — final removal", use_container_width=True)

        st.subheader("Latency")
        latency = result["latency"]
        df = pd.DataFrame(
            {
                "Stage": [
                    "Grounding DINO Tiny",
                    "EfficientSAM-S encoder",
                    "EfficientSAM-S decoder",
                    "EfficientSAM-S total",
                    "MI-GAN",
                    "End-to-end total",
                ],
                "Time (ms)": [
                    latency["grounding_ms"],
                    latency["efficient_sam_encoder_ms"],
                    latency["efficient_sam_decoder_ms"],
                    latency["segmentation_total_ms"],
                    latency["migan_ms"],
                    latency["total_ms"],
                ],
            }
        )
        df["Time (ms)"] = df["Time (ms)"].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Grounding detections")
        detection_rows = [
            {
                "label": det["label"],
                "score": round(det["score"], 4),
                "box_xyxy": str([round(float(v), 1) for v in det["box"]]),
            }
            for det in result["grounding"]["detections"]
        ]
        st.dataframe(pd.DataFrame(detection_rows), use_container_width=True, hide_index=True)

        st.subheader("Final binary mask")
        st.image(result["mask"], use_container_width=True)

        image_buf = io.BytesIO()
        Image.fromarray(result["result"]).save(image_buf, format="PNG")
        st.download_button(
            "Download removed image",
            image_buf.getvalue(),
            "model2_efficientsam_removed.png",
            "image/png",
            use_container_width=True,
        )

        mask_buf = io.BytesIO()
        Image.fromarray(result["mask"]).save(mask_buf, format="PNG")
        st.download_button(
            "Download mask",
            mask_buf.getvalue(),
            "model2_efficientsam_mask.png",
            "image/png",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        st.exception(exc)
