from pathlib import Path
import io
import streamlit as st
from PIL import Image
from pipeline import Model1Pipeline, PipelineConfig

st.set_page_config(page_title="Model 1 — EfficientSAM-S + PowerPaint v2-1", layout="wide")
st.title("Model 1 — EfficientSAM-S + PowerPaint v2-1")
st.caption("Grounding DINO-Tiny → EfficientSAM-S → PowerPaint v2-1 object removal")

with st.sidebar:
    st.header("Settings")
    prompt = st.text_input("Object to remove", "bottle")
    box_thr = st.slider("Grounding box threshold", 0.10, 0.60, 0.28, 0.01)
    text_thr = st.slider("Grounding text threshold", 0.10, 0.60, 0.22, 0.01)
    dilation = st.slider("Mask dilation", 1, 31, 17, 2)
    roi_padding = st.slider("ROI padding", 0.30, 1.00, 0.65, 0.05)
    steps = st.slider("PowerPaint steps", 15, 50, 30, 1)
    guidance = st.slider("Guidance scale", 7.0, 16.0, 12.0, 0.5)
    max_side = st.select_slider("PowerPaint max side", options=[512, 640, 768, 896], value=768)
    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    device = st.selectbox("Device", ["auto", "mps", "cpu", "cuda"], index=0)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)
    if st.button("Remove object", type="primary"):
        cfg = PipelineConfig(
            grounding_box_threshold=box_thr,
            grounding_text_threshold=text_thr,
            dilation_kernel=dilation,
            roi_padding=roi_padding,
            pp_steps=steps,
            pp_guidance=guidance,
            pp_max_side=max_side,
            seed=int(seed),
        )
        try:
            with st.spinner("Running Grounding DINO → EfficientSAM-S → PowerPaint v2-1..."):
                pipe = Model1Pipeline(device=device, config=cfg)
                out = pipe.remove(image, prompt)
            c1, c2, c3 = st.columns(3)
            c1.image(out["original"], caption="1. Original", use_container_width=True)
            c2.image(out["mask_overlay"], caption="2. EfficientSAM-S mask", use_container_width=True)
            c3.image(out["result"], caption="3. PowerPaint v2-1 removal", use_container_width=True)

            st.subheader("PowerPaint diagnostics")
            d1, d2 = st.columns(2)
            d1.image(out["raw_powerpaint_roi"], caption="Raw PowerPaint ROI", use_container_width=True)
            d2.image(out["roi_mask"], caption="Exact mask supplied to PowerPaint", use_container_width=True)

            st.subheader("Latency")
            st.table(out["latency"])
            st.write({"device": out["device"], "Grounding score": out["grounding_score"], "EfficientSAM IoU": out["efficient_sam_iou"], "ROI": out["roi_box"], "PowerPaint input": out["powerpaint_size"]})
            buf = io.BytesIO()
            out["result"].save(buf, format="PNG")
            st.download_button("Download result", buf.getvalue(), file_name="powerpaint_removed.png", mime="image/png")
        except Exception as e:
            st.exception(e)
