from pathlib import Path
import io

import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import Model1Pipeline, PipelineConfig

st.set_page_config(page_title="Model 1 — MobileSAM Object Removal", page_icon="🖼️", layout="wide")
st.title("Model 1 — MobileSAM + MobileCLIP + LaMa")
st.caption("Text-guided object removal: MobileSAM → MobileCLIP-S0 → MobileSAM refinement → LaMa ONNX")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"


def model_status():
    return {
        "mobile_sam.pt": (MODEL_DIR / "mobile_sam.pt").exists(),
        "mobileclip_s0.pt": (MODEL_DIR / "mobileclip_s0.pt").exists(),
        "lama_fp32.onnx": (MODEL_DIR / "lama_fp32.onnx").exists(),
    }


@st.cache_resource(show_spinner=False)
def load_pipeline(device, pps, ppb, iou, stability, min_area, max_candidates, dilation, roi_padding):
    cfg = PipelineConfig(
        sam_points_per_side=int(pps),
        sam_points_per_batch=int(ppb),
        sam_pred_iou_thresh=float(iou),
        sam_stability_thresh=float(stability),
        sam_min_mask_area=int(min_area),
        max_clip_candidates=int(max_candidates),
        mask_dilation=int(dilation),
        roi_padding=float(roi_padding),
    )
    return Model1Pipeline(device=device, config=cfg)


with st.sidebar:
    st.header("Model weights")
    statuses = model_status()
    for name, ok in statuses.items():
        st.write(("✅ " if ok else "❌ ") + name)
    if not all(statuses.values()):
        st.warning("Add the missing checkpoints to model_wts/.")

    st.divider()
    st.header("Runtime")
    device = st.selectbox("Device", ["auto", "mps", "cpu", "cuda"])

    st.header("MobileSAM")
    pps = st.select_slider("Points per side", [8, 10, 12, 16], value=10)
    ppb = st.select_slider("Points per batch", [16, 32, 64, 128], value=64)
    iou = st.slider("Predicted IoU threshold", 0.70, 0.95, 0.84, 0.01)
    stability = st.slider("Stability threshold", 0.70, 0.98, 0.88, 0.01)
    min_area = st.number_input("Minimum mask area", 10, 5000, 100, 10)
    max_candidates = st.slider("Maximum CLIP candidates", 8, 64, 48, 4)

    st.header("LaMa")
    dilation = st.slider("Mask dilation", 1, 21, 9, 2)
    roi_padding = st.slider("ROI padding", 0.10, 0.60, 0.35, 0.05)

if not all(model_status().values()):
    st.info("Required weights: model_wts/mobile_sam.pt, model_wts/mobileclip_s0.pt, model_wts/lama_fp32.onnx")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
prompt = st.text_input("Object to remove", placeholder="e.g. remove the red car")
run = st.button("Remove object", type="primary", use_container_width=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    a, b = st.columns([1.6, 1])
    with a:
        st.image(image, caption="Original", use_container_width=True)
    with b:
        st.write("**Prompt**", prompt or "—")
        st.write("**Resolution**", f"{image.width} × {image.height}")

if run:
    if uploaded is None:
        st.error("Upload an image first.")
        st.stop()
    if not prompt.strip():
        st.error("Enter an object description.")
        st.stop()

    try:
        with st.spinner("Loading models and running Model 1..."):
            pipe = load_pipeline(device, pps, ppb, iou, stability, int(min_area), max_candidates, dilation, roi_padding)
            output = pipe.remove(image, prompt)

        st.success("Object removal completed.")

        st.subheader("Pipeline stages")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.image(output["original"], caption="1. Original", use_container_width=True)
        with c2:
            st.image(output["selected_candidate_mask"], caption="2. MobileCLIP-selected MobileSAM proposal", use_container_width=True)
        with c3:
            st.image(output["overlay"], caption="3. MobileSAM refined mask", use_container_width=True)

        st.image(output["result"], caption="4. Final LaMa object removal", use_container_width=True)

        st.subheader("Latency")
        latency = output["latency"]
        df = pd.DataFrame({
            "Stage": [
                "MobileSAM proposals",
                "MobileCLIP scoring",
                "MobileSAM refinement",
                "LaMa inpainting",
                "Total",
            ],
            "Time (ms)": [
                latency["mobile_sam_proposal_ms"],
                latency["mobileclip_ms"],
                latency["mobile_sam_refine_ms"],
                latency["lama_ms"],
                latency["total_ms"],
            ],
        })
        df["Time (ms)"] = df["Time (ms)"].round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total", f'{latency["total_ms"]:.1f} ms')
        m2.metric("SAM proposals", output["proposal_count"])
        m3.metric("CLIP candidates", output["clip_candidate_count"])
        m4.metric("CLIP score", f'{output["clip_score"]:.3f}')

        st.subheader("Selection diagnostics")
        sel = output["selected_candidate"]
        diag = pd.DataFrame({
            "Property": ["Prompt","Selected bbox","SAM predicted IoU","SAM stability","Refinement score","ROI","Main device","SAM device","MobileCLIP device"],
            "Value": [
                output["prompt"], str(sel["bbox"]), f'{sel["predicted_iou"]:.4f}', f'{sel["stability_score"]:.4f}',
                f'{output["refine_score"]:.4f}', str(output["roi_box"]), output["device"], output["sam_device"], output["clip_device"]
            ],
        })
        st.dataframe(diag, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        Image.fromarray(output["result"]).save(buf, format="PNG")
        st.download_button("Download removed image", buf.getvalue(), "mobilesam_model1_removed.png", "image/png", use_container_width=True)

        mbuf = io.BytesIO()
        Image.fromarray(output["mask"]).save(mbuf, format="PNG")
        st.download_button("Download final mask", mbuf.getvalue(), "mobilesam_model1_mask.png", "image/png", use_container_width=True)

    except Exception as exc:
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        st.exception(exc)

st.divider()
st.caption("Model 1: MobileSAM + MobileCLIP-S0 + LaMa FP32 ONNX. COCO is not required at runtime.")
