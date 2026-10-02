from pathlib import Path
import io

import pandas as pd
import streamlit as st
from PIL import Image

from pipeline import Model1Pipeline, PipelineConfig


st.set_page_config(
    page_title="Model 1 — EfficientSAM-S + LaMa",
    page_icon="🪄",
    layout="wide",
)

st.title("Model 1 — EfficientSAM-S + LaMa")
st.caption("Grounding DINO-Tiny → EfficientSAM-S → LaMa FP32 ONNX")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_wts"


def model_status():
    return {
        "Grounding DINO config": (MODEL_DIR / "grounding_dino" / "config.json").exists(),
        "Grounding DINO weights": (MODEL_DIR / "grounding_dino" / "model.safetensors").exists(),
        "EfficientSAM-S encoder": (MODEL_DIR / "efficient_sam" / "encoder.onnx").exists(),
        "EfficientSAM-S decoder": (MODEL_DIR / "efficient_sam" / "decoder.onnx").exists(),
        "LaMa FP32": (MODEL_DIR / "lama_fp32.onnx").exists(),
    }


@st.cache_resource(show_spinner=False)
def load_pipeline(
    box_threshold,
    text_threshold,
    dilation,
    roi_padding,
):
    cfg = PipelineConfig(
        grounding_box_threshold=float(box_threshold),
        grounding_text_threshold=float(text_threshold),
        dilation_kernel=int(dilation),
        roi_padding=float(roi_padding),
    )
    return Model1Pipeline(device="cpu", config=cfg)


with st.sidebar:
    st.header("Model weights")

    statuses = model_status()

    for name, ok in statuses.items():
        st.write(("✅ " if ok else "❌ ") + name)

    if not all(statuses.values()):
        st.warning("Run `bash download_wts.sh` first.")

    st.divider()

    st.header("Grounding DINO")
    box_threshold = st.slider(
        "Box threshold",
        0.10,
        0.60,
        0.28,
        0.01,
    )
    text_threshold = st.slider(
        "Text threshold",
        0.10,
        0.60,
        0.22,
        0.01,
    )

    st.header("Mask → LaMa")
    dilation = st.slider(
        "Mask dilation",
        1,
        31,
        15,
        2,
    )
    roi_padding = st.slider(
        "ROI padding",
        0.20,
        0.80,
        0.50,
        0.05,
    )


if not all(model_status().values()):
    st.info(
        "Required weights are missing. Run:\n\n"
        "`chmod +x download_wts.sh && ./download_wts.sh`"
    )
    st.stop()


uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

prompt = st.text_input(
    "Object to remove",
    placeholder="e.g. remove the red bottle",
)

run = st.button(
    "🪄 Remove object",
    type="primary",
    use_container_width=True,
)


if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns([1.6, 1])

    with left:
        st.image(
            image,
            caption="Original",
            use_container_width=True,
        )

    with right:
        st.write("**Prompt:**", prompt or "—")
        st.write(
            "**Resolution:**",
            f"{image.width} × {image.height}",
        )


if run:
    if uploaded is None:
        st.error("Upload an image first.")
        st.stop()

    if not prompt.strip():
        st.error("Enter an object description.")
        st.stop()

    try:
        with st.spinner(
            "Grounding DINO → EfficientSAM-S → LaMa..."
        ):
            pipe = load_pipeline(
                box_threshold,
                text_threshold,
                dilation,
                roi_padding,
            )

            output = pipe.remove(
                image,
                prompt,
            )

        st.success("Object removal completed.")

        st.subheader("Pipeline stages")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(
                output["original"],
                caption="1. Original",
                use_container_width=True,
            )

        with c2:
            st.image(
                output["efficient_sam_mask_overlay"],
                caption="2. EfficientSAM-S mask",
                use_container_width=True,
            )

        with c3:
            st.image(
                output["result"],
                caption="3. Final LaMa removal",
                use_container_width=True,
            )

        st.subheader("LaMa debugging")

        d1, d2 = st.columns(2)

        with d1:
            st.image(
                output["lama_raw_roi"],
                caption="Raw LaMa output inside ROI",
                use_container_width=True,
            )

        with d2:
            st.image(
                output["roi_mask"],
                caption="Exact mask supplied to LaMa",
                use_container_width=True,
            )

        dbg = output["lama_debug_stats"]
        st.write(
            "**LaMa diagnostics:** "
            f"raw range = `{dbg['raw_min']:.3f} .. {dbg['raw_max']:.3f}`, "
            f"mean absolute change inside mask = `{dbg['mean_abs_change_in_mask']:.3f}`"
        )

        if dbg["mean_abs_change_in_mask"] < 2.0:
            st.warning(
                "LaMa is producing almost the same pixels as the input inside "
                "the removal mask. This indicates an inference/preprocessing "
                "problem rather than a compositing problem."
            )

        st.subheader("Latency")

        latency = output["latency"]

        df = pd.DataFrame(
            {
                "Stage": [
                    "Grounding DINO",
                    "EfficientSAM-S encoder",
                    "EfficientSAM-S decoder",
                    "LaMa",
                    "Total",
                ],
                "Time (ms)": [
                    latency["grounding_ms"],
                    latency["efficient_sam_encoder_ms"],
                    latency["efficient_sam_decoder_ms"],
                    latency["lama_ms"],
                    latency["total_ms"],
                ],
            }
        )

        df["Time (ms)"] = df["Time (ms)"].round(2)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Total",
            f'{latency["total_ms"]:.1f} ms',
        )

        m2.metric(
            "Grounding score",
            f'{output["grounding_score"]:.3f}',
        )

        m3.metric(
            "EfficientSAM IoU",
            f'{output["efficient_sam_iou"]:.3f}',
        )

        m4.metric(
            "LaMa",
            f'{latency["lama_ms"]:.1f} ms',
        )

        st.subheader("Detection")

        st.write(
            "Selected box:",
            output["selected_box"],
        )

        st.write(
            "ROI:",
            output["roi_box"],
        )

        buf = io.BytesIO()
        Image.fromarray(output["result"]).save(
            buf,
            format="PNG",
        )

        st.download_button(
            "⬇️ Download removed image",
            buf.getvalue(),
            "efficient_sam_lama_removed.png",
            "image/png",
            use_container_width=True,
        )

        mbuf = io.BytesIO()
        Image.fromarray(output["mask"]).save(
            mbuf,
            format="PNG",
        )

        st.download_button(
            "⬇️ Download final mask",
            mbuf.getvalue(),
            "efficient_sam_final_mask.png",
            "image/png",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            f"Pipeline failed: {type(exc).__name__}: {exc}"
        )
        st.exception(exc)


st.divider()

st.caption(
    "Model 1: Grounding DINO-Tiny + EfficientSAM-S + LaMa FP32 ONNX. "
    "All weights are local; no checkpoint is downloaded during inference."
)
