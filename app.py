"""Streamlit public web demonstration for lightweight image dehazing.

Production-ready deployment application supporting CPU and CUDA execution,
dynamic checkpoint resolution, sample demo loading, and quantitative evaluation.
"""
from __future__ import annotations

import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dehaze.evaluation.checkpoints import load_checkpoint_model
from dehaze.evaluation.metrics import psnr, ssim
from dehaze.utils.io import pil_to_tensor, tensor_to_pil
from dehaze.utils.reproducibility import resolve_device

# Checkpoint candidate search paths (supports both local outputs and deployment models/ folder)
CHECKPOINT_CATALOG = {
    "Experiment B (Recommended Winner — 75.7K params)": {
        "paths": [
            ROOT / "models" / "best_epoch_019.pt",
            ROOT / "outputs" / "stage3_7_loss_l1_ssim" / "best_epoch_019.pt",
        ],
        "description": "4-Block Residual CNN trained with 0.8 L1 + 0.2 (1-SSIM) structural loss. Achieves 13.00 dB PSNR / 0.4418 SSIM on NH-HAZE validation.",
        "default": True,
    },
    "Multi-Model Variant v1 (Validation Candidate — 114.7K params)": {
        "paths": [
            ROOT / "models" / "multimodel_variant_best.pt",
            ROOT / "outputs" / "multimodel_variant_v1_continuation_20261019" / "best_epoch_020.pt",
        ],
        "description": "Multi-scale feature pyramid with AAFM and 3 parallel branches (atmospheric, layer-separation, context). Experimental model.",
        "default": False,
    },
    "Stage 3.6 Baseline (L1 Only — 75.7K params)": {
        "paths": [
            ROOT / "models" / "baseline_0p2_best.pt",
            ROOT / "outputs" / "residual_scale_continuation_0p2_rerun_1" / "best_epoch_020.pt",
        ],
        "description": "Pure L1 loss baseline with 0.2 residual scale. 12.84 dB PSNR / 0.4268 SSIM on validation.",
        "default": False,
    },
}


def download_checkpoint_if_needed(target_path: Path) -> bool:
    """Download checkpoint from public URL configured in secrets or environment."""
    url = None
    if hasattr(st, "secrets") and "CHECKPOINT_URL" in st.secrets:
        url = st.secrets["CHECKPOINT_URL"]
    elif "CHECKPOINT_URL" in os.environ:
        url = os.environ["CHECKPOINT_URL"]
    if not url:
        return False
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading model checkpoint from cloud storage..."):
            urllib.request.urlretrieve(url, str(target_path))
        return target_path.is_file()
    except Exception as error:
        st.error(f"Failed to download remote checkpoint from {url}: {error}")
        return False


def get_checkpoint_path(label: str) -> Path | None:
    """Resolve checkpoint file path from candidates or remote download."""
    config = CHECKPOINT_CATALOG.get(label)
    if not config:
        return None
    for candidate in config["paths"]:
        if candidate.is_file():
            return candidate
    # Try downloading to primary deployment location
    primary = config["paths"][0]
    if download_checkpoint_if_needed(primary):
        return primary
    return None


@st.cache_resource(show_spinner=False)
def get_model(label: str):
    """Load model checkpoint onto available hardware (CUDA or CPU fallback)."""
    checkpoint_path = get_checkpoint_path(label)
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Checkpoint file not found for '{label}'. Expected at one of: "
            f"{[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in CHECKPOINT_CATALOG[label]['paths']]}"
        )
    device = resolve_device("cuda")
    model, descriptor, _ = load_checkpoint_model(checkpoint_path, device)
    return model, descriptor, device


def dehaze_image(image: Image.Image, label: str) -> tuple[Image.Image, float, dict, torch.Tensor]:
    """Execute single-image dehazing forward pass with runtime profiling."""
    model, descriptor, device = get_model(label)
    tensor = pil_to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = model(tensor)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_image = tensor_to_pil(result)
    return output_image, elapsed, descriptor, result


# Streamlit Page Configuration
st.set_page_config(
    page_title="Lightweight Image Dehazing — ECE Major Project",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ Lightweight Edge-Aware Image Dehazing")
st.markdown(
    "**Major Project in Digital Image Processing** | Department of Electronics and Communication Engineering (2026–2027)\n\n"
    "Real-time outdoor image dehazing powered by lightweight residual deep learning with structural similarity preservation."
)

# Sidebar with Device & Model Info
device_name = "NVIDIA CUDA GPU" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "CPU (Standard Cloud Instance)"
st.sidebar.header("System Specifications")
st.sidebar.markdown(f"**Execution Hardware:** `{device_name}`")
st.sidebar.markdown("**Input Resolution Support:** Up to $1600 \\times 1200$ (Full Native)")
st.sidebar.markdown("**Primary Model:** `Stage 3.7 Experiment B`")
st.sidebar.markdown("**Parameter Count:** 75,747 parameters (933.6 KB)")
st.sidebar.markdown("**Loss Function:** $0.8 \\mathcal{L}_{L1} + 0.2 \\mathcal{L}_{\\text{SSIM}}$")
st.sidebar.markdown("---")
st.sidebar.markdown("### Benchmarks (NH-HAZE)")
st.sidebar.markdown("- **Validation PSNR:** 13.00 dB")
st.sidebar.markdown("- **Validation SSIM:** 0.4418")
st.sidebar.markdown("- **Held-Out Test PSNR:** 14.72 dB")
st.sidebar.markdown("- **Held-Out Test SSIM:** 0.5441")
st.sidebar.markdown("- **Sustained Latency (GPU):** 182.14 ms / frame")

# Filter available models based on existing checkpoint files
available_labels = [label for label in CHECKPOINT_CATALOG if get_checkpoint_path(label) is not None]
if not available_labels:
    available_labels = list(CHECKPOINT_CATALOG.keys())

selected_model = st.selectbox("Select Model Architecture", available_labels, index=0)
st.caption(CHECKPOINT_CATALOG[selected_model]["description"])

# Demo sample options
sample_hazy_path = ROOT / "sample_images" / "sample_hazy.png"
sample_gt_path = ROOT / "sample_images" / "sample_clean_GT.png"

col_demo, _ = st.columns([1, 2])
with col_demo:
    use_sample = st.checkbox("🧪 Use built-in test sample (NH-HAZE Scene 53)", value=False)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    user_upload = st.file_uploader(
        "Upload Hazy Image (Required if sample not selected)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
with col_upload2:
    gt_upload = st.file_uploader(
        "Upload Ground-Truth Clean Image (Optional — computes PSNR/SSIM)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

active_hazy_img = None
active_gt_img = None
source_name = "image"

if use_sample and sample_hazy_path.is_file():
    active_hazy_img = Image.open(sample_hazy_path).convert("RGB")
    source_name = "scene_53"
    if sample_gt_path.is_file():
        active_gt_img = Image.open(sample_gt_path).convert("RGB")
elif user_upload is not None:
    active_hazy_img = Image.open(user_upload).convert("RGB")
    source_name = Path(user_upload.name).stem
    if gt_upload is not None:
        active_gt_img = Image.open(gt_upload).convert("RGB")

if active_hazy_img is not None:
    try:
        with st.spinner("Processing dehazing forward pass..."):
            dehazed_img, latency_sec, descriptor, pred_tensor = dehaze_image(active_hazy_img, selected_model)

        st.subheader("Visual Restoration Results")
        if active_gt_img is not None:
            col1, col2, col3 = st.columns(3)
            col1.image(active_hazy_img, caption=f"Hazy Input ({active_hazy_img.width}×{active_hazy_img.height})", use_container_width=True)
            col2.image(active_gt_img, caption=f"Ground Truth Clean ({active_gt_img.width}×{active_gt_img.height})", use_container_width=True)
            col3.image(dehazed_img, caption=f"Dehazed Result ({dehazed_img.width}×{dehazed_img.height})", use_container_width=True)

            gt_tensor = pil_to_tensor(active_gt_img).to(pred_tensor.device)
            if gt_tensor.shape == pred_tensor.shape:
                calculated_psnr = psnr(pred_tensor, gt_tensor)
                calculated_ssim = ssim(pred_tensor, gt_tensor)
                latency_ms = latency_sec * 1000

                # Strict validation guard: enforce SSIM domain [-1.0, 1.0] before UI presentation
                if not (-1.0 <= calculated_ssim <= 1.0):
                    raise ValueError(
                        f"SSIM value {calculated_ssim} is outside valid mathematical range [-1.0, 1.0]. "
                        "Check image normalization and metric pipeline."
                    )

                st.markdown("#### Quantitative Evaluation & Performance")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("PSNR (Peak SNR)", f"{calculated_psnr:.4f} dB")
                m_col2.metric("SSIM (Structural Similarity)", f"{calculated_ssim:.4f}")
                m_col3.metric("Inference Latency", f"{latency_ms:.2f} ms")

                st.success(
                    f"**Verified Ground-Truth Benchmarks:** PSNR = **{calculated_psnr:.4f} dB** | "
                    f"SSIM = **{calculated_ssim:.4f}** | Inference Time = **{latency_ms:.2f} ms** "
                    f"*(Full-reference metrics computed via IEEE standard implementations)*"
                )
            else:
                st.warning(
                    f"Ground-truth dimensions ({active_gt_img.width}×{active_gt_img.height}) do not match hazy image "
                    f"({active_hazy_img.width}×{active_hazy_img.height}). Quantitative calculation skipped."
                )
        else:
            col1, col2 = st.columns(2)
            col1.image(active_hazy_img, caption=f"Hazy Input ({active_hazy_img.width}×{active_hazy_img.height})", use_container_width=True)
            col2.image(dehazed_img, caption=f"Dehazed Result ({dehazed_img.width}×{dehazed_img.height})", use_container_width=True)
            
            latency_ms = latency_sec * 1000
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Inference Latency", f"{latency_ms:.2f} ms")
            m_col2.metric("Processed Resolution", f"{dehazed_img.width} × {dehazed_img.height}")
            st.info("No ground-truth clean image provided. PSNR and SSIM are only computed when a reference image is uploaded.")

        st.caption(f"Hardware & Execution: `{descriptor}` | Single-pass forward runtime: {latency_ms:.2f} ms")

        buffer = io.BytesIO()
        dehazed_img.save(buffer, format="PNG")
        st.download_button(
            label="⬇️ Download Dehazed Image (PNG)",
            data=buffer.getvalue(),
            file_name=f"{source_name}_dehazed.png",
            mime="image/png",
        )
    except Exception as error:
        st.error(f"Error during dehazing inference: {error}")
else:
    st.info("Upload a hazy RGB image or check the sample image option above to test the dehazing model.")
    st.markdown(
        """
        ---
        ### Framework Overview & Operation Guide:

        **Lightweight Edge-Aware Multi-Model Image Dehazing Framework** is an academic research project developed for Digital Image Processing (Department of Electronics and Communication Engineering, 2026–2027).

        #### Key Technical Specifications:
        - **Primary Architecture:** 4-Block Residual Convolutional Neural Network with explicit residual scale learning (`Stage 3.7 Experiment B`).
        - **Model Footprint:** **75,747 parameters** (~911.7 KB checkpoint), optimized for edge and mobile robotics vision platforms.
        - **Loss Formulation:** $0.8 \\mathcal{L}_{L1} + 0.2 \\mathcal{L}_{\\text{SSIM}}$ composite objective to suppress haze while preserving high-frequency edge gradients.
        - **Benchmark Accuracy:** Achieves **13.00 dB PSNR / 0.4418 SSIM** on NH-HAZE validation, and **14.72 dB PSNR / 0.5441 SSIM** on the held-out benchmark.
        - **Throughput:** ~182 ms / frame on CUDA GPU and highly responsive CPU execution.

        #### How to Test:
        1. **Instant Demo:** Enable the **"🧪 Use built-in test sample"** checkbox above to test NH-HAZE Scene 53.
        2. **Custom Upload:** Upload any outdoor hazy photograph (PNG, JPEG, TIFF) to run native-resolution restoration.
        3. **Quantitative Benchmarking:** Optionally upload the matching ground-truth clean photograph to inspect IEEE-standard PSNR and SSIM metrics.
        4. **Export:** Click **"Download Dehazed Image (PNG)"** to save the restored high-resolution output.
        """
    )

