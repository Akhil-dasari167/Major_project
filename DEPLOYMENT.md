# Public Web Deployment Guide: Streamlit Community Cloud

This guide provides step-by-step instructions to deploy the **Lightweight Edge-Aware Image Dehazing** framework as a public web application accessible via a global URL (e.g., `https://dehaze-dip.streamlit.app`).

---

## 1. Deployment Architecture Overview

- **Web Framework:** Streamlit (Python 3.11)
- **Deployment Platform:** [Streamlit Community Cloud](https://share.streamlit.io) (Free hosting tier)
- **Runtime Environment:** Linux Debian headless container
- **Compute Hardware:** Cloud CPU instance (automatically detected; optional CUDA GPU if deployed on self-hosted GPU nodes)
- **Primary Deployed Model:** Stage 3.7 Experiment B (`models/best_epoch_019.pt`, 75.7K parameters, 911 KB)
- **Requirements File:** [`requirements-deployment.txt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/requirements-deployment.txt) (Streamlined runtime stack without CUDA/GUI build dependencies)

---

## 2. Checkpoint Storage Strategy

The production checkpoint is staged at [`models/best_epoch_019.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/models/best_epoch_019.pt).

### Git Storage Compliance:
- **File size:** 933,606 bytes (~0.89 MB).
- GitHub permits direct file commits up to **100 MB** (warning threshold: 25 MB).
- Because our model is under 1 MB, it is committed directly to the repository without requiring Git LFS or external cloud storage.

### Optional External Hosting (Without Git Storage):
If you prefer not storing `.pt` binary weights in Git:
1. Upload `best_epoch_019.pt` to a public GitHub Release, Hugging Face Model Hub, or cloud bucket (AWS S3, Google Cloud Storage).
2. In your Streamlit Cloud Dashboard, navigate to **App Settings > Secrets** and define:
   ```toml
   CHECKPOINT_URL = "https://your-public-url.com/best_epoch_019.pt"
   ```
3. The app will automatically download and cache the checkpoint on container boot.

---

## 3. Step-by-Step Deployment Instructions

### Step 3.1: Push Repository to GitHub
Ensure your local project is pushed to your personal or organization GitHub account:
```bash
# 1. Initialize git if not already done
git init
git add .
git commit -m "feat: complete lightweight edge-aware dehazing framework and web demo"

# 2. Add your GitHub remote and push
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git branch -M main
git push -u origin main
```

> [!IMPORTANT]
> The updated [`.gitignore`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/.gitignore) automatically excludes `.venv`, heavy raw datasets (`data/NH-HAZE/` 330 MB+), intermediate outputs (`outputs/`), and private files while preserving `models/best_epoch_019.pt` and `sample_images/`.

---

### Step 3.2: Connect and Deploy on Streamlit Community Cloud

1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Select **I already have an app**.
4. Configure the deployment settings:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Choose a custom subdomain (e.g., `outdoor-dehazing.streamlit.app`).
5. Open **Advanced settings**:
   - **Python version:** Select `3.11`.
6. Click **Deploy!**

---

### Step 3.3: Deployment Build Process

Streamlit Community Cloud will automatically:
1. Spin up a secure container.
2. Detect `requirements-deployment.txt` and install:
   - `streamlit>=1.30.0`
   - `torch>=2.2.0` (standard lightweight Linux wheel)
   - `torchvision>=0.17.0`
   - `Pillow>=10.0.0`
   - `numpy>=1.24.0`
   - `scikit-image>=0.21.0`
3. Launch `app.py`.
4. Provide a public URL (typically live within 60–90 seconds).

---

## 4. Web Application Features Available Online

Once deployed, the public webpage provides:
1. **Interactive Image Dehazing:** Upload any outdoor photograph (`PNG`, `JPEG`, `BMP`, `TIFF`).
2. **One-Click Built-in Demo:** Toggle "Use built-in test sample (Scene 53)" to instantly run dehazing without needing an image file.
3. **Optional Ground Truth Comparison:** Upload a reference clean image to calculate real-time **PSNR** and **SSIM** metrics.
4. **Side-by-Side Visual Inspection:** High-resolution side-by-side preview with responsive scaling.
5. **One-Click PNG Download:** Save the dehazed output directly to local storage.
6. **Hardware Transparency:** Automatically displays whether the instance is running on CPU or GPU and profiles per-image latency.

---

## 5. Troubleshooting & FAQ

### Q1: Will the app crash on CPU-only Streamlit Cloud instances?
**No.** All checkpoint loading logic in `app.py` uses `resolve_device()`, falling back to CPU. CUDA synchronizations are strictly bypassed when running on CPU. CPU inference takes ~370 ms on full $1600 \times 1200$ images.

### Q2: Why use `requirements-deployment.txt` instead of the full `requirements.txt`?
The full development `requirements.txt` includes CUDA 12.1-specific PyTorch wheels and OpenCV GUI bindings which can cause build errors or missing `libGL.so.1` warnings on Linux servers. `requirements-deployment.txt` contains only pure Python and headless scientific wheels, ensuring 100% build reliability.

### Q3: How do I restart or clear cache on Streamlit Cloud?
In the bottom-right corner of your deployed app, click the **Manage app** menu (three dots) > **Clear cache and rerun** or **Reboot app**.
