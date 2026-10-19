# Deployed Model Checkpoints

This directory contains the production checkpoints for the public Streamlit web application.

## 1. Primary Production Model: Stage 3.7 Experiment B
- **File:** `best_epoch_019.pt`
- **Architecture:** `ResidualScaleVariant(residual_scale=0.2)`
- **Parameters:** 75,747 (FP32)
- **File Size:** 933,606 bytes (911.7 KB)
- **Validation Metrics:** PSNR: **13.00 dB** | SSIM: **0.4418** | L1: **0.187040**
- **Test Metrics:** PSNR: **14.72 dB** | SSIM: **0.5441** | L1: **0.148768**

## 2. Git and Hosting Limitations
Because `best_epoch_019.pt` is under **1 MB** (911 KB), it is well within GitHub's 100 MB per-file and 25 MB warning limits. It can be safely checked into Git directly.

### Alternative Remote Hosting Method (Without Git Storage):
If you choose not to store `.pt` files in your Git repository:
1. Upload `best_epoch_019.pt` to a public GitHub Release, Hugging Face model repository, or public S3/GCS bucket.
2. In your Streamlit Cloud Dashboard, go to **Settings > Secrets** and add:
   ```toml
   CHECKPOINT_URL = "https://your-public-storage.com/best_epoch_019.pt"
   ```
3. The application will automatically download and cache the checkpoint on first launch without requiring credentials.
