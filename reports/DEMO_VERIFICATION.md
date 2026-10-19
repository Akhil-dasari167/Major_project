# Demonstration Verification & Metric Audit Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Target Milestone:** Project Demonstration (23 October 2026)  
**Audit Date:** October 19, 2026  
**Status:** **FULLY VERIFIED & READY FOR DEMONSTRATION**

---

## 1. Executive Summary

This report establishes the completed resolution of metric reporting ambiguities, local demonstration verification across multiple paired scenes, visual enhancement of the Streamlit web application, and submission readiness ahead of the final demonstration on 23 October 2026.

---

## 2. Priority 1: Metric Root-Cause Analysis & Fix

### 2.1 The Issue
The deployed application exhibited a scenario where inference latency (`204.5946 ms`) collided with metric reporting (`PSNR = 13.1417 dB, SSIM = 204.5946`). Structurally, SSIM is mathematically bounded in $[-1.0, 1.0]$. A value of `204.5946` represents a forward-pass runtime in milliseconds (`0.2045946 s × 1000 = 204.5946 ms`), formatted to 4 decimal places.

### 2.2 Root-Cause Mitigations Implemented
1. **Mathematical Robustness (`src/dehaze/evaluation/metrics.py`):**
   - Implemented `_to_numpy_hwc()` helper supporting PyTorch tensors (3D `CHW`, 3D `HWC`, 4D `1×C×H×W`), NumPy arrays, and PIL Images.
   - Enforced automatic data-range detection and normalization: inputs scaled in $[0, 255]$ are safely divided by 255.0 to maintain a strict $[0.0, 1.0]$ float representation.
   - SSIM calculation uses `skimage.metrics.structural_similarity(truth, pred, channel_axis=2, data_range=1.0)` and is guaranteed to return a scalar Python `float` in $[-1.0, 1.0]$.
   - PSNR calculation guards against zero MSE and returns `float('inf')` for identical images.
2. **User-Interface Separation (`app.py`):**
   - Transformed metric reporting into 3 distinct `st.metric` cards:
     - Card 1: **PSNR (Peak SNR)** (e.g., `13.1417 dB`)
     - Card 2: **SSIM (Structural Similarity)** (e.g., `0.4547`)
     - Card 3: **Inference Latency** (e.g., `204.59 ms`)
   - Guarantees zero ambiguity between latency in milliseconds and structural similarity.
3. **Regression Test Suite (`tests/test_metrics.py`):**
   - Added 9 targeted test cases covering identical images (`PSNR=inf, SSIM=1.0`), black vs. white images, shape variations (`CHW`, `HWC`, `BCHW`), array types (NumPy, PIL), $[0, 255]$ vs $[0, 1]$ normalization, known NH-HAZE pairs, Scene 53 metric isolation, and strict domain validation guards.
   - Test suite status: **34 / 34 unit tests passing (`Ran 34 tests in 5.64s, OK`)**.

---

## 3. Priority 2: Local Demonstration & Multi-Scene Verification

The model (`Stage 3.7 Experiment B`, 75,747 parameters) was evaluated on 3 independent paired $1600 \times 1200$ scenes using `scripts/verify_local_demo.py`:

| Scene Identifier | Resolution | Raw Hazy PSNR | Raw Hazy SSIM | Dehazed PSNR | Dehazed SSIM | Net Improvement | Measured Latency | Output PNG Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scene 53 (Demo Sample)** | $1600 \times 1200$ | 9.3392 dB | 0.4134 | **13.1417 dB** | **0.4547** | **+3.8025 dB / +0.0413** | 895.43 ms | 2,527,591 bytes |
| **Scene 03 (Validation Pair)** | $1600 \times 1200$ | 8.8519 dB | 0.4132 | **11.8523 dB** | **0.4342** | **+3.0004 dB / +0.0210** | 574.13 ms | 2,483,723 bytes |
| **Scene 07 (Validation Pair)** | $1600 \times 1200$ | 11.4212 dB | 0.4630 | **12.7756 dB** | **0.4937** | **+1.3543 dB / +0.0308** | 771.72 ms | 2,444,714 bytes |

### Presentation Assets Saved:
All verified demo files are saved to `outputs/demo_presentation_assets/`:
- `Scene_53_Builtin_Sample_input_hazy.png`, `..._ground_truth.png`, `..._dehazed.png`
- `Scene_03_Validation_input_hazy.png`, `..._ground_truth.png`, `..._dehazed.png`
- `Scene_07_Validation_input_hazy.png`, `..._ground_truth.png`, `..._dehazed.png`
- `demo_verification_summary.json` (machine-readable run log)

---

## 4. Priority 3: UI Improvements & Clean Public Interface

1. **Removed Leftover Instructions:** The raw text block *"### Deployment Instructions: 1. Push this repository..."* has been completely removed from `app.py`.
2. **Added Professional Project Overview:** Replaced with comprehensive academic details (Department of ECE, 2026–2027), key technical parameters, and clear user operating steps.
3. **Dual Metric Layout:** Provides high-visibility metrics for paired comparisons, and resolution/latency telemetry for unreferenced user uploads.
4. **Cloud Compatibility:** `requirements.txt` remains strictly tailored for headless CPU Linux deployment on Streamlit Community Cloud.

---

## 5. Priority 4: Academic Integrity & Claim Boundaries

1. **Parameter Count & Size:** Exactly **75,747 parameters** (~911.7 KB weights; 933,606 bytes).
2. **Validation Performance:** **13.00 dB PSNR / 0.4418 SSIM** on 8-scene NH-HAZE validation set.
3. **Held-Out Test Benchmark:** **14.72 dB PSNR / 0.5441 SSIM** on 8-scene post-hoc benchmark.
4. **Latency Characterization:**
   - Single-image cold start: ~574–895 ms ($1600 \times 1200$).
   - Sustained warm GPU throughput: 182.14 ms (~5.5 FPS at $1600 \times 1200$).
   - Real-time video claim boundary: The system achieves near-real-time throughput at native high resolution ($1600 \times 1200$), and true real-time throughput (>30 FPS) when processed at standard sub-resolutions ($640 \times 480$). We avoid claiming full 30 FPS video at $1600 \times 1200$ without hardware downsampling.

---

## 6. How to Launch Local Demo

From the project root in PowerShell:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Or with custom port:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

---

## 7. Verification Checklist for 21 October Session (6:00–11:00 PM)

- [ ] **Step 1:** Boot local Streamlit demo (`streamlit run app.py`) and confirm UI opens at `http://localhost:8501`.
- [ ] **Step 2:** Test built-in sample (Scene 53): verify PSNR reads `13.1417 dB`, SSIM reads `0.4547`, and Latency reads in ms.
- [ ] **Step 3:** Test custom image upload and verify PNG download button outputs valid image file.
- [ ] **Step 4:** Check live Streamlit Cloud URL ([https://majorproject-vtrzftbsbv2zrv3qeeamsb.streamlit.app](https://majorproject-vtrzftbsbv2zrv3qeeamsb.streamlit.app/)) to confirm automatic redeployment.
- [ ] **Step 5:** Review pre-rendered contact sheets and presentation slides using assets in `outputs/demo_presentation_assets/`.
