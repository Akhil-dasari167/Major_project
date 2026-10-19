# Final Pre-Submission Audit Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Audit Execution Date:** October 19, 2026  
**Auditor:** Antigravity Autonomous Pre-Submission Auditor  

---

## 1. Itemized Pre-Submission Audit Checklist

| Item # | Verification Task | Rating | Evidence & File Paths | Findings & Discrepancies |
|---|---|:---:|---|---|
| **1** | **Read Core Project Reports** | **PASS** | [`reports/FINAL_HANDOVER.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_HANDOVER.md)<br>[`reports/FINAL_PROJECT_REPORT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_PROJECT_REPORT.md)<br>[`reports/FINAL_RESULTS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_RESULTS.md)<br>[`reports/PRESENTATION_AND_VIVA_GUIDE.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/PRESENTATION_AND_VIVA_GUIDE.md)<br>[`reports/PROJECT_PROGRESS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/PROJECT_PROGRESS.md) | All 5 documents inspected. Minor ambiguities regarding test set terminology and latency were safely corrected. |
| **2** | **Unit Test Suite Verification** | **PASS** | Command: `python -m unittest discover tests -v`<br>Duration: 2.538s | **26 passed, 0 failed, 0 skipped.** 100% pass rate across all test modules. |
| **3** | **Streamlit Application (`app.py`)** | **PASS** | File: [`app.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/app.py)<br>Test script: `streamlit.testing.v1.AppTest`<br>Input: `data/NH-HAZE/53_hazy.png` | App initializes with **0 exceptions**. Dehazing routine produces valid $1600 \times 1200$ RGB output with finite values in $[0.001, 0.989]$. |
| **4** | **Documented CLI Inference Execution** | **PASS** | Script: [`scripts/infer.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/infer.py)<br>Output: `outputs/dehazed_sample.png` (2,527,591 bytes)<br>Metadata: `outputs/dehazed_sample.summary.json` | Command exited with code 0. Image verified: $1600 \times 1200$, 3-channel RGB, valid dynamic range $[0, 252]$, non-empty (mean 115.14). |
| **5** | **Metrics Cross-Check Against Artifacts** | **PASS** | Artifacts verified:<br>- `outputs/stage3_7_loss_l1_ssim/summary.json`<br>- `outputs/stage3_8_architectural_edge_guidance/summary.json`<br>- `analysis/residual_scale_continuation_report/continuation_summary.json`<br>- `outputs/final_evaluation_20261019/validation_comparison/summary.json`<br>- `outputs/stage3_9_posthoc_comparison/summary.json` | Every single metric in `FINAL_RESULTS.md` matches its underlying evaluation artifact bit-for-bit: Exp B Val PSNR 12.9984 dB, SSIM 0.4418, L1 0.187040; Test PSNR 14.7239 dB, SSIM 0.5441. |
| **6** | **Scientific Test Split Distinction** | **PASS** | Files: `FINAL_PROJECT_REPORT.md`, `FINAL_RESULTS.md`, `TEST_BENCHMARK_PREFLIGHT.md`, `README.md` | The 8 test scenes (`09, 15, 16, 18, 48, 02, 08, 41`) are explicitly designated as **post-hoc held-out scenes** (quarantined during training/tuning, but evaluated post-hoc). Zero claims of an untouched blind test set. |
| **7** | **Latency & Real-Time Performance Audit** | **PASS** | Files: `FINAL_PROJECT_REPORT.md`, `PRESENTATION_AND_VIVA_GUIDE.md`, `README.md` | Distinction established: **347–396 ms** is a cold-start measurement without GPU warmup; **182.14 ms / frame** (~5.5 FPS) is the sustained warm benchmark at full $1600 \times 1200$. True $>30$ FPS robotics operation requires lower resolutions or FP16/TensorRT. |
| **8** | **Architecture & Presentation Consistency** | **PASS** | File: [`src/dehaze/models/multimodel_variant.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/models/multimodel_variant.py) | All 10 modules (stem, multi-scale, AAFM, 3 branches, fusion, edge gating, color/texture, bounded residual tail) physically exist and are verified. Exp B (4-block CNN) clearly distinguished from experimental multi-model variant. |
| **9** | **README Operational Commands Audit** | **PASS** | File: [`README.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/README.md) | Setup, test, inference, and demo commands match the codebase and execute without errors. |

---

## 2. Detailed Technical Evidence

### 2.1 Unit Test Execution Evidence
```text
Command: .\.venv\Scripts\python.exe -m unittest discover tests -v
Ran 26 tests in 2.538s
OK (26 passed, 0 failed, 0 skipped)
```

### 2.2 CLI Inference Verification Evidence
```text
Command: .\.venv\Scripts\python.exe scripts/infer.py --checkpoint outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt --input data/NH-HAZE/53_hazy.png --output outputs/dehazed_sample.png --device cuda
Exit Code: 0
Generated File: outputs/dehazed_sample.png
Dimensions: (1600, 1200) | Mode: RGB | Channels: 3
Dynamic Range: [0, 252] | Pixel Mean: 115.14
```

### 2.3 Streamlit Application Verification Evidence
```text
AppTest Title: "Lightweight Edge-Aware Multi-Model Image Dehazing Framework"
Exceptions: 0
Image Dehazing Execution: Success (Output size: 1600x1200, Range: [0.001, 0.989])
```

---

## 3. Discrepancies and Corrective Actions Taken

1. **Test Set Description:**
   - *Previous state:* Some summaries referred to the 8 test scenes as "quarantined held-out test partition" without emphasizing their post-hoc nature.
   - *Corrective Action:* Refined `FINAL_PROJECT_REPORT.md`, `PRESENTATION_AND_VIVA_GUIDE.md`, and `README.md` to consistently designate them as **post-hoc held-out scenes**, noting that while quarantined during Stage 3.7/3.8 training and checkpoint selection, baseline scores had been evaluated in Stage 3.6.
2. **Inference Latency Qualification:**
   - *Previous state:* A single-pass timing of ~396 ms was reported alongside the sustained 182 ms latency without highlighting cold-start context.
   - *Corrective Action:* Explicitly documented that ~347–396 ms is a **cold-start measurement** on an initial forward pass, while **182.14 ms / frame** (~5.5 FPS) represents sustained warm inference with CUDA synchronization at full $1600 \times 1200$ resolution.

---

## 4. Final Verdict

# **READY FOR SUBMISSION**

**Justification:**  
All 10 project modules are implemented in code and backed by automated unit tests (26/26 passed). The recommended model (**Stage 3.7 Experiment B**) is backed by verifiable checkpoints, machine-readable metrics, and visual comparison sheets. All operational commands (CLI inference and Streamlit GUI) execute flawlessly with zero errors. All academic and defense reports are complete, accurate, and scientifically defensible.
