# Final Project Handover Document

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Completion Date:** October 19, 2026  
**Overall Status:** **PASS (All Project Phases Successfully Verified and Completed)**

---

## 1. Executive Handover Summary

The major project has reached full completion. All technical requirements, architectural components, test suites, evaluation scripts, CLI inference workflows, web demonstration interfaces, and academic reports are verified and delivered.

---

## 2. Recommended Model Checkpoint

- **Recommended Checkpoint:** **Stage 3.7 Experiment B**
- **Exact File Path:** [`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt)
- **Model Architecture:** `ResidualScaleVariant(residual_scale=0.2)`
- **Loss Objective:** $0.8 \mathcal{L}_{L1} + 0.2 [1 - \text{SSIM}]$
- **Trainable Parameters:** Exactly **75,747 parameters**
- **Checkpoint Footprint:** **933,606 bytes** (933.6 KB FP32)
- **Primary Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1)

### Verified Validation Performance (8 Canonical Validation Scenes):
- **Validation PSNR:** **12.9984 dB** (~13.00 dB)
- **Validation SSIM:** **0.4418**
- **Validation L1 Error:** **0.187040** (Lowest error recorded in the project)
- **Mean Edge Gradient Energy:** **0.0992** (+7.1% over baseline)
- **Near-Bound Saturation:** 27.5% (Smooth, unstaturated residual scaling)
- **Clamping Fraction:** 0.02% (Virtually zero pixel clipping)

### Verified Held-Out Test Performance (8 Post-Hoc Test Scenes):
- **Test PSNR:** **14.7239 dB** (+0.14 dB over Stage 3.6 baseline)
- **Test SSIM:** **0.5441** (+0.0086 over Stage 3.6 baseline; improved on 8/8 test scenes)
- **Test L1 Error:** **0.148768** (-1.2% error reduction over baseline)
- **Mean Full-Resolution Latency:** **182.14 ms / frame** (~5.5 FPS at $1600 \times 1200$)

---

## 3. Comparison of Models Evaluated

| Model Variant | Checkpoint Path | Parameters | Val L1 Error | Val PSNR | Val SSIM | Recommendation Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Stage 3.6 Baseline** | `outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt` | 75,747 | 0.192185 | 12.84 dB | 0.4268 | Historical Baseline |
| **Stage 3.7 Exp B** | `outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt` | **75,747** | **0.187040** | **13.00 dB** | **0.4418** | **RECOMMENDED WINNER** |
| **Stage 3.7 Exp C** | `outputs/stage3_7_loss_l1_ssim_sobel/best_epoch_019.pt` | 75,747 | 0.188690 | 12.97 dB | 0.4371 | Experimental Loss Variant |
| **Stage 3.8 EC-Head** | `outputs/stage3_8_architectural_edge_guidance/best_epoch_020.pt` | 76,035 | 0.191811 | 12.75 dB | 0.4227 | Experimental Edge Head |
| **Multi-Model v1** | `outputs/multimodel_variant_v1_continuation_20261019/best_epoch_020.pt` | 114,753 | 0.224439 | 11.64 dB | 0.4039 | Experimental Architecture |

---

## 4. Test Suite and Regression Results

- **Command:** `python -m unittest discover tests`
- **Pass Rate:** **26 / 26 unit tests passed** in **2.73s**.
- **Skipped Checks:** Zero tests skipped; zero failures; zero errors.

---

## 5. Ready-to-Run Operational Commands

### 5.1 Activate Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 5.2 Launch Streamlit Interactive Demo App
```powershell
streamlit run app.py
```

### 5.3 Single-Image CLI Inference
```powershell
python scripts/infer.py --checkpoint outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt --input data/NH-HAZE/53_hazy.png --output outputs/dehazed_53.png --device cuda
```

### 5.4 Batch Folder CLI Inference
```powershell
python scripts/infer.py --checkpoint outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt --input data/NH-HAZE --output outputs/batch_predictions --device cuda
```

### 5.5 Run Unit Tests
```powershell
python -m unittest discover tests
```

---

## 6. Directory and Artifact Inventory

| Category | Artifact Path | Description |
|---|---|---|
| **Academic Report** | [`reports/FINAL_PROJECT_REPORT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_PROJECT_REPORT.md) | Comprehensive academic report with abstract, methodology, math, Mermaid diagram, and references. |
| **Results Summary** | [`reports/FINAL_RESULTS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_RESULTS.md) | Consolidated metrics, comparison tables, and visual contact sheets. |
| **Dataset Audit** | [`reports/DATASET_AUDIT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/DATASET_AUDIT.md) | Full audit of NH-HAZE pairs, OHaze limitations, and split disjointness. |
| **Test Preflight** | [`reports/TEST_BENCHMARK_PREFLIGHT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/TEST_BENCHMARK_PREFLIGHT.md) | Held-out benchmark reproducibility and reuse audit. |
| **Remaining Work Audit** | [`reports/REMAINING_WORK_AUDIT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/REMAINING_WORK_AUDIT.md) | Technical environment and task verification audit. |
| **Progress Tracker** | [`reports/PROJECT_PROGRESS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/PROJECT_PROGRESS.md) | Milestone tracking across Phases A through K. |
| **Changelog** | [`reports/CHANGELOG.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/CHANGELOG.md) | File-by-file record of all additions and modifications. |
| **Experiment Log** | [`reports/EXPERIMENT_LOG.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/EXPERIMENT_LOG.md) | Chronological log of all training and evaluation experiments. |
| **Validation Sheet** | [`outputs/final_evaluation_20261019/validation_comparison/validation_comparison.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/final_evaluation_20261019/validation_comparison/validation_comparison.png) | 4-way comparison contact sheet (Input, GT, Exp B, Candidate). |
| **Validation CSV** | [`outputs/final_evaluation_20261019/validation_comparison/per_image_metrics.csv`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/final_evaluation_20261019/validation_comparison/per_image_metrics.csv) | Machine-readable per-image metrics on validation scenes. |
| **Test Evaluation CSV**| [`outputs/stage3_9_posthoc_comparison/per_image_metrics.csv`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_9_posthoc_comparison/per_image_metrics.csv) | Machine-readable per-image metrics on 8 post-hoc test scenes. |
| **Web Application** | [`app.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/app.py) | Streamlit demonstration app. |
| **Inference CLI** | [`scripts/infer.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/infer.py) | Generic inference entry point. |

---

## 7. Limitations & Honest Assessment

1. **Dataset Limitation:** NH-HAZE consists of 55 real outdoor scenes. While it represents real outdoor smoke/fog capture, training complex deep networks (e.g., Multi-Model v1) from scratch without pretraining leads to underfitting/generalization trade-offs within small epoch budgets.
2. **OHaze Directory:** `data/OHazeResultsSmall` contains baseline output images only (no hazy inputs or ground truth clean images). It cannot be used for quantitative PSNR/SSIM.
3. **Edge Guidance Dynamics:** Explicit edge concatenation (Stage 3.8) boosts edge energy significantly (+18.1%), but requires careful gating in diffuse fog to avoid residual noise modulation.

---

## 8. User Action Required

**None.** All phases have executed safely and autonomously. All acceptance criteria are satisfied, all tests pass, and all deliverables are saved in the project repository.
