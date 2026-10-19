# Stage 3.6 — Residual-Scale Continuation Report

**Status: Complete (Validation-Only).**  
**Audit & Report Date:** 2026-10-17 23:51 IST  
**Canonical Split Manifest:** [`outputs/v0_pilot_001/split.json`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/v0_pilot_001/split.json) (Seed 42, SHA-256: `582d66de9c2f91b5a86e67f733a910a4abd133d91e96ccfc07098f88e30e99ba`)  
**Partitioning:** 39 Train / 8 Validation / 8 Test  
**Test Set Guardrail:** Verified 100% intact. Zero test images were loaded, read, or evaluated during this continuation study.

---

## 1. Executive Summary & Decision Rule Application

In accordance with the pre-registered protocol defined in [`stage3_6_residual_scale_continuation_plan.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/residual_scale_continuation_plan/stage3_6_residual_scale_continuation_plan.md), both the reference (±0.1) and alternative (±0.2) residual-scale models were trained for a fixed budget of **20 epochs** (400 optimizer steps and 160 validation forwards per variant) on the NVIDIA GeForce RTX 3050 Laptop GPU.

### Decision Rule Evaluation
> **Pre-Registered Selection Criterion:**  
> Minimum full-image validation L1 on the canonical 8-scene validation split.  
> If $|L1_{\pm0.2} - L1_{\pm0.1}| \le 0.001$, the comparison is deemed inconclusive.  
> If the difference exceeds $0.001$, the variant with the lower validation L1 is designated the provisional winner for exactly one subsequent held-out test evaluation.

| Variant | Residual Bound | Best Epoch | Validation L1 (Primary) | Validation PSNR | Validation SSIM | Near-Bound Saturation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | $\pm 0.1$ | Epoch 20 | **0.220420** | 11.6421 dB | 0.4137 | 41.27% |
| **Alternative** | $\pm 0.2$ | Epoch 20 | **0.192185** | **12.8367 dB** | **0.4268** | **33.89%** |
| **Delta ($\pm0.2 - \pm0.1$)** | — | — | **-0.028235** | **+1.1947 dB** | **+0.0131** | **-7.38%** |

* **Decision Outcome:** The validation L1 reduction of **0.028235** (a **12.81% relative error reduction**) substantially exceeds the pre-registered $0.001$ margin.
* **Provisional Winner:** **Residual scale $\pm0.2$** (`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`) is unambiguously declared the provisional winner under the pre-registered criterion.

---

## 2. Experimental Protocol & Run Identifiers

To guarantee fair, single-factor evaluation, all parameters other than the residual scale parameter in $\text{clamp}(x + \text{scale} \cdot \tanh(R), 0, 1)$ were kept strictly identical:

| Controlled Dimension | Reference Run | Alternative Run |
| :--- | :--- | :--- |
| **Run Directory** | [`outputs/residual_scale_continuation_0p1/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p1) | [`outputs/residual_scale_continuation_0p2_rerun_1/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p2_rerun_1) |
| **Model Class** | `ResidualScaleVariant(scale=0.1)` | `ResidualScaleVariant(scale=0.2)` |
| **Trainable Parameters** | 75,747 | 75,747 (Identical architecture: 3→32, 4 ResBlocks, 32→3) |
| **Training Budget** | 20 epochs (20 batches/epoch = 400 steps) | 20 epochs (20 batches/epoch = 400 steps) |
| **Crop Size / Batch Size**| 256 × 256 / 2 | 256 × 256 / 2 |
| **Optimizer / LR / Loss**| Adam / $\eta = 2\times 10^{-4}$ / Normalized RGB L1 | Adam / $\eta = 2\times 10^{-4}$ / Normalized RGB L1 |
| **Random Seed / Sampler**| 42 / Deterministic `EpochShuffleSampler` | 42 / Deterministic `EpochShuffleSampler` |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU (CUDA 12.1) | NVIDIA RTX 3050 Laptop GPU (CUDA 12.1) |
| **Wallclock Runtime** | 276.94 seconds (~4.6 min) | 290.19 seconds (~4.8 min) |

---

## 3. Documentation of Interrupted Run & Reproducibility Audit

During initial execution of the continuation suite:
1. `outputs/residual_scale_continuation_0p1` executed completely for all 20 epochs.
2. The initial `outputs/residual_scale_continuation_0p2` run was halted after saving epoch 10 due to an environment interruption.
3. The run was resumed with exact parameters via `python scripts/run_residual_scale_continuation.py --scales 0.2 --suffix rerun_1` into `outputs/residual_scale_continuation_0p2_rerun_1`, executing all 20 epochs.
4. **Reproducibility Audit:** We compared model state weights and validation loss at the checkpoint intersection point (Epoch 10):
   * $\sum |\mathbf{W}_{\text{interrupted}} - \mathbf{W}_{\text{rerun}}| = \mathbf{0.0}$
   * $\text{Validation Loss}_{\text{interrupted}} = \mathbf{0.219393659}$
   * $\text{Validation Loss}_{\text{rerun}} = \mathbf{0.219393659}$
   * **Conclusion:** The rerun is bit-for-bit identical to the initial run up to epoch 10 and completed epochs 11–20 deterministically. `outputs/residual_scale_continuation_0p2_rerun_1` is the authoritative run representing the $\pm0.2$ variant.

---

## 4. Full 20-Epoch Metric Progression

The complete progression across all 20 epochs is tabulated below from the verified `history.csv` logs:

| Epoch | $\pm0.1$ Tr L1 | $\pm0.1$ Val L1 | $\pm0.1$ PSNR | $\pm0.1$ SSIM | $\pm0.1$ Near% | $\pm0.2$ Tr L1 | $\pm0.2$ Val L1 | $\pm0.2$ PSNR | $\pm0.2$ SSIM | $\pm0.2$ Near% |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.2243 | 0.239273 | 10.91 dB | 0.4088 | 0.0% | 0.2200 | 0.233267 | 11.17 dB | 0.4073 | 0.0% |
| 2 | 0.2085 | 0.229799 | 11.33 dB | 0.4076 | 57.9% | 0.1944 | 0.227231 | 11.57 dB | 0.3955 | 9.5% |
| 3 | 0.2099 | 0.230183 | 11.32 dB | 0.4074 | 99.4% | 0.2028 | 0.226594 | 11.59 dB | 0.3961 | 28.5% |
| 4 | 0.2030 | 0.230187 | 11.32 dB | 0.4074 | 99.3% | 0.1927 | 0.224928 | 11.63 dB | 0.3998 | 18.3% |
| 5 | 0.1938 | 0.230162 | 11.32 dB | 0.4074 | 98.7% | 0.1792 | 0.225812 | 11.62 dB | 0.3966 | 29.9% |
| 6 | 0.2077 | 0.230211 | 11.32 dB | 0.4074 | 99.6% | 0.1898 | 0.226946 | 11.59 dB | 0.3946 | 35.3% |
| 7 | 0.2023 | 0.230205 | 11.32 dB | 0.4074 | 99.5% | 0.1855 | 0.225306 | 11.63 dB | 0.3973 | 32.3% |
| 8 | 0.1936 | 0.230194 | 11.32 dB | 0.4074 | 99.2% | 0.1800 | 0.223984 | 11.68 dB | 0.3990 | 30.5% |
| 9 | 0.2099 | 0.229986 | 11.33 dB | 0.4076 | 93.9% | 0.1883 | 0.223261 | 11.71 dB | 0.3992 | 30.2% |
| 10 | 0.1958 | 0.229659 | 11.34 dB | 0.4078 | 73.5% | 0.1848 | 0.219394 | 11.83 dB | 0.4039 | 20.6% |
| 11 | 0.1840 | 0.229448 | 11.35 dB | 0.4080 | 71.2% | 0.1679 | 0.215623 | 11.94 dB | 0.4088 | 7.9% |
| 12 | 0.1721 | 0.229354 | 11.35 dB | 0.4081 | 72.6% | 0.1626 | 0.214767 | 12.00 dB | 0.4081 | 21.1% |
| 13 | 0.1984 | 0.230277 | 11.32 dB | 0.4074 | 99.9% | 0.1707 | 0.210395 | 12.13 dB | 0.4134 | 9.5% |
| 14 | 0.2009 | 0.230275 | 11.32 dB | 0.4074 | 99.9% | 0.1784 | 0.204923 | 12.34 dB | 0.4171 | 13.2% |
| 15 | 0.2089 | 0.230227 | 11.32 dB | 0.4074 | 99.7% | 0.1778 | 0.201521 | 12.47 dB | 0.4183 | 15.1% |
| 16 | 0.2009 | 0.230178 | 11.32 dB | 0.4075 | 99.3% | 0.1794 | 0.201197 | 12.46 dB | 0.4213 | 11.8% |
| 17 | 0.2005 | 0.229524 | 11.34 dB | 0.4080 | 83.7% | 0.1759 | 0.201672 | 12.48 dB | 0.4170 | 27.8% |
| 18 | 0.1832 | 0.228165 | 11.37 dB | 0.4088 | 15.8% | 0.1566 | 0.206079 | 12.24 dB | 0.4190 | 12.4% |
| 19 | 0.1832 | 0.224700 | 11.50 dB | 0.4105 | 34.4% | 0.1633 | 0.194666 | 12.75 dB | 0.4240 | 35.0% |
| 20 | 0.1951 | **0.220420** | 11.64 dB | 0.4137 | 41.3% | 0.1685 | **0.192185** | **12.84 dB** | **0.4268** | **33.9%** |

*Note: `Near%` represents the percentage of validation pixels with absolute correction $\ge 95\%$ of the respective scale ceiling.*

---

## 5. In-Depth Residual & Saturation Diagnostics

1. **Ceiling Saturation Relief:**
   * In the $\pm0.1$ reference, the network quickly pinned against its bounding tanh limits: from epochs 3 to 8 and 13 to 16, **over 99% of all pixels were saturating at $\ge 0.095$ absolute correction**. This confirmed the Stage 3.4 diagnosis that the model was artificially restrained from reaching outdoor scene haze depths.
   * In the $\pm0.2$ variant, near-bound saturation dropped to an average of **$20.7\%$** across training and settled at **$33.89\%$** at epoch 20. The optimization trajectory was free to model darker, denser haze regions.
2. **Output Value Clamping to $[0, 1]$:**
   * A primary risk flagged in the continuation plan was that allowing larger residual swings might cause severe pixel clipping to $0$ or $1$.
   * Empirically, output clamping was **negligible**: for $\pm0.2$, the peak clipping across 20 epochs was only **$0.135\%$** (Epoch 6), dropping to **$0.022\%$** ($< 1$ in 4,500 pixels) at Epoch 20.
3. **Correction Directionality:**
   * The mean signed correction for $\pm0.2$ at epoch 20 is **$-0.111143$** (with mean absolute correction of $0.119599$).
   * Over $96\%$ of active corrections are negative (darkening). Because NH-HAZE images suffer from bright white scattering veil, negative residuals directly subtract the airlight component.

---

## 6. Qualitative Validation Observations

We inspected the multi-panel contact sheets generated from the best checkpoint of each variant:
* **Artifact:** [`validation_four_way_contact_sheet.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/residual_scale_continuation_report/validation_four_way_contact_sheet.png) (all 8 validation scenes: `53`, `03`, `28`, `38`, `06`, `35`, `44`, `07`).

**Verified Visual Properties:**
1. **Dehazing Depth & Foliage Contrast:** In dense haze regions (e.g., Scene `03` and Scene `38`), $\pm0.2$ restores significantly deeper green foliage, sharper bark textures, and crisper building silhouettes than $\pm0.1$, which looks washed out and cloudy.
2. **Distant Structural Recovery:** In background trees and distant terrain (Scene `53`), $\pm0.2$ recovers high-frequency edge contours that remain completely veiled under $\pm0.1$.
3. **Color Balance:** $\pm0.2$ introduces a moderately darker, warmer tone in heavily hazy foregrounds. While this contributes to a higher SSIM and PSNR, it highlights the need for edge-aware loss and chromatic preservation in subsequent architectural stages (Stages 3.7+).

---

## 7. Artifact Manifest

| Path | Description |
| :--- | :--- |
| [`outputs/residual_scale_continuation_0p1/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p1/best_epoch_020.pt) | Reference $\pm0.1$ best checkpoint (Val L1: `0.220420`) |
| [`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt) | Alternative $\pm0.2$ best checkpoint (Val L1: `0.192185`) |
| [`analysis/residual_scale_continuation_report/validation_loss_comparison.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/residual_scale_continuation_report/validation_loss_comparison.png) | 4-panel comparison: Loss, PSNR, SSIM, and Saturation curves |
| [`analysis/residual_scale_continuation_report/validation_four_way_contact_sheet.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/residual_scale_continuation_report/validation_four_way_contact_sheet.png) | 8-scene 4-way visual sheet: Input, Target, $\pm0.1$, $\pm0.2$ |
| [`analysis/residual_scale_continuation_report/continuation_summary.json`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/residual_scale_continuation_report/continuation_summary.json) | Consolidated machine-readable delta metrics |

---

## 8. Proposed Next Action (Awaiting Approval)

Stage 3.6 is complete on the validation set. In accordance with the pre-registered protocol:
* The provisional winner is **$\pm0.2$** (`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`).
* **Next Authorized Step:** Execute **exactly one final test evaluation** of this winning checkpoint on the 8 held-out test scenes (`09`, `15`, `16`, `18`, `48`, `02`, `08`, `41`) using [`scripts/test_v0.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/test_v0.py).
* This test run will generate:
  * Per-image and mean test PSNR and SSIM.
  * Measured inference time per image on the RTX 3050 GPU.
  * Model parameter counts and convolution MAC estimates.
  * A test prediction contact sheet and `metrics.csv`.
* **Guardrail:** No test evaluation will be launched until you review this report and grant explicit approval.
