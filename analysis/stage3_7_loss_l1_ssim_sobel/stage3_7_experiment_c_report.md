# Stage 3.7 Experiment C: Edge-Gradient Loss Investigation Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Experiment:** Stage 3.7 Experiment C — Edge-Gradient Loss ($L_{\text{total}} = 0.8 \times L_1 + 0.15 \times [1 - \text{SSIM}] + 0.05 \times L_{\text{Sobel}}$)  
**Date:** October 18, 2026  
**Status:** Completed & Validated (Validation Only; Held-Out Test Set Quarantined)

---

## 1. Executive Summary

In Stage 3.7 Experiment C, we investigated whether augmenting the multi-objective loss with an explicit differentiable Sobel edge-gradient loss ($L_{\text{Sobel}}$) recovers suppressed high-frequency structural energy and improves validation PSNR and SSIM over the Reference Baseline ($L_1$-only) and Experiment B ($0.8 L_1 + 0.2 [1 - \text{SSIM}]$).

### Key Takeaways
1. **Edge Recovery vs. Ref Baseline:** Experiment C significantly improves edge sharpness and structural fidelity over the Reference Baseline ($L_1$-only). Best validation edge energy rose from **0.0925** to **0.0970** (a **+4.9%** increase at best epoch 19, reaching **0.0991** at epoch 20). Validation PSNR improved by **+0.13 dB** (from **12.84 dB** to **12.97 dB**) and SSIM improved by **+0.0103** (from **0.4268** to **0.4371**).
2. **Comparison Against Experiment B ($L_1 + \text{SSIM}$):** Experiment B ($0.8 L_1 + 0.2 [1 - \text{SSIM}]$) remains marginally superior across global reconstruction metrics:
   - **Validation L1:** Exp B achieved **0.187040** vs. Exp C's **0.188690** (Exp B is 0.88% lower error).
   - **Validation PSNR:** Exp B achieved **13.00 dB** vs. Exp C's **12.97 dB** (Exp B is +0.03 dB higher).
   - **Validation SSIM:** Exp B achieved **0.4418** vs. Exp C's **0.4371** (Exp B is +0.0047 higher).
   - **Validation Edge Energy:** Exp B achieved **0.0992** vs. Exp C's **0.0970** at best epoch 19 (Exp C reached 0.0991 at epoch 20).
3. **Loss Dynamics & Gradient Behavior:** The gradient and loss values remained strictly finite and well-conditioned throughout all 20 epochs (400 training steps, 160 validation steps). No gradient clipping or instability occurred.
4. **Test Set Quarantine:** The 8 held-out test scenes (`09, 15, 16, 18, 48, 02, 08, 41`) remained **100% untouched and unaccessed**. All analyses and decisions are strictly based on the 8 validation scenes.

---

## 2. Experimental Configuration & Methodology

All experiments were executed on an identical training partition, random seed, optimizer configuration, and model architecture to guarantee strict comparability.

### Hyperparameters & Architecture
- **Model Architecture:** `ResidualScaleVariant` with fixed residual scale $\pm 0.2$.
- **Parameter Count:** Exactly **75,747 parameters** (303.0 KB FP32 checkpoint weights).
- **Dataset:** NH-HAZE (55 paired outdoor hazy/clean images).
- **Split:** Canonical Seed-42 split:
  - **Train (39 pairs):** Random $256 \times 256$ patches, horizontal/vertical flips, deterministic `EpochShuffleSampler`.
  - **Validation (8 pairs):** Full-resolution evaluation (`03, 06, 07, 28, 35, 38, 44, 53`).
  - **Test (8 pairs):** Strictly quarantined (`02, 08, 09, 15, 16, 18, 41, 48`).
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$), initial learning rate $\eta = 2 \times 10^{-4}$.
- **Batch Size:** 2 patches per batch (20 batches / epoch).
- **Epoch Budget:** 20 epochs (400 total training optimizer steps).
- **Hardware & Runtime:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1). Total training time: **321.5 seconds** (~5.3 minutes).

### Loss Formulation
$$L_{\text{total}} = 0.8 \times L_1(\hat{Y}, Y) + 0.15 \times [1 - \text{SSIM}(\hat{Y}, Y)] + 0.05 \times L_{\text{Sobel}}(\hat{Y}, Y)$$

Where:
- $L_1(\hat{Y}, Y) = \frac{1}{C \cdot H \cdot W} \|\hat{Y} - Y\|_1$
- $\text{SSIM}(\hat{Y}, Y)$ is computed across RGB channels using an $11 \times 11$ Gaussian window ($\sigma = 1.5$, $C_1 = 0.01^2$, $C_2 = 0.03^2$).
- $L_{\text{Sobel}}(\hat{Y}, Y) = \frac{1}{H \cdot W} \left( \|\nabla_x \hat{Y}_{\text{gray}} - \nabla_x Y_{\text{gray}}\|_1 + \|\nabla_y \hat{Y}_{\text{gray}} - \nabla_y Y_{\text{gray}}\|_1 \right)$, using standard integer Sobel kernels convolved over standard ITU-R luminance ($Y_{\text{gray}} = 0.2989 R + 0.5870 G + 0.1140 B$).

---

## 3. Quantitative Comparison

### 3.1 Macro Benchmark Summary (8 Validation Scenes)

| Metric | Reference Baseline ($L_1$ Only) | Stage 3.7 Exp B ($L_1 + \text{SSIM}$) | Stage 3.7 Exp C ($L_1 + \text{SSIM} + \text{Sobel}$) | Exp C vs. Ref ($\Delta$) | Exp C vs. Exp B ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Best Checkpoint** | `best_epoch_020.pt` | `best_epoch_019.pt` | `best_epoch_019.pt` | — | — |
| **Validation $L_1$ (Lower is Better)** | 0.192185 | **0.187040** | 0.188690 | **-0.00350** (-1.8%) | +0.00165 (+0.9%) |
| **Validation PSNR (dB)** | 12.8367 | **12.9984** | 12.9656 | **+0.1289 dB** | -0.0328 dB |
| **Validation SSIM** | 0.4268 | **0.4418** | 0.4371 | **+0.0103** (+2.4%) | -0.0047 (-1.1%) |
| **Mean Edge Gradient Energy** | 0.0925 | **0.0992** | 0.0970 | **+0.0045** (+4.9%) | -0.0022 (-2.2%) |
| **Mean Abs Correction** | 0.1189 | 0.1281 | 0.1271 | +0.0082 | -0.0010 |
| **Near-Bound Fraction ($\ge 0.19$)** | 22.3% | 29.8% | 33.2% | +10.9% | +3.4% |
| **Clamped Fraction ($\le 0 \lor \ge 1$)** | 0.01% | 0.02% | 0.02% | +0.01% | 0.00% |
| **Parameters** | 75,747 | 75,747 | 75,747 | 0 (Identical) | 0 (Identical) |

*Note: In Epoch 20 of Exp C, edge energy reached 0.0991 with Val SSIM of 0.4402 and Val PSNR of 12.60 dB; however, epoch 19 achieved the lowest validation L1 (0.188690) and highest validation PSNR (12.97 dB).*

---

### 3.2 Scene-by-Scene Validation Breakdown

The table below breaks down the 8 held-out validation scenes, comparing PSNR, SSIM, and Sobel edge energy against Ground Truth (GT) edge energy.

| Scene | Ref PSNR | Exp B PSNR | Exp C PSNR | Ref SSIM | Exp B SSIM | Exp C SSIM | Ref Edge | Exp B Edge | Exp C Edge | GT Edge |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **53** | 13.06 | 13.14 | 13.11 | 0.4469 | **0.4547** | 0.4520 | 0.1019 | **0.1060** | 0.1042 | 0.3650 |
| **03** | 11.84 | **11.85** | 11.79 | 0.4296 | **0.4342** | 0.4298 | 0.1153 | **0.1203** | 0.1179 | 0.4008 |
| **28** | 12.27 | **12.49** | 12.15 | 0.3362 | **0.3499** | 0.3431 | 0.0629 | **0.0687** | 0.0665 | 0.4473 |
| **38** | 13.24 | **13.44** | 13.43 | 0.4354 | **0.4571** | 0.4512 | 0.0846 | **0.0940** | 0.0899 | 0.3013 |
| **06** | 13.71 | **13.91** | 13.81 | 0.4793 | **0.4906** | 0.4847 | 0.0866 | **0.0912** | 0.0900 | 0.3704 |
| **35** | 14.17 | 14.24 | **14.68** | 0.3915 | **0.4163** | 0.4141 | 0.1006 | **0.1116** | 0.1085 | 0.4751 |
| **44** | 11.83 | **12.14** | 11.83 | 0.4162 | **0.4376** | 0.4294 | 0.0834 | **0.0907** | 0.0895 | 0.3393 |
| **07** | 12.57 | 12.78 | **12.93** | 0.4794 | **0.4937** | 0.4927 | 0.1051 | **0.1110** | 0.1100 | 0.3904 |
| **Mean** | **12.84** | **13.00** | **12.97** | **0.4268** | **0.4418** | **0.4371** | **0.0925** | **0.0992** | **0.0970** | **0.3862** |

#### Observations on Scene-Level Dynamics:
- **Exp C Wins in Dense Textured Scenes:** On Scene `35` (dense tree foliage and heavy foreground haze), Exp C achieved **14.68 dB PSNR**, outperforming Exp B (**14.24 dB**) by **+0.44 dB** and Ref (**14.17 dB**) by **+0.51 dB**. Similarly on Scene `07`, Exp C achieved **12.93 dB PSNR** vs Exp B's **12.78 dB** (+0.15 dB).
- **Exp B Retains Broad SSIM Dominance:** Exp B achieved the highest SSIM on all 8 validation scenes, demonstrating that allocating the full 0.20 structural weight to SSIM ($1 - \text{SSIM}$) aligns better with perceptual structure indices than splitting it into $0.15 \times [1 - \text{SSIM}] + 0.05 \times L_{\text{Sobel}}$.
- **Edge Gradient Recovery:** Both Exp B and Exp C substantially elevated edge gradient energy compared to the Reference Baseline across every scene (e.g. Scene 38 edge energy grew from 0.0846 to ~0.090-0.094).

---

## 4. Qualitative Analysis & Visual Comparison

Visual comparisons were generated at full resolution and are saved in the analysis directory:
- **4-Way Comparison Sheet:** `analysis/stage3_7_loss_l1_ssim_sobel/validation_four_way_comparison_sheet.png` (Columns: Hazy Input, Ground Truth, Exp B, Exp C).
- **5-Way Comparison Sheet:** `analysis/stage3_7_loss_l1_ssim_sobel/validation_five_way_comparison_sheet.png` (Columns: Hazy Input, Ground Truth, Ref Baseline, Exp B, Exp C).
- **Comparative Metric Curves:** `analysis/stage3_7_loss_l1_ssim_sobel/validation_metrics_comparison.png` (Validation L1, PSNR, SSIM, and Edge Gradient Energy across 20 epochs).

### Visual Inspection Findings:
1. **Edge Sharpness & Boundary Transition:**
   - Predictions from both Exp B and Exp C show visibly crisper branch outlines, bark textures, and railing silhouettes compared to the blurrier Ref Baseline.
   - The explicit Sobel loss in Exp C produced subtle edge sharpening in high-frequency textured regions (foliage in Scene 35), but in uniform sky or distant hazy backgrounds, it slightly amplified residual noise gradients.
2. **Color Fidelity & Haze Clearing:**
   - Both Exp B and Exp C retain natural color balance without chrominance drift, oversaturation, or hue shifting.
   - Dehazing penetration in deep fog remains constrained by the $\pm 0.2$ residual bound, with mean absolute correction at ~0.127.
3. **Artifact Analysis:**
   - No haloing, checkerboard patterns, or boundary ringing were observed in Exp C outputs. Clamping fraction remained negligible ($0.02\%$).

---

## 5. Non-Functional Requirements & Quarantine Audit

| Check | Requirement | Result | Evidence |
|---|---|:---:|---|
| **Test Set Quarantine** | Zero test scenes accessed, loaded, or evaluated | **VERIFIED** | `scripts/run_stage3_7_loss_l1_ssim_sobel.py` and `scratch/generate_expc_artifacts.py` strictly loaded `manifest.train` and `manifest.validation`. Test directory timestamp (`17-10-2026 23:57:13`) unchanged. |
| **Model Architecture** | `ResidualScaleVariant` with scale $\pm 0.2$ | **VERIFIED** | Model architecture unchanged; parameter count exactly 75,747. |
| **Loss Weights** | Exactly $0.8 L_1 + 0.15 (1-\text{SSIM}) + 0.05 L_{\text{Sobel}}$ | **VERIFIED** | Verified in `config.json`, runner logs, and `summary.json`. |
| **Numerical Stability** | Finite losses and gradients | **VERIFIED** | All 400 training steps and 160 validation steps executed without NaN or Inf. |
| **Artifact Preservation** | Previous runs intact | **VERIFIED** | `outputs/stage3_7_loss_l1_ssim` and `outputs/residual_scale_continuation_0p2_rerun_1` untouched. |

---

## 6. Interpretation & Recommended Next Project Decisions

### Synthesis of Stage 3.7 Experiments:
1. **Experiment A (Diagnosis):** Identified that the $L_1$-only pilot suffered from edge attenuation (recovering only 24% of GT high-frequency edge energy).
2. **Experiment B ($0.8 L_1 + 0.2 [1 - \text{SSIM}]$):** Produced substantial gains (+0.16 dB PSNR, +0.0150 SSIM, -2.7% L1 error, +7.2% edge energy), achieving the best overall validation metrics (**13.00 dB PSNR, 0.4418 SSIM**).
3. **Experiment C ($0.8 L_1 + 0.15 [1 - \text{SSIM}] + 0.05 L_{\text{Sobel}}$):** Confirmed edge enhancement (+4.9% edge energy over Ref, +0.13 dB PSNR, +0.0103 SSIM), but demonstrated that allocating weight to pixel-space Sobel gradients slightly trades off global SSIM compared to allocating the full 0.20 weight to SSIM.

### Options for User Approval:
- **Option 1 (Select Experiment B as Stage 3.7 Winner):** Adopt the Experiment B model (`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`) as the top-performing Stage 3.7 model, given its superior validation PSNR (13.00 dB), SSIM (0.4418), and L1 loss (0.187040).
- **Option 2 (Proceed to Stage 3.7 Experiment D — Architecture/Feature Edge Guidance):** Explore if an explicit architectural edge branch (e.g. lightweight high-frequency feature concatenation or Laplacian pyramid feature) improves edge energy without loss-weight trade-offs.
- **Option 3 (Approve Final Test Evaluation for Exp B):** Conduct a single, controlled evaluation of Experiment B on the held-out test set to measure the test PSNR/SSIM impact against Stage 3.6 reference (14.58 dB / 0.5355).

*Awaiting user review and direction before taking any further action.*
