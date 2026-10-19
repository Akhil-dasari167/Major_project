# Stage 3.8 Formal Evaluation Report: Lightweight Architectural Edge Guidance

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Experiment:** Stage 3.8 — Lightweight Architectural Edge Guidance (`EdgeConditionedVariant`, 4-channel input)  
**Date:** October 18, 2026  
**Status:** Completed & Evaluated (Validation Only; Held-Out Test Set Strictly Quarantined)

---

## 1. Executive Summary

In Stage 3.8, we implemented and evaluated the proposed **Edge-Conditioned Head (EC-Head)** architecture to determine whether explicit architectural edge guidance can resolve the edge suppression observed in earlier stages while matching or exceeding the top-performing metrics of Stage 3.7 Experiment B ($0.8 L_1 + 0.2 [1-\text{SSIM}]$).

### Key Findings
1. **Dramatic Elevation in Edge Recovery:** The Edge-Conditioned architecture achieved the highest edge gradient energy recorded in the project to date:
   - **Mean Edge Energy:** rose from **0.0925** (Ref Baseline) and **0.0992** (Exp B) to **0.1092** in Stage 3.8.
   - This represents a **+18.1% increase over Reference Baseline** and a **+10.1% increase over Experiment B**.
   - Edge energy improved across **8 out of 8 validation scenes**.
2. **Reconstruction Metric Trade-Off:** While edge energy surged, global reconstruction metrics showed a measurable trade-off:
   - **Validation L1:** 0.191811 (vs. Exp B's **0.187040**; +2.5% error, though still better than Ref Baseline's 0.192185).
   - **Validation PSNR:** 12.75 dB (vs. Exp B's **13.00 dB** and Ref Baseline's 12.84 dB; -0.25 dB vs. Exp B).
   - **Validation SSIM:** 0.4227 (vs. Exp B's **0.4418** and Ref Baseline's 0.4268; -0.0191 vs. Exp B).
3. **Scene-Level Variance:**
   - On scenes with prominent geometric structures (Scene 03 and Scene 44), Stage 3.8 achieved higher PSNR than Experiment B (Scene 03: **12.14 dB** vs. 11.85 dB; Scene 44: **12.23 dB** vs. 12.14 dB).
   - On scenes with heavy diffuse fog and smooth sky/backgrounds (Scene 28 and 38), injecting raw Sobel gradients directly into layer 0 caused subtle high-frequency variation in hazy areas, slightly penalizing global SSIM and PSNR.
4. **Pre-Registered Protocol Verdict:**
   - In accordance with the pre-registered decision criteria (Outcome B: Trade-off), **Stage 3.7 Experiment B (`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`) remains the provisional top-performing model** for global image reconstruction quality.
5. **Test Set Quarantine:** The 8 held-out test scenes (`02, 08, 09, 15, 16, 18, 41, 48`) remained **100% quarantined and untouched**.

---

## 2. Architectural Specification & Computational Profile

### 2.1 Model Architecture: `EdgeConditionedVariant`
- **Location:** [`src/dehaze/models/edge_conditioned_variant.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/models/edge_conditioned_variant.py)
- **Input Formulation:** Concatenation of the 3-channel hazy RGB image with an on-the-fly, parameter-free Sobel gradient magnitude map computed from input luminance ($0.2989 R + 0.5870 G + 0.1140 B$).
- **Head:** `nn.Conv2d(4, 32, kernel_size=3, padding=1)` + `nn.ReLU(inplace=True)`
- **Trunk:** 4 standard `ResidualBlock(32)` modules (identical to baseline).
- **Tail:** `nn.Conv2d(32, 3, kernel_size=3, padding=1)` followed by bounded residual addition: $(I_{\text{hazy}} + \tanh(\cdot) \times 0.2)\text{.clamp}(0, 1)$.

### 2.2 Complexity & Latency Audit
- **Trainable Parameters:** **76,035 parameters** ($+288$ parameters, $+0.38\%$ over baseline 75,747).
- **Checkpoint Footprint:** 304.1 KB FP32.
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1).
- **Inference Latency (Full $1200 \times 1600$ Resolution):** **275.98 ms/image** ($\approx 3.62$ FPS).
- **VRAM Utilization:** Peak memory remained $< 1.8$ GB during validation inference.

---

## 3. Quantitative Comparison Across Iterations

### 3.1 Macro Benchmark Summary (8 Validation Scenes)

| Metric | Ref Baseline ($L_1$ Only) | Stage 3.7 Exp B ($L_1 + \text{SSIM}$) | Stage 3.7 Exp C ($+ L_{\text{Sobel}}$) | Stage 3.8 Candidate (EC-Head) | Stage 3.8 vs. Exp B ($\Delta$) | Stage 3.8 vs. Ref ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model Class** | `ResidualScaleVariant` | `ResidualScaleVariant` | `ResidualScaleVariant` | `EdgeConditionedVariant` | — | — |
| **Input Channels** | 3 (RGB) | 3 (RGB) | 3 (RGB) | **4 (RGB + Edge)** | +1 channel | +1 channel |
| **Parameters** | 75,747 | 75,747 | 75,747 | **76,035** | +288 (+0.38%) | +288 (+0.38%) |
| **Best Checkpoint** | `best_epoch_020.pt` | `best_epoch_019.pt` | `best_epoch_019.pt` | `best_epoch_020.pt` | — | — |
| **Validation $L_1$** (Lower is Better) | 0.192185 | **0.187040** | 0.188690 | 0.191811 | +0.00477 (+2.5%) | **-0.00037** (-0.2%) |
| **Validation PSNR (dB)** | 12.8367 | **12.9984** | 12.9656 | 12.7507 | -0.2477 dB | -0.0860 dB |
| **Validation SSIM** | 0.4268 | **0.4418** | 0.4371 | 0.4227 | -0.0191 (-4.3%) | -0.0041 (-1.0%) |
| **Mean Edge Gradient Energy** | 0.0925 | 0.0992 | 0.0970 | **0.1092** | **+0.0100 (+10.1%)** | **+0.0167 (+18.1%)** |
| **Mean Abs Correction** | 0.1189 | 0.1281 | 0.1271 | 0.1076 | -0.0205 | -0.0113 |
| **Near-Bound Fraction ($\ge 0.19$)** | 22.3% | 29.8% | 33.2% | 17.7% | -12.1% | -4.6% |
| **Clamped Fraction ($\le 0 \lor \ge 1$)** | 0.01% | 0.02% | 0.02% | 0.003% | -0.017% | -0.007% |

---

### 3.2 Scene-by-Scene Validation Breakdown

| Scene | Ref PSNR | Exp B PSNR | Stage 3.8 PSNR | Ref SSIM | Exp B SSIM | Stage 3.8 SSIM | Ref Edge | Exp B Edge | Stage 3.8 Edge | GT Edge |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **53** | 13.06 | **13.14** | 13.10 | 0.4469 | **0.4547** | 0.4532 | 0.1019 | 0.1060 | **0.1227** | 0.3650 |
| **03** | 11.84 | 11.85 | **12.14** | 0.4296 | 0.4342 | **0.4396** | 0.1153 | 0.1203 | **0.1316** | 0.4008 |
| **28** | 12.27 | **12.49** | 11.78 | 0.3362 | **0.3499** | 0.3198 | 0.0629 | 0.0687 | **0.0790** | 0.4473 |
| **38** | 13.24 | **13.44** | 12.67 | 0.4354 | **0.4571** | 0.4107 | 0.0846 | 0.0940 | **0.1110** | 0.3013 |
| **06** | 13.71 | **13.91** | 13.82 | 0.4793 | **0.4906** | 0.4702 | 0.0866 | 0.0912 | **0.0945** | 0.3704 |
| **35** | 14.17 | **14.24** | 13.66 | 0.3915 | **0.4163** | 0.3851 | 0.1006 | 0.1116 | **0.1220** | 0.4751 |
| **44** | 11.83 | 12.14 | **12.23** | 0.4162 | **0.4376** | 0.4217 | 0.0834 | 0.0907 | **0.0972** | 0.3393 |
| **07** | 12.57 | **12.78** | 12.59 | 0.4794 | **0.4937** | 0.4817 | 0.1051 | 0.1110 | **0.1152** | 0.3904 |
| **Mean** | **12.84** | **13.00** | **12.75** | **0.4268** | **0.4418** | **0.4227** | **0.0925** | **0.0992** | **0.1092** | **0.3862** |

---

## 4. Qualitative Analysis & Visual Comparison

Visual inspection was performed using the generated 4-way full-resolution validation contact sheet:
[`analysis/stage3_8_architectural_edge_guidance/validation_four_way_comparison_sheet.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/stage3_8_architectural_edge_guidance/validation_four_way_comparison_sheet.png).

### Visual Findings:
1. **Edge Sharpness & Structural Clarity:**
   - In Scene `03` (paved stone paths, fences, and building walls), Stage 3.8 produces visibly sharper boundaries than Experiment B. This corresponds directly with the measured gain in PSNR (+0.29 dB) and SSIM (+0.0054) on Scene 03.
   - Railings, branch silhouettes, and tree trunks across all scenes exhibit enhanced contrast and sharpness.
2. **Diffuse Haze Behavior:**
   - In Scene `38` and Scene `28` (broad hazy sky regions with distant foliage), the Sobel operator computes non-zero gradients on noise and fog turbulence in the input.
   - Because the 4-channel head forwards these edge activations directly into the small 4-block trunk, the model slightly modulates the residual in diffuse fog, producing faint high-frequency variations that lower pixel-level PSNR.
3. **Color Balance & Clamping:**
   - Color rendition remains clean and natural without chrominance drift or hue skewing.
   - The clamped fraction dropped to a negligible **0.003%**, and near-bound saturation dropped from 29.8% to 17.7%, indicating smooth residual scaling without harsh saturation.

---

## 5. Pre-Registered Decision Criteria Evaluation

| Proposal Criterion | Threshold Target | Stage 3.8 Result | Verdict |
|---|:---:|:---:|:---:|
| **Criterion 1: Validation PSNR** | $\ge 13.00$ dB | **12.75 dB** | **Missed** (-0.25 dB vs. Exp B) |
| **Criterion 2: Validation SSIM** | $\ge 0.4418$ | **0.4227** | **Missed** (-0.0191 vs. Exp B) |
| **Criterion 3: Mean Edge Energy** | $> 0.0992$ | **0.1092** | **Exceeded** (+10.1% over Exp B; +18.1% over Ref) |
| **Criterion 4: Validation L1** | $\le 0.1875$ | **0.191811** | **Missed** (+0.00477 vs. Exp B) |
| **Criterion 5: Parameter Budget** | $\le 80,000$ | **76,035** | **Achieved** (+288 params / +0.38%) |
| **Criterion 6: Qualitative Artifacts** | Artifact-free | No ringing; minor texture in diffuse haze | **Partially Achieved** |

### Formal Decision: Outcome B (Trade-off)
Under the pre-registered protocol, because Criteria 1 and 2 were not met despite dramatic gains in edge recovery (Criterion 3), **Experiment B (`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`) remains the overall top-performing model**.

---

## 6. Non-Functional Safeguards & Quarantine Audit

| Check | Requirement | Result | Evidence |
|---|---|:---:|---|
| **Test Set Quarantine** | Zero test scenes accessed, loaded, or evaluated | **VERIFIED** | Training runner strictly loaded `manifest.train` (39) and `manifest.validation` (8). Zero test scenes loaded. Test directory timestamp unchanged. |
| **Model Code Isolation** | Existing models preserved intact | **VERIFIED** | `ResidualScaleVariant` untouched; `EdgeConditionedVariant` implemented in separate module. |
| **Loss & Split Integrity** | Unchanged Exp B loss and Seed-42 split | **VERIFIED** | Loss: $0.8 L_1 + 0.2 (1-\text{SSIM})$. Split manifest hash identical. |
| **Reproducibility** | Seed 42, deterministic loops | **VERIFIED** | All 20 epochs completed deterministically. |

---

## 7. Conclusions & Recommended Next Steps

### Scientific Insight:
Stage 3.8 confirmed the core hypothesis: **architectural edge conditioning directly solves edge suppression** (+18.1% edge recovery over baseline, achieving 0.1092 edge energy). However, raw input-layer concatenation in a small 4-block CNN allows fog noise to propagate into diffuse background regions, trading off global SSIM and PSNR.

### Standing Project Ranking:
1. **Rank 1 (Best Global Reconstruction):** **Stage 3.7 Experiment B** (`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`)
   - Val PSNR: **13.00 dB** | Val SSIM: **0.4418** | Val L1: **0.187040** | Edge: 0.0992 | Params: 75,747
2. **Rank 2 (Best Edge Recovery / Structural Sharpness):** **Stage 3.8 Candidate** (`outputs/stage3_8_architectural_edge_guidance/best_epoch_020.pt`)
   - Val PSNR: 12.75 dB | Val SSIM: 0.4227 | Val L1: 0.191811 | Edge: **0.1092** | Params: 76,035
3. **Rank 3 (Hybrid Loss):** **Stage 3.7 Experiment C** (`outputs/stage3_7_loss_l1_ssim_sobel/best_epoch_019.pt`)
   - Val PSNR: 12.97 dB | Val SSIM: 0.4371 | Val L1: 0.188690 | Edge: 0.0970 | Params: 75,747
4. **Rank 4 (Reference Baseline):** **Stage 3.6 Model** (`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`)
   - Val PSNR: 12.84 dB | Val SSIM: 0.4268 | Val L1: 0.192185 | Edge: 0.0925 | Params: 75,747

### Recommended Options for User Direction:
- **Option 1 (Crown Experiment B & Proceed to Final Test Evaluation):** Formally approve Experiment B as the overall project winner and conduct a single, controlled held-out test evaluation to benchmark against the Stage 3.6 reference (14.58 dB / 0.5355).
- **Option 2 (Edge-Gated Architecture / Stage 3.9):** If both high edge energy AND top PSNR/SSIM are desired simultaneously, investigate a lightweight *edge-gated trunk* (e.g. learning an edge-attention gate that applies edge sharpening only where structural contrast exceeds background fog).
- **Option 3 (Conclude Stage 3 Phase):** Finalize Stage 3 documentation and transition to the multi-model / deployment phase of the major project.

*Awaiting your direction.*
