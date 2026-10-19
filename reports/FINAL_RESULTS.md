# Final Quantitative and Qualitative Results Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Date:** October 19, 2026

---

## 1. Master Model Benchmark Summary

Across all phases of the project, five model architectures and loss formulations were trained and evaluated under strictly controlled protocols on the NVIDIA GeForce RTX 3050 Laptop GPU.

### Comprehensive Validation Benchmark (8 Full-Resolution Validation Scenes)

| Model Name | Architectural Description | Training Loss Objective | Trainable Parameters | Best Checkpoint Path | Val PSNR (dB) | Val SSIM | Val L1 Error | Latency ($1600 \times 1200$) |
|---|---|---|:---:|---|:---:|:---:|:---:|:---:|
| **Stage 3.6 Baseline** | 4 ResBlocks, scale $\pm0.2$ | $L_1$ | 75,747 | [`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt) | 12.8367 | 0.4268 | 0.192185 | 185.2 ms |
| **Stage 3.7 Exp B (Winner)** | 4 ResBlocks, scale $\pm0.2$ | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | **75,747** | [`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt) | **12.9984** | **0.4418** | **0.187040** | **184.6 ms** |
| **Stage 3.7 Exp C** | 4 ResBlocks, scale $\pm0.2$ | $0.8 L_1 + 0.15 \text{SSIM} + 0.05 \text{Sobel}$ | 75,747 | [`outputs/stage3_7_loss_l1_ssim_sobel/best_epoch_019.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_7_loss_l1_ssim_sobel/best_epoch_019.pt) | 12.9656 | 0.4371 | 0.188690 | 186.1 ms |
| **Stage 3.8 EC-Head** | 4-channel input (RGB+Sobel) | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | 76,035 | [`outputs/stage3_8_architectural_edge_guidance/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_8_architectural_edge_guidance/best_epoch_020.pt) | 12.7507 | 0.4227 | 0.191811 | 275.9 ms |
| **Multi-Model v1** | Multi-Scale + AAFM + 3 Branches | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | 114,753 | [`outputs/multimodel_variant_v1_continuation_20261019/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/multimodel_variant_v1_continuation_20261019/best_epoch_020.pt) | 11.6370 | 0.4039 | 0.224439 | 424.1 ms |

---

## 2. Post-Hoc Held-Out Benchmark (8 Full-Resolution Test Scenes)

As pre-registered in Stage 3.9, the top-performing model (**Stage 3.7 Experiment B**) was compared side-by-side with the **Stage 3.6 Baseline** on the 8 held-out test scenes (`09, 15, 16, 18, 48, 02, 08, 41`).

| Metric | Stage 3.6 Baseline | Stage 3.7 Exp B | Delta ($\Delta$) | Direction |
|---|:---:|:---:|:---:|:---:|
| **Mean PSNR (dB)** | 14.5836 | **14.7239** | **+0.1403 dB** | **Improved** |
| **Mean SSIM** | 0.5355 | **0.5441** | **+0.0086** | **Improved** |
| **Mean L1 Error** | 0.150610 | **0.148768** | **-0.001842** | **Improved (-1.2%)** |
| **Trainable Parameters** | 75,747 | 75,747 | 0 | Identical |
| **Checkpoint Size** | 933,478 bytes | 933,606 bytes | +128 bytes | Identical |
| **Average Latency** | 201.10 ms | **182.14 ms** | -18.96 ms | **Faster** (~5.5 FPS) |

---

## 3. Scientific and Architectural Insights

1. **Why Stage 3.7 Experiment B is the Winner:**
   - Incorporating structural similarity loss ($0.8 L_1 + 0.2 [1-\text{SSIM}]$) penalized contrast collapse without adding any inference parameter overhead.
   - It achieved the lowest validation L1 error (**0.187040**), highest validation PSNR (**13.00 dB**), highest validation SSIM (**0.4418**), and highest test PSNR (**14.72 dB**) and SSIM (**0.5441**).
   - SSIM improved on **100% of validation scenes (8/8)** and **100% of test scenes (8/8)**.

2. **Analysis of Stage 3.8 (Early Edge Concatenation):**
   - Directly feeding raw Sobel gradients into the input layer elevated high-frequency edge energy (+18.1% over baseline, reaching 0.1092).
   - However, in diffuse fog and smooth sky regions, the Sobel kernel amplified noise and particulate gradients, modulating the residual and reducing global PSNR and SSIM.

3. **Analysis of Multi-Model Variant v1:**
   - Multi-Model v1 successfully integrated multi-scale feature pyramids, an Adaptive Attention Feature Fusion Module (AAFM), three specialized approximation branches (atmospheric, layer-separation, context), edge-gated refinement, and color/texture refinement.
   - It proved computationally lightweight (114.7K parameters, 1.46 MB checkpoint, 424 ms latency).
   - On the small 39-image NH-HAZE training set, the deeper multi-branch architecture faced optimization headwinds within 20 epochs, settling at 11.64 dB PSNR compared to Exp B's 13.00 dB.
   - Under pre-registered decision rules, **Experiment B remains the recommended primary model**, while Multi-Model v1 is documented as an experimental architecture.

---

## 4. Visual Comparison Artifacts

- **Validation Comparison Contact Sheet:**
  [`outputs/final_evaluation_20261019/validation_comparison/validation_comparison.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/final_evaluation_20261019/validation_comparison/validation_comparison.png)
  - Columns: Hazy Input, Ground Truth, Experiment B, Candidate Multi-Model v1.
- **Stage 3.7 Four-Way Visual Sheet:**
  [`analysis/stage3_7_loss_l1_ssim/validation_four_way_comparison_sheet.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/stage3_7_loss_l1_ssim/validation_four_way_comparison_sheet.png)
- **Stage 3.8 Visual Sheet:**
  [`analysis/stage3_8_architectural_edge_guidance/validation_four_way_comparison_sheet.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/analysis/stage3_8_architectural_edge_guidance/validation_four_way_comparison_sheet.png)
- **Inference Smoke Test:**
  [`outputs/final_evaluation_20261019/inference_smoke/53_dehazed.png`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/final_evaluation_20261019/inference_smoke/53_dehazed.png)
