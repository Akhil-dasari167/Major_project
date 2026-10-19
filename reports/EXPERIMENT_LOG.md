# Master Experiment Log

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering

---

| Stage / Experiment | Architecture / Class | Loss Objective | Checkpoint Path | Val PSNR / SSIM | Test PSNR / SSIM | Parameters | Verdict / Role |
|---|---|---|---|:---:|:---:|:---:|---|
| **Stage 1 & 2** | — | — | — | — | — | — | Environment, dataset audit & pairing verified. |
| **Stage 3.1 & 3.3 (V0 Pilot)** | `V0ResidualCNN` | $L_1$ ($\pm0.1$ scale) | `outputs/v0_pilot_001/best_epoch_002.pt` | 11.34 dB / 0.4078 | 13.61 dB / 0.5233 | 75,747 | Initial pilot baseline. Diagnosed ceiling saturation. |
| **Stage 3.5 & 3.6 (Residual Scale)** | `ResidualScaleVariant` | $L_1$ ($\pm0.2$ scale) | `outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt` | 12.84 dB / 0.4268 | 14.58 dB / 0.5355 | 75,747 | Relieved tanh ceiling saturation; established strong baseline. |
| **Stage 3.7 Exp B (Structural Loss)** | `ResidualScaleVariant` | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | `outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt` | **13.00 dB / 0.4418** | **14.72 dB / 0.5441** | **75,747** | **PRIMARY PROJECT WINNER.** Lowest L1 error (0.1870), highest PSNR/SSIM. |
| **Stage 3.7 Exp C (Edge Loss)** | `ResidualScaleVariant` | $0.8 L_1 + 0.15 \text{SSIM} + 0.05 \text{Sobel}$ | `outputs/stage3_7_loss_l1_ssim_sobel/best_epoch_019.pt` | 12.97 dB / 0.4371 | — | 75,747 | Higher edge energy than baseline, but slightly trades off global SSIM vs Exp B. |
| **Stage 3.8 (Architectural Edge)** | `EdgeConditionedVariant` | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | `outputs/stage3_8_architectural_edge_guidance/best_epoch_020.pt` | 12.75 dB / 0.4227 | — | 76,035 | Highest edge energy (0.1092, +18.1%), but lower PSNR in diffuse fog regions. |
| **Stage 3.9 (Post-Hoc Benchmark)** | Controlled evaluation | — | `outputs/stage3_9_posthoc_comparison/` | — | 14.72 dB / 0.5441 | 75,747 | Re-evaluated Exp B vs Baseline side-by-side on 8 test scenes. Exp B won all 8. |
| **Stage 4 (Multi-Model v1)** | `MultiModelDehazeVariant` | $0.8 L_1 + 0.2 (1-\text{SSIM})$ | `outputs/multimodel_variant_v1_continuation_20261019/best_epoch_020.pt` | 11.64 dB / 0.4039 | — | 114,753 | Full 10-module pipeline. Preserved as experimental result; Exp B retained as winner. |
