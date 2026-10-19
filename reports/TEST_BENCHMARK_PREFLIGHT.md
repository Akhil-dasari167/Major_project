# Held-Out Benchmark Preflight Audit Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Date:** October 19, 2026  
**Status:** Evaluated & Verified (Reusing Existing Post-Hoc Benchmark)

---

## 1. Executive Summary & Preflight Findings

A benchmark was proposed comparing the **Stage 3.6 Baseline** (`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`) against **Stage 3.7 Experiment B** (`outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`).

Before executing any redundant GPU runs, an audit of existing test artifacts was conducted.

### Preflight Question 1: Does the two-model comparison already exist?
**YES.** A rigorous, controlled side-by-side post-hoc evaluation comparing both models under identical conditions already exists in:
[`outputs/stage3_9_posthoc_comparison/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/stage3_9_posthoc_comparison)
- **Artifacts:**
  - `report.md`: Complete methodology, per-image breakdown, and analysis.
  - `per_image_metrics.csv`: Per-scene PSNR, SSIM, L1 error, and CUDA-synchronized inference latency for each model across all 8 test scenes.
  - `summary.json`: Aggregate metrics, hardware details, parameter counts, and model descriptors.

### Preflight Question 2: Are the metrics and checkpoint metadata reproducible?
**YES.**
- Both checkpoints exist, are intact, and have verified SHA/size signatures:
  - Baseline: `933,478 bytes` (75,747 parameters).
  - Experiment B: `933,606 bytes` (75,747 parameters).
- Both checkpoints store self-describing metadata in `payload['config']` (`residual_scale: 0.2`, `model_type: residual_scale_variant`).
- The canonical split manifest ([`outputs/v0_pilot_001/split.json`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/v0_pilot_001/split.json)) defines the 8 test scenes (`09, 15, 16, 18, 48, 02, 08, 41`).
- Historical test metrics from Stage 3.6 (`14.5836 dB` PSNR, `0.5355` SSIM) matched bit-for-bit with the re-evaluated baseline numbers in `outputs/stage3_9_posthoc_comparison/summary.json`.

### Preflight Question 3: Does the evaluator support both architectures correctly?
**YES.**
- Both [`scripts/test_v0.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/test_v0.py) and [`scripts/evaluate_posthoc_comparison.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/evaluate_posthoc_comparison.py) use the unified model factory:
  `dehaze.evaluation.checkpoints.load_checkpoint_model`.
- This factory correctly inspects checkpoint metadata and instantiates `ResidualScaleVariant(0.2)` for both models.
- Preprocessing, tensor normalization in `[0, 1]`, and metric routines (`psnr`, `ssim`) are 100% shared and identical.

### Preflight Question 4: Would a new test run add useful information?
**NO.**
- The existing post-hoc evaluation already executed deterministic evaluation on the exact same 8 test images on the same NVIDIA RTX 3050 Laptop GPU with CUDA synchronization and warmup.
- Running the identical command again into a new directory would produce redundant duplicate numbers without providing new scientific insight.
- Furthermore, because test scenes `09, 15, 16, 18, 48, 02, 08, 41` were previously evaluated in Stage 3.6 and Stage 3.9, they represent **post-hoc held-out scenes** rather than an unseen double-blind test set.

---

## 2. Verified Benchmark Comparison (Reused from Stage 3.9)

| Model | Checkpoint | Test PSNR (dB) | Test SSIM | Test L1 Error | Parameters | Checkpoint Size | Latency ($1600 \times 1200$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 3.6 Baseline** | `outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt` | 14.5836 | 0.5355 | 0.150610 | 75,747 | 933.5 KB | 201.10 ms |
| **Stage 3.7 Exp B** | `outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt` | **14.7239** | **0.5441** | **0.148768** | **75,747** | **933.6 KB** | **182.14 ms** |
| **Delta ($\text{Exp B} - \text{Base}$)** | — | **+0.1403 dB** | **+0.0086** | **-0.001842 (-1.2%)** | 0 | +128 bytes | **-18.96 ms** |

### Per-Image Breakdown on the 8 Post-Hoc Test Scenes:

| Scene ID | Baseline PSNR | Exp B PSNR | Baseline SSIM | Exp B SSIM | Baseline L1 | Exp B L1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **09** | 12.7386 | **13.0781** | 0.5034 | **0.5162** | 0.194553 | **0.187089** |
| **15** | **14.6497** | 14.6089 | 0.6104 | **0.6138** | **0.146723** | 0.149762 |
| **16** | 14.7802 | **15.1546** | 0.6359 | **0.6455** | 0.148928 | **0.143866** |
| **18** | 15.3994 | **15.6008** | 0.5533 | **0.5648** | 0.129173 | **0.127178** |
| **48** | **13.4488** | 13.4157 | 0.3095 | **0.3213** | **0.167706** | 0.168674 |
| **02** | **17.3668** | 17.2649 | 0.5896 | **0.5901** | **0.104521** | 0.106900 |
| **08** | 15.7665 | **16.1714** | 0.6145 | **0.6329** | 0.129135 | **0.122881** |
| **41** | **12.5186** | 12.4970 | 0.4671 | **0.4687** | 0.184138 | **0.183797** |
| **Mean** | **14.5836** | **14.7239** | **0.5355** | **0.5441** | **0.150610** | **0.148768** |

- **Key Finding:** Experiment B improved SSIM on **8 out of 8 test scenes**, reduced mean L1 error, and increased mean PSNR by $+0.14$ dB while maintaining identical parameter count.

---

## 3. Protocol Decision & Next Action

In accordance with Phase B rules:
- Because the benchmark already exists, is completely verifiable, and matches all reproducibility requirements, **we reuse `outputs/stage3_9_posthoc_comparison/` directly**.
- No redundant test runs are launched.
- Test metrics were never used for model selection, epoch tuning, or training loss weighting.
