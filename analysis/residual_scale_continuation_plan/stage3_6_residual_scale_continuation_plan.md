# Stage 3.6 — Residual-Scale Continuation Plan

**Status: pre-registered plan; no new training or held-out test evaluation has started.** Existing V0 code, checkpoints, manifests, datasets, requirements, and environment remain preserved.

## Preflight verification

| Check | Result |
|---|---|
| Canonical manifest | `outputs/v0_pilot_001/split.json` |
| Manifest SHA-256 | `582d66de9c2f91b5a86e67f733a910a4abd133d91e96ccfc07098f88e30e99ba` |
| Split counts | 39 train / 8 validation / 8 test |
| Train–validation / train–test / validation–test overlap | None / none / none |
| Stage 3.5 manifests | Both byte-equivalent to the canonical manifest |
| Stage 3.5 checkpoints and validation sheets | Present for ±0.1 and ±0.2 |
| Existing evaluator | Correct for ±0.1 only; it reconstructs `V0ResidualCNN` and cannot correctly load a ±0.2 experimental checkpoint |

## Fair continuation experiment

The sole experimental factor is the residual scale in `clamp(hazy + scale * tanh(residual), 0, 1)`.

| Setting | Reference | Alternative |
|---|---:|---:|
| Residual scale | ±0.1 | ±0.2 |
| Output directory | `outputs/residual_scale_continuation_0p1` | `outputs/residual_scale_continuation_0p2` |
| Source split | Canonical immutable manifest | Same canonical immutable manifest |
| Seed | 42 | 42 |
| Architecture | V0-equivalent 3→32, four residual blocks, 32→3 | Identical |
| Epoch budget | 20 exactly | 20 exactly |
| Patch / batch size | 256×256 / 2 | Identical |
| Optimizer / LR / loss | Adam / 0.0002 / normalized RGB L1 | Identical |
| Training sampler / augmentation | Same epoch-seeded sampler and horizontal flip | Identical |
| Validation | Eight unaugmented, full-resolution validation images each epoch | Identical |

This fixed budget gives **400 optimizer steps per variant** (20 batches/epoch × 20 epochs) and 160 full-resolution validation forwards per variant. Based on the Stage 3.5 run, expected workload is roughly 3.5–5 minutes total on the RTX 3050 Laptop GPU, plus artifact generation.

## Pre-registered records and selection rule

For both variants and every epoch, record train/validation L1, validation PSNR/SSIM under the same `[0,1]` RGB convention, mean absolute/signed residual correction, near-bound fraction, output clipping fraction, checkpoints, and immutable configuration.

Generate a cumulative train/validation loss plot and an all-eight-image validation contact sheet from each variant's **best validation-L1 checkpoint**. The primary selection metric is minimum validation L1. PSNR/SSIM and contact sheets are required diagnostics, not extra tuning criteria.

If best validation L1 values differ by **0.001 or less**, label the comparison inconclusive and do not select a variant for test evaluation. Otherwise, the lower-validation-L1 checkpoint is the provisional winner. This rule is fixed before training.

## Required implementation work after approval

1. Extend the dedicated residual-scale runner for the fixed continuation budget and validation PSNR/SSIM logging, preserving current study artifacts.
2. Save validation contact sheets from each best checkpoint rather than only the final epoch.
3. Add a scale-aware evaluation entry point (or shared model factory), because `scripts/test_v0.py` cannot load ±0.2 correctly.
4. Add focused automated tests for checkpoint-scale matching, split leakage validation, and validation metric logging.

No held-out test evaluation is authorized in the continuation phase. After training and validation, provide a separate comparison report and request approval for exactly one final test evaluation of the pre-registered winner. That later evaluation must save per-image/mean PSNR and SSIM, inference time, parameter count, convolution-MAC estimate, machine-readable metrics, and a test contact sheet.

## Existing evidence, not a final decision

The five-epoch study found validation L1 0.224928 for ±0.2 versus 0.229799 for ±0.1, but ±0.2 also made stronger negative/darkening corrections. These validation-only observations motivate the continuation experiment; they do not establish perceptual quality or held-out performance.
