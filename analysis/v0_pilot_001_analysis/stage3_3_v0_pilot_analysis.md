# Stage 3.3 — V0 Pilot Results Analysis

**Status: preliminary.** This analysis uses the fixed eight-image held-out test
split from a project with only 55 paired images. It is not evidence of
generalization, superiority, or final image quality. No parameters were tuned
from these test results and no additional training was run.

## Inputs and metric convention

- Run: `outputs/v0_pilot_001`
- Selected checkpoint: `outputs/v0_pilot_001/best_epoch_002.pt`
- Held-out evaluation set: IDs `09, 15, 16, 18, 48, 02, 08, 41`
- Metrics: RGB PSNR and RGB SSIM, evaluated on `[0, 1]` tensors using the same
  `dehaze.evaluation.metrics` functions for both the original hazy input and
  the V0 prediction. Predictions were recomputed from the selected checkpoint
  before PNG quantization.

## Training and validation trend

| Epoch | Train L1 | Validation L1 |
|---:|---:|---:|
| 1 | 0.224323 | 0.239273 |
| 2 | 0.208500 | **0.229799** |
| 3 | 0.209934 | 0.230183 |
| 4 | 0.203019 | 0.230187 |
| 5 | 0.193810 | 0.230162 |

Training L1 decreased overall from 0.2243 to 0.1938. Validation improved most
between epochs 1 and 2, then remained essentially flat (0.2298–0.2302).
`best_epoch_002.pt` is therefore the correct validation-selected checkpoint;
the held-out test split was not used in this selection.

The widening train/validation gap after epoch 2 is a signal to investigate,
not a conclusion. Plausible causes include the small 39-pair training split,
patch-to-full-image validation mismatch, limited model capacity, and/or a
learning-rate schedule that is not yet appropriate. Five epochs are too few
to choose among those explanations. Any such investigation must use only the
training and validation splits.

The cumulative pilot loss curve is available at
`outputs/v0_pilot_001/loss_epoch_005.png`.

## Quantitative comparison

| ID | Hazy PSNR | V0 PSNR | Delta PSNR | Hazy SSIM | V0 SSIM | Delta SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 09 | 11.9583 | 11.6342 | -0.3240 | 0.4836 | 0.4751 | -0.0085 |
| 15 | 11.6449 | 13.6648 | +2.0199 | 0.5760 | 0.6075 | +0.0315 |
| 16 | 15.8850 | 15.7446 | -0.1404 | 0.6551 | 0.6531 | -0.0019 |
| 18 | 13.5005 | 14.8635 | +1.3630 | 0.5316 | 0.5428 | +0.0112 |
| 48 | 10.7543 | 11.9162 | +1.1620 | 0.2844 | 0.2857 | +0.0012 |
| 02 | 14.2020 | 16.1124 | +1.9104 | 0.5850 | 0.5708 | -0.0142 |
| 08 | 14.7516 | 13.7159 | -1.0357 | 0.5897 | 0.5766 | -0.0130 |
| 41 | 9.4537 | 11.2101 | +1.7564 | 0.4628 | 0.4751 | +0.0122 |
| **Mean** | **12.7688** | **13.6077** | **+0.8389** | **0.5210** | **0.5233** | **+0.0023** |

On these eight images, V0 improves mean PSNR by 0.8389 dB and mean SSIM by
0.0023. The improvement is not uniform:

- PSNR worsens for IDs **09, 16, and 08**.
- SSIM worsens for IDs **09, 16, 02, and 08**.
- ID 02 improves PSNR while worsening SSIM, illustrating why both metrics
  should be reported together.

## Visual comparison

The contact sheet at `v0_pilot_001_contact_sheet.png` contains, for each fixed
test ID, the hazy input, ground truth, and V0 prediction in that order.

At contact-sheet scale, V0 predictions are mostly conservative and remain
close to their hazy inputs. The most heavily obscured scenes (notably 09, 16,
18, 48, 02, and 41) retain substantial haze relative to ground truth. There
is no obvious large haloing, geometric distortion, or severe color cast in the
downsampled comparison. The corrections are generally too small to recover
the ground-truth contrast and detail in difficult scenes. Some small
brightness/color and local-contrast changes are visible, but a full-resolution
inspection would be required to make strong claims about oversmoothing or fine
edge artifacts.

## Output artifacts

| Artifact | Purpose |
|---|---|
| `v0_pilot_001_contact_sheet.png` | Eight-row hazy / ground-truth / prediction visual comparison. |
| `metrics_comparison.json` | Machine-readable per-image and mean input/prediction metrics. |
| `stage3_3_v0_pilot_analysis.md` | This preliminary analysis. |

## Next-step caution

Do not tune on the test IDs or use this result as a final claim. Before a full
V0 experiment, decide the training/validation-only procedure for investigating
epoch-2 validation flattening, then preserve this test split for one final
evaluation of the selected configuration.
