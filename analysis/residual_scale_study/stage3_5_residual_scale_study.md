# Controlled Residual-Scale Study

**Status: preliminary, validation-only.** This study did not read or evaluate
the held-out test split. It does not establish generalization or final image
quality. The existing V0 implementation, pilot checkpoints, manifests,
datasets, requirements, and environment were preserved.

## Design

Two V0-equivalent CNN variants were trained for five epochs. The only changed
factor was the residual scale in `clamp(hazy + scale * tanh(residual), 0, 1)`.
The dedicated experimental model duplicates V0's layer layout and initialization
but leaves `V0ResidualCNN` unchanged.

| Controlled factor | Both variants |
|---|---|
| Train / validation manifest | Existing `outputs/v0_pilot_001/split.json` (39 / 8 pairs) |
| Test split | Not loaded or evaluated |
| Seed / epoch ordering | 42; same deterministic epoch sampler |
| Epochs / patches / batch size | 5 / 256×256 / 2 |
| Optimizer / learning rate / loss | Adam / 0.0002 / normalized RGB L1 |
| Validation | Eight full-resolution, unaugmented validation images |

| Variant | Residual scale |
|---|---:|
| Reference | ±0.1 |
| Alternative | ±0.2 |

The reference's epoch-wise losses exactly reproduce the prior V0 pilot log,
verifying that the experimental training setup is matched.

## Results

| Epoch | ±0.1 train L1 | ±0.1 validation L1 | ±0.2 train L1 | ±0.2 validation L1 |
|---:|---:|---:|---:|---:|
| 1 | 0.224323 | 0.239273 | 0.220040 | 0.233267 |
| 2 | 0.208500 | 0.229799 | 0.194387 | 0.227231 |
| 3 | 0.209934 | 0.230183 | 0.202842 | 0.226594 |
| 4 | 0.203019 | 0.230187 | 0.192662 | **0.224928** |
| 5 | 0.193810 | 0.230162 | 0.179235 | 0.225812 |

The best validation L1 is 0.229799 for ±0.1 (epoch 2) and 0.224928 for ±0.2
(epoch 4). The absolute difference is -0.004871 L1 (about 2.12% relative to
the ±0.1 best value). This is verified for this fixed validation set and
five-epoch budget only.

## Residual-correction diagnostics

| Variant / epoch | Mean absolute correction | Mean signed correction | Near bound | Output-clamped pixels |
|---|---:|---:|---:|---:|
| ±0.1, best epoch 2 | 0.094695 | -0.094695 | 57.94% | 0.0060% |
| ±0.1, epoch 5 | 0.099549 | -0.099549 | 98.65% | 0.0135% |
| ±0.2, best epoch 4 | 0.139359 | -0.139351 | 18.30% | 0.0716% |
| ±0.2, epoch 5 | 0.161143 | -0.161143 | 29.92% | 0.1056% |

`Near bound` is the fraction of validation pixels with absolute output
correction at least 95% of the configured residual scale. The ±0.2 variant is
less constrained by its residual ceiling and has a lower validation L1 in this
study. Both variants learn predominantly negative (darkening) corrections.

## Qualitative validation inspection

The validation-only four-way sheet at
`validation_four_way_contact_sheet.png` compares hazy input, ground truth, and
the best ±0.1/±0.2 predictions for all eight validation IDs.

Verified visual observations at contact-sheet scale:

- ±0.2 produces visibly stronger correction than ±0.1, primarily a darker and
  somewhat warmer/browner appearance in several scenes.
- It still leaves substantial haze and does not recover the ground-truth
  contrast and detail in heavily obscured scenes.
- No obvious geometric distortion appears in this scale-only study, but
  contact-sheet inspection cannot rule out fine texture loss, clipping, or
  color fidelity problems at full resolution.

The lower validation L1 therefore should not be interpreted as proof of a
better perceptual dehazing result. Its darker correction direction is a risk
that needs validation-only analysis before any final test evaluation.

## Findings: verified versus hypotheses

| Statement | Status |
|---|---|
| The ±0.1 residual ceiling becomes heavily active by epochs 3–5. | Verified. |
| Increasing the scale to ±0.2 lowers the best validation L1 in this fixed study. | Verified. |
| ±0.2 reduces near-bound saturation while increasing output-clamped pixels modestly. | Verified. |
| ±0.2 will generalize better or improve held-out test quality. | Not established; not tested. |
| A larger scale solves residual haze or color fidelity. | Hypothesis only; visual inspection shows residual haze remains. |

## Recommended next action

Do not use test data to decide. The next justified step is a predeclared,
training/validation-only continuation comparing ±0.1 and ±0.2 with early
stopping and validation diagnostics for correction direction, clipping, and
color/edge behavior. Keep all other factors fixed. If ±0.2 remains preferred
by the predeclared validation criterion, perform exactly one final held-out
test evaluation afterward. Do not add a larger scale or a new loss until that
controlled comparison is complete.

## Artifacts

| Path | Contents |
|---|---|
| `outputs/residual_scale_study_0p1` | Five-epoch reference checkpoints, history, validation contact sheet, and manifest snapshot. |
| `outputs/residual_scale_study_0p2` | Five-epoch ±0.2 checkpoints, history, validation contact sheet, and manifest snapshot. |
| `analysis/residual_scale_study/validation_loss_comparison.png` | Validation-loss comparison. |
| `analysis/residual_scale_study/validation_four_way_contact_sheet.png` | Corrected four-way validation-only visual comparison. |
| `analysis/residual_scale_study/study_summary.json` | Machine-readable best/final diagnostics. |
