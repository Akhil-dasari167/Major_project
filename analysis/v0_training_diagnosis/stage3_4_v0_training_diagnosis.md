# Stage 3.4 — V0 Training Diagnosis

**Scope:** diagnosis only. No training was run, no test-image metrics were used,
and no model, split, package, or checkpoint was modified. Observations below
use the V0 source, pilot log, and the fixed training/validation partitions only.

## 1. Verified implementation facts

| Component | Verified behavior |
|---|---|
| Architecture | 3→32 convolution, four 32-channel residual blocks, 32→3 output convolution; 75,747 trainable parameters. |
| Output formulation | `clamp(hazy + 0.1 * tanh(residual), 0, 1)`. Each learned residual channel is bounded to `[-0.1, +0.1]` before output clipping. |
| Normalization | RGB images are loaded with Pillow and converted from `uint8` to float tensors in `[0, 1]`. |
| Loss | Mean L1 loss between normalized RGB prediction and target. |
| Training sampling | One deterministic, random 256×256 crop per one of 39 train pairs per epoch; horizontal flip is randomly applied. Batch size is 2. |
| Validation sampling | Eight full-resolution images, batch size 1, no crop and no augmentation. |
| Optimizer | Adam, learning rate 0.0002, no scheduler or weight decay configured. |
| Checkpoint selection | Lowest validation L1 only. The 8 test IDs are absent from train/validation and are not loaded by the training script. |

## 2. Verified pilot observations

| Epoch | Train L1 | Validation L1 |
|---:|---:|---:|
| 1 | 0.224323 | 0.239273 |
| 2 | 0.208500 | **0.229799** |
| 3 | 0.209934 | 0.230183 |
| 4 | 0.203019 | 0.230187 |
| 5 | 0.193810 | 0.230162 |

The training loss decreases overall, while validation reaches its lowest value
at epoch 2 and then changes by only about 0.0004. This is a plateau over this
short five-epoch pilot; it does not, by itself, establish overfitting.

### Training/validation difficulty and preprocessing checks

The same normalized L1 convention was used to compare hazy input with target
without involving the model:

| Partition / sampling | Mean input-target L1 | Min | Max |
|---|---:|---:|---:|
| Train, full image (39 pairs) | 0.221658 | 0.132632 | 0.361632 |
| Validation, full image (8 pairs) | 0.248470 | 0.187395 | 0.286446 |
| Train patches, epoch 1 | 0.229830 | 0.054053 | 0.574044 |
| Train patches, epoch 2 | 0.236600 | 0.074083 | 0.478855 |
| Train patches, epoch 3 | 0.235664 | 0.063307 | 0.609900 |
| Train patches, epoch 4 | 0.231758 | 0.068989 | 0.541725 |
| Train patches, epoch 5 | 0.235103 | 0.100948 | 0.390360 |

This verifies a protocol mismatch, not a bug: training optimizes highly varied
256×256 crops while validation reports full-image L1. The validation partition
also has a higher mean input-target L1 than the training partition (0.24847 vs
0.22166), so it is modestly harder by this baseline measure. L1 normalization,
RGB channel order, and target range match between the two paths.

### Residual-bound diagnostic on validation only

For `best_epoch_002.pt` evaluated on the validation set:

| Measurement | Value |
|---|---:|
| Mean absolute output correction | 0.094695 |
| Maximum representable pre-clamp correction | 0.100000 |
| Pixels with `abs(correction) >= 0.095` | 57.94% |
| Pixels with `abs(correction) >= 0.099` | 11.67% |
| Mean signed correction | -0.094695 |
| Mean output-clamped pixel fraction | 0.0060% |

The residual branch is therefore operating close to its explicit bound across
much of validation, predominantly darkening the input. This is verified. It
does **not** prove that raising/removing the bound will help: the bound may also
be preventing unstable or unrealistic corrections.

## 3. Causes: observations versus hypotheses

| Item | Status | Reasoning |
|---|---|---|
| Validation plateau after epoch 2 | Observed | Logged validation L1 remains approximately 0.230 through epochs 3–5. |
| Train/validation crop mismatch | Observed | Train uses augmented 256×256 crops; validation uses full images. |
| Validation set is harder by baseline L1 | Observed | Its mean hazy-target full-image L1 is 0.0268 higher. |
| Residual correction cap is active | Observed | Mean correction is 0.0947 with 57.9% of pixels above 0.095 absolute correction. |
| Classic overfitting | Hypothesis | Falling train loss with flat validation loss is consistent with it, but five epochs and 39 images are insufficient to establish it. |
| Under-capacity / residual-range limitation | Hypothesis with supporting evidence | The cap is active; the small receptive-field CNN may be unable to model scene-scale haze. Controlled ablations are needed. |
| Learning rate is unsuitable | Hypothesis | A constant 0.0002 Adam rate and no scheduler were used. The pilot does not isolate learning-rate effects. |
| L1 alone is insufficient for perceptual dehazing | Hypothesis | L1 optimizes pixel error and does not explicitly preserve edges, contrast, or perceptual appearance. This should be tested only after a stable baseline. |

## 4. Prioritized V0 plan (not executed)

| Priority | Controlled change / check | Expected benefit | Risk | Validation-only test |
|---:|---|---|---|---|
| 1 | Run a longer **unchanged** V0 baseline with early stopping and fixed seed/split. | Determines whether the epoch-2 plateau persists beyond the five-epoch pilot. | Uses compute without resolving the cap issue. | Select by validation L1 only; retain the existing test split untouched. |
| 2 | Add logging of validation correction magnitude, saturation fraction, and deterministic validation-patch L1 alongside full-image L1. | Separates optimization behavior from crop/full-image evaluation mismatch. | Additional reporting complexity only. | Compare curves on train/validation; do not view test results. |
| 3 | Ablate only the V0 residual scale (`0.1` versus a documented alternative) while holding split, seed, epochs, optimizer, and patch protocol fixed. | Tests whether the verified bound limits correction. | Larger corrections can cause clipping, color shifts, or unstable outputs. | Compare validation L1 and correction/clipping diagnostics; use a single predeclared selection rule. |
| 4 | If the baseline still plateaus, test a validation-only learning-rate schedule or lower post-plateau learning rate. | May refine a plateaued optimizer trajectory. | More hyperparameter search can overfit the eight-image validation split. | Predeclare a small candidate set and report every run. |
| 5 | Only after a stable baseline, test an edge-aware auxiliary loss or a modest capacity/receptive-field adjustment. | Better aligns training with edge retention and spatially varying haze. | Changes the V0 baseline and increases confounding factors. | One factor at a time; choose by validation only. |

## 5. Recommendation

A longer run with early stopping is justified **as a controlled unchanged-V0
baseline**, but should not be treated as the first proof of a solution. The
active residual bound and crop/full-image mismatch should be logged during that
run. If validation remains flat under early stopping, perform the single-factor
residual-scale ablation before adding loss terms or more capacity.

All prospective decisions must use only the fixed train/validation partitions.
The held-out test split should be used once for the final selected V0 setup,
not for choosing any item in this plan.
