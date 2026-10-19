# Project State Snapshot — 2026-10-19

## Preservation boundary

This snapshot was created before the multi-model architecture implementation.
Existing datasets, prior checkpoints, outputs, analysis artifacts, and reports
remain preserved in place.

## Verified reference

- Reference checkpoint: `outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt`
- Model: `ResidualScaleVariant`, residual scale `0.2`, 75,747 parameters.
- Validation metrics recorded in the existing summary: PSNR `12.9984` dB,
  SSIM `0.4418`, L1 `0.187040`.
- Post-hoc test comparison recorded in
  `outputs/stage3_9_posthoc_comparison/`: PSNR `14.7239` dB, SSIM `0.5441`,
  L1 `0.148768`. This split was previously examined and is not a blind test.

## Environment

- Python `3.11.5`
- PyTorch `2.5.1+cu121`; torchvision `0.20.1+cu121`
- CUDA `12.1`; GPU `NVIDIA GeForce RTX 3050 Laptop GPU`
- NumPy `2.4.6`, OpenCV `5.0.0`, Pillow `12.3.0`, SciPy `1.17.1`,
  scikit-image `0.26.0`, Matplotlib `3.11.2`, Streamlit `1.66.0`
- `python -m unittest discover -s tests -v`: 19 tests passed.

## Dataset and split

- NH-HAZE: 55 RGB hazy/ground-truth pairs.
- Canonical split: 39 train / 8 validation / 8 post-hoc test pairs, seed 42.
- O-Haze output set remains qualitative-only because paired inputs and targets
  are not present.

## Planned isolated additions

- `src/dehaze/models/multimodel_variant.py`
- New tests and metadata-driven inference/training scripts.
- New `outputs/multimodel_variant_*` directories only.

