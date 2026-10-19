# Dataset Audit and Split Validation Report

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Date:** October 19, 2026

---

## 1. Dataset Inventory & Verification

The local workspace was thoroughly inspected for paired outdoor dehazing datasets.

| Dataset Name | Directory Path | File Format | File Count | Channels / Resolution | Ground Truth Availability | Usability Role |
|---|---|:---:|:---:|:---:|:---:|---|
| **NH-HAZE** | `data/NH-HAZE` | PNG | **110 files** (55 pairs) | RGB / $1600 \times 1200$ px | Full (1:1 paired) | **Training, Validation, & Held-Out Benchmarking** |
| **O-HAZE Baseline Outputs** | `data/OHazeResultsSmall` | JPEG | **315 files** (7 methods $\times$ 45) | RGB / $575-800 \times 486-801$ px | None (Outputs only) | **Qualitative Baseline Comparisons Only** |

### Critical Dataset Notes:
1. **NH-HAZE Pairing Integrity:**
   - Every single hazy image (`NN_hazy.png`) matches exactly one ground-truth image (`NN_GT.png`).
   - All 110 files decode without corruption or truncation.
   - All images share identical dimensions: $1600 \times 1200$ pixels.
2. **O-HAZE Limitation:**
   - `data/OHazeResultsSmall` contains outputs from 7 published methods: `2011_DCP`, `2013_ICCV_Meng`, `2014_SIGGRAPH_Fattal`, `2016_CVPR_Berman`, `2016_DehazeNet`, `2016_ECCV_SID`, `2016_ICIP_Ancuti`.
   - The original hazy images and ground-truth clean images are not present in the workspace.
   - Therefore, numerical PSNR and SSIM **cannot** be computed on O-HAZE.

---

## 2. Canonical Split Manifest

The project uses the canonical Seed-42 split defined in [`outputs/v0_pilot_001/split.json`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/v0_pilot_001/split.json) (SHA-256: `582d66de9c2f91b5a86e67f733a910a4abd133d91e96ccfc07098f88e30e99ba`).

- **Total Paired Scenes:** 55
- **Train Partition (39 scenes, ~70.9%):**
  `31, 24, 50, 12, 17, 21, 51, 54, 36, 20, 10, 34, 42, 45, 30, 55, 04, 22, 47, 05, 29, 11, 43, 23, 39, 25, 01, 26, 52, 19, 32, 37, 27, 13, 40, 33, 49, 14, 46`
- **Validation Partition (8 scenes, ~14.5%):**
  `53, 03, 28, 38, 06, 35, 44, 07`
- **Held-Out Test Partition (8 scenes, ~14.5%):**
  `09, 15, 16, 18, 48, 02, 08, 41`

### Partition Disjointness Check:
- $\text{Train} \cap \text{Validation} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- $\text{Validation} \cap \text{Test} = \emptyset$
- $\text{Total Unique Scenes} = 39 + 8 + 8 = 55$.

---

## 3. Preprocessing & Data Pipeline Specification

- **Training Pipeline:**
  - Random cropping to $256 \times 256$ patches.
  - Paired deterministic horizontal flip augmentation.
  - Floating point tensor conversion in range $[0.0, 1.0]$.
  - `EpochShuffleSampler` with deterministic per-epoch permutation seed.
  - Batch size: 2 patches per batch (20 batches per epoch = 40 training patches / epoch).
- **Validation Pipeline:**
  - Full-resolution evaluation ($1600 \times 1200$ pixels).
  - Deterministic evaluation order.
  - Batch size: 1 image per forward pass.
  - No cropping or resizing; preserves original aspect ratio and geometry.
- **Quarantine Safeguards:**
  - During training runs, the DataLoader receives only `split.train` and `split.validation`.
  - The 8 test scenes are never sampled during optimizer steps or checkpoint selection.
