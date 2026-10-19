# Stage 2 — Dataset Validation and Environment Audit

**Scope:** Read-only validation of the existing workspace, datasets, and virtual environment. This report does not add a model, training code, package changes, or modifications to datasets.

## 1. Executive summary

The workspace is ready for a carefully controlled V0 implementation stage, but not yet for reproducible training. NH-HAZE is structurally sound: all 55 hazy images have one matching ground-truth image, and all 110 files decoded successfully. The O-Haze directory contains seven sets of 45 baseline outputs with matching IDs and dimensions, suitable for qualitative comparisons only; the source hazy images and ground truth required for PSNR/SSIM are absent.

The existing CUDA environment works for a small PyTorch operation on the RTX 3050 Laptop GPU. The main environment concern is a declared/installed NumPy mismatch: `requirements.txt` requires NumPy `<2.0`, while the installed OpenCV 5 package on Python 3.11 declares `numpy>=2`. The installed runtime imports successfully, but the current requirement file cannot reproduce the installed scientific stack as written.

## 2. Workspace inventory

| Area | Contents / result |
|---|---|
| Project source | Stage 1 package namespaces only; no preprocessing, model, training, inference, or evaluation implementation exists. |
| `data/NH-HAZE` | 110 PNG files; 55 hazy inputs and 55 ground-truth images; 316.96 MiB. |
| `data/OHazeResultsSmall` | 315 JPEG baseline outputs in seven method directories; 148.96 MiB. |
| `data/` total | 425 images; 465.91 MiB. |
| `raw/`, `processed/` | Present and empty. |
| `.venv/` | Present; 5,412,215,294 bytes (5.04 GiB) across files. |
| Metadata/unrelated files | No dataset metadata files or unrelated files found under `data/`. |
| Checkpoints/results/metrics | No checkpoint, metric, log, CSV, JSON, or notebook artifacts found before this report. |

All Stage 1 files exist: `.gitignore`, `README.md`, `configs/base.yaml`, the six `src/dehaze` namespace initializers, and `tests/__init__.py`.

`requirements.txt` remains 482 bytes with SHA-256 `05ce8a24cdf6ec4dc014e7566963c1fb1b213a6bc625cbf7298fe37056c115f9`.

### Dataset directory counts

| Directory | Files | Format | Size |
|---|---:|---|---:|
| `data/NH-HAZE` | 110 | PNG | 332,351,815 bytes |
| `data/OHazeResultsSmall/2011_DCP` | 45 | JPEG | 23,312,433 bytes |
| `data/OHazeResultsSmall/2013_ICCV_Meng` | 45 | JPEG | 21,744,937 bytes |
| `data/OHazeResultsSmall/2014_SIGGRAPH_Fattal` | 45 | JPEG | 23,125,480 bytes |
| `data/OHazeResultsSmall/2016_CVPR_Berman` | 45 | JPEG | 23,799,671 bytes |
| `data/OHazeResultsSmall/2016_DehazeNet` | 45 | JPEG | 17,711,776 bytes |
| `data/OHazeResultsSmall/2016_ECCV_SID` | 45 | JPEG | 19,730,278 bytes |
| `data/OHazeResultsSmall/2016_ICIP_Ancuti` | 45 | JPEG | 26,766,093 bytes |

All 425 images are RGB. Pillow decoded and fully loaded every image; there are no zero-byte files, unreadable files, or duplicate-content SHA-256 groups.

The only filename inconsistency is six uppercase `.JPG` extensions (IDs 03 and 21 in the Berman, DehazeNet, and SID folders). Their contents decode normally; case-insensitive extension handling is required in future loaders.

## 3. NH-HAZE validation

Images were paired by the documented filename stem: `NN_hazy.png` with `NN_GT.png`, never by directory order.

| Check | Result |
|---|---|
| Candidate hazy files | 55 |
| Candidate ground-truth files | 55 |
| Valid one-to-one pairs | 55 |
| Missing pairs / duplicate IDs / unmatched files | 0 / 0 / 0 |
| Unexpected files | 0 |
| Readable / corrupted / zero-byte | 110 / 0 / 0 |
| Format, mode, channels | PNG, RGB, 3 channels for every image |
| Paired size/mode mismatches | 0 |
| Image dimensions | Every file is 1600×1200 px |

Thus each valid hazy input has exactly one ground-truth counterpart. No image was modified or normalized during validation.

## 4. O-Haze baseline-output audit

| Method folder | Images | IDs | Duplicate or missing IDs | Read failures |
|---|---:|---|---|---:|
| `2011_DCP` | 45 | 01–45 | None | 0 |
| `2013_ICCV_Meng` | 45 | 01–45 | None | 0 |
| `2014_SIGGRAPH_Fattal` | 45 | 01–45 | None | 0 |
| `2016_CVPR_Berman` | 45 | 01–45 | None | 0 |
| `2016_DehazeNet` | 45 | 01–45 | None | 0 |
| `2016_ECCV_SID` | 45 | 01–45 | None | 0 |
| `2016_ICIP_Ancuti` | 45 | 01–45 | None | 0 |

Every baseline output decoded successfully. For each ID, the image dimensions match across all seven methods. Sizes vary by sample (width 575–800 px; height 486–801 px), so future evaluation must preserve per-image geometry or apply the same declared resize/pad policy to every comparison.

These files alone are **not sufficient** for numerical PSNR or SSIM. Both metrics require a same-scene, geometrically aligned ground-truth clean image; fair comparison also requires the corresponding original hazy input and a documented preprocessing/color-space policy. The available directories support qualitative side-by-side comparisons only; they do not establish that any baseline is better or worse.

## 5. Provisional split design (not created)

Use a deterministic 39 / 8 / 8 train/validation/test split of the 55 NH-HAZE pairs, generated by filename ID with random seed **42**. Keep both members of a pair in the same split. If scene variants or related capture sequences are identified later, group them before splitting and assign each group as a unit; the current dataset has no scene metadata to prove such grouping.

This is provisional, not a finalized dataset plan. Validation data selects architecture and hyperparameters; using it as the final test set would bias the reported result. With only 55 real paired scenes, deep models can overfit and metric estimates have high variance. Later work should document splits, augment only training images, retain an untouched test set, and state whether external training data are used.

## 6. Python dependency compatibility

| Component | Installed version | Evidence / status |
|---|---|---|
| Python | 3.11.5 | `.venv\\Scripts\\python.exe` used explicitly |
| NumPy | 2.4.6 | Imports successfully |
| PyTorch | 2.5.1+cu121 | Imports successfully |
| torchvision | 0.20.1+cu121 | Imports successfully; exact Torch pairing declared |
| CUDA | 12.1 | Available to PyTorch |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU | One CUDA device detected |
| OpenCV | 5.0.0.93 | Imports; Unicode-path caveat below |
| SciPy | 1.17.1 | Imports successfully |
| scikit-image | 0.26.0 | Imports successfully |
| Pillow | 12.3.0 | Imports and decodes all images |
| Matplotlib | 3.11.2 | Imports successfully |
| pandas | 3.0.6 | Imports successfully |
| Streamlit | 1.66.0 | Imports successfully |

`pip check` reported: **No broken requirements found**. That check verifies installed distribution metadata; it does not validate that the project’s declared requirements can rebuild this environment or that every future image pipeline will work.

### NumPy evidence and resolution recommendation

`requirements.txt` declares `numpy>=1.24.0,<2.0.0`. Installed package metadata on Python 3.11 declares:

| Installed package | NumPy metadata requirement |
|---|---|
| `opencv-python==5.0.0.93` | `numpy>=2` |
| `scipy==1.17.1` | `numpy>=1.26.4,<2.7` |
| `scikit-image==0.26.0` | `numpy>=1.24` |
| `torchvision==0.20.1+cu121` | `numpy` (unbounded) |

The compatible metadata interval for this installed stack is `numpy>=2,<2.7`. It does not overlap the declared `<2.0` upper bound. The current NumPy 2.4.6 does satisfy the installed package metadata and passed the import, conversion, and representative image tests in this audit, but no full downstream project pipeline exists yet. Before any environment change, preserve a lockfile or package inventory and decide whether to align the requirement with the current stack (for example `numpy>=2,<2.7`) or intentionally use older, mutually compatible OpenCV/SciPy builds. Do not mix the current OpenCV 5 package with a NumPy `<2` requirement.

## 7. GPU and image-processing readiness

| Check | Result |
|---|---|
| CUDA allocation | Successful |
| Small tensor operation | `mean((arange(16) * 2) + 1) = 16.0` on CUDA |
| Representative Pillow decode | NH-HAZE PNG and O-Haze JPEG successful |
| Direct OpenCV `imread` | Failed for both representative paths |
| OpenCV `imdecode` from path bytes | Successful for both images |
| NumPy → PyTorch → NumPy RGB round trip | Shape preserved; values exactly equal |
| Warnings from Python check | None; OpenCV emitted file-path warnings for `imread` |

The OpenCV `imread` failure is path-related: this workspace sits below a Unicode-normalized OneDrive folder name. Pillow can open those paths and OpenCV can decode the same bytes via `cv2.imdecode`. Future code should avoid `cv2.imread` for these paths (read bytes then use `imdecode`, or use Pillow), use `pathlib.Path`, and test from the intended launch directory.

## 8. Proposed future evaluation protocol

No metric values are reported in this stage. Later evaluations should use the same immutable, held-out paired test split for every model version and baseline that has matching inputs/targets.

| Measure | Valid when | Protocol |
|---|---|---|
| PSNR | Aligned clean ground truth exists | Compare predicted and target RGB images after identical, documented crop/resize and range handling. |
| SSIM | Aligned clean ground truth exists | Use one declared implementation and fixed data range/channel policy. |
| Inference time/image | Hardware and timing policy are fixed | Warm up, synchronize CUDA, measure multiple images, report mean/dispersion and resolution. |
| Parameter count | Model is implemented | Count trainable parameters from the evaluated checkpoint architecture. |
| FLOPs | A reliable model-aware counter supports every operation | Report input resolution, tool, and unsupported operations; otherwise mark unavailable. |
| Peak GPU memory | CUDA inference is used | Reset/measure around synchronized inference, report batch size and resolution. |
| Qualitative review | Inputs and outputs are available | Compare color fidelity, residual haze, halos, texture retention, and edge preservation at fixed display settings. |

## 9. Issues and readiness risks

| Severity | Issue | Evidence | Recommended action |
|---|---|---|---|
| High | Non-reproducible NumPy declaration | Requirements require `<2`, installed OpenCV 5 requires `>=2` on Python 3.11. | Resolve and test dependency policy before Stage 3 environment-dependent work. |
| Medium | O-Haze cannot support full-reference metrics | Only baseline output images are present; no matched hazy inputs or clean targets. | Obtain/document matching inputs and ground truth before numerical benchmarking. |
| Medium | Very small paired training set | NH-HAZE has 55 pairs. | Use an untouched test split, conservative capacity, augmentation only on train, and report limitations. |
| Medium | Unicode path breaks OpenCV `imread` | Reproduced on one PNG and one JPEG; `imdecode` succeeds. | Standardize path-safe image loading before implementation. |
| Low | Filename extension casing varies | Six valid `.JPG` files among otherwise lowercase `.jpg`. | Use case-insensitive suffix matching. |
| Low | Missing scene metadata | No metadata files found. | Document provenance and related-scene groupings before final split. |

Priority for Stage 3: (1) decide and document a dependency policy, (2) add a path-safe paired dataset reader with validation tests, (3) finalize and store a group-aware split manifest after confirming scene metadata, (4) define the held-out evaluation data and preprocessing policy, then (5) begin a minimal V0 model only after those controls are in place.

## 10. Commands and checks actually performed

1. Recursive workspace and dataset inventory with PowerShell.
2. Pillow `Image.open`, `verify`, and `load` for all 425 images.
3. SHA-256 content scan for all dataset images; zero-byte and duplicate-content checks.
4. Filename-based NH-HAZE pairing and pair size/mode comparison.
5. O-Haze method/ID enumeration and cross-method per-ID dimension comparison.
6. Explicit `.venv` package import/version inspection and `pip check`.
7. CUDA availability, allocation, and small-tensor arithmetic using PyTorch.
8. Representative Pillow/OpenCV decode checks and NumPy/PyTorch round trip.
9. Final report review and post-report preservation checks.

## 11. Checks not completed

No required inspection was skipped. Numerical dehazing metrics, FLOPs, inference benchmarks, and training checks were intentionally not run because no project model exists and this stage forbids training/model implementation.
