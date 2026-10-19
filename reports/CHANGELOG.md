# Project Changelog

## 2026-10-19 — Major Project Completion Phase

### Architecture & Models
- Added [`src/dehaze/models/multimodel_variant.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/models/multimodel_variant.py):
  - Implemented 10-module pipeline: Lightweight stem CNN, MultiScaleFeatureExtractor (1x, 0.5x, 0.25x), AdaptiveAttentionFeatureFusion (AAFM), AtmosphericContextBranch (learned smoothing), LayerSeparationBranch (base vs detail), ContextualFeatureBranch (dilated convolutions), AdaptiveMultiModelFusion (pixel-wise softmax gating), EdgeAwareRefinement (Sobel magnitude feature gating), ColorTextureRefinement (channel mixing + depthwise texture), and bounded residual reconstruction.
  - Parameter count: 114,753 trainable parameters.
- Updated [`src/dehaze/models/__init__.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/models/__init__.py) to expose `MultiModelDehazeVariant`.
- Updated [`src/dehaze/evaluation/checkpoints.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/evaluation/checkpoints.py) to support metadata-driven instantiation of `MultiModelDehazeVariant`.

### Testing & Verification
- Added [`tests/test_multimodel_model.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/tests/test_multimodel_model.py):
  - Unit tests covering input/output shapes, valid range `[0, 1]`, finite values, backpropagation gradient flow, AAFM scale attention, branch fusion, edge gating, and parameter counting.
- Updated [`tests/test_checkpoints.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/tests/test_checkpoints.py) to verify multi-model checkpoint reconstruction.
- Test suite expanded from 19 to 26 unit tests; verified 100% pass rate.

### Training & Comparison
- Added [`scripts/train_multimodel_variant.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/train_multimodel_variant.py):
  - Added training pipeline supporting seed reproducibility, random $256 \times 256$ crops, validation tracking, checkpoint saving, and loss curve generation.
- Trained Multi-Model Variant v1 for 20 epochs ([`outputs/multimodel_variant_v1_continuation_20261019/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/multimodel_variant_v1_continuation_20261019)).
- Added [`scripts/evaluate_validation_comparison.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/evaluate_validation_comparison.py) and generated comparative artifacts in [`outputs/final_evaluation_20261019/validation_comparison/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/final_evaluation_20261019/validation_comparison).
- Preserved Stage 3.7 Experiment B as the authoritative winning model based on superior validation L1 error (0.1870 vs 0.2244).

### Deployment & Demonstration
- Added [`scripts/infer.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/scripts/infer.py):
  - Metadata-driven single-image and batch inference CLI supporting CUDA/CPU, preserving input image aspect ratios, and logging JSON execution summaries.
- Enhanced [`app.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/app.py):
  - Upgraded Streamlit demonstration application with model selection dropdown, hazy image upload, optional ground-truth clean image upload with dynamic PSNR/SSIM evaluation, side-by-side visualization, and PNG download.

### Documentation & Reports
- Added [`reports/REMAINING_WORK_AUDIT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/REMAINING_WORK_AUDIT.md) (Phase A audit).
- Added [`reports/TEST_BENCHMARK_PREFLIGHT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/TEST_BENCHMARK_PREFLIGHT.md) (Phase B preflight).
- Added [`reports/DATASET_AUDIT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/DATASET_AUDIT.md) (Dataset audit).
- Added [`reports/FINAL_RESULTS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_RESULTS.md) (Master results summary).
- Added [`reports/FINAL_PROJECT_REPORT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_PROJECT_REPORT.md) (Academic major project report).
- Added [`reports/FINAL_HANDOVER.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_HANDOVER.md) (Operational handover report).
- Added [`reports/PRESENTATION_AND_VIVA_GUIDE.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/PRESENTATION_AND_VIVA_GUIDE.md) (12-slide presentation outline and viva voce examiner Q&A defense guide).
- Updated [`README.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/README.md) to document full framework, execution commands, and project findings.

### Pre-Submission Audit & Scientific Precision
- Refined test-split terminology across `FINAL_PROJECT_REPORT.md`, `PRESENTATION_AND_VIVA_GUIDE.md`, and `README.md` to explicitly designate the 8 test scenes as a *post-hoc held-out benchmark* (quarantined during training, but evaluated post-hoc against baseline).
- Refined latency reporting across reports to distinguish *sustained warm throughput* (182.14 ms / ~5.5 FPS at $1600 \times 1200$) from *single-image cold start* (~347–396 ms).
- Created [`reports/FINAL_SUBMISSION_AUDIT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/FINAL_SUBMISSION_AUDIT.md) containing itemized PASS/FAIL ratings and pre-submission verdict.

### Public Webpage Deployment Preparation
- Created [`requirements-deployment.txt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/requirements-deployment.txt) with minimal headless runtime dependencies for Streamlit Community Cloud.
- Staged production checkpoint at [`models/best_epoch_019.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/models/best_epoch_019.pt) (911.7 KB) and documented hosting options in [`models/README.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/models/README.md).
- Provisioned sample demo images in [`sample_images/`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/sample_images) for 1-click evaluation by public web visitors.
- Updated [`.gitignore`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/.gitignore) to exclude heavy training datasets, `.venv`, outputs, and secrets while permitting the 911 KB deployment model and sample images.
- Upgraded [`app.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/app.py) with dynamic path resolution, CPU execution fallback, hardware status badges, built-in demo toggle, and remote secret URL download support.
- Added [`DEPLOYMENT.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/DEPLOYMENT.md) with exact step-by-step instructions for deploying to Streamlit Community Cloud.
- Generated [`reports/DEPLOYMENT_READINESS.md`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/reports/DEPLOYMENT_READINESS.md) documenting local CPU/GPU smoke tests and the exact remaining manual steps.
