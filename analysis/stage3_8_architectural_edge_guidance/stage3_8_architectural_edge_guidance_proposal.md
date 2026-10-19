# Stage 3.8 Proposal: Lightweight Architectural Edge Guidance

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Document:** Stage 3.8 Proposal & Experimental Protocol  
**Date:** October 18, 2026  
**Status:** Proposal Draft — Awaiting User Approval (Zero Training / Zero Code Changes Executed)

---

## 1. Executive Summary & Context

Throughout Stages 3.5 to 3.7, the project established:
1. **Residual Scale Tuning (Stage 3.5 & 3.6):** The $\pm 0.2$ residual scale significantly outperformed $\pm 0.1$, achieving **12.84 dB PSNR / 0.4268 SSIM** on validation and **14.58 dB PSNR / 0.5355 SSIM** on held-out test.
2. **Diagnosis of Edge Attenuation (Stage 3.7 Exp A):** The baseline $L_1$ model suppressed high-frequency structural edges, recovering only ~24% of Ground Truth edge energy (0.0926 vs. GT 0.3862).
3. **Loss-Based Optimization (Stage 3.7 Exp B & Exp C):**
   - **Experiment B ($0.8 L_1 + 0.2 [1-\text{SSIM}]$):** Emerged as the provisional top performer: **Val L1: 0.187040, Val PSNR: 12.9984 dB, Val SSIM: 0.4418, Mean Edge Energy: 0.0992** (a +7.2% edge recovery increase).
   - **Experiment C ($0.8 L_1 + 0.15 [1-\text{SSIM}] + 0.05 L_{\text{Sobel}}$):** Confirmed edge enhancement (+4.9% over baseline to 0.0970 at best epoch 19 and 0.0991 at epoch 20) and achieved strong PSNR gains on dense foliage scenes (Scene 35: +0.44 dB over Exp B). However, allocating loss weight to pixel-space Sobel differences slightly compromised global SSIM (0.4371 vs. 0.4418) because a single convolutional trunk had to simultaneously optimize low-frequency transmission clearing and high-frequency edge matching.

### Objective of Stage 3.8
Stage 3.8 explores whether **architectural edge guidance**—providing explicit structural edge representations directly to the neural network rather than relying purely on loss penalties—can boost high-frequency edge recovery beyond 0.0992 while preserving or improving the global PSNR ($\ge 13.00$ dB) and SSIM ($\ge 0.4418$) achieved by Experiment B.

---

## 2. Baseline Architecture Inspection & Bottleneck Analysis

### 2.1 Existing Model: `ResidualScaleVariant`
The current model implementation in [`src/dehaze/models/residual_scale_variant.py`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/src/dehaze/models/residual_scale_variant.py) is a homogeneous spatial CNN:

```text
Input (3, H, W)
  │
  ▼
Head: Conv2d(3, 32, kernel=3, pad=1) + ReLU                 [896 params]
  │
  ▼
Blocks: 4 × ResidualBlock(32 channels)                     [73,984 params]
  │       (Each block: Conv-ReLU-Conv with residual add)
  ▼
Tail: Conv2d(32, 3, kernel=3, pad=1)                        [867 params]
  │
  ▼
Residual: tanh(tail_out) × 0.2
  │
  ▼
Output: clamp(Input + Residual, 0.0, 1.0)
```
- **Total Parameters:** **75,747** ($303.0$ KB FP32 checkpoint).
- **FLOPs:** $\approx 145.4$ GFLOPs at full $1200 \times 1600$ resolution.
- **Inference Latency:** $\approx 167.5$ ms/image on NVIDIA RTX 3050 Laptop GPU.

### 2.2 Identified Architectural Bottleneck
1. **Blind Input Processing:** The network receives only the 3-channel RGB image. Because haze acts as a low-pass optical filter (scattering high-frequency light), early convolutional features are dominated by low-frequency haze gradients.
2. **Capacity Allocation Dilemma:** The 4 residual blocks must simultaneously learn to:
   - Estimate the large-scale atmospheric veil / transmission map (low frequency).
   - Detect and sharpen faint underlying object boundaries (high frequency).
   With only 32 intermediate channels and 4 blocks, the network naturally prioritizes the low-frequency haze veil to minimize global $L_1$ loss, resulting in smoothed edges.

---

## 3. Proposed Minimal Architectural Modification: Edge-Conditioned Input Head (EC-Net)

To introduce edge awareness without redesigning the core trunk or substantially increasing parameter count, we propose a single minimal modification: **Edge-Conditioned Input Concatenation (EC-Head)**.

### 3.1 Architectural Specification
Instead of passing only the 3-channel RGB hazy image into the head convolution, we compute an on-the-fly, parameter-free gradient magnitude map from the input luminance and concatenate it along the channel dimension.

1. **Parameter-Free Edge Extractor:**
   $$I_{\text{gray}} = 0.2989 R + 0.5870 G + 0.1140 B$$
   $$G_x = I_{\text{gray}} \circledast K_x, \quad G_y = I_{\text{gray}} \circledast K_y$$
   $$E_{\text{in}} = \sqrt{G_x^2 + G_y^2 + \epsilon}$$
   *(Normalized to $[0, 1]$ via fixed scaling or standard deviation).*
2. **Augmented 4-Channel Input:**
   $$X_{\text{in}} = [I_{\text{hazy}}, E_{\text{in}}] \in \mathbb{R}^{B \times 4 \times H \times W}$$
3. **Head Layer Modification:**
   - Change `head.0` from `nn.Conv2d(3, 32, kernel_size=3, padding=1)` to `nn.Conv2d(4, 32, kernel_size=3, padding=1)`.
   - All 4 residual blocks (73,984 params) and the tail convolution (867 params) remain **100% unchanged**.

```text
Input Hazy RGB (3, H, W) ──┬──► Fixed Sobel Filter ──► Edge Map E_in (1, H, W)
                           │                                  │
                           └──────────────────┬───────────────┘
                                              ▼
                               Concatenated Input (4, H, W)
                                              │
                                              ▼
                      Modified Head: Conv2d(4, 32, 3, pad=1) + ReLU
                                              │
                                              ▼
                             4 × ResidualBlock(32 channels)  [UNCHANGED]
                                              │
                                              ▼
                             Tail: Conv2d(32, 3, 3, pad=1)   [UNCHANGED]
                                              │
                                              ▼
                             tanh(tail_out) × 0.2 + Input RGB
```

---

## 4. Parameter & Computational Cost Estimation

| Component | Current Baseline (`ResidualScaleVariant`) | Proposed Stage 3.8 (`EdgeConditionedVariant`) | Difference ($\Delta$) |
|---|:---:|:---:|:---:|
| **Input Channels** | 3 (RGB) | 4 (RGB + Edge Map) | +1 channel |
| **Head Weights** | $32 \times 3 \times 3 \times 3 = 864$ | $32 \times 4 \times 3 \times 3 = 1,152$ | +288 weights |
| **Head Biases** | 32 | 32 | 0 |
| **Residual Blocks** | 73,984 | 73,984 | 0 |
| **Tail Layer** | 867 | 867 | 0 |
| **Total Parameters** | **75,747** | **76,035** | **+288 (+0.38%)** |
| **Model Size (FP32)** | 303.0 KB | 304.1 KB | +1.1 KB |
| **Edge Operator Cost** | None | 1 Depthwise Conv ($3 \times 3$) | $\approx 0.15$ ms GPU |
| **Inference Time (1200×1600)** | $\approx 167.5$ ms | $\approx 167.8$ ms | $< +0.3$ ms ($< +0.2\%$) |
| **FPS (RTX 3050 Laptop)** | $\approx 6.0$ FPS | $\approx 6.0$ FPS | 0 (No perceptible degradation) |

**Assessment:** The proposed modification adds only **288 parameters** (+0.38%), keeping the model well within the lightweight edge-device boundary (<100k parameters) while preserving real-time throughput.

---

## 5. Implementation Risks & Mitigation Strategies

| Risk | Description | Impact | Mitigation Strategy |
|---|---|:---:|---|
| **Risk 1: Noise Injection in Deep Haze** | Computing Sobel gradients on noisy or heavily compressed hazy inputs could inject high-frequency noise into the network's first layer. | Low / Medium | Pass $I_{\text{gray}}$ through a normalized Gaussian blur ($\sigma=1.0$) or standard Sobel smoothing kernel before gradient extraction; clamp gradient map to $[0, 1]$. |
| **Risk 2: Color Distortions / Chrominance Imbalance** | Adding a scalar edge channel could bias the initial 32 filter responses toward luminance gradients, potentially disturbing color balance. | Low | The network is trained with the proven Experiment B multi-scale loss ($0.8 L_1 + 0.2 [1-\text{SSIM}]$), which penalizes RGB chrominance deviations heavily. |
| **Risk 3: Code Regression / Incompatibility** | Editing the existing `ResidualScaleVariant` in-place would invalidate existing checkpoints (`best_epoch_019.pt`, `best_epoch_020.pt`). | High | **Do NOT modify existing files.** Create a dedicated new model module: `src/dehaze/models/edge_conditioned_variant.py`, preserving `residual_scale_variant.py` completely intact. |
| **Risk 4: Optimistic Bias** | Assuming that adding an edge channel automatically guarantees higher SSIM/PSNR without rigorous validation. | Medium | Enforce pre-registered validation criteria and bounded comparisons. Do not claim improvements prior to empirical verification. |

---

## 6. Dataset Limitation Documentation (10,000-Image Guideline)

### 6.1 Current Dataset Scope
The project currently relies exclusively on the **NH-HAZE** benchmark dataset:
- **Total Paired Scenes:** 55 outdoor scenes (captured with real non-homogeneous outdoor fog generated by professional haze machines).
- **Split Breakdown:**
  - **Training:** 39 paired images (70.9%).
  - **Validation:** 8 paired images (14.5%).
  - **Held-Out Test:** 8 paired images (14.5%, strictly quarantined).

### 6.2 Comparison Against the 10,000-Image Major Project Guideline
- Standard deep learning benchmarks (such as RESIDE-OTS or large synthetic collections) frequently feature 10,000+ paired images.
- **Specific Limitations of NH-HAZE:**
  1. **Small Scene Diversity:** 39 training scenes represent a restricted sample of outdoor terrain, tree types, and lighting conditions.
  2. **Patch Augmentation Trade-off:** While random $256 \times 256$ cropping yields thousands of unique patches per epoch, these patches share semantic, lighting, and camera response profiles from the same 39 physical scenes.
  3. **High Resolution / Memory Constraint:** Full-resolution images are $1200 \times 1600$, requiring patch-based training to fit the 4 GB VRAM envelope of the RTX 3050 GPU.
- **Preservation Policy:** Under user instructions, **no external datasets will be downloaded or introduced**. Stage 3.8 will execute strictly on the canonical 39-train / 8-val split to ensure clean scientific control and direct comparability with Stages 3.4–3.7.

---

## 7. Validation-Only Experimental Protocol for Stage 3.8

To ensure rigorous scientific attribution, all hyperparameters and loss weights will be held strictly identical to Stage 3.7 Experiment B, isolating the architecture as the sole independent variable.

### 7.1 Protocol Matrix
- **Reference Model (Control):** `outputs/stage3_7_loss_l1_ssim/best_epoch_019.pt` (Val L1: 0.187040, Val PSNR: 12.9984 dB, Val SSIM: 0.4418, Edge: 0.0992).
- **Candidate Model (Treatment):** `EdgeConditionedVariant` (4-channel input, scale $\pm 0.2$, 76,035 parameters).
- **Loss Function:** Exactly $L_{\text{total}} = 0.8 \times L_1 + 0.2 \times (1 - \text{SSIM})$ (Identical to Exp B).
- **Optimizer & Budget:** Adam ($\text{lr} = 2 \times 10^{-4}$), batch size 2, patch size $256 \times 256$, exactly 20 epochs (400 training steps, 160 validation steps).
- **Reproducibility:** Seed 42, deterministic `EpochShuffleSampler`, identical canonical split manifest (`outputs/v0_pilot_001/split.json`).
- **Target Output Directory:** `outputs/stage3_8_architectural_edge_guidance/` (new, isolated directory).
- **Analysis Directory:** `analysis/stage3_8_architectural_edge_guidance/`.

### 7.2 Quarantine Protocol
- Held-out test scenes (`02, 08, 09, 15, 16, 18, 41, 48`) remain **100% quarantined**.
- Evaluator scripts will load only `manifest.validation` (8 scenes).

---

## 8. Pre-Registered Decision Criteria

| Criteria | Metric / Requirement | Target vs. Exp B Reference | Interpretation / Action |
|---|---|:---:|---|
| **Criterion 1 (PSNR)** | Validation PSNR | $\ge 13.00$ dB | Must match or exceed Exp B (12.9984 dB). |
| **Criterion 2 (SSIM)** | Validation SSIM | $\ge 0.4418$ | Must match or exceed Exp B (0.4418). |
| **Criterion 3 (Edge Recovery)** | Mean Edge Gradient Energy | $> 0.0992$ | Must exceed Exp B (0.0992) to confirm edge guidance effectiveness. |
| **Criterion 4 (L1 Error)** | Validation $L_1$ Loss | $\le 0.1875$ | Must remain competitive with Exp B (0.187040). |
| **Criterion 5 (Complexity)** | Parameter Budget | $\le 80,000$ params | Enforces lightweight edge deployment budget (Candidate: 76,035). |
| **Criterion 6 (Qualitative)** | Visual Contact Sheet | No artifacts | No edge haloing, false boundary ringing, or color shifting. |

### Decision Outcomes:
- **Outcome A (Clear Winner):** If Criteria 1, 2, 3, 5, and 6 are met, the Edge-Conditioned architecture is declared the Stage 3.8 winner and supersedes Experiment B.
- **Outcome B (Equivalence / Trade-off):** If Edge Energy improves but SSIM/PSNR drops below Exp B, Experiment B remains the overall reference, and edge guidance is documented as a scene-selective trade-off.
- **Outcome C (Underperformance):** If the candidate fails to improve edge energy or causes instability, the architectural modification is rejected, and Experiment B remains the recommended candidate.

---

## 9. Conclusion & Awaiting User Approval

This proposal defines a minimal, low-risk, scientifically controlled modification adding only 288 parameters (+0.38%) to test whether architectural edge guidance resolves the edge attenuation discovered in Stage 3.7.

**No code modifications, dependencies, or training runs have been initiated.** Awaiting user approval to proceed with Stage 3.8.
