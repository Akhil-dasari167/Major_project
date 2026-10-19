# Stage 3.7 — Diagnosis and PSNR/SSIM Improvement Experiment Plan

**Document:** Stage 3.7 Diagnosis & Controlled Experiment Plan  
**Date:** 2026-10-18  
**Scope:** Validation-only diagnosis and prospective experiment planning. No training executed, no code or dependencies modified, no test split accessed or evaluated.  
**Reference Checkpoint:** [`outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt)  
**Dataset & Canonical Partition:** NH-HAZE (55 scenes, Seed 42): 39 Train / 8 Validation / 8 Held-Out Test (Untouched).  

---

## Phase 1 — In-Depth Audit and Validation Diagnosis

We performed a comprehensive diagnostic audit of the reference model ($\pm 0.2$ residual-scale variant, Epoch 20) using **training and validation data only**. Held-out test scenes (`09, 15, 16, 18, 48, 02, 08, 41`) were strictly quarantined and unread.

### 1. Residual Saturation & Darkening Directionality
* **Predominant Darkening:** Mean signed correction on the 8 validation scenes is **$-0.1111$** (mean absolute correction: $0.1196$). Over **$96\%$** of active corrections are negative. Because NH-HAZE outdoor scenes are blanketed in bright, scattering fog, the network learns to subtract airlight luminance.
* **Persistent Saturation:** While the $\pm 0.2$ bound dramatically reduced the severe $>99\%$ saturation observed in $\pm 0.1$, **$33.89\%$ of validation pixels remain saturated** ($\ge 0.19$ absolute correction). In dense haze regions (Scenes `03` and `38`), the true ground-truth difference requires subtracting $>0.25$ to $0.35$ in intensity, which the bounded tanh branch physically cannot output.

### 2. Dehazing Quality: Residual Haze, Color Shifts, and Edge Attenuation
We conducted a quantitative scene-by-scene analysis across all 8 validation images:

| Scene ID | Hazy Mean RGB | Ground Truth Mean RGB | Prediction Mean RGB | Hazy-GT L1 | Pred-GT L1 | Val PSNR | Val SSIM | Hazy Edge | GT Edge | Pred Edge | Pred/GT Edge Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **53** | `[0.55, 0.58, 0.64]` | `[0.40, 0.37, 0.32]` | `[0.42, 0.44, 0.48]` | 0.2773 | 0.1792 | 13.06 dB | 0.4469 | 0.1014 | 0.3650 | 0.1019 | **0.28** |
| **03** | `[0.57, 0.63, 0.71]` | `[0.36, 0.39, 0.41]` | `[0.39, 0.45, 0.52]` | 0.2858 | 0.2162 | 11.84 dB | 0.4296 | 0.1158 | 0.4008 | 0.1153 | **0.29** |
| **28** | `[0.53, 0.55, 0.59]` | `[0.57, 0.58, 0.55]` | `[0.47, 0.47, 0.45]` | 0.2338 | 0.2131 | 12.27 dB | **0.3362** | 0.0612 | 0.4473 | 0.0629 | **0.14** |
| **38** | `[0.57, 0.61, 0.66]` | `[0.34, 0.34, 0.34]` | `[0.44, 0.46, 0.48]` | 0.2864 | 0.1785 | 13.24 dB | 0.4354 | 0.0850 | 0.3013 | 0.0846 | **0.28** |
| **06** | `[0.41, 0.43, 0.46]` | `[0.46, 0.43, 0.39]` | `[0.38, 0.37, 0.34]` | 0.1874 | 0.1639 | 13.71 dB | 0.4793 | 0.0717 | 0.3704 | 0.0866 | **0.23** |
| **35** | `[0.53, 0.55, 0.57]` | `[0.37, 0.38, 0.38]` | `[0.49, 0.48, 0.45]` | 0.2146 | 0.1613 | 14.17 dB | 0.3915 | 0.0997 | 0.4751 | 0.1006 | **0.21** |
| **44** | `[0.64, 0.67, 0.72]` | `[0.54, 0.55, 0.54]` | `[0.52, 0.54, 0.55]` | 0.2783 | 0.2284 | 11.83 dB | 0.4162 | 0.0835 | 0.3393 | 0.0834 | **0.25** |
| **07** | `[0.48, 0.47, 0.49]` | `[0.49, 0.44, 0.40]` | `[0.49, 0.45, 0.41]` | 0.2242 | 0.1969 | 12.57 dB | 0.4794 | 0.0971 | 0.3904 | 0.1051 | **0.27** |
| **Mean** | — | — | — | **0.2485** | **0.1922** | **12.84 dB** | **0.4268** | **0.0894** | **0.3862** | **0.0926** | **0.24** |

#### Critical Findings from the Table:
* **Severe Edge Attenuation:** Ground truth images exhibit strong high-frequency edge energy (Sobel mean: **0.3862**). Haze suppresses this energy down to **0.0894**. The model predictions show an edge energy of only **0.0926** (an edge preservation ratio of only **$24\%$**). The network is performing smooth low-frequency tone mapping rather than edge reconstruction.
* **Residual Blue Haze Shift:** In Scenes `53` and `03`, the haze veil has a strong cool/blue cast ($B > 0.64$, $R < 0.57$). In the ground truth, the vegetation and ground are warm ($R > B$). While the model darkens all channels, the prediction retains an excess blue component ($B - R \approx +0.06$ to $+0.13$), leaving an unnatural cool veil.
* **Indiscriminate Darkening (Scene 28):** In Scene `28`, the ground truth is actually *brighter* in Red ($0.57$ GT vs $0.53$ Hazy). Because the model is heavily biased toward negative residuals, it inappropriately darkens the Red channel to $0.47$, yielding the worst SSIM on the validation split (**0.3362**).

### 3. Convergence & Loss Trajectory: Underfitting vs. Overfitting
* **No Divergence:** Training L1 decreased monotonically from $0.2200 \to 0.1685$. Validation L1 decreased from $0.2333 \to 0.1922$.
* **Narrow Generalization Gap:** The gap between train and validation L1 at Epoch 20 is only **$0.0237$**. Validation loss did not degrade or oscillate.
* **Diagnosis:** The network is **substantially underfitting**. The underfitting stems from three distinct architectural and training constraints:
  1. **Tiny Receptive Field:** 4 residual blocks without downsampling/pooling provide an effective receptive field of only **$21 \times 21$ pixels**. On a $1200 \times 1600$ image, $21$ pixels is ~1.3% of the image width, preventing the network from estimating scene-level atmospheric light $A$ or transmission $t(x)$.
  2. **Model Capacity:** 75,747 parameters is very lean, limiting high-order feature representation.
  3. **Loss Function Limitation:** Pure L1 loss computes an absolute pixel difference that is dominated by bulk luminance shifts and completely insensitive to structural coherence or gradient sharpness.

### 4. Implementation Integrity Verification
* **Data Pipelines & Normalization:** Images are loaded via `Pillow` as float tensors normalized in $[0.0, 1.0]$ in true RGB order.
* **Metric Implementations:** `psnr` correctly uses $MAX_I = 1.0$; `ssim` implements the standard 11×11 Gaussian filter ($\sigma=1.5, C_1=10^{-4}, C_2=9\times 10^{-4}$).
* **Output Clamping:** Predictions are properly clamped to $[0.0, 1.0]$.
* **Conclusion:** There are **zero arithmetic, indexing, or normalization bugs**. The performance ceiling is dictated strictly by the loss formulation and the architecture.

### 5. Evidence Summary: Verified Facts vs. Working Hypotheses

| Observation | Status | Verified Evidence |
| :--- | :---: | :--- |
| Model fails to recover edge gradients | **Verified Fact** | Sobel edge energy is only 24% of GT (0.0926 vs 0.3862). |
| Residual saturation limits dense haze removal | **Verified Fact** | 33.89% of pixels reach $\ge 0.19$ bound; Scenes 03 & 38 remain veild. |
| Model suffers from underfitting, not overfitting | **Verified Fact** | Monotonic val L1 decrease; narrow 0.023 train-val gap. |
| Receptive field is local (21×21) | **Verified Fact** | Layer composition: $3 + 4\times 4 + 2 = 21$ pixels. |
| Adding SSIM loss will improve structural fidelity | **Hypothesis** | Supported by the fact that SSIM correlates with visual perception and penalizes local luminance/contrast distortion (like Scene 28). Needs controlled ablation. |
| Adding Sobel edge loss will restore high-frequency detail | **Hypothesis** | Supported by the 24% edge ratio deficit. Needs controlled ablation to avoid edge ringing or artifacts. |
| Direct prediction removes saturation limits | **Hypothesis** | Allows output to reach zero/clean values without bound, but may destabilize training convergence without the identity residual anchor. |
| Cosine LR scheduling refines convergence | **Hypothesis** | May help Adam escape late plateaus without hyperparameter explosion. |

---

## Phase 2 — Controlled Experiment Matrix (Stage 3.7)

All experiments will use the **canonical split (Seed 42: 39 train / 8 validation)**, identical patch extraction ($256\times 256$, batch size 2, 20 epochs), and identical evaluation protocol. All runs will be logged in dedicated output folders without modifying the baseline $\pm 0.2$ checkpoint.

```
                                 [Reference V0.2]
                                 (L1 Loss, ±0.2)
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
      [Branch B]                   [Branch C]                   [Branch D]
   Loss: L1 + SSIM             Loss: L1 + Sobel Edge       Formulation: Direct Clean
  (Structural Focus)             (Gradient Focus)             (Unbounded Output)
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                                   [Branch E]
                               Training Strategy:
                             Cosine Annealing / LR
```

### Experiment A: Reference Baseline (Completed)
* **Configuration:** Architecture $V0$, Residual scale $\pm 0.2$, Loss: Pure L1.
* **Checkpoint:** `outputs/residual_scale_continuation_0p2_rerun_1/best_epoch_020.pt`
* **Baseline Validation Metrics:** Val L1: `0.192185`, PSNR: `12.8367 dB`, SSIM: `0.4268`.

---

### Experiment B: Structural Loss Enhancement ($\mathcal{L}_{L1} + \lambda_{ssim} \mathcal{L}_{SSIM}$)
* **Hypothesis:** Pixel-wise L1 loss treats pixels independently, allowing contrast collapse in ambiguous scenes (Scene 28 SSIM: 0.3362). Adding a differentiable SSIM loss penalizes local luminance and contrast distortion, directly raising structural preservation.
* **Exact Change:**
  $$\mathcal{L}_{total} = (1 - \alpha)\mathcal{L}_{L1} + \alpha (1 - \text{SSIM}(I_{pred}, I_{gt})) \quad \text{with } \alpha = 0.2$$
* **Expected Benefit:** Measurable improvement in validation SSIM (target: $>0.47$) and reduced blurriness.
* **Possible Risk:** Minor gradient conflict between L1 and SSIM terms; slightly increased backpropagation time (~10-15%).
* **Pre-Registered Validation Criterion:** Win if Validation SSIM increases by $\ge +0.020$ without Validation L1 increasing by more than $0.005$.

---

### Experiment C: Edge-Gradient Loss Enhancement ($\mathcal{L}_{L1} + \lambda_{edge} \mathcal{L}_{Edge}$)
* **Hypothesis:** The model achieves only 24% of the ground truth's edge gradient energy. Explicitly penalizing the difference in Sobel edge gradients will force the convolutional kernels to preserve high-frequency foliage and boundary details.
* **Exact Change:**
  $$\mathcal{L}_{edge} = \frac{1}{N} \|\nabla_x I_{pred} - \nabla_x I_{gt}\|_1 + \frac{1}{N} \|\nabla_y I_{pred} - \nabla_y I_{gt}\|_1$$
  $$\mathcal{L}_{total} = \mathcal{L}_{L1} + \lambda_{edge} \mathcal{L}_{edge} \quad \text{with } \lambda_{edge} = 0.05$$
* **Expected Benefit:** Significant recovery in the Pred/GT edge energy ratio (target: $>0.50$), sharpening foliage and architectural lines.
* **Possible Risk:** Setting $\lambda_{edge}$ too high can cause high-frequency checkerboard noise or halo artifacts along contrast edges.
* **Pre-Registered Validation Criterion:** Win if Pred/GT edge energy ratio exceeds $0.40$ and Validation PSNR improves or remains within $\pm 0.2$ dB of baseline.

---

### Experiment D: Output Formulation (Direct Clean Prediction vs. Residual)
* **Hypothesis:** 33.89% of pixels remain saturated against the $\pm 0.2$ ceiling because dense haze requires intensity drops $>0.25$. Formulating the network to directly predict the clean image $I_{clean} = \text{clamp}(f(I_{hazy}), 0, 1)$ or unbounded residual removes the artificial clipping ceiling.
* **Exact Change:**
  Replace output layer $I_{hazy} + 0.2 \cdot \tanh(R)$ with $I_{clean} = \text{clamp}(f(I_{hazy}), 0, 1)$ using an identical 4-ResBlock backbone.
* **Expected Benefit:** Complete elimination of near-bound residual saturation; capability to remove dense fog veils in Scenes `03` and `38`.
* **Possible Risk:** Losing the identity-shortcut anchor can increase optimization variance, slow initial convergence, or lead to color drift.
* **Pre-Registered Validation Criterion:** Win if Validation L1 decreases by $\ge 0.005$ and near-bound saturation drops to $0\%$.

---

### Experiment E: Training Strategy (Cosine Annealing Learning Rate)
* **Hypothesis:** A constant learning rate ($2\times 10^{-4}$) underfits the training patches and prevents fine-grained convergence in late epochs. A Cosine Annealing scheduler tapering to $\eta_{min} = 1\times 10^{-6}$ will allow the optimizer to settle into sharper local minima.
* **Exact Change:**
  Add `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)` without changing any loss or architecture.
* **Expected Benefit:** Smoother loss trajectory, improved late-epoch validation loss.
* **Possible Risk:** Premature learning rate decay if the learning rate drops too fast in early epochs.
* **Pre-Registered Validation Criterion:** Win if Validation L1 improves by $\ge 0.002$ over constant LR reference.

---

## Phase 3 — Strict Evaluation Protocol

For all Stage 3.7 prospective experiments:
1. **Partition Isolation:** Training is strictly confined to the 39 training scenes (`train` list in canonical manifest).
2. **Validation-Only Selection:** All decisions, ablation comparisons, and hyperparameter selections will be made using the **8 validation scenes** (`53, 03, 28, 38, 06, 35, 44, 07`).
3. **Quarantine of Held-Out Test Set:** **Zero test evaluations** will be performed during Stage 3.7. The 8 test scenes (`09, 15, 16, 18, 48, 02, 08, 41`) will not be read or evaluated until the complete architectural and loss search is concluded.
4. **Logged Outputs:** Every run will produce:
   * `config.json`: Immutable hyperparameter record.
   * `history.csv`: Per-epoch train/val L1, PSNR, SSIM, edge energy, saturation %, and runtime.
   * `best_epoch_*.pt`: Checkpoint corresponding to minimum validation loss.
   * `validation_contact_sheet_best.png`: Full-resolution 8-scene visual comparison sheet.
   * `summary.json`: Parameter count, Conv2d MACs, and final validation metrics.

---

## Phase 4 — Dataset Limitations and Future Scaling Policy

### Current Limitations:
1. **Sample Size:** 39 training scenes is very small for deep learning. Over 20 epochs with 1 crop/image, the network sees only 780 total $256\times 256$ patches.
2. **Non-Homogeneous Real Fog:** Real outdoor fog has spatially non-uniform density and depth. Unlike synthetic datasets with clear transmission maps $t(x) = e^{-\beta d(x)}$, NH-HAZE lacks depth maps, making purely local models struggle.

### Policy for Stage 3.7:
* Continue strictly with the existing NH-HAZE dataset and the canonical 39/8/8 split.
* **Do not download or add external datasets during Stage 3.7.**

### Future Dataset Scaling Trigger:
* Additional large paired datasets (such as RESIDE-OTS or RESIDE-INDOOR) will become necessary only when:
  1. We advance to **Stage 4** to build a multi-scale U-Net or edge-guided dual-branch model with larger capacity (>500k parameters), where 39 scenes would risk severe overfitting.
  2. We need cross-dataset synthetic-to-real transfer evaluation.

---

## Phase 5 — Estimated Workload & Recommended Next Safe Action

### Workload Estimate (NVIDIA RTX 3050 Laptop GPU):
* **Single 20-epoch training run:** 400 optimizer steps + 160 validation forwards takes **~4.5 to 5.0 minutes**.
* **Disk Footprint:** Each run consumes ~15 MB for checkpoints and contact sheets (ample room in 40.5 GB free space).

### Recommended Next Action (Awaiting Your Approval):
Implement and execute **Experiment B** (Loss-Function Experiment: $\mathcal{L}_{L1} + 0.2 \cdot \mathcal{L}_{SSIM}$) as the first isolated ablation:
* **Output Directory:** `outputs/stage3_7_loss_l1_ssim/`
* **Model:** Unchanged $\pm 0.2$ architecture (`ResidualScaleVariant`).
* **Evaluation:** Validation split only.
* **Report:** Compare Validation L1, PSNR, SSIM, and contact sheets against the reference `outputs/residual_scale_continuation_0p2_rerun_1`.

*No files have been modified, no code has been changed, and no training has started. Awaiting your approval before proceeding.*
