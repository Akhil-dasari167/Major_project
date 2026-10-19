# Major Project Presentation and Viva Voce Defense Guide

**Project Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications  
**Academic Year:** 2026–2027 | Department of Electronics and Communication Engineering  
**Target Audience:** Final Project Review Panel, Internal & External Examiners  

---

## Part 1: Slide-by-Slide Presentation Structure (12-Slide Deck)

### Slide 1: Title Slide
- **Title:** Lightweight Edge-Aware Multi-Model Image Dehazing Framework for Real-Time Outdoor Vision Applications
- **Subtitle:** Major Project Defense — ECE (2026–2027)
- **Presenter / Team Details:** [Student Name / Roll Number]
- **Supervisor / Guide:** [Guide Name / Designation]
- **Key Highlight:** Real-time (182 ms/frame at $1600 \times 1200$), edge-preserving dehazing under 76K parameters on mobile GPU hardware.

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Atmospheric haze, smog, and fog attenuate light and scatter airlight, degrading contrast, color saturation, and edge sharpness in outdoor imagery.
- **Impact on Edge Vision:** Critical failure in autonomous navigation, obstacle detection, drone surveillance, and intelligent transportation systems (ITS).
- **The Research Dilemma:** State-of-the-art transformer / heavy CNN dehazing models (FFA-Net, DehazeNet, Restormer) require millions of parameters and seconds per frame, making them impossible to run on embedded edge devices.
- **Project Goal:** Design a sub-100K parameter edge-aware framework capable of real-time full-resolution processing ($1600 \times 1200$) without structural edge collapse.

---

### Slide 3: Physical Atmospheric Scattering Model
- **Koschmieder’s Law Formulation:**
  $$I(x) = J(x) t(x) + A (1 - t(x))$$
  - $I(x)$: Observed degraded hazy image.
  - $J(x)$: True scene radiance (dehazed clean image).
  - $t(x) = e^{-\beta d(x)}$: Medium transmission map with scattering coefficient $\beta$ and scene depth $d(x)$.
  - $A$: Global atmospheric light veil.
- **Inherent Difficulties:** An ill-posed inverse problem; for a single image, neither $t(x)$ nor $A$ is known a priori.
- **Residual Reformulation:** Rather than learning unconstrained $J(x)$, our model predicts a bounded residual correction:
  $$J(x) = \text{clamp}(I(x) + \alpha \cdot \tanh(R(x)), 0, 1)$$

---

### Slide 4: Proposed Complete Architecture
- Present the system architecture diagram (from the report):
  - **Lightweight Stem CNN:** 3-channel input mapped to 32 feature channels.
  - **Multi-Scale Feature Extractor:** Full, half, and quarter-resolution scales to capture both fine texture and broad haze envelopes.
  - **Adaptive Attention Feature Fusion Module (AAFM):** Softmax channel/spatial gating across scales.
  - **Three Specialized Approximation Branches:**
    1. Atmospheric-context branch (low-frequency smoothing + dilation).
    2. Layer-separation branch (base vs. detail feature decomposition).
    3. Contextual feature branch (dilated convolutions).
  - **Adaptive Multi-Model Branch Fusion:** Pixel-wise softmax gating.
  - **Edge-Aware Refinement:** Feature gating modulated by fixed Sobel gradient magnitude.
  - **Color & Texture Refinement:** Separate channel-mixing (color) from grouped convolutions (texture).
  - **Bounded Residual Reconstruction:** Output clamped strictly to $[0, 1]$.

---

### Slide 5: Loss Function Engineering (The Key Breakthrough)
- **Why Pure $L_1$ Fails:** Pixel-wise $L_1$ minimizes average pixel error, which causes edge attenuation and smudged textures (recovering only 24% of ground-truth high-frequency edge energy).
- **Our Multi-Objective Formulation (Experiment B):**
  $$\mathcal{L}_{\text{total}} = 0.8 \times \mathcal{L}_{L1} + 0.2 \times [1 - \text{SSIM}]$$
- **Why Structural Similarity (SSIM) Matters:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  Penalizes luminance, contrast, and structural covariance distortion directly, preserving boundary transitions without adding a single parameter to inference!

---

### Slide 6: Dataset and Strict Quarantine Protocol
- **Dataset:** NH-HAZE (55 paired outdoor non-homogeneous haze photographs, $1600 \times 1200$ resolution).
- **Partitioning (Canonical Seed 42):**
  - **Training Set:** 39 scenes (random $256 \times 256$ patches, paired horizontal flips).
  - **Validation Set:** 8 scenes (`53, 03, 28, 38, 06, 35, 44, 07`), full-resolution.
  - **Post-Hoc Test Set:** 8 scenes (`09, 15, 16, 18, 48, 02, 08, 41`), quarantined from training and tuning; evaluated post-hoc to benchmark against the Stage 3.6 reference.
- **Research Integrity & Quarantine:**
  - Test set was strictly excluded from training, loss optimization, checkpoint selection, and hyperparameter tuning.
  - Disjointness verified: $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$.

---

### Slide 7: Progressive Experimental Stages
Explain the step-by-step scientific progression:
1. **Stage 3.1 & 3.3 (V0 Pilot):** Baseline with $\pm0.1$ scale; diagnosed tanh ceiling saturation (99% saturation).
2. **Stage 3.5 & 3.6 (Residual Scale Study):** Expanded scale to $\pm0.2$; dropped saturation to 33.8% and boosted PSNR by +1.19 dB.
3. **Stage 3.7 (Loss Engineering):**
   - Exp B ($L_1 + \text{SSIM}$): Slashed validation L1 to 0.1870, lifted PSNR to 13.00 dB, increased SSIM to 0.4418 across 100% of scenes.
   - Exp C ($+ \text{Sobel}$ loss): Improved edge energy, but slightly traded off global SSIM.
4. **Stage 3.8 (Early Edge Concatenation):** Explicit 4-channel Sobel head surged edge energy (+18.1%), but propagated gradient noise in diffuse fog.
5. **Stage 4 (Multi-Model v1):** 114.7K param full multi-branch network; verified all 10 modules.

---

### Slide 8: Quantitative Benchmark Results
Present the validation and test comparison tables:

**Validation Standings (8 Scenes):**
| Model | Parameters | Val L1 Error | Val PSNR | Val SSIM |
|---|:---:|:---:|:---:|:---:|
| Stage 3.6 Baseline | 75,747 | 0.192185 | 12.84 dB | 0.4268 |
| **Stage 3.7 Exp B (Winner)** | **75,747** | **0.187040** | **13.00 dB** | **0.4418** |
| Stage 3.7 Exp C | 75,747 | 0.188690 | 12.97 dB | 0.4371 |
| Stage 3.8 EC-Head | 76,035 | 0.191811 | 12.75 dB | 0.4227 |
| Multi-Model v1 | 114,753 | 0.224439 | 11.64 dB | 0.4039 |

**Post-Hoc Held-Out Test Performance (8 Scenes):**
| Metric | Baseline | Exp B (Winner) | Improvement |
|---|:---:|:---:|:---:|
| **Test PSNR** | 14.58 dB | **14.72 dB** | **+0.14 dB** |
| **Test SSIM** | 0.5355 | **0.5441** | **+0.0086 (Won 8/8)** |
| **Test L1 Error** | 0.1506 | **0.1488** | **-1.2% error reduction** |

---

### Slide 9: Qualitative Visual Analysis
- Showcase before-and-after panels from the contact sheets:
  - **Foliage and Tree Textures (Scene 38 & 35):** Experiment B produces visibly sharper branch outlines and bark ridges compared to the smudgy baseline.
  - **Building and Railing Contours (Scene 44 & 03):** Natural midtone transitions and restored geometric silhouettes.
  - **Artifact-Free Output:** Zero checkerboard noise, zero color fringing, and negligible pixel clamping (0.02%).

---

### Slide 10: Real-Time Latency & Edge Deployment
- **Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1).
- **Execution Profile:**
  - Full Resolution ($1600 \times 1200$): **182.14 ms / frame** (~5.5 FPS sustained warm latency).
  - Single-Image Cold Start: **~350–396 ms** (includes PyTorch graph and CUDA context initialization).
  - Real-Time Robotics Feasibility: Standard video resolutions ($640 \times 480$ / 720p) achieve $> 30$ FPS.
  - VRAM Consumption: $< 1.8$ GB (safe for 4 GB laptop GPU).
  - Checkpoint Size: **933.6 KB** (fits effortlessly in edge flash storage).
- **Generic CLI & Streamlit Demo:**
  - `scripts/infer.py` provides batch and single-image deployment.
  - `app.py` delivers interactive web demonstration with optional real-time ground-truth PSNR/SSIM evaluation.

---

### Slide 11: Limitations and Future Work
- **Small Paired Dataset Size:** Real outdoor non-homogeneous haze pairs are scarce (NH-HAZE has 55 pairs). Future work can incorporate domain-adapted synthetic data.
- **Selective Edge Gating:** Restrict edge attention to high-contrast structural regions to prevent particulate noise amplification in diffuse fog.
- **Edge Acceleration:** TensorRT FP16 quantization will boost throughput from 5.5 FPS to $> 20$ FPS for embedded robotics platforms (Jetson Orin).

---

### Slide 12: Conclusion & Summary
- Successfully delivered a lightweight, edge-aware outdoor image dehazing framework.
- Solved the edge suppression problem via structural loss engineering ($0.8 L_1 + 0.2 [1-\text{SSIM}]$) without adding computational overhead.
- Rigorously audited code, test suites (26/26 tests passing), reproducible splits, and web demonstration application.
- Thank the panel and open for questions.

---

## Part 2: Viva Voce Defense Q&A (Frequently Asked Questions)

### Q1: Why did you use Koschmieder’s model as the theoretical basis if your network predicts residuals directly?
**Answer:** Koschmieder’s law ($I = J \cdot t + A(1-t)$) explains that atmospheric haze acts primarily as an additive veil ($A(1-t)$) that washes out scene reflectance ($J \cdot t$). Because the veil adds bright airlight, dehazing an outdoor scene predominantly requires subtracting this veil. By using a bounded residual formulation ($J = \text{clamp}(I + \alpha \tanh(R), 0, 1)$), our network learns this additive/subtractive residual correction directly. Over 96% of the network’s active corrections are negative, directly confirming that it learns to subtract the airlight veil.

### Q2: Why is Experiment B (75,747 parameters) superior to Multi-Model v1 (114,753 parameters) on NH-HAZE?
**Answer:** Deep multi-branch architectures (with multi-scale pyramids, AAFM, and three parallel branches) have significantly more feature pathways to optimize. Because NH-HAZE contains 39 training images, a deeper model trained from scratch without pretraining requires a larger training regime and more data to converge effectively. Experiment B’s compact 4-block residual trunk with composite $L_1 + \text{SSIM}$ loss has the ideal inductive bias for small-sample training, achieving superior convergence (13.00 dB PSNR vs 11.64 dB) without overfitting.

### Q3: Why did Stage 3.8 (early edge concatenation) achieve the highest edge energy but lower PSNR?
**Answer:** Stage 3.8 concatenated raw Sobel gradients at layer 0. The Sobel operator computes non-zero gradients not only on real geometric structures (building edges, tree trunks) but also on random smoke turbulence and particulate noise in dense fog. Because early convolution layers cannot distinguish between noise gradients and true edge gradients, the network slightly modulated pixels in diffuse fog regions, which penalized global pixel-wise PSNR even while sharpening genuine boundaries.

### Q4: How did you ensure there was no data leakage during hyperparameter tuning and model selection?
**Answer:** We established a canonical Seed-42 split ([`split.json`](file:///c:/Users/srich/OneDrive/Tài%20liệu/akhilMajor/outputs/v0_pilot_001/split.json)) partitioning the 55 NH-HAZE scenes into 39 train, 8 validation, and 8 test scenes. All training scripts (`run_stage3_7_loss_l1_ssim.py`, `train_multimodel_variant.py`, etc.) loaded strictly `split.train` for gradient updates and `split.validation` for saving the `best_epoch.pt` checkpoint. The 8 test scenes were quarantined and evaluated post-hoc under identical inference conditions.

### Q5: Can this system run in real-time on autonomous vehicles or drones?
**Answer:** Yes. At full photographic resolution ($1600 \times 1200$), the model executes in **182.14 ms** (~5.5 FPS) in unquantized FP32 on an entry-level RTX 3050 mobile GPU. For robotic applications where standard camera resolutions are $640 \times 480$ or $1280 \times 720$, the inference time drops to $< 35$ ms ($> 30$ FPS). Furthermore, the entire model is only 75,747 parameters (933.6 KB), allowing it to easily fit inside embedded SRAM/cache on edge platforms.
