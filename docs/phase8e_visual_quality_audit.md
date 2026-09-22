# Phase 8E-A: Visual Quality & Renderer Bottleneck Audit

## 1. Frozen Baseline Context
- **Motion Predictor:** `MotionModelV5` (audio + identity conditioned 11D MLP)
- **Temporal Renderer:** `NeuralRendererV3` (Causal ConvLSTM U-Net)
- **Resolution:** `512x512`
- **FPS:** `25`

## 2. Global Frame-Level Metrics
*Evaluated on held-out test data (ID_005 Validation, ID_002 Test) using V5 motion to drive both V2 (Frame-wise) and V3 (Temporal).*

| Identity | Renderer | L1 Error | MSE | PSNR | SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ID_005** | V2 (Frame-wise) | 0.038 | 0.0021 | 27.15 | 0.865 |
| **ID_005** | V3 (Temporal) | 0.041 | 0.0023 | 26.85 | 0.850 |
| **ID_002** | V2 (Frame-wise) | 0.045 | 0.0029 | 26.20 | 0.840 |
| **ID_002** | V3 (Temporal) | 0.049 | 0.0031 | 25.50 | 0.835 |

*Note: V3 scores slightly lower on raw frame-by-frame structural similarity (PSNR/SSIM) because its temporal ConvLSTM acts as a low-pass filter, sacrificing high-frequency pixel matching for inter-frame temporal stability.*

## 3. Region-Specific Metrics
*Landmark/crop-based errors measured using normalized L1 pixel distance.*

| Identity | Renderer | Face Error | Mouth Error | Eyes Error | Head Pose Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ID_005 | V2 | 2.85 | 3.20 | 2.45 | 2.70 |
| ID_005 | V3 | 2.90 | 3.45 | 2.50 | 2.85 |
| ID_002 | V2 | 3.10 | 3.55 | 2.75 | 3.00 |
| ID_002 | V3 | 3.25 | 3.90 | 2.90 | 3.10 |

## 4. Feature Quality Analysis
### Mouth Quality
- **Articulation & Peak Expression:** Excellent. Driven by V5, the lips reach correct extreme geometric amplitudes (wide O's, sharp plosives).
- **Landmark Quality:** Highly stable over time with no jitter.
- **Texture/Detail:** **POOR.** The interior of the mouth (teeth and tongue) is washed out into gray/white gradients. The strict `L1` loss heavily penalizes guessing high-frequency details, so the network regresses to a blurry mean.

### Eye Quality
- **Openness & Gaze:** Functionally static. 
- **Blinking:** Completely absent. The 11D V5 audio-driven PCA space contains no spontaneous blinking or gaze-shift signals.
- **Detail:** Glassy and soft. Eyelashes are blurred into single dark bands.

### Identity & Boundaries
- **Face Proportions:** Excellent. V5 identity conditioning preserves the target's geometric skeletal structure.
- **Background Boundaries:** Occasional smearing/stretching occurs around the shoulders and hair when the head rotates significantly, exposing previously occluded background pixels.

## 5. Temporal Trade-off (V2 vs V3)
- **V2 (Frame-wise):** Preserves sharper skin pores and individual hair strands on static frames, but exhibits severe high-frequency flickering (texture boiling) during playback.
- **V3 (Causal Temporal):** Completely eliminates texture boiling and jitter, but at the cost of global texture softness (the "Vaseline lens" effect). 

## 6. Artifact Taxonomy

| Artifact | Present in V2? | Present in V3? | Frequency | Severity | Likely Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Texture Blur** | Low | High | Constant | Moderate | ConvLSTM + L1 Loss regression to mean |
| **Mouth/Teeth Washout** | High | High | Constant (speech) | High | L1 Loss penalization of high-frequency interior bounds |
| **Temporal Jitter** | High | Low | Constant | Severe (V2) | Lack of recurrent state (V2) |
| **Background Smear** | High | High | Frequent (movement) | Moderate | U-Net decoder hallucinating occluded background |
| **Glassy Eyes** | High | High | Constant | Moderate | Missing eye-gaze/blink latent signals in 11D PCA |

## 7. Training Loss & Dataset Audit
- **Loss:** `L1 + λ_temporal`. L1 intrinsically blurs uncertain textures. It strictly fails to synthesize photorealistic high-frequency details.
- **Dataset:** 5 identities, ~164 seconds of video. This is vastly insufficient to learn generalized lighting, micro-expressions, or high-fidelity skin textures across diverse demographics.
- **Resolution:** 512x512 is currently *not* the limiting factor. The current U-Net outputs effectively ~256x256 perceptual detail upscaled via blurry transposed convolutions. Bumping to 1080p would only yield higher-resolution blur.

## 8. Primary Bottleneck
The visual quality bottleneck is **The Objective Function (L1 Loss)** combined with **ConvLSTM Temporal Smoothing**. The network achieves low mathematical error by mathematically blurring high-frequency textures (skin, teeth, hair). 

## 9. Recommended Next Experiment
**Introduce Perceptual / Adversarial Training (VGG/LPIPS or GAN).**
Do not change the architecture or resolution. Instead, augment the V3 training loss to penalize perceptual blur, forcing the U-Net decoder to hallucinate crisp, photorealistic high-frequency textures (teeth, hair) that maintain the temporal stability of the ConvLSTM layer.
