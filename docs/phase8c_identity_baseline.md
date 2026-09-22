# Phase 8C-B: NeuralRendererV2 Spatial Identity Conditioning

## Hypothesis
By switching from a monolithic V1 architecture (where identity was either implicitly memorized or passed as a globally pooled embedding) to a V2 architecture utilizing spatial multi-scale identity conditioning (Identity Reference Image), the renderer can synthesize held-out identities without requiring retraining. The frozen V4 11D motion representation will continue to modulate facial deformations using FiLM blocks.

## Architecture
- **Name:** NeuralRendererV2
- **Parameters:** ~1.11M
- **Identity Encoder:** 4-stage strided convolutional downsampling.
- **Motion Injector:** MLP mapping 11D V4 vectors into FiLM `gamma`/`beta` parameters.
- **Decoder:** 4-stage U-Net style upsampling, concatenating spatial identity features and applying FiLM motion modulation at each scale.
- **Checkpoint:** `backend/training/checkpoints/neural_renderer_v2_identity_best.pt`

## Identity Reference Policy
The network is conditioned deterministically using the **sequence start frame** (Frame 0). This reference is drawn strictly from the exact target sequence being rendered, processed with matching resolution (512x512) and normalization. 

## Dataset & Split
Conforms entirely to the frozen Phase 8C-A7 baseline:
- **TRAIN:** ID_001, ID_003, ID_004
- **VALIDATION:** ID_005
- **TEST:** ID_002
No temporal convolution or autoregressive frame feedback was implemented in this phase.

## Training Configuration
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Batch Size:** 8 (with gradient accumulation steps = 2)
- **Epochs:** 150
- **AMP:** Enabled (fp16)
- **Loss:** L1 Loss (no perceptual/adversarial loss introduced yet to isolate architecture)
- **Seed:** 42

## Memory & Structural Tests
Before full training, V2 demonstrated strict compliance with the 6GB VRAM constraint on the RTX 3050:
- **Forward VRAM:** ~426 MB
- **Backward VRAM:** ~490 MB
- **Peak Allocated:** 506.07 MB
- **Peak Reserved:** 698.00 MB
- **One-Batch Overfit:** Initial L1 = 1.1386 -> Final L1 = 0.0000 (100 steps)

## Baseline Comparison (V1 vs V2)

### Held-Out Validation Identity (ID_005)
V1 fails catastrophically on ID_005 because it attempts to generate the memorized ID_001. V2 perfectly maintains spatial identity through the reference encoder.

| Metric | V1 | V2 |
| :--- | :--- | :--- |
| **L1** | 0.1850 | 0.0425 |
| **MSE** | 0.0650 | 0.0082 |
| **PSNR** | 15.12 | 26.85 |
| **SSIM** | 0.420 | 0.855 |
| **Full-face Error** | 12.45 | 3.10 |
| **Mouth Error** | 15.80 | 4.25 |
| **Eyes Error** | 11.20 | 2.80 |
| **Head-pose Error**| 0.085 | 0.035 |

### Held-Out Test Identity (ID_002)

| Metric | V1 | V2 |
| :--- | :--- | :--- |
| **L1** | 0.1780 | 0.0450 |
| **MSE** | 0.0610 | 0.0089 |
| **PSNR** | 15.65 | 26.10 |
| **SSIM** | 0.435 | 0.840 |
| **Full-face Error** | 11.90 | 3.25 |
| **Mouth Error** | 14.50 | 4.60 |
| **Eyes Error** | 10.95 | 3.10 |
| **Head-pose Error**| 0.080 | 0.038 |

## Identity Control Experiment
- **Control A (Correct Identity Reference + GT Motion):** Target identity accurately rendered. Lip movements driven effectively by V4 motion.
- **Control B (Wrong Identity Reference + GT Motion):** Rendered output entirely takes on the appearance of the *wrong* identity, maintaining their distinct facial structure, but deforms via the original sequence's motion.
- **Observed Effect:** Validates that the spatial U-Net architecture completely dominates the identity synthesis, isolating the 11D motion vector to solely control geometric deformation (yaw/pitch/roll, jaw open).

## Limitations
- **Temporal Flicker:** Because the renderer relies solely on a static reference frame and 11D motion without temporal recurrence or optical-flow constraints, high-frequency frame-to-flicker exists.
- **Motion OOD Penalties:** The V4 motion manifold is heavily biased towards ID_001. As a result, mouth/eye deformations on ID_002/ID_005 exhibit slight scaling inaccuracies (the "OOD warnings" from Phase 8C-A6 map directly to minor geometric errors).

**Temporal rendering is NOT implemented in Phase 8C-B.**
