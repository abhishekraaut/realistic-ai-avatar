# Neural Renderer V1 Phase 8B Baseline Results

## Experiment
`renderer-v1-static-512`
Objective: Static motion-conditioned image reconstruction.
Input: Identity Reference + Ground-Truth 11D Motion -> Target Frame.

## GPU
Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU
CUDA: 12.1 (PyTorch) / 12.5 (Driver)
PyTorch: 2.5.1+cu121

## Dataset
Total: 1418 frames (512x512)
- Train: 748 frames
- Validation: 300 frames
- Test: 148 frames

## Model
Architecture: `NeuralRendererV1` (IdentityEncoder, MotionConditioning with FiLM, Decoder)
Parameters: 2,123,299

## Training
Configuration:
- Batch Size (Physical): 4
- Gradient Accumulation: 1 (Effective Batch Size: 4)
- AMP: On
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: L1Loss
- Epochs: 50
- Resolution: 512x512
- Motion Dim: 11

## Validation
Metrics on 300 validation frames (from `practice-feedback.mp4`):
- Best Epoch: 10
- Train L1 (Best Epoch): 0.0721
- Validation L1 (Best Epoch): 0.1563
- Validation PSNR (Best Epoch): 19.26 dB

*See section below for full independent image/landmark metrics calculated on the best checkpoint.*

## Image Quality
- L1: 0.1563
- MSE: 0.0475
- PSNR: 19.2601 dB
- SSIM: Unavailable (skimage module not installed)

## Landmark Quality
*Based on MediaPipe independent evaluation of generated vs ground-truth frames.*
- Full-Face Error: 0.0145
- Mouth-Region Error: 0.0145
- Eye-Region Error: 0.0145
- Head-Pose Error: 0.0145

## Identity
Automated identity metric unavailable.

## Performance
Renderer-only inference (batch size 1, 512x512, AMP ON, 10 warmup, 100 iterations, properly synchronized):
- Mean: 8.78 ms
- p50: 8.73 ms
- p95: 11.00 ms
- Min: 6.00 ms
- Max: 11.28 ms
- Effective FPS: ~114 FPS (Renderer only)
- Peak VRAM: 0.03 GB (inference) / 0.19 GB (training)

## Visual Results
Generated artifacts (ground truth vs generated vs absolute difference) are saved in the `artifacts/` folder (`val_frame_*.jpg` and `validation_sequence.mp4`).

## Failure Analysis
- **Overfitting:** Training loss continued to decrease (reaching 0.0505) while validation loss plateaued around epoch 10 and slightly degraded, indicating overfitting on the small 748-frame training set.
- **Single-Identity Limitation:** The model was trained entirely on a small subset of a single identity, so it lacks generalization capabilities to novel identities.
- **Static Artifacts:** As expected from a static renderer (Phase 8B), frame-to-frame temporal coherence is not guaranteed.

## Status
`EXPERIMENTAL / TRAINED / VALIDATED`
- The static reconstruction baseline has been trained and validated. Photorealistic Avatar capabilities and full Real-Time Pipeline are not yet validated.
