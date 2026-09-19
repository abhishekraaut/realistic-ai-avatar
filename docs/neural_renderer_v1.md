# Neural Renderer V1 (Static Reconstruction Proof-of-Concept)

## Architecture
**Model**: `NeuralRendererV1`
**Type**: Static Image Reconstruction (UNet + FiLM)
**Encoder**: `IdentityEncoder` (Conv2D -> ResidualBlocks) mapping 512x512 -> 16x16 latent grid.
**Conditioning**: `MotionConditioning` (MLP mapping 11D motion vector to FiLM parameters for affine modulation of identity features).
**Decoder**: `Decoder` (ConvTranspose2D -> Tanh) mapping 16x16 latent back to 512x512 RGB.

## Purpose
Phase 8B objective: Prove that the 11D PCA Facial Latent (from V4) contains sufficient structural signal to drive the deformation of a static reference identity frame without leaking the target frame directly.

## GPU / CPU Constraints
Because the current environment remains **CPU-only**, a full multi-epoch, full-dataset video-diffusion training job cannot be practically executed. 
Instead, the architecture was explicitly built to run a `ONE-BATCH OVERFIT` test to validate forward passes, loss propagation, gradient descent, and checkpointing. 

## Inputs
1.  **Identity Representation**: A single extracted 512x512 RGB reference frame (Frame 0).
2.  **Motion Representation**: 11D PCA Dense Latent target (3 Euler pose + 8 PCA latents).

## Target
The corresponding ground-truth 512x512 RGB video frame at time `t`.

## Future Temporal Support (Phase 8C)
Currently, temporal context is omitted to isolate the spatial mapping capacity of the PCA latents. The next iteration will add an autoregressive temporal input `[prev_frame]` to constrain structural jitter over time.
