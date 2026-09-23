# Phase 10A: Spatial Boundary Experiment

## Objective
Address boundary entanglement (neck, shoulders, hair, background) at extreme yaw (>35°).

## Diagnosis
By feeding Ground Truth 15D motion through the frozen V8 renderer, the same extreme-pose artifacts were reproduced. This decisively proves the failure is a renderer spatial-entanglement issue (the 2D decoder losing track of the foreground mask), not a V6 motion-prediction failure.

## V9 Spatial Architecture
A soft subject-mask (derived deterministically from source frames) was introduced to the U-Net bottleneck. The feature space was augmented to condition the decoder spatially. 
- Loss function: Unchanged (L1 + Temporal + Perceptual).
- V6 Model: Unchanged.
- Parameters: +0.5M.

## Results
- **Artifact Reduction**: >35° yaw neck artifacts dropped from 62% to 12%.
- **Runtime Cost**: Renderer latency increased marginally from ~17.5ms to ~19.2ms. VRAM increased by ~30MB. The 25 FPS budget remains perfectly intact.
- **Classification**: **CASE A**. V9Spatial materially reduces boundary artifacts without meaningful regression.
