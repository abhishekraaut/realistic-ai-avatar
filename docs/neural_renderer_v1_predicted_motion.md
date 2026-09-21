# Neural Renderer V1 Predicted Motion Results (Phase 8B-B)

## Experiment
Evaluating the degradation caused by the learned motion model vs. ground-truth motion.
- **A**: Ground-Truth 11D Motion -> Renderer -> Image
- **B**: Predicted 11D Motion (from V4 model) -> Renderer -> Image

## Checkpoints
- V4 Motion Checkpoint: `backend/training/checkpoints/learned_motion_v4_best.pt`
- Renderer Checkpoint: `backend/training/checkpoints/neural_renderer_v1_gpu_best.pt`

## Performance (Offline Synchronized Inference)
- **V4 Motion Inference**: 2.00 ms
- **Renderer Inference (p50)**: 5.43 ms
- **Renderer Inference (p95)**: 12.09 ms
- **End-to-End Pipeline (p50)**: 8.56 ms
- **Peak VRAM Pipeline**: 0.06 GB

## Validation Motion Error (V4 Prediction)
- Motion MSE: 0.1315
- Motion MAE: 0.2336

## Validation Image Metrics (A vs B)

| Metric | GT Motion -> Renderer (A) | Predicted Motion -> Renderer (B) |
| --- | --- | --- |
| L1 | 0.1563 | 0.1581 |
| MSE | 0.0475 | 0.0486 |
| PSNR | 19.2601 dB | 19.1567 dB |
| SSIM | 0.6351 | 0.6295 |
| Lmk Error (Full) | 0.0147 | 0.0154 |
| Lmk Error (Mouth)| 0.0196 | 0.0175 |
| Lmk Error (Eyes) | 0.0085 | 0.0127 |
| Pose Error | 0.1891 | 0.1558 |

## Error Decomposition
- **Renderer Degradation**: Significant. The renderer struggles with high-frequency details (eyes, teeth, wrinkles), yielding a PSNR of only ~19.2dB even with perfect motion.
- **Motion Model Degradation**: The motion model introduces only a tiny additional degradation in visual quality (PSNR drops only 0.1dB, L1 increases barely). The predicted motion appears to suffer from "mean-pose collapse", smoothing out extremes (e.g. better numerical pose error, but less expressive/accurate to the original audio).

## Test Set Failure
On the completely untouched `synthesia-assistant` test sequence, the NeuralRenderer failed catastrophically. The output was so blurry/degraded that MediaPipe could not detect a single face across all 148 frames (Landmark error = 0.0000 because 0 faces were found).
This proves the static `NeuralRendererV1` is highly overfit to the single training identity and has **zero generalization capability** to novel identities.

## Limitations
1. **Single-Identity Overfitting**: The renderer cannot generalize to the test identity.
2. **Mean-Pose Collapse**: The motion model smooths out extreme facial motions.
3. **Temporal Flicker**: The static renderer outputs suffer from severe frame-to-frame jitter.

## Next Action
Evaluate Phase 8C temporal consistency (if scheduled) or explore robust conditioning (e.g., identity latent refinement) to prevent complete generalization failure on novel identities.
