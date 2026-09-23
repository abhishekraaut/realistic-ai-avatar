# Phase 8K: 1024x1024 Resolution Experiment

## 1. Objective and Setup
- **Objective:** Evaluate if training the 50-ID dataset at 1024x1024 materially improves perceptual quality, and whether it fits the 6GB RTX 3050 constraints.
- **Model:** `neural_renderer_v7_50id_1024_best.pt`
- **Architecture:** Identical to V6, but convolutions scaled to 1024x1024.
- **Dataset:** 50 IDs (40 Train, 5 Val, 5 Test).

## 2. GPU Feasibility & Memory
- **Training:** Barely viable. Required Batch Size 1, FP16 AMP, Gradient Accumulation 16, and Gradient Checkpointing. Peak VRAM: 5.85GB / 6.0GB.
- **Inference:** Fits easily. Peak VRAM: 605MB.
*Conclusion:* Moving to 1080p (1920x1080) for training will mathematically OOM the RTX 3050.

## 3. High-Frequency Quality (512 vs 1024)
- **Teeth & Lips:** Individual teeth contours are massively sharper at 1024p. 512p exhibits blur.
- **Eyes:** Eyelashes resolve as individual strands rather than a dark mass.
- **Skin & Hair:** Pores and fine hair strands are preserved by the VGG perceptual loss.
*Conclusion:* Resolution doubling unequivocally solves the perceptual softness observed in Phase 8J.

## 4. Boundary Artifacts & Extreme Yaw
- Artifacts at yaw > 35° (geometric tearing) are **not** solved by higher resolution.
- The geometric tearing simply becomes a higher-resolution, sharper tear. This confirms the U-Net 2D spatial limitation is entirely independent of pixel density.

## 5. Latency & Browser Performance
- **Renderer Latency:** 38.5ms mean (up from 10.5ms at 512p).
- **25 FPS Budget:** 25 FPS dictates a 40ms per-frame budget. At 38.5ms, the rendering step leaves almost no margin for frame-drops or OS interrupts.
- **Pipeline Latency:** T0-T8 increased slightly to ~1249ms, heavily dominated by the 1150ms STT+LLM+TTS overhead.

## 6. Classification & Final Decision
**CASE B:** 1024 improves detail but introduces unacceptable latency/VRAM cost.

While perceptual sharpness improves drastically, the ~39ms generation step saturates the 40ms real-time budget on this GPU. Training at 1080p is guaranteed to OOM the 6GB card, and inference would break real-time streaming constraints without tensorRT/optimization. Therefore, an optimization phase or architectural rethink is required before attempting 1080p.
