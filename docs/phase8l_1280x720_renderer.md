# Phase 8L: 1280x720 Renderer Evaluation

## 1. Objective and Training
Trained a 1280x720 model using the verified 50-ID dataset. Preprocessing correctly mapped native 16:9 widescreen imagery without stretching. Training consumed ~5.1GB VRAM, comfortably fitting the 6GB RTX 3050 limit (unlike 1024x1024, which peaked at 5.85GB).

## 2. Quality and Aspect Ratio
Visual fidelity (teeth, hair, eyelashes) matches the high-frequency sharpness of the 1024x1024 model. The key advantage is the native 16:9 aspect ratio, preventing unnatural crops of shoulders and background.

## 3. Real-Time Performance
With 921,600 pixels, 1280x720 is mathematically ~12% lighter than 1024x1024 (1,048,576 pixels). Using the optimized inference stack (FP16, channels_last, torch.compile), generation latency dropped to **17.50ms**, easily preserving the 40ms real-time budget for 25 FPS.
