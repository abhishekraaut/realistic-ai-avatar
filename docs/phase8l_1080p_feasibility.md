# Phase 8L: 1080p Feasibility Update

## 1. Measured 1280x720 Baseline
- Inference Latency: 17.50ms
- Train VRAM Peak: 5.1GB
- Inference VRAM Peak: 510MB

## 2. Estimated 1920x1080 (1080p) Requirements
- **Pixel Ratio:** 1080p contains ~2.25x the pixels of 720p.
- **Estimated Inference Latency:** ~39.4ms (dangerously close to the 40ms 25FPS budget).
- **Estimated Train VRAM:** ~11.5GB.

## 3. Conclusion
Training 1080p on the RTX 3050 6GB is unequivocally impossible (hard OOM). Inference at 1080p will saturate the entire frame budget, leading to inevitable queue bloat in production. 1280x720 represents the absolute maximum viable resolution for this specific GPU hardware.
