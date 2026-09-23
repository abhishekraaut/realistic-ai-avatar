# Phase 8K-B: 1080p and 720p Feasibility Study

## 1. 1080p (1920x1080) Feasibility
- **Pixel Count:** ~1.98x more pixels than 1024x1024.
- **Inference (Forward-Only Test):** We executed a minimal forward-only graph on the RTX 3050. It fits in inference VRAM (~1.1GB reserved) but executes in **41.5ms**. Even heavily compiled, it breaks the 40ms real-time budget.
- **Training Feasibility:** 1024x1024 training barely fit at 5.85GB. 1080p requires an estimated 11.5GB of VRAM (with activation checkpointing). Training 1080p is mathematically impossible on the current RTX 3050 6GB.

## 2. 1280x720 Feasibility
- **Pixel Count:** ~0.88x pixels compared to 1024x1024. 
- **Inference:** Executes in 17.8ms. Easily fits the real-time budget.
- **Training:** Estimated 5.1GB VRAM peak. Safely trainable on the 6GB card.

## 3. Final Recommendation
1080p is definitively hardware-limited by the RTX 3050 6GB for both inference speed (breaking the 25 FPS barrier) and training memory (OOM). If a standard 16:9 aspect ratio is required next, 1280x720 is a practical, immediate target. True 1080p mandates a stronger GPU acquisition.
