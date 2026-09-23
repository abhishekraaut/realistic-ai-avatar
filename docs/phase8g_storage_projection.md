# Phase 8G: Storage Projection Audit

## 1. Measured 10-Identity Pilot Storage
- **Source Video:** 4.2 GB
- **Canonical Frames (512x512 PNG, 25 FPS):** 17.5 GB (22,500 frames)
- **Audio (16kHz WAV):** 150 MB
- **Landmarks/Metadata (.npy, .json):** 300 MB
- **Derived Tensors (15D targets):** 15 MB
- **Total:** ~22.1 GB

## 2. Projections
Assuming an average of 3 minutes of usable footage per identity:

| Component | 10 Identities | 50 Identities | 100 Identities |
| :--- | :--- | :--- | :--- |
| **Total Frames** | 22,500 | 112,500 | 225,000 |
| **Frame Storage** | ~17.5 GB | ~87.5 GB | ~175.0 GB |
| **Total Storage** | ~22.1 GB | ~110.5 GB | ~221.0 GB |

## 3. Storage Policy
- **Git Repository:** Strictly code, manifests, and documentation only.
- **Git LFS:** Weights (`.pt`) and minimal verification assets (< 200MB).
- **External Object Storage (S3/GCS):** All raw videos, canonical frames, audio, and landmarks must be hosted externally. 
- *Conclusion:* Scaling to 50+ identities fundamentally requires cloud object storage integration into the training dataloader to prevent local SSD exhaustion.
