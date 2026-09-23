# Phase 8H-B: 50-ID Storage Projection & Measurement

## 1. Actual Storage Consumed (50 Identities)
- **Total Duration:** 158.0 Minutes (237,000 Frames @ 25 FPS)
- **Source Video (Raw):** ~20.5 GB
- **Canonical Frames (512x512 PNG):** ~178.5 GB (~750 KB/frame)
- **Audio (16kHz WAV):** ~1.5 GB
- **Landmarks/Metadata (.npy, .json):** ~3.2 GB
- **Derived Tensors (15D targets):** ~150 MB
- **Total Disk Usage:** **~204 GB**

## 2. Updated Projection Scaling
| Component | 10 IDs | 25 IDs | 50 IDs (Actual) | 100 IDs (Projected) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Frames** | 22,500 | 56,250 | 237,000 | ~475,000 |
| **Frame Storage** | 17.5 GB | 43.5 GB | 178.5 GB | ~360.0 GB |
| **Total Storage** | 22.1 GB | 53.0 GB | 204.0 GB | ~415.0 GB |

*Note: The 50-ID actuals grew non-linearly against the 10-ID projection because average sequence yield increased from 1.5 mins/ID to 3.1 mins/ID due to the inclusion of long-form emotional monologues.*

## 3. Storage Architecture Recommendation
- **Git Repo:** Must exclusively track code, preprocessing scripts, `dataset_manifest_*.json`, and documentation.
- **Git LFS:** Checkpoints only.
- **Cloud Object Storage (GCS/S3):** All frames, videos, and `.npy` landmarks must reside in a cloud bucket. Local SSD (1 TB limit on training rig) will reach saturation if scaling past 100 IDs without a streaming dataloader.
