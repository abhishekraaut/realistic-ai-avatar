# Phase 8C Dataset Freeze

## 1. Identity Inventory & Specifications

The dataset strictly limits to the 5 designated identities. No external identities will be introduced during Phase 8C-B.

| Identity | Source | Sequences | Usable Frames | Duration | FPS | Resolution |
| -------- | ------ | --------: | ------------: | -------: | --: | ---------- |
| **ID_001** | Original Synthetic Actor | 6 | 1196 | 47.8s | 25.0 | 512×512 |
| **ID_002** | Wikimedia Commons (Obama) | 1 | 750 | 30.0s | 25.0 | 1280×720 |
| **ID_003** | Wikimedia Commons (Trump) | 1 | 700 | 28.0s | 25.0 | 1280×720 |
| **ID_004** | Wikimedia Commons (Biden) | 1 | 800 | 32.0s | 25.0 | 1280×720 |
| **ID_005** | Wikimedia Commons (Bush) | 1 | 650 | 26.0s | 25.0 | 1280×720 |

- **Total Identities:** 5
- **Total Sequences:** 10
- **Total Frames:** 4096
- **Total Duration:** ~163.8 seconds

## 2. Dataset Split

Strict isolation mathematically enforced. Identities do not overlap across splits.

- **TRAIN:** `ID_001`, `ID_003`, `ID_004`
- **VALIDATION:** `ID_005`
- **TEST:** `ID_002`

## 3. Storage Policy

| Asset | Size | Git Strategy | Reason |
| ----- | ---: | ------------ | ------ |
| **Source Code & Manifests** | < 10 MB | Normal Git | Small text files vital to project execution and dataset lineage. |
| **Model Checkpoints** | < 50 MB | Git LFS | `.pt` artifacts (e.g. `neural_renderer_v1_gpu_best.pt` = ~25MB). They change frequently and are binary, making them ideal for Git LFS. |
| **Raw Video (.mp4/.webm)** | ~75 MB | External | Large binary blobs unnecessary for Git history. Can be retrieved from sources or internal buckets. |
| **Extracted Frames (.jpg)** | ~333 MB | External | Massive file counts (4000+ files) would pollute Git index. |
| **Extracted Audio (.wav)** | ~10 MB | External | Derived reproducible binaries. |
| **Landmarks/Motion** | ~16 MB | External | Can be deterministically regenerated using the manifest and ingestion scripts. |
| **Dataset Tensors (.pt)** | ~3.7 GB | External | Monolithic tensors (e.g., `dataset_v4_images.pt`) are far too large for Git/LFS and should reside on S3/GCS or local scratch. |

## 4. Pipeline & Contract

- **Preprocessing Version:** `1.0`
- **Face Detection:** MediaPipe Holistic (Stable across all 4096 frames)
- **V4 Contract:** Frozen. 11D PCA parameters are mapped strictly against `ID_001`'s variance bounds without refitting.

## 5. Known Limitations & OOD

**Quality Limitation:**
This 5-identity dataset is sufficient for an INITIAL identity-generalization architectural experiment, but is NOT sufficient to establish broad/general production identity generalization.
- Only 3 identities exist in the training manifold.
- Only 1 acquired sequence per identity for ID_002–ID_005.

**OOD (Out-of-Distribution) Observations:**
All newly ingested identities trigger significant spatial OOD warnings in their V4 projection due to fundamental facial geometry differences. The spatial extremes frequently violate the `[-3, 3]` threshold established by `ID_001`. Phase 8C-B architecture updates must account for these domain shifts contextually, as the PCA space was not recalibrated.
