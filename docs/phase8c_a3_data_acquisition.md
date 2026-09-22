# Phase 8C-A3: Data Acquisition and Validation Pipeline

## 1. Current Data Gap

The repository currently contains only 1 unique identity (ID_001). To proceed with Phase 8C identity-generalization training, we require a multi-identity dataset. Relying on a single actor prevents meaningful testing of spatial conditioning and identity feature extraction, completely blocking Phase 8C-B.

## 2. Target Identity Count & Requirements

To begin initial multi-identity experiments, the minimum target is:
- **Total Unique Identities:** >= 5 (ID_001 + 4 new identities).
- **Sequences per identity:** 3-5 sequences.
- **Duration per identity:** 1-2 minutes total.

### Source Video Requirements:
- **Visuals:** Real human subjects, strictly frontal or near-frontal faces, consistently visible. Varied facial expressions and natural head movements (yaw/pitch). Appropriate studio-style or consistent indoor lighting.
- **Resolution:** Minimum 512x512.
- **FPS:** Strictly 25.00 FPS to match the existing 16-frame 16kHz audio context pipeline without artificial desynchronization.
- **Audio:** High-quality, clean spoken dialogue without background music or excessive noise.
- **Licensing:** Fully cleared for AI training / commercial use, or academic/open-source use (depending on final project scope).

## 3. Candidate Datasets / Sources

| Candidate | Identities | Speech | Video | FPS | Resolution | License | Training Usage | Suitable? |
| --------- | ---------: | ------ | ----- | --- | ---------- | ------- | -------------- | --------- |
| **HDTF** | ~300 | Clean | Frontal | 25 | 720p+ | Academic | Non-commercial only | YES (Research only) |
| **MEAD** | 60 | Clean | Multi-view | 30 | 1080p | Academic | Non-commercial only | NO (30 FPS conflicts) |
| **VoxCeleb2** | 6000+ | Mixed | Wild | Mixed | Variable | Academic | Non-commercial only | NO (Low-res/FPS) |
| **CC-0 Stock** | ~10-20 | Varies | Frontal | 25/30 | 4K | CC-0 / Royalty Free | Fully Commercial | YES (Requires manual curation) |
| **Custom Recording** | N/A | Studio | Frontal | 25 | 4K | Owned | Fully Commercial | YES (Best quality) |

**Acquisition Decision:**
*Identities still required:* 4
*Source selected:* None currently downloaded.
*Reason:* Automatic downloading of large 10GB+ external datasets is paused until storage limits and exact licensing preferences are approved by the user.

## 4. Ingestion Structure

To properly structure acquired identities without sequence leakage, the new directory architecture is:

```text
synthesia_training_data/
  raw/
    ID_001/
      seq_01/
      seq_02/
    ID_002/
      seq_01/
  processed/
    ID_002/
      seq_01/
        frames/
        audio.wav
        landmarks.pkl
        motion.pt
  manifests/
    ID_001_seq_01_manifest.json
    ID_002_seq_01_manifest.json
```

## 5. Ingestion Manifest Schema

Every processed sequence will strictly output a JSON manifest to track metadata and prevent processing errors:
- `identity_id` (str)
- `sequence_id` (str)
- `source` (str)
- `source_license` (str)
- `video_path` (str)
- `audio_path` (str)
- `fps` (float)
- `width` (int)
- `height` (int)
- `duration` (float)
- `frame_count` (int)
- `landmark_status` (str)
- `face_detection_status` (str)
- `v4_motion_status` (str)
- `preprocessing_version` (str)

## 6. Data Validation Rules

A deterministic validation script (`backend/data_pipeline/ingestion_validator.py`) has been added. It enforces:
1. File existence and read access.
2. OpenCV readable video stream (no corruption).
3. Minimum dimensions >= 512x512.
4. Exact FPS == 25.00 (within 0.01 tolerance).
5. Non-zero frame counts.

## 7. Identity-Aware Split Design

The split generator (`backend/data_pipeline/identity_split_generator.py`) operates solely at the `identity_id` level. 
- **Train:** Identities [A, B, C]
- **Validation:** Identities [D]
- **Test:** Identities [E]
- **Leakage Protection:** Mathematical set intersection checks ensure `Train ∩ Val = ∅`, `Train ∩ Test = ∅`, and `Val ∩ Test = ∅`.

*(Currently blocked locally as `len(identities) == 1`).*

## 8. V4 Compatibility Constraints

- **11D:** New video sequences will be passed through the existing Landmark -> PCA -> 11D extraction logic.
- **PCA:** The existing V4 PCA basis will NOT be recomputed.
- **Normalization:** Existing min/max bounding is frozen.
- **OOD Warnings:** Since PCA is overfitted to ID_001, novel facial bone structures will inevitably produce out-of-distribution mappings. The V4 motion status in the manifest will track these warnings, but the PCA logic remains frozen.

## 9. Storage Handling

- Current large datasets (`dataset_v4_images.pt` ~3.5GB) are already `.gitignore`'d.
- Future raw videos, extracted frames, and `dataset_v5_*.pt` multi-identity tensors will also be ignored to prevent accidental Git staging.
- Git LFS / External Object Storage (S3/GCS) configuration will be executed later depending on the selected data source (e.g., HDTF vs Custom).
