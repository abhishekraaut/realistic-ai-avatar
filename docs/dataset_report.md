# Avatar Motion Dataset Report (v2)

## Source Data
Real recordings discovered in `public/assets`:
1. `emotional-rollercoaster.mp4`
2. `guess-the-animal.mp4`
3. `guess-where.mp4`
4. `how-i-ai.mp4`
5. `practice-feedback.mp4`
6. `synthesia-assistant.mp4`

## Dataset Size
**Sequences**: 6
**Actors**: 1 (all videos appear to feature the same or similar Synthesia-style avatar format)
**Total Video FPS**: 25.0
**Total Duration**: ~56.7 seconds
**Total Frames**: ~1,418

## Splits
Strict sequence-level isolation applied:
*   **TRAIN**: 4 sequences (`emotional-rollercoaster`, `guess-the-animal`, `guess-where`, `how-i-ai`).
*   **VALIDATION**: 1 sequence (`practice-feedback`).
*   **TEST**: 1 sequence (`synthesia-assistant`).
No temporal windows cross split boundaries.

## Alignment
Audio (24,000 Hz) and Video (25.0 FPS) are identically aligned. 
Each video frame timestamp `t` directly fetches the Audio Feature Windows that overlap `[t, t + (1/FPS))`. 
Audio extraction is perfectly synchronized with the exact source video length using `moviepy`.

## Features
**Representation**: Averaged Log-Mel Spectrogram.
**Shape**: `[Sequence Length, 80]`
**Limitation Noted**: The current pipeline aggregates sub-frame (93.75 Hz) audio features via `np.mean(mels, axis=0)`. While this preserves sequence-level temporal continuity (one vector per video frame), it intentionally smooths intra-frame high-frequency audio fluctuations. The model correctly receives this temporal stack `(Batch, Sequence Length, 80)`, but a future iteration may need a local context window `[Batch, Sequence Length, Context, 80]` if lip-sync fidelity demands it.

## Targets
**Extraction Tool**: MediaPipe FaceLandmarker Task API.
**Representations**:
1. `jaw_open`: Directly from `jawOpen` face blendshape (0.0 to 1.0).
2. `lip_pucker`: Directly from `mouthPucker` face blendshape (0.0 to 1.0).
3. `head_pitch`: Euler angle (X) derived from facial transformation matrix, normalized via `value / 45.0` and clamped to `[-1, 1]`.
4. `head_yaw`: Euler angle (Y) derived from facial transformation matrix, normalized via `value / 45.0` and clamped to `[-1, 1]`.

## Leakage Check
**Passed**. Genuine sequence-level separation. Train data is structurally forbidden from appearing in Validation/Test.

## Validation
Genuine out-of-sample validation now **EXISTS**. We can measure the model's performance on previously unseen phoneme combinations and sentences within the same identity.

## Quality
See `validation_report.json` for precise invalid/dropped sample statistics. Extreme NaNs, drops, and timestamp mismatches are explicitly flagged.

## Limitations
*   **Single Actor Bias**: All 6 sequences belong to a singular aesthetic/speaker profile. The model will heavily overfit to this specific individual's articulation style.
*   **Size**: ~56 seconds of total data is sufficient to prove out-of-sample inference structure and sequence loss, but vastly insufficient for true generalized production-quality synthesis. 
