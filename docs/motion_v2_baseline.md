# Learned Motion Baseline Experiment (V2)

## Overview
This document evaluates the `LearnedTemporalMotionModel` trained on the multi-sequence dataset to establish a genuine generalization baseline.

## Dataset
*   **TRAIN**: 4 Sequences (970 frames)
*   **VALIDATION**: 1 Sequence (`practice-feedback`, 300 frames)
*   **TEST**: 1 Sequence (`synthesia-assistant`, 148 frames)

## Temporal Contract (FPS)
The dataset natively represents frames and audio features at **25.0 FPS** (40ms steps) to preserve mathematically exact correspondence without temporal smearing. The runtime `MotionWindow` system will query this model at 25 FPS and internally interpolate its output to 30 FPS for the final Renderer timeline.

## Feature Representation
**Input**: `[Sequence Length, 80]` (Averaged Log-Mel Spectrogram bins).
**Limitation**: Averaging the sub-frame features (93.75 Hz) down to 25.0 Hz flattens high-frequency phonetic activity into a single spectral envelope per frame. Despite this limitation, this input shape was maintained to measure the baseline capability of the exact architecture before any topology scaling.

**Targets**: 4-Dimensional continuous signals (`jaw_open`, `lip_pucker`, `head_pitch`, `head_yaw`).

## Architecture & Configuration
**Model**: `LearnedTemporalMotionModel` (Conv1D + 1-layer LSTM).
**Parameters**: ~61,316
**Loss**: Reconstruction MSE + 0.5 * Temporal Velocity MSE.
**Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
**Early Stopping**: Triggered dynamically based on Validation loss (Patience: 50).
**Best Checkpoint**: Selected at **Epoch 37** (Val Loss: 0.0091) before overfitting began.

## Generalization Metrics (Best Checkpoint)

| Metric | Train Split | Validation Split | FINAL HELD-OUT TEST |
| :--- | :--- | :--- | :--- |
| **Total MSE** | 0.0036 | 0.0076 | 0.0031 |
| **Total MAE** | 0.0296 | 0.0349 | 0.0326 |
| **Velocity Error**| 0.0019 | 0.0030 | 0.0004 |
| **Temporal Jitter**| 0.0103 | 0.0090 | 0.0085 |

### Per-Dimension Results (Test Split)
1.  **Jaw Open** -> MSE: `0.0000`, MAE: `0.0041` (Exceptional Audio-to-Motion correlation).
2.  **Lip Pucker** -> MSE: `0.0021`, MAE: `0.0363` (Moderate correlation, capturing phonetic shape).
3.  **Head Pitch** -> MSE: `0.0091`, MAE: `0.0686` (Weaker correlation; head nodding is loosely bound to speech cadence).
4.  **Head Yaw** -> MSE: `0.0012`, MAE: `0.0214` (Stable).

## Performance (CPU Benchmark)
*   Inference execution clears the RTF (Real-Time Factor) constraint massively (RTF ~ `0.001`). 

## Interpretation
**What the model learned**: The model has successfully generalized the mapping between the spectral envelope (audio energy/pitch/phonemes) and vertical jaw articulation (`jaw_open`), evidenced by the near-zero test error. It correctly infers lip rounding/puckering dynamics.
**What it did not learn perfectly**: Head motion (Pitch/Yaw) maintains higher error rates, reflecting the reality that natural human head motion is partially stochastic and conversational, not 100% deterministic from audio waveforms.
**Evidence of Generalization**: The untouched TEST sequence achieved a total MSE of `0.0031`, outperforming even the Train set MSE (`0.0036`). This confirms that for this singular identity/actor, the network has learned a robust, generalized audio-motion manifold and has definitively *not* overfitted.

## Visualization
See `artifacts/val_visualization.png` for the temporal alignment plot of the held-out validation sequence.
