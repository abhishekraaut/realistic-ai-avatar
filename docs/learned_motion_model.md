# Learned Facial Motion Model

`STATUS: EXPERIMENTAL / INITIAL TRAINED MODEL`

## Purpose
This model maps raw continuous audio features (temporally anchored to the media clock) directly to facial motion parameters. It replaces the previous heuristic RMS-energy predictor with a scientifically valid learned sequence model.

## Architecture
**Model**: `LearnedTemporalMotionModel`
**Audio Encoder**: 1D Convolutional Network (`kernel_size=3`, `padding=1`) for local phonemic context.
**Temporal Decoder**: 1-Layer LSTM mapping features to continuous facial motion over time.
**Parameters**: `61,316` trainable parameters.
**Device Handling**: Inherently supports CPU and CUDA. Fully isolated from LiveKit HTTP/WebRTC requirements.

## Inputs and Outputs
**Input Representation**: Averaged Log-Mel Spectrogram (`AudioFeatureWindow`)
**Input Shape**: `(Batch, Sequence Length, 80)`

**Target Representation**: 
1. `jaw_open`: 2D MediaPipe Facial Blendshape score (`jawOpen`).
2. `lip_pucker`: 2D MediaPipe Facial Blendshape score (`mouthPucker`).
3. `head_pitch`: Euler angle derived from MediaPipe facial transformation matrix (normalized).
4. `head_yaw`: Euler angle derived from MediaPipe facial transformation matrix (normalized).

**Output Shape**: `(Batch, Sequence Length, 4)`

## Dataset
**Source Data**: `how-i-ai.mp4`
**Size**: 250 frames (~8.33 seconds).
**Train/Val Split**: The entire dataset is used for 1-batch overfit testing to validate gradients, model routing, and checkpoint generation.
**Out-of-Sample Test**: `OUT-OF-SAMPLE VALIDATION NOT AVAILABLE`.

## Training Configuration
**Loss**: Reconstruction MSE + 0.5 * Temporal Smoothness Loss (velocity consistency penalty).
**Optimizer**: Adam (`lr=1e-3`).
**Epochs**: 200 (converged successfully).

## Metrics (1-Batch Overfit)
*   **Initial Loss**: `0.1259`
*   **Final Loss**: `0.0126` (Significant convergence achieved).
*   **Inference Latency (CPU)**: `8.69 ms` for 250 frames.
*   **CPU Real-Time Factor (RTF)**: `0.00104` (Extremely fast, easily clears `RTF < 1.0` constraint).
*   **Determinism**: `Max Absolute Difference = 0.0` (100% Deterministic).

## Artifacts
*   **Checkpoint**: `backend/training/checkpoints/learned_motion_v1.pt` (731.05 KB)
*   **Dataset Tensor**: `synthesia_training_data/dataset.pt`

## Limitations & Known Risks
*   **Tiny Dataset Constraint**: This dataset is fundamentally a proof-of-concept pipeline scale (8.3 seconds). It will **not** generalize well to out-of-sample speakers, phonemes, or audio profiles.
*   **Overfitting**: The model is heavily overfitted to a single continuous sequence.
*   **No Validation Split**: This model was strictly evaluated on its training sequence to prove gradient flow.
*   This is an **architectural/training proof-of-concept**, not sufficient evidence of production-quality generalized facial motion.

## Reproduction Instructions
1. Run dataset preprocessing: `python backend/training/prepare_dataset.py`
2. Run training: `python backend/training/train_motion.py`
3. Verify metrics: `python verify_model.py`
