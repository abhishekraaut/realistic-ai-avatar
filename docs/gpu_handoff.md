# Phase 7 to Phase 8 GPU Handoff Document

## Repository Status
*   **Commit**: `910fe34` (`feat(avatar): implement Phase 7 learned audio-to-motion pipeline`)
*   **Status**: Verified Working, Clean Tree.

## Current Verified Pipeline
The architectural orchestration logic is successfully locked:
REAL AUDIO → AUDIO FEATURES → LEARNED MOTION → MOTION WINDOWS → FRAME SCHEDULER

## Relevant Files for Handoff
*   `backend/models/motion_model.py` (The Audio-to-Motion Model architecture)
*   `backend/training/train_motion.py` (Training script with losses and optimizer)
*   `backend/preprocessing/data_pipeline.py` (MediaPipe FaceLandmarker target extraction)
*   `backend/training/prepare_dataset.py` (Audio stream feature mapping & temporal alignment)
*   `backend/engine/learned_motion_timeline.py` (Inference contract wrapper)

## Dataset Properties
*   **Frames**: 250 (30.0 FPS)
*   **Duration**: ~8.33 seconds
*   **Audio Sample Rate**: 24000Hz (extracted via `moviepy`)
*   **Input Features**: 80-bin Log-Mel Spectrogram
*   **Target Labels**: 4-Dimensional (`jaw_open`, `lip_pucker`, `head_pitch`, `head_yaw`)

## Model Profile
*   **Architecture**: `LearnedTemporalMotionModel` (Conv1D + LSTM)
*   **Total Parameters**: 61,316
*   **Checkpoint**: `backend/training/checkpoints/learned_motion_v1.pt` (731 KB)

## Execution Environment
*   **Python**: 3.12
*   **Frameworks**: PyTorch, NumPy, MediaPipe (`0.10.x`), OpenCV, Librosa
*   **Training Command**: `python backend/training/train_motion.py`
*   **Inference Verification**: `python verify_model.py`
*   **Runtime Simulator**: `python backend/engine/compare_motion.py`

## Heavy GPU Next Step

**DO NOT immediately assume this model is final or production-ready.** The current checkpoint is explicitly an overfitted proof-of-concept proving tensor shapes, timestamps, orchestration, and gradient flow on CPU.

The Heavy GPU Machine should strictly follow this sequence:
1. **Load Checkpoint & Verify Inference**: Run `verify_model.py` to assert determinism and that tensor shapes match expectations.
2. **GPU Benchmark**: Profile the LSTM + Conv1D throughput on CUDA to ensure RTF remains safely < 1.0.
3. **Scale the Dataset**: Ingest the full Synthesia-like datasets using `data_pipeline.py` to extract tens of thousands of frames, keeping a rigid 80/10/10 Split.
4. **Generalization Training**: Retrain `train_motion.py` tracking validation loss on genuinely held-out, non-overlapping sequences.
5. **Architectural Scaling**: Expand the hidden dimensions or switch to a Temporal Transformer/DiT only if the larger dataset necessitates it. 
6. **Phase 8 Transition**: Begin Neural Video Renderer architecture using the continuously generated `MotionWindow` representations.
