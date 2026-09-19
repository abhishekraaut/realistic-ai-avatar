# GPU Environment Requirements (Phase 8)

To reproduce and train the Neural Renderer V1 in Phase 8B, the following environment specifications are required.

## Hardware Requirements
*   **GPU**: NVIDIA GPU (e.g., RTX 3090, RTX 4090, A100, or equivalent).
*   **VRAM**: Minimum 16GB required for 512x512 training at `batch_size=4` with mixed precision. 24GB+ recommended.
*   **Disk Storage**: Minimum 50GB free space (Dataset V4 requires extraction of raw video frames + checkpoint caching).
*   **System RAM**: 32GB+ recommended (to hold decoded frames and audio features).

## Software Requirements
*   **OS**: Ubuntu 20.04/22.04 LTS (recommended) or Windows 11 with WSL2.
*   **CUDA Driver**: Compatible with CUDA 11.8 or 12.1.
*   **Python**: Version 3.10.x
*   **Environment**: Virtual environment (`venv` or `conda`).

## Required Python Packages
A locked `requirements.txt` is provided in `backend/requirements.txt`:
*   `torch==2.1.0` (Must be compiled with CUDA support corresponding to system driver)
*   `torchvision==0.16.0`
*   `opencv-python==4.8.1.78`
*   `numpy==1.26.0`
*   `mediapipe==0.10.8`

## Verifying CUDA Availability
Before launching the training loop, always run the validation script:
```bash
python backend/training/verify_renderer_gpu.py
```
This ensures the model can allocate VRAM and perform the correct forward/backward passes.
