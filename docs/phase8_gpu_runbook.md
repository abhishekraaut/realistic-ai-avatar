# Phase 8B GPU Runbook

Execute this exact sequence when transitioning the repository to the high-configuration GPU system.

## 1. Checkout and Verification
```bash
git clone <repository_url>
cd Avtar-Project
git status
git log --oneline -5
```
Verify the latest commit includes the Phase 8 GPU handoff preparation.

## 2. Environment Setup
```bash
python3.10 -m venv engine/venv
source engine/venv/bin/activate
pip install -r backend/requirements.txt
```
*(Ensure the PyTorch installation includes CUDA support for your specific driver via standard `index-url` overrides if needed).*

## 3. Hardware Verification
```bash
nvidia-smi
```
Verify the driver version and available VRAM.

## 4. CUDA Smoke Test
```bash
python backend/training/verify_renderer_gpu.py
```
This tests PyTorch CUDA tensor allocation and basic UNet forward propagation.

## 5. Dataset Verification
```bash
python backend/training/validate_renderer_dataset.py
```
Ensures `dataset_v4.pt` and all source MP4s are present and intact.

## 6. Forward & Overfit Test
```bash
# Edit train_renderer_v1.py to set epochs=20, then run:
python backend/training/train_renderer_v1.py
```
Verify that the L1 loss drops on a single batch.

## 7. Full Training Loop (Experiment A: Ground-Truth Motion)
```bash
# Ensure renderer_config.json is set correctly for 100+ epochs
python backend/training/train_renderer_v1.py
```
Select the best validation checkpoint based on the held-out validation sequence.

## 8. Validation Rendering
Using the best checkpoint, render the complete validation sequence:
1. Extract ground-truth frames.
2. Render frames using `identity_frame[0] + motion[t]`.
3. Save as an MP4 side-by-side with ground-truth.

## 9. Metrics Calculation
Calculate exact validation metrics across the sequence:
*   Image Quality: L1, MSE, PSNR, SSIM.
*   Landmark Consistency: Run MediaPipe on generated frames, compare to ground-truth coordinates.

## 10. Proceed to Experiment B
Only after Experiment A succeeds:
1. Run Audio through V4 learned motion model.
2. Run predicted 11D motion through NeuralRendererV1.
3. Compare structural degradation.

*Do NOT start Phase 8C (Temporal Video Diffusion) until Experiment A and B static results are fully documented and visually validated.*
