# Setup Guide
1. Ensure NVIDIA GPU (>= 6GB VRAM), CUDA 12.1.
2. `git checkout v0.1.0-avatar-rc`
3. `pip install -r artifacts/release/python_dependencies.txt`
4. `npm ci`
5. Place checkpoints in `backend/training/checkpoints/`
6. Copy `.env.example` to `.env` and fill secrets.
