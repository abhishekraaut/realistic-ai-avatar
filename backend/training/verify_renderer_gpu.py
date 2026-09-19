import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1

def verify_gpu():
    print("Verifying GPU environment for Phase 8B...")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! Must run on a high-config GPU.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\nAllocating NeuralRendererV1 to CUDA...")
    model = NeuralRendererV1(motion_dim=11).to(device)
    print("Model successfully moved to CUDA.")

    identity = torch.randn(1, 3, 512, 512, device=device)
    motion = torch.randn(1, 11, device=device)

    print("\nExecuting forward pass...")
    try:
        output = model(identity, motion)
        assert output.shape == (1, 3, 512, 512), f"Unexpected output shape: {output.shape}"
        assert output.is_cuda, "Output tensor is not on CUDA."
        print("Forward pass successful. Output shape is correct.")
        print("\nGPU Readiness: VALIDATED")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_gpu()
