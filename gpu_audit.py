import os
import sys
import time
import torch
import platform
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.models.motion_model import LearnedTemporalMotionModel

def audit_environment():
    print("\n--- GPU ENVIRONMENT AUDIT ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f" GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f" Compute Capability: {torch.cuda.get_device_capability(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f" Total VRAM: {props.total_memory / 1024**2:.2f} MB")
            
        # Run a tiny tensor op to prove it works
        try:
            x = torch.ones((3, 3), device='cuda')
            y = x * 2
            print(f" CUDA Tensor Op Result: {y[0][0].item()} (Success)")
        except Exception as e:
            print(f" CUDA Tensor Op Failed: {e}")
    else:
        print("WARNING: CUDA is not available. Falling back to CPU.")
        
    return torch.device("cuda" if cuda_available else "cpu")

def audit_dataset():
    print("\n--- DATASET AUDIT ---")
    dataset_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\dataset.pt"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return None, None
        
    data = torch.load(dataset_path, map_location='cpu')
    X = data["X"]
    Y = data["Y"]
    
    print(f"Dataset Path: {dataset_path}")
    print(f"Audio Features Shape (X): {X.shape} - (Frames, Log-Mel Bins)")
    print(f"Facial Targets Shape (Y): {Y.shape} - (Frames, Latent Dimensions)")
    print(f"Total Frames: {X.shape[0]}")
    print(f"Target Dimensions: jaw_open, lip_pucker, head_pitch, head_yaw")
    print(f"Unique Source Sequences: 1 (how-i-ai.mp4)")
    print(f"Speakers/Actors: 1")
    print(f"Multiple Takes/Conditions: False")
    print(f"Genuine Validation Data: False (No held-out split available in this small proof-of-concept)")
    
    return X, Y

def audit_checkpoint_and_benchmark(device, X):
    print("\n--- CHECKPOINT & BENCHMARK AUDIT ---")
    ckpt_path = r"C:\Users\iabhi\Downloads\Avtar-Project\backend\training\checkpoints\learned_motion_v1.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    print("Checkpoint Config:", config)
    
    model = LearnedTemporalMotionModel(
        input_dim=config.get("input_dim", 80),
        hidden_dim=config.get("hidden_dim", 64),
        output_dim=config.get("output_dim", 4)
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: LearnedTemporalMotionModel")
    print(f"Trainable Parameters: {params:,}")
    
    model.load_state_dict(ckpt["model_state"])
    print("Checkpoint weights loaded successfully.")
    
    model.eval()
    
    # 1. Determinism Test
    X_dev = X.unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = model(X_dev)
        out2 = model(X_dev)
        
    diff = torch.abs(out1 - out2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"\nDeterminism Check:")
    print(f"  Max Absolute Difference: {max_diff}")
    print(f"  Mean Absolute Difference: {mean_diff}")
    if max_diff == 0.0:
        print("  Status: DETERMINISTIC")
    else:
        print("  Status: NON-DETERMINISTIC")
        
    # 2. Benchmark
    print(f"\nInference Benchmark ({device.type.upper()}):")
    seq_len = X_dev.shape[1]
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(X_dev)
            
    # Measure
    runs = 50
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(X_dev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    avg_latency_ms = ((t1 - t0) / runs) * 1000
    fps = seq_len / (avg_latency_ms / 1000.0)
    rtf = (avg_latency_ms / 1000.0) / (seq_len / 30.0)
    
    print(f"  Sequence Length: {seq_len} frames")
    print(f"  Batch Size: {X_dev.shape[0]}")
    print(f"  Average Inference Latency: {avg_latency_ms:.2f} ms")
    print(f"  Throughput: {fps:.2f} frames/sec")
    print(f"  Real-Time Factor (RTF): {rtf:.5f} (< 1.0 is real-time)")
    
    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  VRAM Allocated: {mem_alloc:.2f} MB")

if __name__ == "__main__":
    device = audit_environment()
    X, Y = audit_dataset()
    if X is not None:
        audit_checkpoint_and_benchmark(device, X)
