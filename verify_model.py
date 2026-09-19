import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.motion_model import LearnedTemporalMotionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_dataset(dataset_path):
    print(f"\n--- VERIFYING DATASET ---")
    data = torch.load(dataset_path)
    X = data["X"]
    Y = data["Y"]
    print(f"Dataset Path: {dataset_path}")
    print(f"X shape (Audio Features): {X.shape} - (Frames, Log-Mel Bins)")
    print(f"Y shape (Motion Targets): {Y.shape} - (Frames, Latent Dim)")
    print(f"Sample Count: {X.shape[0]} frames")
    print(f"Train/Val Split: NONE (1-batch overfit test)")
    print(f"Out-of-sample data: OUT-OF-SAMPLE VALIDATION NOT AVAILABLE")
    return X, Y

def verify_checkpoint(ckpt_path):
    print(f"\n--- VERIFYING CHECKPOINT ---")
    ckpt_size = os.path.getsize(ckpt_path)
    print(f"Checkpoint size: {ckpt_size / 1024:.2f} KB")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint Keys:", list(ckpt.keys()))
    print("Config:", ckpt.get("config", "Missing"))
    print("Dataset Info:", ckpt.get("dataset_info", "Missing"))
    return ckpt

def verify_inference_and_determinism(model, X):
    print(f"\n--- VERIFYING INFERENCE & DETERMINISM ---")
    model.eval()
    
    # Run 1
    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model(X.unsqueeze(0))
    t1 = time.perf_counter()
    
    # Run 2
    with torch.no_grad():
        out2 = model(X.unsqueeze(0))
        
    diff = torch.abs(out1 - out2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Inference Time for {X.shape[0]} frames: {(t1 - t0)*1000:.2f} ms")
    rtf = (t1 - t0) / (X.shape[0] / 30.0)
    print(f"Real-Time Factor (RTF): {rtf:.5f} (lower is better, < 1.0 is real-time)")
    
    print(f"Determinism Check:")
    print(f"  Max Absolute Difference: {max_diff}")
    print(f"  Mean Absolute Difference: {mean_diff}")
    if max_diff == 0.0:
        print("  Status: DETERMINISTIC")
    else:
        print("  Status: NON-DETERMINISTIC")

if __name__ == "__main__":
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset.pt")
    ckpt_path = os.path.join(base_dir, "backend", "training", "checkpoints", "learned_motion_v1.pt")
    
    X, Y = verify_dataset(dataset_path)
    ckpt = verify_checkpoint(ckpt_path)
    
    model = LearnedTemporalMotionModel(
        input_dim=ckpt["config"]["input_dim"],
        hidden_dim=ckpt["config"]["hidden_dim"],
        output_dim=ckpt["config"]["output_dim"]
    )
    
    params = count_parameters(model)
    print(f"\n--- VERIFYING MODEL ---")
    print(f"Architecture: LearnedTemporalMotionModel (Conv1D + LSTM)")
    print(f"Total Trainable Parameters: {params:,}")
    
    model.load_state_dict(ckpt["model_state"])
    print("Model weights loaded successfully.")
    
    verify_inference_and_determinism(model, X)
