import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model import LearnedTemporalMotionModel

def compute_metrics(preds, targets):
    # preds, targets: (1, T, 4)
    mse = nn.MSELoss()(preds, targets).item()
    mae = nn.L1Loss()(preds, targets).item()
    
    # Velocity Error
    p_vel = preds[:, 1:, :] - preds[:, :-1, :]
    t_vel = targets[:, 1:, :] - targets[:, :-1, :]
    vel_error = nn.MSELoss()(p_vel, t_vel).item()
    
    # Temporal Jitter (Mean Absolute First Difference)
    jitter = torch.mean(torch.abs(p_vel)).item()
    
    # Per-dimension MSE & MAE
    dims = ["jaw_open", "lip_pucker", "head_pitch", "head_yaw"]
    dim_metrics = {}
    for i, dim_name in enumerate(dims):
        dim_metrics[dim_name] = {
            "mse": nn.MSELoss()(preds[..., i], targets[..., i]).item(),
            "mae": nn.L1Loss()(preds[..., i], targets[..., i]).item()
        }
        
    return {
        "mse": mse,
        "mae": mae,
        "vel_error": vel_error,
        "jitter": jitter,
        "per_dimension": dim_metrics
    }

def evaluate():
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v2.pt")
    ckpt_path = os.path.join(base_dir, "backend", "training", "checkpoints", "learned_motion_v2_best.pt")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    device = torch.device("cpu")
    data = torch.load(dataset_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    x_mean = ckpt["norm_stats"]["x_mean"]
    x_std = ckpt["norm_stats"]["x_std"]
    
    model = LearnedTemporalMotionModel(input_dim=80, hidden_dim=64, output_dim=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"Loaded BEST Checkpoint from Epoch {ckpt['epoch']}")
    
    results = {}
    
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            if data[split]["X"].shape[0] == 0:
                continue
                
            X = data[split]["X"].unsqueeze(0)
            Y = data[split]["Y"].unsqueeze(0)
            
            X_norm = (X - x_mean) / x_std
            preds = model(X_norm)
            
            results[split] = compute_metrics(preds, Y)
            
            # Plotting for Val
            if split == "val":
                plt.figure(figsize=(15, 10))
                dims = ["Jaw Open", "Lip Pucker", "Head Pitch", "Head Yaw"]
                
                # Numpy conversion
                p_np = preds[0].numpy()
                t_np = Y[0].numpy()
                time_axis = np.arange(p_np.shape[0]) * (1.0 / 25.0) # 25 FPS
                
                for i in range(4):
                    plt.subplot(4, 1, i+1)
                    plt.plot(time_axis, t_np[:, i], label="Ground Truth", color="black", alpha=0.7)
                    plt.plot(time_axis, p_np[:, i], label="Predicted", color="blue", alpha=0.8)
                    plt.title(dims[i])
                    plt.ylabel("Value")
                    plt.grid(True, alpha=0.3)
                    if i == 0: plt.legend()
                
                plt.xlabel("Time (s)")
                plt.tight_layout()
                plot_path = os.path.join(artifacts_dir, "val_visualization.png")
                plt.savefig(plot_path)
                print(f"Saved Visualization to {plot_path}")
                plt.close()
                
    # Print report
    print("\n--- GENERALIZATION REPORT ---")
    for split, mets in results.items():
        print(f"\n[{split.upper()} SPLIT]")
        print(f"Total MSE: {mets['mse']:.4f}")
        print(f"Total MAE: {mets['mae']:.4f}")
        print(f"Velocity Error: {mets['vel_error']:.4f}")
        print(f"Temporal Jitter: {mets['jitter']:.4f}")
        for dim, dim_mets in mets['per_dimension'].items():
            print(f"  {dim} -> MSE: {dim_mets['mse']:.4f}, MAE: {dim_mets['mae']:.4f}")

if __name__ == "__main__":
    evaluate()
