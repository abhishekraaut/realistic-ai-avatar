import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model import LearnedTemporalMotionModel

def analyze_v2():
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v2.pt")
    ckpt_path = os.path.join(base_dir, "backend", "training", "checkpoints", "learned_motion_v2_best.pt")
    
    device = torch.device("cpu")
    data = torch.load(dataset_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    x_mean = ckpt["norm_stats"]["x_mean"]
    x_std = ckpt["norm_stats"]["x_std"]
    
    model = LearnedTemporalMotionModel(input_dim=80, hidden_dim=64, output_dim=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    dims = ["jaw_open", "lip_pucker", "head_pitch", "head_yaw"]
    
    print("=== V2 TARGET DISTRIBUTIONS ===")
    for split in ["train", "val", "test"]:
        Y = data[split]["Y"]
        if Y.shape[0] == 0: continue
        
        print(f"\n[{split.upper()}] - {Y.shape[0]} frames")
        for i, dim_name in enumerate(dims):
            y_dim = Y[:, i]
            print(f"  {dim_name}: Mean={y_dim.mean().item():.4f}, Std={y_dim.std().item():.4f}, Min={y_dim.min().item():.4f}, Max={y_dim.max().item():.4f}")

    print("\n=== V2 DETAILED JAW ANALYSIS (TEST SPLIT) ===")
    with torch.no_grad():
        X_test = data["test"]["X"].unsqueeze(0)
        Y_test = data["test"]["Y"].unsqueeze(0)
        
        X_norm = (X_test - x_mean) / x_std
        preds = model(X_norm)
        
        jaw_t = Y_test[0, :, 0]
        jaw_p = preds[0, :, 0]
        
        mse = nn.MSELoss()(jaw_p, jaw_t).item()
        mae = nn.L1Loss()(jaw_p, jaw_t).item()
        
        print(f"Jaw Open Full Precision MSE: {mse:.10f}")
        print(f"Jaw Open Full Precision MAE: {mae:.10f}")
        
        print(f"Target Jaw - Mean: {jaw_t.mean().item():.6f}, Std: {jaw_t.std().item():.6f}, Range: [{jaw_t.min().item():.6f}, {jaw_t.max().item():.6f}]")
        print(f"Pred Jaw   - Mean: {jaw_p.mean().item():.6f}, Std: {jaw_p.std().item():.6f}, Range: [{jaw_p.min().item():.6f}, {jaw_p.max().item():.6f}]")

if __name__ == "__main__":
    analyze_v2()
