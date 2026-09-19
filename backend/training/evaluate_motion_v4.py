import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model_v4 import LearnedTemporalMotionModelV4

def compute_metrics(preds, targets):
    mse = nn.MSELoss()(preds, targets).item()
    mae = nn.L1Loss()(preds, targets).item()
    p_vel = preds[:, 1:, :] - preds[:, :-1, :]
    t_vel = targets[:, 1:, :] - targets[:, :-1, :]
    vel_error = nn.MSELoss()(p_vel, t_vel).item()
    jitter = torch.mean(torch.abs(p_vel)).item()
    return {"mse": mse, "mae": mae, "vel_error": vel_error, "jitter": jitter}

def evaluate_v4():
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    ckpt_path = os.path.join(base_dir, "backend", "training", "checkpoints", "learned_motion_v4_best.pt")
    pca_path = os.path.join(base_dir, "synthesia_training_data", "pca_v4.pkl")
    
    device = torch.device("cpu")
    data = torch.load(dataset_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
        
    x_mean = ckpt["norm_stats"]["x_mean"]
    x_std = ckpt["norm_stats"]["x_std"]
    
    model = LearnedTemporalMotionModelV4(input_dim=80, context_size=16, hidden_dim=64, output_dim=11).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print(f"Loaded BEST Checkpoint V4 from Epoch {ckpt['epoch']}")
    
    MOUTH_INDICES = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
    
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            if data[split]["X"].shape[0] == 0: continue
                
            X = data[split]["X"].unsqueeze(0)
            Y = data[split]["Y"].unsqueeze(0)
            
            X_norm = (X - x_mean) / x_std
            preds = model(X_norm)
            
            lat_metrics = compute_metrics(preds, Y)
            
            # Reconstruction Logic
            # preds: [1, T, 11]. First 3 are pose, last 8 are PCA latents.
            pred_latents = preds[0, :, 3:].numpy()
            targ_latents = Y[0, :, 3:].numpy()
            
            pred_recon = pca.inverse_transform(pred_latents).reshape(-1, 478, 2)
            targ_recon = pca.inverse_transform(targ_latents).reshape(-1, 478, 2)
            
            # Full Face Error
            full_error = np.mean((pred_recon - targ_recon)**2)
            
            # Mouth Error
            pred_mouth = pred_recon[:, MOUTH_INDICES, :]
            targ_mouth = targ_recon[:, MOUTH_INDICES, :]
            mouth_error = np.mean((pred_mouth - targ_mouth)**2)
            
            # Head Pose Error
            pred_pose = preds[0, :, :3].numpy()
            targ_pose = Y[0, :, :3].numpy()
            pose_error = np.mean((pred_pose - targ_pose)**2)
            
            print(f"\n[{split.upper()} SPLIT]")
            print(f"Latent MSE: {lat_metrics['mse']:.6f} | MAE: {lat_metrics['mae']:.6f} | Vel: {lat_metrics['vel_error']:.6f} | Jitter: {lat_metrics['jitter']:.6f}")
            print(f"Landmark Recon MSE (Full Face): {full_error:.6f}")
            print(f"Landmark Recon MSE (Mouth Only): {mouth_error:.6f}")
            print(f"Head Pose MSE: {pose_error:.6f}")

if __name__ == "__main__":
    evaluate_v4()
