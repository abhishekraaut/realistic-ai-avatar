import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model_v4 import LearnedTemporalMotionModelV4

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_velocity_loss(preds, targets):
    pred_vel = preds[:, 1:, :] - preds[:, :-1, :]
    target_vel = targets[:, 1:, :] - targets[:, :-1, :]
    return nn.MSELoss()(pred_vel, target_vel)

def train_v4():
    set_seed(42)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    ckpt_dir = os.path.join(base_dir, "backend", "training", "checkpoints")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    data = torch.load(dataset_path, map_location=device)
    
    X_train = data["train"]["X"].unsqueeze(0)
    Y_train = data["train"]["Y"].unsqueeze(0)
    X_val = data["val"]["X"].unsqueeze(0)
    Y_val = data["val"]["Y"].unsqueeze(0)
    
    x_mean = X_train.mean(dim=(1, 2), keepdim=True)
    x_std = X_train.std(dim=(1, 2), keepdim=True) + 1e-8
    
    X_train_norm = (X_train - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std
    
    model = LearnedTemporalMotionModelV4(input_dim=80, context_size=16, hidden_dim=64, output_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse_loss_fn = nn.MSELoss()
    
    epochs = 400
    patience = 50
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    metrics_history = []
    
    print("Starting V4 Full Training Loop...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_train_norm)
        recon_loss = mse_loss_fn(preds, Y_train)
        vel_loss = compute_velocity_loss(preds, Y_train)
        train_total = recon_loss + 0.5 * vel_loss
        
        train_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val_norm)
            val_recon = mse_loss_fn(preds_val, Y_val)
            val_vel = compute_velocity_loss(preds_val, Y_val)
            val_total = val_recon + 0.5 * val_vel
            
        metrics_history.append({
            "epoch": epoch,
            "train_total": train_total.item(),
            "val_total": val_total.item()
        })
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_total.item():.4f} | Val: {val_total.item():.4f}")
            
        ckpt_state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": {"input_dim": 80, "context_size": 16, "hidden_dim": 64, "output_dim": 11, "architecture": "Context_Conv1D_LSTM_V4"},
            "norm_stats": {"x_mean": x_mean, "x_std": x_std},
            "fps_policy": "NATIVE_25FPS",
            "target_representation": "3_Pose_8_PCA_Latents"
        }
        
        torch.save(ckpt_state, os.path.join(ckpt_dir, "learned_motion_v4_latest.pt"))
        
        if val_total.item() < best_val_loss:
            best_val_loss = val_total.item()
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt_state, os.path.join(ckpt_dir, "learned_motion_v4_best.pt"))
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (Val Loss: {best_val_loss:.4f})")
            break
            
    print("Training Complete!")

if __name__ == "__main__":
    train_v4()
