import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model import LearnedTemporalMotionModel

def train_smoke_test():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    dataset_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\dataset_v2.pt"
    ckpt_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\backend\training\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data = torch.load(dataset_path, map_location=device)
    
    X_train = data["train"]["X"].unsqueeze(0) # (1, T, 80)
    Y_train = data["train"]["Y"].unsqueeze(0) # (1, T, 4)
    
    X_val = data["val"]["X"].unsqueeze(0) if data["val"]["X"].shape[0] > 0 else None
    Y_val = data["val"]["Y"].unsqueeze(0) if data["val"]["Y"].shape[0] > 0 else None
    
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
    if X_val is not None:
        print(f"Val Shape: X={X_val.shape}, Y={Y_val.shape}")
    
    model = LearnedTemporalMotionModel(input_dim=80, hidden_dim=64, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss_fn = nn.MSELoss()
    
    # Tiny Smoke Test (5 Epochs)
    epochs = 5
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_train)
        recon_loss = mse_loss_fn(preds, Y_train)
        pred_vel = preds[:, 1:, :] - preds[:, :-1, :]
        target_vel = Y_train[:, 1:, :] - Y_train[:, :-1, :]
        vel_loss = mse_loss_fn(pred_vel, target_vel)
        loss = recon_loss + 0.5 * vel_loss
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")
        
    if X_val is not None:
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val)
            val_loss = mse_loss_fn(preds_val, Y_val)
            print(f"Validation MSE: {val_loss.item():.4f}")
            
    print("Smoke Test Passed!")

if __name__ == "__main__":
    train_smoke_test()
