import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.motion_model import LearnedTemporalMotionModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\dataset.pt"
    ckpt_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\backend\training\checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load dataset
    data = torch.load(dataset_path)
    X_full = data["X"] # (250, 80)
    Y_full = data["Y"] # (250, 4)
    
    # We treat the entire sequence as a single batch for this tiny dataset
    X_full = X_full.unsqueeze(0).to(device) # (1, 250, 80)
    Y_full = Y_full.unsqueeze(0).to(device) # (1, 250, 4)
    
    model = LearnedTemporalMotionModel(input_dim=80, hidden_dim=64, output_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    mse_loss_fn = nn.MSELoss()
    
    # Training Loop
    epochs = 200
    global_step = 0
    
    print("Starting ONE-BATCH OVERFIT test...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_full)
        
        # 1. Reconstruction loss
        recon_loss = mse_loss_fn(preds, Y_full)
        
        # 2. Temporal Smoothness loss (velocity consistency)
        # Minimize the difference between consecutive frames in predictions
        # to prevent jitter.
        pred_vel = preds[:, 1:, :] - preds[:, :-1, :]
        target_vel = Y_full[:, 1:, :] - Y_full[:, :-1, :]
        vel_loss = mse_loss_fn(pred_vel, target_vel)
        
        # Total loss
        loss = recon_loss + 0.5 * vel_loss
        
        loss.backward()
        optimizer.step()
        
        global_step += 1
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} (Recon: {recon_loss.item():.6f}, Vel: {vel_loss.item():.6f})")
            
    # Checkpoint
    ckpt_path = os.path.join(ckpt_dir, "learned_motion_v1.pt")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "input_dim": 80,
            "hidden_dim": 64,
            "output_dim": 4,
            "architecture": "Conv1D_LSTM"
        },
        "dataset_info": {
            "frames": 250,
            "sequence_length": 250
        }
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    print("Overfit test completed successfully!")

if __name__ == "__main__":
    train()
