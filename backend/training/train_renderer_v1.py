import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1

class RendererDataset(Dataset):
    def __init__(self, data_v4, images_v4, split):
        # data_v4 is dict with keys X, Y
        # images_v4 is list of dicts {"identity_frame": tensor, "target_frame": tensor}
        self.X = data_v4[split]["X"]
        self.Y = data_v4[split]["Y"]
        self.images = images_v4[split]
        
        assert self.Y.shape[0] == len(self.images), f"Mismatch in {split} split! Y: {self.Y.shape[0]}, Images: {len(self.images)}"
        
    def __len__(self):
        return self.Y.shape[0]
        
    def __getitem__(self, idx):
        # X: audio features (not used in Phase 8B static motion-only experiment, but available)
        # Y: ground truth 11D motion
        return {
            "identity_img": self.images[idx]["identity_frame"],
            "motion_vector": self.Y[idx],
            "target_img": self.images[idx]["target_frame"]
        }

def calculate_psnr(mse):
    if mse == 0: return 100
    max_pixel = 2.0 # range is [-1, 1] so diff can be 2
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def train_renderer_v1():
    print("Checking CUDA Status...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    if device.type == "cpu":
        print("ERROR: GPU is required for Phase 8B.")
        return
        
    epochs = 150
    batch_size = 4 # Adjust based on 6GB VRAM. 512x512 images.
    learning_rate = 1e-4
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_dir = os.path.join(base_dir, "backend", "training", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    data_v4_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    images_v4_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4_images.pt")
    
    print("Loading datasets...")
    data_v4 = torch.load(data_v4_path, map_location="cpu", weights_only=False)
    images_v4 = torch.load(images_v4_path, map_location="cpu", weights_only=False)
    
    train_dataset = RendererDataset(data_v4, images_v4, "train")
    val_dataset = RendererDataset(data_v4, images_v4, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = NeuralRendererV1(motion_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(device='cuda')
    
    print("\n--- MODEL ARCHITECTURE ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    best_val_loss = float('inf')
    
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "epoch_times": []
    }
    
    print("\nStarting Training...")
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        
        train_loss_accum = 0.0
        for batch in train_loader:
            ident = batch["identity_img"].to(device)
            motion = batch["motion_vector"].to(device)
            target = batch["target_img"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                pred = model(ident, motion)
                loss = loss_fn(pred, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_accum += loss.item() * ident.size(0)
            global_step += 1
            
        train_loss = train_loss_accum / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ident = batch["identity_img"].to(device)
                motion = batch["motion_vector"].to(device)
                target = batch["target_img"].to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(ident, motion)
                    loss = loss_fn(pred, target)
                    mse = mse_fn(pred, target)
                    
                val_loss_accum += loss.item() * ident.size(0)
                val_mse_accum += mse.item() * ident.size(0)
                
        val_loss = val_loss_accum / len(val_dataset)
        val_mse = val_mse_accum / len(val_dataset)
        val_psnr = calculate_psnr(val_mse)
        
        epoch_time = time.time() - epoch_start
        
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_psnr"].append(val_psnr)
        metrics["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}dB | Peak VRAM: {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB")
        
        # Save checkpoints
        ckpt_state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "config": {
                "motion_dim": 11,
                "resolution": 512,
                "identity_policy": "first_frame"
            }
        }
        
        torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_gpu_latest.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_gpu_best.pt"))
            print(f"  -> Saved new best checkpoint (Val L1: {best_val_loss:.4f})")
            
    # Save metrics
    with open(os.path.join(base_dir, "artifacts", "renderer_v1_training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("\nTraining Complete.")
    print(f"Best Val L1: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_renderer_v1()
