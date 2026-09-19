import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1

def load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize to [-1, 1]
    frame = (frame / 127.5) - 1.0
    return torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

def train_renderer_v1_overfit():
    print("CUDA is False. Running CPU ONE-BATCH OVERFIT test to validate architecture.")
    
    device = torch.device("cpu")
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    asset_dir = os.path.join(base_dir, "public", "assets")
    ckpt_dir = os.path.join(base_dir, "backend", "training", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load dataset to get the motion vectors
    data = torch.load(dataset_path, map_location=device)
    
    # We will pick a single video (e.g. emotional-rollercoaster.mp4) and a few frames
    video_name = "emotional-rollercoaster.mp4"
    video_path = os.path.join(asset_dir, video_name)
    
    # Pick frame 0 as identity, and frames 10, 20, 30, 40 as targets
    # Wait, dataset_v4.pt has targets ordered by how we extracted them.
    # To be safe for the smoke test, we'll just extract them directly or assume indices.
    # Since we just want to prove the model can train on real inputs, we can use 
    # the first 4 frames of Y from dataset_v4.pt, and read the first 4 frames of the video.
    
    Y_batch = data["train"]["Y"][:4] # [4, 11]
    
    identity_frame = load_frame(video_path, 0)
    target_frames = [load_frame(video_path, i) for i in range(1, 5)]
    
    if None in target_frames or identity_frame is None:
        print("Failed to load frames. Cannot proceed.")
        return
        
    identity_batch = identity_frame.unsqueeze(0).repeat(4, 1, 1, 1) # [4, 3, 512, 512]
    target_batch = torch.stack(target_frames, dim=0) # [4, 3, 512, 512]
    
    model = NeuralRendererV1(motion_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.L1Loss()
    
    epochs = 20
    print(f"Starting 1-Batch Overfit (Batch Size: 4, Image: 512x512, Epochs: {epochs})")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(identity_batch, Y_batch)
        loss = loss_fn(preds, target_batch)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:03d} | L1 Loss: {loss.item():.4f}")
        
    # Save checkpoint
    ckpt_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {"motion_dim": 11, "resolution": 512, "architecture": "UNet_FiLM"},
        "dataset_version": "v4",
        "identity_representation": "reference_frame_0",
        "status": "EXPERIMENTAL_SMOKE_TEST"
    }
    torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_latest.pt"))
    torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_best.pt"))
    
    print("Renderer V1 Overfit Test Complete. Checkpoint saved.")

if __name__ == "__main__":
    train_renderer_v1_overfit()
