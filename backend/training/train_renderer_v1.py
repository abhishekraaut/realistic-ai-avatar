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
        self.X = data_v4[split]["X"]
        self.Y = data_v4[split]["Y"]
        self.images = images_v4[split]
        assert self.Y.shape[0] == len(self.images)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "identity_img": self.images[idx]["identity_frame"],
            "motion_vector": self.Y[idx],
            "target_img": self.images[idx]["target_frame"]
        }

def calculate_psnr(mse):
    if mse == 0: return 100
    max_pixel = 2.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def benchmark_renderer(model, device, batch_size=1, resolution=512, amp=True):
    print("\n--- RENDERER FORWARD BENCHMARK ---")
    model.eval()
    ident = torch.randn(batch_size, 3, resolution, resolution, device=device)
    motion = torch.randn(batch_size, 11, device=device)

    warmup = 10
    reps = 100

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            if amp:
                with torch.amp.autocast('cuda'):
                    _ = model(ident, motion)
            else:
                _ = model(ident, motion)

        torch.cuda.synchronize()

        latencies = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.time()
            if amp:
                with torch.amp.autocast('cuda'):
                    _ = model(ident, motion)
            else:
                _ = model(ident, motion)
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000)

    latencies.sort()
    print(f"Warmup count: {warmup}")
    print(f"Measured iteration count: {reps}")
    print(f"Synchronization: torch.cuda.synchronize() before and after timing")
    print(f"Batch size: {batch_size}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"AMP: {'On' if amp else 'Off'}")
    print(f"Mean latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"Min latency: {latencies[0]:.2f} ms")
    print(f"Max latency: {latencies[-1]:.2f} ms")
    print(f"p50 latency: {latencies[int(len(latencies)*0.5)]:.2f} ms")
    print(f"p95 latency: {latencies[int(len(latencies)*0.95)]:.2f} ms")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")

def one_batch_overfit(model, batch, device, scaler, optimizer, loss_fn):
    print("\n--- ONE BATCH OVERFIT TEST ---")
    model.train()
    ident = batch["identity_img"].to(device)
    motion = batch["motion_vector"].to(device)
    target = batch["target_img"].to(device)

    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            pred = model(ident, motion)
            loss = loss_fn(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i == 0 or i == 9:
            print(f"Overfit Step {i+1}/10 - Loss: {loss.item():.4f}")
    print(f"Peak VRAM after backward: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
    print("One-batch overfit passed (loss decreased).")

def train_renderer_v1():
    print("Checking CUDA Status...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ERROR: GPU is required for Phase 8B.")
        return

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

    # Configuration
    epochs = 50
    physical_batch_size = 4
    effective_batch_size = 4
    accum_steps = effective_batch_size // physical_batch_size
    learning_rate = 1e-4

    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=physical_batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = NeuralRendererV1(motion_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    print("\n--- MODEL ARCHITECTURE ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    benchmark_renderer(model, device, batch_size=1, resolution=512, amp=True)

    # 1 batch overfit
    sample_batch = next(iter(train_loader))
    one_batch_overfit(model, sample_batch, device, scaler, optimizer, loss_fn)

    # Reset model to fresh state
    model = NeuralRendererV1(motion_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    metrics = {"train_loss": [], "val_loss": [], "val_psnr": [], "epoch_times": []}
    global_step = 0

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_accum = 0.0

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            ident = batch["identity_img"].to(device, non_blocking=True)
            motion = batch["motion_vector"].to(device, non_blocking=True)
            target = batch["target_img"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                pred = model(ident, motion)
                loss = loss_fn(pred, target)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            train_loss_accum += loss.item() * accum_steps * ident.size(0)

        train_loss = train_loss_accum / len(train_dataset)

        model.eval()
        val_loss_accum, val_mse_accum = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                ident = batch["identity_img"].to(device, non_blocking=True)
                motion = batch["motion_vector"].to(device, non_blocking=True)
                target = batch["target_img"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    pred = model(ident, motion)
                    val_loss_accum += loss_fn(pred, target).item() * ident.size(0)
                    val_mse_accum += mse_fn(pred, target).item() * ident.size(0)

        val_loss = val_loss_accum / len(val_dataset)
        val_mse = val_mse_accum / len(val_dataset)
        val_psnr = calculate_psnr(val_mse)
        epoch_time = time.time() - epoch_start

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_psnr"].append(val_psnr)
        metrics["epoch_times"].append(epoch_time)

        print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}dB | Peak VRAM: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")

        ckpt_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "config": {"batch_size": physical_batch_size, "resolution": 512, "motion_dim": 11}
        }
        torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_gpu_latest.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, os.path.join(ckpt_dir, "neural_renderer_v1_gpu_best.pt"))

    with open(os.path.join(base_dir, "artifacts", "renderer_v1_training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining Complete.")
    print(f"Best Val L1: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_renderer_v1()
