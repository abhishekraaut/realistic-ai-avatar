import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
import hashlib
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def get_fingerprint(model):
    # Deterministic fingerprint: sum of abs of all weights
    return sum(p.abs().sum().item() for p in model.parameters())

def load_data(res):
    img_data = torch.load(f"synthesia_training_data/dataset_v5_{res}_images.pt", map_location='cpu')
    motion_data = torch.load("synthesia_training_data/dataset_v5.pt", map_location='cpu')
    
    # img_data['train'] is a list of 397 dicts
    # motion_data['train']['Y'] is [397, 52]
    
    def format_split(split):
        images = img_data[split]
        motions = motion_data[split]['Y']
        
        # We need sequence chunks of T=16. Let's just stride by 16.
        T = 16
        seqs = []
        for i in range(0, len(images) - T + 1, T):
            ident = images[i]['identity_frame'] # [3, 512, 512]
            
            target = torch.stack([images[i+t]['target_frame'] for t in range(T)]) # [16, 3, 512, 512]
            motion = motions[i:i+T] # [16, 52]
            
            seqs.append((ident, target, motion))
        return seqs

    return format_split('train'), format_split('val')

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--motion_dim', type=int, default=52)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default='backend/training/checkpoints/neural_renderer_v6_512_best.pt')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_seqs, val_seqs = load_data(args.resolution)
    print(f"Loaded {len(train_seqs)} train sequences, {len(val_seqs)} val sequences.")
    
    model = NeuralRendererV3(motion_dim=args.motion_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.checkpoint):
        print("Reloading checkpoint...")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch = state['epoch'] + 1
        best_val_loss = state.get('best_validation_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val: {best_val_loss:.4f}")
    
    criterion = nn.MSELoss()
    
    B, T, res = args.batch_size, args.sequence_length, args.resolution
    
    fp_before = get_fingerprint(model)
    first_step_done = False
    fp_after = None
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        t0 = time.time()
        for seq_idx, (ident, target, motion) in enumerate(train_seqs):
            ident = ident.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            motion = motion.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            h = torch.zeros(1, 256, res//16, res//16).to(device)
            c = torch.zeros(1, 256, res//16, res//16).to(device)
            
            loss = 0
            for t in range(T):
                out, (h, c) = model(ident, motion[:, t], (h, c))
                loss += criterion(out, target[:, t])
                
            loss = loss / T
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if not first_step_done:
                fp_after = get_fingerprint(model)
                first_step_done = True

        train_loss /= len(train_seqs)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ident, target, motion in val_seqs:
                ident = ident.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                motion = motion.unsqueeze(0).to(device)
                
                h = torch.zeros(1, 256, res//16, res//16).to(device)
                c = torch.zeros(1, 256, res//16, res//16).to(device)
                loss = 0
                for t in range(T):
                    out, (h, c) = model(ident, motion[:, t], (h, c))
                    loss += criterion(out, target[:, t])
                val_loss += (loss / T).item()
        val_loss /= len(val_seqs)
        
        epoch_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}s | VRAM: {peak_vram:.2f}GB")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'step': (epoch + 1) * len(train_seqs),
                'best_validation_loss': best_val_loss,
                'config': vars(args),
                'seed': args.seed,
                'dataset_version': 'v5_512'
            }
            torch.save(state, args.checkpoint)

    print("--- FINGERPRINT CHECK ---")
    print(f"Fingerprint BEFORE 1st step: {fp_before}")
    print(f"Fingerprint AFTER 1st step: {fp_after}")
    diff = abs(fp_after - fp_before) if fp_after else 0
    print(f"Params changed footprint diff: {diff}")
    
    print("Training complete.")

if __name__ == '__main__':
    train()
