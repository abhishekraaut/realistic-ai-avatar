import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def load_data(res):
    img_data = torch.load(f'synthesia_training_data/dataset_v5_{res}_images.pt', map_location='cpu')
    motion_data = torch.load('synthesia_training_data/dataset_v5.pt', map_location='cpu')
    def format_split(split):
        images = img_data[split]
        motions = motion_data[split]['Y']
        T = 16
        seqs = []
        for i in range(0, len(images) - T + 1, T):
            ident = images[i]['identity_frame']
            target = torch.stack([images[i+t]['target_frame'] for t in range(T)])
            motion = motions[i:i+T]
            seqs.append((ident, target, motion))
        return seqs
    return format_split('train'), format_split('val')

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_epoch', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    print("Batch size: 1")
    print("Sequence length: 16")
    print("Learning rate: 5e-5")
    print("Optimizer: Adam")
    print("Scheduler: StepLR (gamma=0.5, step=5)")
    print(f"Epochs target: {args.stop_epoch}")
    print("Precision: fp32")
    print("Gradient accumulation: 1")
    print("Seed: 42")
    
    torch.manual_seed(42)
    train_seqs, val_seqs = load_data(768)
    model = NeuralRendererV3(motion_dim=52).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    
    final_checkpoint = 'backend/training/checkpoints/neural_renderer_v6_768_best.pt'
    init_checkpoint = 'backend/training/checkpoints/neural_renderer_v6_512_converged_best.pt'
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(final_checkpoint):
        state = torch.load(final_checkpoint, map_location=device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state'])
        start_epoch = state['epoch'] + 1
        best_val_loss = state.get('best_validation_loss', float('inf'))
    else:
        state = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(state['model_state'])

    T, res = 16, 768
    
    for epoch in range(start_epoch, args.stop_epoch):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for seq_idx, (ident, target, motion) in enumerate(train_seqs):
            ident = ident.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            motion = motion.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            h = torch.zeros(1, 256, res//16, res//16, device=device)
            c = torch.zeros(1, 256, res//16, res//16, device=device)
            
            loss = 0
            for t in range(T):
                out, (h, c) = model(ident, motion[:, t], (h, c))
                loss += criterion(out, target[:, t])
                
            loss = loss / T
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_seqs)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ident, target, motion in val_seqs:
                ident = ident.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)
                motion = motion.unsqueeze(0).to(device)
                h = torch.zeros(1, 256, res//16, res//16, device=device)
                c = torch.zeros(1, 256, res//16, res//16, device=device)
                loss = 0
                for t in range(T):
                    out, (h, c) = model(ident, motion[:, t], (h, c))
                    loss += criterion(out, target[:, t])
                val_loss += (loss / T).item()
        val_loss /= len(val_seqs)
        
        epoch_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_out = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'step': (epoch + 1) * len(train_seqs),
                'best_validation_loss': best_val_loss,
                'config': {'res': 768, 'motion_dim': 52},
                'seed': 42,
                'dataset_version': 'v5_768'
            }
            torch.save(state_out, final_checkpoint)

if __name__ == '__main__':
    train()
