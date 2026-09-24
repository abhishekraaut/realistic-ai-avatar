import os
import time
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Load datasets
    motion_data = torch.load('synthesia_training_data/dataset_v5.pt', map_location='cpu')
    image_data = torch.load('synthesia_training_data/dataset_v5_images.pt', map_location='cpu')
    
    Y_train = motion_data['train']['Y'].to(device) # [397, 52]
    Y_val = motion_data['val']['Y'].to(device)     # [100, 52]
    
    train_images = image_data['train']
    val_images = image_data['val']
    
    ident_train = train_images[0]['identity_frame'].unsqueeze(0).to(device) # [1, 3, 256, 256]
    ident_val = val_images[0]['identity_frame'].unsqueeze(0).to(device)
    
    target_train = torch.stack([img['target_frame'] for img in train_images]).unsqueeze(0).to(device) # [1, T, 3, 256, 256]
    target_val = torch.stack([img['target_frame'] for img in val_images]).unsqueeze(0).to(device)
    Y_train = Y_train.unsqueeze(0) # [1, T, 52]
    Y_val = Y_val.unsqueeze(0)
    
    model = NeuralRendererV3(motion_dim=52).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.L1Loss()
    
    # Smoke test & resume test
    chk_path = 'backend/training/checkpoints/neural_renderer_v5_best.pt'
    os.makedirs('backend/training/checkpoints', exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': 0}, chk_path)
    chk = torch.load(chk_path)
    model.load_state_dict(chk['model'])
    optimizer.load_state_dict(chk['opt'])
    print('Resume test passed')
    
    epochs = 50
    chunk_size = 16
    best_val_loss = float('inf')
    initial_loss = None
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        steps = 0
        state = None
        
        optimizer.zero_grad(set_to_none=True)
        for t in range(0, Y_train.shape[1] - chunk_size, chunk_size):
            loss = 0
            # Detach state to prevent BP through entire sequence
            if state is not None:
                state = (state[0].detach(), state[1].detach())
            
            with torch.amp.autocast('cuda'):
                for i in range(chunk_size):
                    out, state = model(ident_train, Y_train[:, t+i], state)
                    loss += loss_fn(out, target_train[:, t+i])
            
            loss = loss / chunk_size
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item()
            steps += 1
        
        epoch_loss /= max(1, steps)
        if epoch == 0: initial_loss = epoch_loss
        
        # Validation
        model.eval()
        val_loss = 0
        state = None
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for t in range(Y_val.shape[1]):
                    out, state = model(ident_val, Y_val[:, t], state)
                    val_loss += loss_fn(out, target_val[:, t]).item()
        val_loss /= Y_val.shape[1]
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, chk_path)
    
    t1 = time.time()
    print(f'Training completed in {t1 - t0:.2f}s')
    print(f'Initial Loss: {initial_loss:.4f}')
    print(f'Best Val Loss: {best_val_loss:.4f}')
    
    # Eval checkpoint
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk['model'])
    model.eval()
    
    # Benchmarks on GPU
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Warmup
            state = None
            for _ in range(10):
                out, state = model(ident_val, Y_val[:, 0], state)
            torch.cuda.synchronize()
            
            latencies = []
            state = None
            for _ in range(100):
                t_start = time.time()
                out, state = model(ident_val, Y_val[:, 0], state)
                torch.cuda.synchronize()
                latencies.append((time.time() - t_start)*1000)
                
    latencies = np.array(latencies)
    print(f'Latency Mean: {latencies.mean():.2f}ms')
    print(f'Latency p50: {np.percentile(latencies, 50):.2f}ms')
    print(f'Latency p95: {np.percentile(latencies, 95):.2f}ms')
    print(f'Latency Max: {latencies.max():.2f}ms')
    print(f'VRAM: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB')
    
    sha256 = hashlib.sha256(open(chk_path, 'rb').read()).hexdigest()
    file_size = os.path.getsize(chk_path)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {num_params}')
    print(f'Size: {file_size}')
    print(f'SHA256: {sha256}')
    print(f'NaN: {torch.isnan(out).any().item()}')
    print(f'Inf: {torch.isinf(out).any().item()}')

if __name__ == '__main__':
    train()
