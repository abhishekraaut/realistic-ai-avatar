import torch
import torch.nn as nn
import torch.optim as optim
import time
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def run_tests():
    print("--- STRUCTURAL TEST ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRendererV3().to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params}")
    
    torch.cuda.reset_peak_memory_stats()
    
    B, C, H, W = 2, 3, 512, 512
    identity_ref = torch.randn(B, C, H, W, device=device)
    motion = torch.randn(B, 11, device=device)
    
    out, state = model(identity_ref, motion)
    print(f"Output shape: {out.shape}")
    
    forward_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Forward VRAM: {forward_vram:.2f} MB")
    
    loss = out.mean()
    loss.backward()
    
    backward_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Backward VRAM: {backward_vram:.2f} MB")
    
    has_nan = torch.isnan(out).any().item()
    print(f"NaN/Inf: {has_nan}")
    
    print("--- ONE-BATCH OVERFIT TEST ---")
    model = NeuralRendererV3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    # 4 contiguous frames
    T = 4
    target = torch.ones(B, T, C, H, W, device=device)
    motions = torch.randn(B, T, 11, device=device)
    
    initial_loss = 0
    final_loss = 0
    
    for epoch in range(50):
        optimizer.zero_grad()
        loss = 0
        state = None
        for t in range(T):
            out, state = model(identity_ref, motions[:, t], state)
            loss += criterion(out, target[:, t])
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item() / T
        if epoch == 49:
            final_loss = loss.item() / T
            
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    
    print("--- BENCHMARK ---")
    with torch.no_grad():
        for _ in range(10):
            state = None
            for t in range(4):
                _, state = model(identity_ref, motion, state)
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            state = None
            for t in range(4):
                _, state = model(identity_ref, motion, state)
        torch.cuda.synchronize()
        end = time.time()
        
    mean_latency = (end - start) / (100 * 4)
    print(f"Mean Latency: {mean_latency*1000:.2f} ms")
    print(f"FPS Equivalent: {1/mean_latency:.2f}")

    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"Peak Allocated VRAM: {peak_allocated:.2f} MB")
    print(f"Peak Reserved VRAM: {peak_reserved:.2f} MB")
    
    torch.save(model.state_dict(), 'backend/training/checkpoints/neural_renderer_v3_temporal_best.pt')

if __name__ == '__main__':
    run_tests()
