import torch
import time
import numpy as np
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def run(gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = NeuralRendererV3(motion_dim=52).to(device)
    state = torch.load('backend/training/checkpoints/neural_renderer_v6_512_best.pt', map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()
    
    i = torch.randn(1, 3, 512, 512, device=device)
    m = torch.randn(1, 52, device=device)
    h = torch.zeros(1, 256, 32, 32, device=device)
    c = torch.zeros(1, 256, 32, 32, device=device)
    
    with torch.no_grad():
        # warmup
        for _ in range(5):
            model(i, m, (h, c))
        torch.cuda.synchronize(device)
        
        latencies = []
        for _ in range(20):
            t0 = time.time()
            model(i, m, (h, c))
            torch.cuda.synchronize(device)
            latencies.append((time.time() - t0) * 1000)
            
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    max_lat = np.max(latencies)
    
    print(f"GPU {gpu_id}: Latency [Mean: {mean_lat:.1f}ms, p50: {p50_lat:.1f}ms, p95: {p95_lat:.1f}ms, Max: {max_lat:.1f}ms] | Peak VRAM {peak:.2f}GB")

import sys
if len(sys.argv) > 1:
    run(int(sys.argv[1]))
else:
    print("Run with python validate_gpu.py <gpu_id>")
