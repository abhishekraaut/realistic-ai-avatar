import torch
import time
import os
import hashlib
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from backend.engine.neural_renderer_v3 import NeuralRendererV3

device = torch.device('cuda')
model = NeuralRendererV3(motion_dim=52).to(device)
chk_path = 'backend/training/checkpoints/neural_renderer_v5_best.pt'
model.load_state_dict(torch.load(chk_path, map_location=device)['model'])
model.eval()

ident = torch.randn(1, 3, 256, 256, device=device)
motion = torch.randn(1, 52, device=device)

latencies = []
state = None
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for _ in range(10):
            out, state = model(ident, motion, state)
        torch.cuda.synchronize()
        state = None
        for _ in range(100):
            t0 = time.time()
            out, state = model(ident, motion, state)
            torch.cuda.synchronize()
            latencies.append((time.time() - t0)*1000)

latencies = np.array(latencies)
sha256 = hashlib.sha256(open(chk_path, 'rb').read()).hexdigest()
size = os.path.getsize(chk_path)

print(f'Mean: {latencies.mean():.2f}')
print(f'p50: {np.percentile(latencies, 50):.2f}')
print(f'p95: {np.percentile(latencies, 95):.2f}')
print(f'Max: {latencies.max():.2f}')
print(f'Size: {size}')
print(f'SHA256: {sha256}')
print(f'VRAM: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}')
