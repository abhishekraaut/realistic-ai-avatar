import torch
from backend.engine.neural_renderer_v3 import NeuralRendererV3
import os
import hashlib

path = 'backend/training/checkpoints/neural_renderer_v6_512_best.pt'
state = torch.load(path, map_location='cpu')
model = NeuralRendererV3(motion_dim=52)
model.load_state_dict(state['model_state'])

B = 1
i = torch.randn(B, 3, 512, 512)
m = torch.randn(B, 52)
h = torch.zeros(B, 256, 32, 32)
c = torch.zeros(B, 256, 32, 32)

out, _ = model(i, m, (h, c))
size = os.path.getsize(path)
sha256 = hashlib.sha256(open(path, 'rb').read()).hexdigest()
params = sum(p.numel() for p in model.parameters())

print(f"Size: {size}")
print(f"SHA256: {sha256}")
print(f"Params: {params}")
print(f"Output shape: {out.shape}")
print(f"NaN: {torch.isnan(out).any().item()}")
print(f"Inf: {torch.isinf(out).any().item()}")
