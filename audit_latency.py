import torch
import time
import gc
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def audit_latency():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRendererV3().to(device)
    model.eval()
    
    # Mock V4
    v4_mock = torch.nn.Sequential(
        torch.nn.Linear(1024, 256), 
        torch.nn.ReLU(), 
        torch.nn.Linear(256, 11)
    ).to(device).eval()
    
    B, C, H, W = 1, 3, 512, 512
    ident = torch.randn(B, C, H, W, device=device)
    mot = torch.randn(B, 11, device=device)
    audio_feat = torch.randn(B, 1024, device=device)
    
    # Warmup
    with torch.inference_mode():
        state = None
        for _ in range(20):
            out, state = model(ident, mot, state)
    
    torch.cuda.synchronize()
    
    # Measure V3 Step
    iters = 200
    v3_times = []
    with torch.inference_mode():
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out, state = model(ident, mot, state)
            torch.cuda.synchronize()
            v3_times.append((time.perf_counter() - start) * 1000)
    
    v3_times = torch.tensor(v3_times)
    
    # Measure V4 + V3 Step
    v4_v3_times = []
    with torch.inference_mode():
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            m = v4_mock(audio_feat)
            out, state = model(ident, m, state)
            torch.cuda.synchronize()
            v4_v3_times.append((time.perf_counter() - start) * 1000)
            
    v4_v3_times = torch.tensor(v4_v3_times)
    
    print("--- V3 STEP LATENCY ---")
    print(f"Mean: {v3_times.mean().item():.2f} ms")
    print(f"P50: {torch.median(v3_times).item():.2f} ms")
    print(f"P95: {torch.quantile(v3_times, 0.95).item():.2f} ms")
    print(f"P99: {torch.quantile(v3_times, 0.99).item():.2f} ms")
    print(f"Max: {v3_times.max().item():.2f} ms")

    print("\n--- V4 + V3 STEP LATENCY ---")
    print(f"Mean: {v4_v3_times.mean().item():.2f} ms")
    print(f"P50: {torch.median(v4_v3_times).item():.2f} ms")
    print(f"P95: {torch.quantile(v4_v3_times, 0.95).item():.2f} ms")
    print(f"P99: {torch.quantile(v4_v3_times, 0.99).item():.2f} ms")
    print(f"Max: {v4_v3_times.max().item():.2f} ms")
    
    # Pure Inference VRAM
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.inference_mode():
        state = None
        for _ in range(50):
            m = v4_mock(audio_feat)
            out, state = model(ident, m, state)
            
    inf_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    inf_res = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"\n--- VRAM INFERENCE (B=1) ---")
    print(f"Peak Allocated: {inf_alloc:.2f} MB")
    print(f"Peak Reserved: {inf_res:.2f} MB")

if __name__ == '__main__':
    audit_latency()
