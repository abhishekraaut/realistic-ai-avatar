import torch
import sys
import os

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(_base, "backend"))

from models.motion_model_v4 import LearnedTemporalMotionModelV4
from models.neural_renderer_v1 import NeuralRendererV1
import time

def smoke_test():
    print("--- MODEL SMOKE TEST ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        # 1. Load V4
        motion_v4 = LearnedTemporalMotionModelV4().to(device)
        v4_ckpt = torch.load(os.path.join(_base, "backend", "training", "checkpoints", "learned_motion_v4_best.pt"), map_location=device)
        motion_v4.load_state_dict(v4_ckpt.get("model_state", v4_ckpt))
        motion_v4.eval()
        print("V4 checkpoint loaded successfully.")
    except Exception as e:
        print(f"V4 load failed: {e}")
        return

    try:
        # 2. Load Renderer
        renderer = NeuralRendererV1().to(device)
        rend_ckpt = torch.load(os.path.join(_base, "backend", "training", "checkpoints", "neural_renderer_v1_gpu_best.pt"), map_location=device)
        renderer.load_state_dict(rend_ckpt.get("model_state_dict", rend_ckpt))
        renderer.eval()
        print("NeuralRendererV1 checkpoint loaded successfully.")
    except Exception as e:
        print(f"Renderer load failed: {e}")
        return

    # 3. Inference Benchmark
    print("Running inference benchmark...")
    try:
        B = 1
        dummy_audio = torch.randn(B, 1, 16, 80).to(device)
        dummy_identity = torch.randn(B, 3, 512, 512).to(device)
        
        # Warmup
        for _ in range(10):
            out = motion_v4(dummy_audio)
            dummy_motion = out[0]
            _ = renderer(dummy_identity, dummy_motion)
            
        torch.cuda.synchronize()
        
        # Benchmark V4
        v4_times = []
        for _ in range(100):
            t0 = time.time()
            out = motion_v4(dummy_audio)
            torch.cuda.synchronize()
            v4_times.append((time.time() - t0) * 1000)
            
        # Benchmark Renderer
        dummy_motion = out[0]
        rend_times = []
        for _ in range(100):
            t0 = time.time()
            _ = renderer(dummy_identity, dummy_motion)
            torch.cuda.synchronize()
            rend_times.append((time.time() - t0) * 1000)
            
        # Benchmark Combined
        comb_times = []
        for _ in range(100):
            t0 = time.time()
            out = motion_v4(dummy_audio)
            _ = renderer(dummy_identity, out[0])
            torch.cuda.synchronize()
            comb_times.append((time.time() - t0) * 1000)
            
        def report(name, times):
            times.sort()
            mean_t = sum(times) / len(times)
            p50 = times[len(times)//2]
            p95 = times[int(len(times)*0.95)]
            max_t = times[-1]
            print(f"{name}: mean={mean_t:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, max={max_t:.2f}ms")
            
        print("\n--- BENCHMARK RESULTS ---")
        report("V4 Motion", v4_times)
        report("Renderer", rend_times)
        report("Combined", comb_times)
        
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"Peak Allocated VRAM: {peak_allocated:.2f} MB")
        print(f"Peak Reserved VRAM: {peak_reserved:.2f} MB")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    smoke_test()
