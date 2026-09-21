import torch
import sys
import time
import os

print("--- GPU VERIFICATION REPORT ---")

print("\n1. PYTORCH INSTALLATION")
print("PyTorch Version:", torch.__version__)
print("CUDA Version reported by PyTorch:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())

print("\n2. CUDA OPERATION TEST")
try:
    device = torch.device("cuda")
    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)
    z = x @ y
    print("device:", z.device)
    print("dtype:", z.dtype)
    print("shape:", z.shape)
    print("gpu:", torch.cuda.get_device_name(0))
    print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")
    print("CUDA Test: SUCCESS")
except Exception as e:
    print("CUDA Test: FAILED", e)

print("\n3. AMP (MIXED PRECISION) TEST")
try:
    with torch.amp.autocast(device_type="cuda"):
        out = x @ y
    print("AMP operation completed.")
    print("AMP Test: SUCCESS")
except Exception as e:
    print("AMP Test: FAILED", e)

print("\n4. GPU MEMORY")
total_mem = torch.cuda.get_device_properties(0).total_memory
alloc_mem = torch.cuda.memory_allocated()
res_mem = torch.cuda.memory_reserved()
free_mem = total_mem - res_mem
print(f"Total VRAM: {total_mem / 1024**3:.2f} GB")
print(f"Allocated: {alloc_mem / 1024**3:.2f} GB")
print(f"Reserved: {res_mem / 1024**3:.2f} GB")
print(f"Free: {free_mem / 1024**3:.2f} GB")
torch.cuda.empty_cache()
print("After empty_cache():")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n5. RENDERER FORWARD BENCHMARK")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from backend.models.neural_renderer_v1 import NeuralRendererV1
    model = NeuralRendererV1(motion_dim=11).to(device)
    ident = torch.randn(1, 3, 512, 512, device=device)
    motion = torch.randn(1, 11, device=device)
    
    # Cold start
    t0 = time.time()
    out = model(ident, motion)
    cold_lat = (time.time() - t0) * 1000
    
    # Warmup
    for _ in range(10):
        _ = model(ident, motion)
        
    # Benchmark
    latencies = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(ident, motion)
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
        
    print(f"Output shape: {out.shape}")
    print(f"Cold-start latency: {cold_lat:.2f} ms")
    print(f"Average latency: {sum(latencies)/len(latencies):.2f} ms")
    latencies.sort()
    print(f"p50 latency: {latencies[int(len(latencies)*0.5)]:.2f} ms")
    print(f"p95 latency: {latencies[int(len(latencies)*0.95)]:.2f} ms")
    print(f"Peak VRAM during benchmark: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("Renderer Benchmark: SUCCESS")
except Exception as e:
    print("Renderer Benchmark: FAILED", e)
    import traceback
    traceback.print_exc()

print("\n6. MOTION CHECKPOINT VERIFICATION")
try:
    from backend.models.motion_model_v4 import LearnedTemporalMotionModelV4
    ckpt_path = "backend/training/checkpoints/learned_motion_v4_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    motion_model = LearnedTemporalMotionModelV4().to(device)
    motion_model.load_state_dict(ckpt["model_state"])
    
    audio_ctx = torch.randn(1, 1, 16, 80, device=device)
    out_motion = motion_model(audio_ctx)
    print(f"Motion Model Output shape: {out_motion.shape}")
    print("Motion Checkpoint Load: SUCCESS")
except Exception as e:
    print("Motion Checkpoint Load: FAILED", e)
    import traceback
    traceback.print_exc()
