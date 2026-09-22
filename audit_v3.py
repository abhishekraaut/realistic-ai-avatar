import torch
import time
from backend.engine.neural_renderer_v2 import NeuralRendererV2
from backend.engine.neural_renderer_v3 import NeuralRendererV3

def audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRendererV3().to(device)
    
    # 1. Parameter Audit
    def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    enc = count(model.encoder)
    clstm = count(model.conv_lstm)
    film = count(model.bottleneck_film) + count(model.dec4.film) + count(model.dec3.film) + count(model.dec2.film) + count(model.dec1.film)
    unet_dec = count(model.dec4.conv) + count(model.dec3.conv) + count(model.dec2.conv) + count(model.dec1.conv) + count(model.out_conv)
    mlp = count(model.motion_mlp)
    total = count(model)
    
    print("--- PARAMETERS ---")
    print(f"Total: {total}")
    print(f"Encoder: {enc}")
    print(f"ConvLSTM: {clstm}")
    print(f"FiLM: {film}")
    print(f"Decoder (U-Net convs): {unet_dec}")
    print(f"Motion MLP: {mlp}")
    print(f"Other: {total - (enc+clstm+film+unet_dec+mlp)}")
    
    # 2. VRAM Training
    torch.cuda.reset_peak_memory_stats()
    model.train()
    B, C, H, W = 4, 3, 512, 512
    ident = torch.randn(B, C, H, W, device=device)
    mots = torch.randn(B, 4, 11, device=device)
    targ = torch.randn(B, C, H, W, device=device)
    
    opt = torch.optim.Adam(model.parameters())
    opt.zero_grad()
    loss = 0
    state = None
    for t in range(4):
        out, state = model(ident, mots[:, t], state)
        loss += out.mean()
    loss.backward()
    opt.step()
    
    train_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    train_res = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"\n--- VRAM TRAINING (B=4, T=4) ---")
    print(f"Peak Allocated: {train_alloc:.2f} MB")
    print(f"Peak Reserved: {train_res:.2f} MB")
    
    # 3. VRAM Inference
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    ident = torch.randn(1, C, H, W, device=device)
    mot = torch.randn(1, 11, device=device)
    with torch.no_grad():
        state = None
        for t in range(4):
            out, state = model(ident, mot, state)
    
    inf_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    inf_res = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"\n--- VRAM INFERENCE (B=1, T=4) ---")
    print(f"Peak Allocated: {inf_alloc:.2f} MB")
    print(f"Peak Reserved: {inf_res:.2f} MB")

    # 4. Latency
    model_v2 = NeuralRendererV2().to(device).eval()
    
    def measure(fn, iters=100):
        with torch.no_grad():
            for _ in range(10): fn()
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
        return (time.time() - start) / iters

    state_global = [None]
    def v3_step():
        _, state_global[0] = model(ident, mot, state_global[0])
        
    def v3_t4():
        s = None
        for _ in range(4):
            _, s = model(ident, mot, s)

    # Mock V4 (Audio -> 11D MLP)
    v4_mock = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.ReLU(), torch.nn.Linear(256, 11)).to(device).eval()
    audio_feat = torch.randn(1, 1024, device=device)
    
    def v4_v3():
        m = v4_mock(audio_feat)
        _, state_global[0] = model(ident, m, state_global[0])
        
    v2_lat = measure(lambda: model_v2(ident, mot))
    v3_step_lat = measure(v3_step)
    v3_t4_lat = measure(v3_t4)
    v4_v3_lat = measure(v4_v3)

    print(f"\n--- PERFORMANCE ---")
    print(f"V2: {v2_lat*1000:.2f} ms")
    print(f"V3 Step: {v3_step_lat*1000:.2f} ms")
    print(f"V3 T=4: {v3_t4_lat*1000:.2f} ms")
    print(f"V4+V3: {v4_v3_lat*1000:.2f} ms")

if __name__ == '__main__':
    audit()
