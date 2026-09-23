# GPU Capacity & Bottleneck Analysis
The frozen `v0.1.0-avatar-rc` system requires ~750MB of reserved VRAM per active session (encompassing model weights, identity references, ConvLSTM hidden states, and activation memory). Memory scaling is roughly linear.

However, the architecture is definitively **compute-bound**. On the RTX 3050 6GB, 2 sessions consume only 1.5GB of VRAM but entirely saturate the CUDA compute cores (99%), pushing renderer latency from 17.5ms to 34.5ms and dropping FPS. 
