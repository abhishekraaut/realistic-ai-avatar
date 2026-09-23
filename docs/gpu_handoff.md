# GPU Handoff Package
This package enables hardware-independent scale-out readiness without modifying the avatar architecture.

## Adding a New GPU Worker
1. Provision cloud or local GPU instance.
2. Run `gpu_readiness_check` to validate CUDA/PyTorch, checkpoint hashes, and VRAM.
3. Run `benchmark_worker` to determine exact latency distributions on the new GPU.
4. If successful, configure `max_capacity` based strictly on the harness output and register the worker with the Gateway.

## Hardware Status
- RTX 3050 6GB: VALIDATED (1 session)
- Multi-GPU Cluster: NOT_TESTED (Blocked by hardware)
- RTX 4080 / 4090: NOT_TESTED (Blocked by hardware)
