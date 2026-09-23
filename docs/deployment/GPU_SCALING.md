# GPU Scaling Deployment Guide
## Admission Control
Never route traffic to a GPU Worker if:
- `active_sessions >= max_worker_capacity`
- `gpu_utilization > 80%`
- VRAM headroom < 1GB

## Distributed Deployment
Isolate GPU workers at the process or container level. Do not attempt to share ConvLSTM instances across threads. Each worker must independently own its connection to the LiveKit cluster and maintain exclusive control over its conversational state buffer.
