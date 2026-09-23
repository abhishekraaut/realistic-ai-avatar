# Multi-Worker GPU Scale-Out Architecture

## Architecture Flow
Client -> Gateway/Router -> Worker Pool -> [GPU Worker A, GPU Worker B, ...]

## Worker Lifecycle
1. **READY**: Model checkpoints loaded, LiveKit initialized, VRAM allocated, awaiting session.
2. **BUSY**: Actively processing a session. Rejects new session assignments.
3. **DRAINING**: Finishing current session; will not accept new sessions.
4. **UNHEALTHY**: Failed health checks or crashed. Excluded from routing.

## Routing Policy
- Assign only to READY workers.
- Deterministic load balancing if multiple READY workers exist.
- Strict isolation: Each worker process has its own isolated memory space, ConvLSTM state, V6 state, Stochastic Residual state, Eye state, and LiveKit connection.
