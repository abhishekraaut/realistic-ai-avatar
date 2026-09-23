# Production Monitoring
Metrics: renderer latency (T_motion_input_ready -> T_renderer_output_ready), FPS, VRAM, queue depth, stale residual, gate overlap, residual resets, session failures, LiveKit errors, browser latency.
Alerts:
- VRAM >850MB (Leak risk on 6GB limit)
- stale_residual >0 (Barge-in failed)
- gate_overlap >0 (Lip sync corrupted)
- renderer latency >30ms (Threatens 40ms/25fps budget)
