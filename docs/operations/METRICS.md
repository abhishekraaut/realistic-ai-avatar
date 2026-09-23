# Metrics Definitions
- **Renderer Steady State**: Time to forward pass one 15D frame to 1280x720 (Target: 17.5ms).
- **T15 Presentation Latency**: Microphone onset to physical browser video callback (Target: ~906ms).
- **Turn Cancellation Rate**: Drops/stale media evictions due to barge-in (Expected ~0 under normal conditions, spikes during barge-in are correct behavior).
- **Queue Depth**: Video frame buffer max depth (Target: <=3).
