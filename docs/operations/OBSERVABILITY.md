# Observability Standards
1. **Events**: Use `TURN_START`, `STT_PARTIAL`, `LLM_FIRST_TOKEN`, `RENDER_START`, `TURN_CANCEL`.
2. **Metadata**: Turn ID, timestamp, status code, latency.
3. **Log Rotation**: 50MB max per file, 10 backups retained.
4. **Diagnostic Overlay**: Dev-only toggle for FPS, VRAM, and Turn state.
