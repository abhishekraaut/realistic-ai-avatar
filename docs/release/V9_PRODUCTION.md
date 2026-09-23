# V9 Production Guide
- **Default**: `renderer_mode="v9_spatial"`
- **Architecture**: Injects a dynamically warped foreground mask using the identity reference and predicted 15D motion vectors.
- **Fail-Closed**: If the V9 checkpoint or identity reference mask is missing/corrupted, the application will refuse to boot. It will **not** silently fall back to V8.
