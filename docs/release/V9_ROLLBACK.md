# V8 Rollback Guide
If critical defects are identified in V9 in production:
1. Stop the worker nodes via Gateway Draining.
2. Edit `config/production_v9.json` or override via environment flag: `renderer_mode="v8_production"`.
3. Reboot the worker nodes.
The v0.1.0 checkpoint (`neural_renderer_v8_50id_1280x720_best.pt`) remains frozen and accessible on disk for immediate recovery.
