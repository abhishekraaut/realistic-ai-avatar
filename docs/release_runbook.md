# Production Release Runbook v0.3.0

## Emergency Rollback
Set `expression_residual_mode = "disabled"`. 
Expected behavior: Clears residual state in ≈4ms. Returns avatar to V6-exclusive motion (v0.2.0 baseline) visible in browser in ≈46ms.

## Observability
Check dashboard for:
- `stale_residual_count == 0`
- `gate_overlap_count == 0`
- `vram_allocated_mb < 850`
