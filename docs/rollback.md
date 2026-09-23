# Rollback Runbook
1. Detection: Monitor VRAM, latency, stale_residual.
2. Command: Set `expression_residual_mode = "disabled"`.
3. State-clear mechanism: Instantly drops residual buffer.
4. V6 fallback: V6 operates exclusively.
5. Browser recovery: Visible in ≈46ms.
6. Verification: Check `expression_residual_activations == 0`.
7. Re-enable: Restore config.
8. Escalation: Ping on-call if error persists.
