# Disaster Recovery Runbook
## Policy
- Architecture is **RECOVERY READY**, not HIGH AVAILABILITY.
- Active sessions are EPHEMERAL and will terminate on node loss. Clients must create a new session.
- Secrets are injected via secure vault, never committed.
## RTO/RPO
- RTO: ~4 minutes.
- RPO: 0 (ephemeral state). 
## Artifact Restoration
1. `git checkout v0.3.0`
2. Download checkpoints.
3. Run hash verification script.
4. Start Worker.