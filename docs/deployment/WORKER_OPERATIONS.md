# Worker Operations Guide
## 1. Draining for Zero-Downtime Updates
To update a deployment without dropping active users:
1. Signal old workers: `POST /worker/{id}/drain`
2. Spin up new workers with `vN+1`.
3. Gateway will naturally route new users to `vN+1`.
4. Wait for old workers to reach `STOPPED` state, then terminate.

## 2. Heartbeat Monitoring
Gateway expects a heartbeat every 5 seconds. Missing 3 heartbeats marks the worker `UNHEALTHY`.
