# Runbook
## Start LiveKit
`livekit-server --dev` or standard Docker deployment.
## Start Backend
`python backend/server.py`
## Start Frontend
`npm run start`
## Graceful Shutdown
Ctrl+C triggers clean exit, flushes LiveKit publishers, and drops GPU tensors.
