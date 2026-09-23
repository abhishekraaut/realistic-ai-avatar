# Health Checks
- `/health`: Liveness probe. Returns 200 OK.
- `/ready`: Readiness probe. Checks GPU availability, Checkpoint loading, and LiveKit WS connectivity.
