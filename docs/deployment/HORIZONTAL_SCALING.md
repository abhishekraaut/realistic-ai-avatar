# Horizontal Scaling Architecture (Updated)

```mermaid
flowchart TD
    C[Client Browser] --> G[Stateless Gateway]
    G -- "Health & Capacity" --> W1[GPU Worker 1]
    G -- "Health & Capacity" --> W2[GPU Worker 2]
    
    W1 --> L[LiveKit Cluster]
    W2 --> L
```

## Admission Contract
- `429 Too Many Requests`: Capacity/admission rejection (all workers full).
- `503 Service Unavailable`: Worker/service unhealthy (no workers available).

## Heartbeat Contract
Workers emit heartbeats every 5 seconds. Gateways enforce a 15-second timeout before transitioning a worker to `UNHEALTHY`.
