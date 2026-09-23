# Horizontal Scaling Architecture

```mermaid
flowchart TD
    C[Client] --> G[Gateway / Router]
    G -- "Health/Heartbeat" --> W1[Worker 1: RTX 3050]
    G -- "Health/Heartbeat" --> W2[Worker 2: RTX 3050]
    G -- "Health/Heartbeat" --> WN[Worker N: RTX 4090]
    
    W1 --> L[LiveKit Cluster]
    W2 --> L
    WN --> L
```

## Worker Contract
Each worker strictly isolates `active_sessions`. A worker broadcasts its `max_capacity` (e.g., 1 for RTX 3050) and its current `active_sessions`. The gateway routes purely on this deterministic mathematical availability, never on ambiguous queue states.

## Backpressure
If no worker is `READY`, the Gateway returns an explicit HTTP 429/503 response. The system explicitly avoids infinite queuing.
