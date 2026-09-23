# Horizontal Scaling Architecture
```mermaid
flowchart TD
    C[Client Browser] -->|WebSocket/WebRTC| R[Session Router / Load Balancer]
    R --> W1[GPU Worker 1]
    R --> W2[GPU Worker 2]
    R --> WN[GPU Worker N]

    subgraph W1 [Worker 1 Node]
    M1[Models & Identity]
    S1[Session A State]
    L1[LiveKit Publisher]
    end

    subgraph W2 [Worker 2 Node]
    M2[Models & Identity]
    S2[Session B State]
    L2[LiveKit Publisher]
    end
```

## Lifecycle
1. `START`: Worker boots and loads models to GPU.
2. `READY`: Worker signals Router via healthcheck.
3. `ACCEPT SESSION`: Router assigns Turn/Session ID.
4. `RUN SESSION`: Active pipeline.
5. `CLEANUP`: ConvLSTM, Eye state, and LiveKit buffers are thoroughly flushed.
