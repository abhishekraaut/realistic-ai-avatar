# Gateway Architecture
The Production Gateway operates as a stateless HTTP/WS router handling admission control.

## Flow
1. Client connects to Gateway.
2. Gateway queries Worker Health (`active_sessions < max_sessions`).
3. If capacity exists, `SESSION_ADMITTED` and routed.
4. If full, `SESSION_REJECTED` with a safe 429/503 response ("Avatar worker is busy").
