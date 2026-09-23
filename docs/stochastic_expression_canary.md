# Stochastic Expression Canary Rollout

## Purpose
Controlled, percentage-based live validation of the stochastic silent-expression residual (`stochastic_silent`) against the production baseline (`disabled`).

## Rollout States
- `OFF`: 100% baseline (V6+V9).
- `CANARY`: Configurable % (currently 10%) receives `stochastic_silent`.
- `FULL`: 100% receives `stochastic_silent`.

## Assignment Method
Deterministic hash of `session_id`. Ensures users experience consistent behavior across reconnects within a session.

## Monitoring & Acceptance Criteria
- Stale residual = 0 (Must immediately clear on VAD/Barge-in)
- VRAM must remain bounded (<800MB on RTX 3050 6GB)
- Browser presentation boundary must not regress materially (>10ms diff)
- Speech behavior must remain unaltered.

## Rollback
Instant config hot-switch to `disabled`. Drops residual buffer in <10ms.

## Incident Handling
If `speech_gate_overlap_count > 0` or `stale_residual_count > 0` fires an alert, immediately invoke rollback and pull session logs.
