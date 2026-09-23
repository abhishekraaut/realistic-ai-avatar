# Session Lifecycle & Draining
- `REQUEST`: Client asks for an avatar session.
- `ADMISSION`: Gateway verifies capacity.
- `RUNNING`: Active conversational loop.
- `CLEANUP`: ConvLSTM, Eye states, and Identity tensors zeroed out.

## Draining
If a deployment update is required, the worker is marked `DRAINING`. 
- No new sessions are admitted.
- The active session continues uninterrupted.
- Once the active session drops, the worker transitions to `STOPPED`.
