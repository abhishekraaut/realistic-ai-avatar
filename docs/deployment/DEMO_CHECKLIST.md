# Demo Operator Checklist
## Pre-flight
- [ ] GPU Available & Health `/ready` == 200.
- [ ] LiveKit & Browser connected.
- [ ] Microphone permissions granted.
- [ ] Render baseline ~17.5ms.

## During Demo
- [ ] Watch for FPS drops (25 expected).
- [ ] Watch Turn ID monotonicity.
- [ ] Observe clean barge-in flushes.

## Post-flight
- [ ] Verify logs have rotated cleanly without bloating disk.
