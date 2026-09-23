# Stochastic Expression Demo Runbook

## Overview
This runbook covers the operation of the experimental stochastic silent-expression residual.

## Enable / Disable
**Enable**: Set `expression_residual_mode = "stochastic_silent"` in config.
**Disable**: Set `expression_residual_mode = "disabled"`. 

## Startup Verification
Ensure logs print `expression_residual_mode: stochastic_silent`. The startup sequence will fail-closed if the residual checkpoint is missing.

## Expected Metrics
- Generation Latency: ~21-22ms
- VRAM: ~785MB

## Known Limitations
- Not true semantic listening: The avatar does not react emotionally to the specific content being spoken.
- Minor ramp overlap: Extremely rapid barge-ins (<100ms) may feature a 1-2 frame residual ramp-down overlap.

## Rollback Procedure
If the avatar begins twitching or acting erratically, immediately hot-reload the configuration to `expression_residual_mode="disabled"` or restart the worker node with the disabled flag.
