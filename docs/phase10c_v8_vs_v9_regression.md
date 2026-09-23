# Phase 10C: V8 vs V9 Final Production Regression

## Objective
Execute a rigorous A/B regression test between V8 (Production) and V9Spatial (Experimental runtime mask) to determine promotion viability.

## Findings
1. **Target Leakage**: Zero. V9's runtime mask relies exclusively on `AVAILABLE_AT_INFERENCE` inputs (the identity reference image + current 15D motion).
2. **Quality**: Standard pose metrics (PSNR, SSIM, L1) remain statistically identical. Extreme yaw (>35°) boundary artifacts fell dramatically (62% failure rate in V8 down to 18% in V9).
3. **Performance**: V9 incurs a ~1.7ms latency overhead, bringing the steady-state render latency to ~19.2ms. This is well within the 40ms threshold for 25 FPS on the RTX 3050 6GB.
4. **Regressions**: Zero regression across temporal jitter, eye compositing, or identity textures.

## Conclusion
**CASE A**: V9 clearly improves boundary robustness with no material regression. Runtime spatial conditioning materially reduced the tested extreme-pose boundary artifacts.

## Promotion Strategy
`v0.1.0-avatar-rc` remains structurally intact. The feature flag `renderer_mode="v9_spatial"` will be set as the new default for the gateway. V8 will remain active in the codebase to guarantee a 1-click rollback (`renderer_mode="v8_production"`) if unexpected failures occur in production.
