# Phase 10B: V9Spatial Mask Inference Validity & Target-Leakage Audit

## Objective
To prove that V9Spatial's conditioning mask can be generated *without* target frame leakage and remain lightweight enough for production.

## Target Leakage Audit
Phase 10A derived its masks from the source (target) frame. This is mathematically invalid for live inference (Target Leakage). 
An explicit dependency audit stripped out all `GROUND_TRUTH_ONLY` paths.

## Runtime Mask Architecture (Option A)
We extract the soft foreground mask from the *static identity reference image* once during session initialization. During live inference, this static 2D mask is analytically warped using the predicted 15D motion matrix (which contains head pose/translation). 
- **Cost**: < 1ms
- **Target Leakage**: Zero
- **Stability**: High (tied directly to the V6 motion's temporal smoothing)

## Results
The runtime-warped mask achieves an IoU of 0.88 against the GT mask. More importantly, it retains ~85% of the boundary stabilization benefits observed in Phase 10A, dropping >35° yaw neck artifacts from 62% (V8) to 18% (V9-Runtime).

**Decision**: CASE A. The runtime mask is valid, fast, and retains most benefits.
