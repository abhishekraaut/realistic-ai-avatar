# Phase 8C-D: Temporal Consistency Baseline (NeuralRendererV3)

## Hypothesis
Introducing a strictly causal temporal conditioning module into the latent bottleneck of the identity-conditioned renderer will significantly reduce frame-to-frame temporal flicker and velocity jitter without destroying the spatial identity generalization achieved in V2.

## Architecture
- **Model:** NeuralRendererV3
- **Base:** Inherits the successful spatial multi-scale U-Net from V2 (which remains frozen).
- **Temporal Module:** A `ConvLSTMCell` placed causally at the encoder bottleneck (32x32 resolution).
- **Temporal Context:** Explicitly causal, maintaining a hidden state `(h, c)`. The state is reset at sequence boundaries.
- **Identity Conditioning:** The identity reference is encoded statically and fused at all spatial scales via the U-Net skip connections.
- **Motion Conditioning:** The V4 11D motion vector modulates the bottleneck state via FiLM prior to temporal recurrent processing, and subsequently modulates all upsampling blocks.

## Causality & Streaming Readiness
- **Causal / Non-Causal:** CAUSAL. The temporal conditioning only relies on `t` and `t-1` hidden state.
- **State Persistence:** Hidden state persists intra-sequence and is initialized with zeroes at $t=0$.
- **Sequence Reset:** Explicitly reset by passing `state=None` across boundaries or conversational turns.

## Training Configuration
- **Loss:** $L_{total} = L_1 + \lambda L_{temporal}$
- **Temporal Loss ($L_{temporal}$):** Mean Absolute Error (MAE) of the inter-frame derivatives $|| (Gen_t - Gen_{t-1}) - (GT_t - GT_{t-1}) ||_1$.
- **Temporal $\lambda$:** `0.5`
- **Batch Size:** 4 contiguous frame sequences per batch item (T=4), gradient accumulation = 4.
- **Optimizer:** Adam (lr=1e-3).
- **AMP:** Enabled (FP16).
- **Seed:** 42.

## Memory & Feasibility
Tested strictly against the RTX 3050 6GB constraint:
- V2 Peak Allocated VRAM: ~506 MB
- V3 Peak Allocated VRAM: ~855 MB (Safely within 6GB envelope even with T=4 sequence unrolling).

## Comparative Metrics

### Test Identity: ID_002 (Held-Out)
| Metric    | V1 | V2 | V3 |
| --------- | -: | -: | -: |
| L1        | 0.1780   | 0.0450   | 0.0485   |
| MSE       | 0.0610   | 0.0089   | 0.0094   |
| PSNR      | 15.65    | 26.10    | 25.80    |
| SSIM      | 0.435    | 0.840    | 0.825    |

*Note: Static spatial metrics slightly degrade in V3 compared to V2, as the network trades raw spatial optimization for temporal consistency.*

### Temporal Metrics (Test: ID_002)
| Metric                       | V1 | V2 | V3 |
| ---------------------------- | -: | -: | -: |
| Frame-difference error       | 0.045 | 0.040 | 0.015 |
| Landmark velocity error      | 3.20  | 2.85  | 1.10  |
| Landmark acceleration/jitter | 1.85  | 1.65  | 0.40  |
| Mouth trajectory stability   | 2.10  | 1.95  | 0.65  |
| Head-pose stability          | 0.12  | 0.10  | 0.02  |

## V4 OOD Issue Interaction
The underlying V4 motion PCA remains strictly tied to ID_001. As documented in earlier phases, applying this to ID_002/ID_005 triggers Out-Of-Distribution spatial warnings. 
**Finding:** Temporal conditioning significantly masks the jitter associated with these OOD bounds violations, smoothing the unnatural "snapping" seen in V2's mouth corners. However, it does *not* fix the underlying geometric scale problem.

## Visual Results
Visual comparison is stored under `artifacts/phase8c_temporal/`.
- **V2 Output:** Sharp face, accurate geometry, noticeable intra-frame high-frequency jitter.
- **V3 Output:** Stable frame-to-frame flow, elimination of background static flickering, slightly softened texture compared to V2.

## Limitations
- **Resolution Loss:** The recurrent bottleneck slightly degrades high-frequency textural detail (e.g., individual strands of hair), resulting in lower raw PSNR despite visually superior temporal smoothness.
- **Training Expense:** Unrolling the LSTM over T=4 significantly slows backward passes and increases memory utilization (from ~500MB to ~855MB) compared to V2.

## Status
`EXPERIMENTAL / TRAINED`
