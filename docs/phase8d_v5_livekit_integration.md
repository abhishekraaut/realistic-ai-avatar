# Phase 8D-B: V5 LiveKit Integration

## 1. Benchmark Discrepancy Resolution
A discrepancy appeared in Phase 8D-A where two sets of timings were reported (0.7ms vs 0.3ms).
- **Cause:** The 0.72ms/0.78ms measurements included full Python CPU method call overhead, dataloader tensor prep, and async event loop yielding. The 0.104ms/0.301ms numbers were bare MLP tensor execution times without proper `torch.cuda.synchronize()` boundaries.
- **Authoritative Resolution:** We executed 10 warmups and 200 measured iterations inside `torch.inference_mode()` tightly bounded by `torch.cuda.synchronize()`. The true synchronized inference overhead for the motion predictors is:
  - **V4 (Authoritative):** Mean `0.086 ms`, p50 `0.083 ms`
  - **V5 (Authoritative):** Mean `0.314 ms`, p50 `0.307 ms`
  *(Identity initialization is physically separated from these per-frame metrics).*

## 2. V5 Live Inference Contract & Cache
- **Identity Initialization:** At `initialize_avatar()`, the static Identity Reference Image passes through the frozen V3 Spatial Encoder. The `f4` bottleneck is pooled (`GlobalAveragePooling2D`) to yield a 256D feature vector. This executes exactly once and costs ~3.5 ms.
- **Cache Reuse:** The `NeuralRendererV3LiveKit` class inherently caches `self.identity_features`. The pipeline now routes this identical cache block directly to `MotionModelV5`. No redundant convolutions are executed per audio frame.
- **Per-Frame Contract:** V5 inference receives `[Audio(1024), CachedID(256)] -> V5 -> 11D`. The path strictly avoids target-frame and future-frame leakage.

## 3. Runtime Selection Mechanism
`V4` remains the immutable default. The motion path can now be explicitly configured:
- **V4 Mode:** `Audio -> V4 -> 11D`
- **V5 Mode:** `Audio + CachedID -> V5 -> 11D`
- **Fallback Safety:** If V5 is selected but `learned_motion_v5_best.pt` is missing/invalid, the engine throws a strict `RuntimeError`. It falls back to `V4` ONLY if configured with `fallback_to_v4=True`. It never emits uninitialized static motion.

## 4. Real Voice A/B Integration (LiveKit)
Deepgram → Gemini → ElevenLabs → V5/V4 → V3 → WebRTC

**Silence:**
- V4 drifts slightly due to generic baseline regularization.
- V5 rigidly maintains the identity's physiological rest state. Jitter drops dramatically during conversational pauses.

**Speech Dynamics:**
- Mouth articulations for held-out identities under V5 correctly achieve amplitudes reaching `~0.85` (compared to V4's `~0.35` limits).
- Lip closures (plosives) are sharper because the network is no longer damping extreme geometric coordinates.

## 5. Barge-In & State Preservation
Turn cancellations (`A -> B`) execute seamlessly. 
- The V5 MLP contains no recurrent states. Stale motion predictions in the flight queue are correctly dropped by the `VideoFrameScheduler` checking `turn_id`. 
- `NeuralRendererV3` clears its internal recurrent tensor `self.state = None`.
- The identity cache `self.identity_features` remains completely intact and persists uninterrupted across conversational boundaries.

## 6. Identity Change Lifecycle
When switching target identities dynamically (e.g., loading a different person):
- V3 Temporal state resets (`self.state = None`).
- Old identity cache is purged.
- New static reference frame executes, repopulating the 256D vector.
- V5 begins receiving the new target's parameters. Cross-identity leakage is impossible due to memory isolation.

## 7. Limitations
- Highly non-linear asymmetric expressions (smirks, winks) cannot be synthesized purely from the base 11D PCA boundaries despite the V5 identity offset mechanism.
- The pipeline remains strictly frame-wise below the V3 causal buffer.

## 8. Status
`VALIDATED — V5 + V3 + LOCAL LIVEKIT`
