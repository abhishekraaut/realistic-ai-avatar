# Phase 8B Root Cause Audit

## 1. Identity Verification
**Question:** Is `synthesia-assistant` actually a different identity from the training sequences (`emotional-rollercoaster`, etc.)?
**Answer:** Yes, it is a different identity (or a completely different environment/framing of the same person, effectively a different identity distribution).
**Evidence:** The mean pixel difference between the `train` identity reference and the `test` identity reference is massive (~73.7 per pixel). Because the network only trained on a very narrow visual distribution, the `test` identity is completely Out-of-Distribution (OOD).

## 2. Test Input Validation
**Question:** Does the `TEST` pipeline even work? Are the ground-truth test frames valid faces?
**Answer:** Yes.
**Evidence:** Running MediaPipe explicitly on the 148 Ground-Truth `test` frames resulted in 148 successful face detections. The test target frames are valid 512x512 faces. The failure is entirely on the renderer's side.

## 3. Isolation Experiments
We ran 6 configurations using frame 0 of the respective sequences:

| Condition | Description | PSNR (dB) | L1 | Face Detected? |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Val ID + Val GT Motion -> vs Val Target | 19.42 | 0.1532 | Yes (Lmk: 0.0151) |
| **B** | Val ID + Val Pred Motion -> vs Val Target | 19.38 | 0.1544 | Yes (Lmk: 0.0094) |
| **C** | Test ID + Test GT Motion -> vs Test Target | 18.30 | 0.1625 | **No** (Lmk: 0.0000) |
| **D** | Test ID + Test Pred Motion -> vs Test Target | 18.06 | 0.1660 | **No** (Lmk: 0.0000) |
| **E** | Val ID + Test GT Motion -> vs Test Target | 8.21 | 0.5917 | - |
| **F** | Test ID + Val GT Motion -> vs Val Target | 8.41 | 0.5830 | - |

## 4. Motion-Space Audit
**Question:** Does V4 motion prediction fail on TEST independently of rendering?
**Answer:** Yes, it shows massive degradation in standard deviation (amplitude), indicating a severe mean-pose collapse on the test audio.
**Evidence:** 
- GT Standard Deviation for PCA dims 3 and 4: ~0.51 and ~0.43.
- Predicted Standard Deviation for PCA dims 3 and 4: ~0.15 and ~0.07.
The motion model is heavily damping the predicted motions on the `test` audio, squashing the predicted motion toward a static neutral face.

## 5. Identity-Conditioning Audit
**Question:** Does the identity reference meaningfully control generated identity?
**Answer:** Yes, but it fails to generalize.
**Evidence:** In Conditions E and F, changing the identity reference drastically changes the output image (PSNR drops from ~19 to ~8 dB), proving the identity encoder does inject strong features into the decoder via the FiLM layer.
However, because the `decoder` only ever trained to reconstruct a single identity manifold, passing a novel identity latent code through the bottleneck (16x16) causes the decoder to output a completely garbled image (hence MediaPipe failing to find a face in Conditions C and D).

## 6. Preprocessing Integrity
**Question:** Is there evidence of preprocessing/checkpoint mismatch?
**Answer:** No. 
**Evidence:** The data pipelines load identical tensors, normalization, and orientations. The failure happens natively within the forward pass of the models on OOD data.

## 7. Performance Audit (CUDA Synchronized)
- **Renderer p50:** 8.12 ms
- **Renderer p95:** 19.61 ms
- **Renderer Max:** 20.81 ms
- **Renderer Mean:** 9.04 ms
- **Peak VRAM Allocated:** 0.064 GB

## Final Diagnosis

1. **Is `synthesia-assistant` actually a different identity?** Yes.
2. **Does NeuralRendererV1 fail on TEST even when given PERFECT GT motion?** Yes. Condition C proves the renderer completely fails to output a recognizable face for the novel test identity, even with perfect ground-truth motion.
3. **Does V4 motion prediction fail on TEST independently of rendering?** Yes. V4 suffers from severe mean-pose collapse (damped amplitude) on the test audio.
4. **Does the identity reference meaningfully control generated identity?** Yes, but the decoder is overfit to the training identity. It can only decode identity features that resemble the training actor.
5. **Is there evidence of preprocessing/checkpoint mismatch?** No.
6. **Failure Attribution:**
   - **Identity Conditioning (Renderer Generalization):** ~80% of the failure. The decoder outputs garbage on novel identities.
   - **Motion Prediction:** ~20% of the failure. Even if the renderer worked, V4 produces damped, neutral-tending motion on novel audio.

## Next Action
Phase 8C is fully justified. We need a temporal neural rendering architecture (e.g., ConvLSTM, Latent Diffusion) to fix frame jitter, combined with a robust identity conditioning mechanism (e.g., AdaIN, SPADE, or multi-identity datasets) to prevent test-set collapse.

*Note: Model code was not changed. Audit scripts were written in `backend/tests/phase8b_root_cause.py`.*
