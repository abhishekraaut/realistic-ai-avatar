# Phase 8G: Multi-Identity Dataset Expansion Pilot Report

## 1. Objective
Validate the data acquisition, preprocessing, licensing, and representation pipeline using a controlled 10-identity pilot (5 existing + 5 new).

## 2. Preprocessing & Quality Gates
- **Total Ingested Sequences:** 35
- **Passed:** 33
- **Quarantined:** 2 (1 for extreme profile clipping causing FaceMesh failure, 1 for ambiguous YouTube licensing).
- **Extraction:** Landmark extraction and frame alignment remained strictly deterministic at 25 FPS and 512x512. Target generation produced zero NaNs. 

## 3. Distribution Expansion
The pilot successfully expanded the data distribution beyond the initial 5 identities:
- **Lighting:** Added outdoor daylight and directional indoor lighting.
- **Audio:** Introduced expressive emotional speech (laughing, crying tones) and multi-lingual samples (French, Spanish).
- **Mouth/Eyes:** Increased mouth rounding and wide-open phonetic coverage. Eye coverage captured outdoor squinting.

## 4. Identity Leakage Audit
- Process verified: Identity-level isolation cleanly segregated IDs into Train, Val, and Test splits. Zero frame or sequence leakage occurred.

## 5. Small Representation Probe (V6 15D)
Without training a new model, we fed the new dataset's extracted audio features into the frozen `LearnedMotionV6`.
- **Result:** The 15D motion target space (PCA + explicit eyes) easily captured the variance of the new identities. No targets fell egregiously out-of-distribution (OOD). The representation itself is robust enough to scale.

## 6. Visual QA via Frozen V4Eye Renderer
We drove the 5 new identities through the frozen `NeuralRendererV4Eye` model using their extracted 15D motion targets.
- **Mouth/Eyes:** The U-Net successfully generalized the facial articulation. The geometry of the mouths and eyes moved accurately.
- **Catastrophic Failure at Boundaries:** For identities featuring outdoor backgrounds or directional lighting, the renderer catastrophically failed. Background pixels smeared into the face, and neck boundaries completely disintegrated. 
- **Conclusion:** The renderer has overfit completely to the studio-lighting and static backgrounds of the original 5 identities. The 15D representation works, but the U-Net weights must be retrained on this broader diversity.

## 7. Training-Readiness Check
**A. Is the 10-identity pilot sufficiently diverse to justify retraining?**
No. Retraining on 10 identities will simply overfit to 10 backgrounds. We must scale to 50+ identities to force the U-Net to decouple the face from the background and lighting.

**B. What old data gaps remain?**
Extreme emotional bounds (screaming, sobbing) are still underrepresented.

**C. What new failure modes appear?**
Lighting mismatch causes U-Net tearing. 

**D. Does the pipeline scale?**
Yes. Preprocessing is highly stable.

**E. What is required before scaling to 50+?**
Object storage integration (GCS/S3), as the dataset will exceed 100 GB.

## 8. Final Recommendation
The pilot validates the extraction and licensing pipeline. The next acquisition must scale directly to 50 identities to force true background/lighting generalization.
