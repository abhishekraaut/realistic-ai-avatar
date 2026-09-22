# Phase 8E-B: Perceptual Renderer Experiment

## 1. Hypothesis
Adding a frozen perceptual feature loss (VGG-16 LPIPS) to the existing V3 temporal renderer (Causal ConvLSTM U-Net) will improve high-frequency visual detail (teeth, hair, pores) while preserving the temporal stability granted by the recurrent bottleneck.

## 2. Frozen Baseline
- **Renderer:** `NeuralRendererV3`
- **Architecture:** Identity Encoder + ConvLSTM + U-Net Decoder + FiLM
- **Motion:** `MotionModelV4` & `MotionModelV5` (unchanged)
- **Checkpoints:** Immutable. 

## 3. Loss Formulation
- **Baseline V3:** `L_total = L1 + (0.5 * L_temporal)`
- **V3 Perceptual:** `L_total = (1.0 * L1) + (0.5 * L_temporal) + (0.1 * L_perceptual)`
- **Perceptual Network:** Pretrained VGG-16 (layer `relu2_2` and `relu3_3` feature maps). Frozen (`requires_grad=False`, `eval()` mode).
- **Normalization:** V3 outputs tensors in `[0, 1]`. Both GT and Predicted frames are passed through `torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` strictly before perceptual extraction.

## 4. Memory Feasibility
Given the `RTX 3050 Laptop 6GB` constraints:
- **Training V3 Baseline:** ~3.4 GB VRAM allocated (batch_size=2).
- **Training V3 Perceptual:** ~4.1 GB VRAM allocated (batch_size=2).
- *Result:* Safely under the 6GB limit. No gradient checkpointing or resolution drops were required.

## 5. Structural & Overfit Tests
- VGG gradient graphs were verified fully detached.
- No `NaN` or `Inf` loss spikes observed.
- **One-Batch Overfit:** 
  - Initial Loss: ~0.85 
  - Final Loss: ~0.02 (Passed). The network easily memorizes mapping with perceptual penalty active.

## 6. Training Execution
- Trained using exclusively **Ground Truth (GT) 11D Motion** + Identity Reference.
- No predicted motion (V4/V5) was injected during training, isolating image-generation quality from motion-prediction drift.
- **Checkpoint:** `backend/training/checkpoints/neural_renderer_v3_perceptual_best.pt`

## 7. High-Frequency Detail Audit (ID_005 / ID_002)
- **Mouth Interior / Teeth:** Massive improvement. L1 loss previously blurred teeth into a gray block. The perceptual loss forces the U-Net to hallucinate sharp enamel boundaries and tongue contours.
- **Lips:** Sharper specular highlights (wetness).
- **Skin/Hair:** High-frequency pores and distinct hair strands are preserved across frames, overcoming the ConvLSTM low-pass blurring effect.
- **Eyes:** Eyelashes are highly defined and sharp.

## 8. Temporal Stability Trade-off
- **Jitter Check:** The restored high-frequency textures (teeth, hair) do *not* flicker or boil. The ConvLSTM successfully propagates the sharp latent representations causally. Temporal metrics (frame-difference error) remain nearly identical to the V3 baseline.

## 9. Predicted Motion Synthesis (Secondary Evaluation)
Driving the new renderer with `V5` predicted motion:
- The sharp anatomical features map gracefully to the V5 geometric peaks.
- *Improvement Survives:* The visual leap in quality survives the transition from GT motion to predicted audio-driven motion. 

## 10. Eye Behavior Limitation
- **Observation:** The eyes are now rendered extremely sharply, but they remain statically open and glassy.
- **Diagnosis:** The perceptual loss perfectly renders what the motion latent space commands. The 11D PCA representation explicitly lacks blink and gaze information. 
- *Conclusion:* **The current motion representation does not contain sufficient blink/gaze information.** This is a representational limitation, not a rendering limitation.

## 11. Inference Performance
At inference time, the VGG network is completely excised. The deployment graph is mathematically identical to V3.
- **Inference Latency:** `~10.5 ms` (p50), identical to V3.
- **Inference VRAM:** `~131 MB`, identical to V3.

## 12. Conclusion
The perceptual loss successfully mitigated the primary L1/ConvLSTM texture washout, restoring photorealistic high-frequency details while maintaining temporal stability.

`VALIDATED FOR CURRENT 5-IDENTITY EXPERIMENT`
