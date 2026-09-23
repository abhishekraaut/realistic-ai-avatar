# Phase 8F: Dataset Breadth, Expression Coverage & Generalization Audit

## 1. Current Dataset Baseline
(See attached `artifacts/phase8f_dataset_audit/dataset_manifest.json` for full manifest).
- **Scale:** 5 Identities, 10 Sequences, 4096 Frames (~164s).
- **Resolution/FPS:** 100% 512x512 @ 25 FPS.
- **Leakage:** Verified clean identity split (Train: 001, 003, 004 | Val: 005 | Test: 002).

## 2. Expression Coverage Audit
**Mouth:** Excellent coverage of standard conversational phonemes (small/medium openings, lip rounding). Near-zero coverage of wide shouting, biting lips, or asymmetrical smirks.
**Head:** Standard conversational yaw (-25° to +25°). Very limited pitch and negligible roll.
**Eyes:** Natural conversational blinking. Almost entirely forward-gaze tracking head yaw. No upward/downward eye rolling.
**Facial Expression:** 95% neutral talking. 5% slight positive (smiling). 0% negative, surprise, or intense concentration.

## 3. Motion-Space Coverage & Audio Audit
- **V6 15D Motion:** PCA 0-2 (Jaw/Mouth) show dense Gaussian distributions across Train/Val/Test. PCA 5-7 (Micro-expressions) show severe sparsity in Validation/Test, confirming that the current model simply defaults to the mean for held-out identities.
- **Audio Coverage:** 100% calm/conversational English. Lacks whispers, shouts, extreme pitch variance, emotional prosody, and multi-lingual phonetics.

## 4. Renderer Identity Generalization Stress Test (V4Eye)
Driving held-out identities (ID_002, ID_005) with their correct audio vs swapped audio.
- **Consistent ID:** Phenomenal texture preservation and geometric consistency.
- **Cross-Condition Swapping (ID_A motion + ID_B reference):** 
  - *Mouth/Eyes:* Animate plausibly.
  - *Boundaries:* Severe tearing and smearing artifacts appear around the neck, shoulders, and hair boundary if ID_A's resting pose differs significantly from ID_B. The U-Net has overfit to the spatial prior of the specific 5 torsos/backgrounds.

## 5. Dataset Gap Analysis
Ranked solely by measured coverage deficiency:
1. **Insufficient Identities (5):** Prevents U-Net from generalizing background/torso decoupling and diverse skin/lighting responses.
2. **Insufficient Lighting/Environment (1 condition):** 100% diffuse frontal. No directional/harsh light robustness.
3. **Insufficient Expression Diversity:** 95% neutral. Restricts PCA bounds from learning genuine emotion.
4. **Insufficient Audio Diversity:** Only calm conversational English.

## 6. Data Expansion Proposal
**Target Composition for Next Expansion Phase (Phase 9):**
- **Identities:** 50-100 diverse identities (balances generalization with RTX 3050 training constraints).
- **Sequences per ID:** 3-5 sequences.
- **Duration:** 1-2 hours total.
- **Resolution:** 512x512 (Keep current bounds to focus on data diversity, not pixel scale).
- **Lighting:** Require multi-environment sourcing (outdoor, directional indoor, mixed color temps).
- **Audio:** Must include expressive, emotional, and multi-lingual speech.

## 7. 30-Minute Eye Stability Closure Test
Validated the frozen eye subsystem via synthetic streaming:
- **Total Blinks:** ~423
- **Deterministic / Stochastic:** ~205 / ~218
- **Suppressed Stochastic:** ~65
- **EAR bounds:** strictly `[0.04, 0.35]`
- **Gaze RMS / Max Excursion:** `0.045` / `0.15` (Clamped safely)
- **Latency Drift:** 0.0ms over 30 minutes. Memory stable.

## 8. Authoritative Runtime Benchmark
*Run strictly via `torch.inference_mode()` with CUDA sync.*
| Scope | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | VRAM (Alloc MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A. V6 Motion | 0.32 | 0.31 | 0.45 | 0.52 | 0.85 | 5.2 |
| B. Eye Scheduler | 0.05 | 0.05 | 0.06 | 0.08 | 0.15 | < 1.0 |
| C. V4Eye Renderer | 10.55 | 10.50 | 12.80 | 15.10 | 18.20 | 43.5 |
| D. Full Generation | 10.92 | 10.86 | 13.31 | 15.70 | 19.20 | 131.0 |
| E. LiveKit Publish | 2.10 | 2.05 | 2.80 | 3.50 | 4.10 | N/A |

## 9. Browser Latency Boundary
- **Instrumentable Timestamps:** Microphone Capture, Deepgram STT, Gemini TTFT, ElevenLabs TTS, Canonical PTS, Generation Pipeline.
- **Browser Receipt:** Instrumentable via WebRTC `rtc.Receiver`.
- **Display/Glass-to-Glass:** **NOT MEASURED.** Awaiting photometric hardware validation.

## 10. Final Decision
**A. Is the dataset large enough for another training phase?**
NO. The 5-identity dataset has reached its absolute representation limit.

**B. What limits generalization?**
The model severely overfits background and torso boundaries, and the PCA space completely collapses on extreme emotional expressions and asynchronous eye saccades due to lack of source representation.

**C. Next Experiment:**
A massive multi-identity (50+) dataset expansion emphasizing environmental, emotional, and phonetic diversity at 512x512. No architectural changes should precede this data expansion.
