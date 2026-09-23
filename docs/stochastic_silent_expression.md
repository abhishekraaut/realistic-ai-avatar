# Stochastic Silent Expression

## 1. Purpose
Introduces plausible, non-repeating idle motion (cheek, brow, jaw drift) during silent listening intervals to eliminate the unnatural "blank stare."

## 2. Architecture
A lightweight Latent VAE (`z ~ N(mu, sigma)`) sampled into a bounded 15D residual. Gated by audio RMS.
`M_final = V6 + G_silence * R_stochastic`

## 3. Why it exists
Deterministic audio-driven models (V6) decay to a strict neutral state when the audio condition is empty (silence). This creates an uncanny "blank stare" while listening.

## 4. What it does NOT do
It does **not** understand the user. The motion is stochastic and synthetic, not a semantic or emotional reaction to the user's spoken words.

## 5. Configuration
`expression_residual_mode = "disabled"` (Default)
`expression_residual_mode = "stochastic_silent"` (Opt-in Canary)

## 6. Checkpoint validation
The system explicitly fails closed (aborts startup/session) if the requested stochastic checkpoint is missing, corrupted, or has mismatched dimensions. No silent fallback overrides the config.

## 7. Observability
Emits structured metrics per turn: activations, duration, RMS, clamp count, and reset counts.

## 8. Known limitations
The avatar may appear to be reacting "appropriately" by chance. Users must be disclosed that this is stochastic synthetic motion.

## 9. Barge-in edge case
Fixed. VAD interruptions instantly flush the residual buffer rather than waiting for the RMS audio gate to decay, eliminating the 1-2 frame overlap observed in 11E.

## 10. Rollback
Hot-switch configuration to `disabled`. The residual state is cleared in <10ms and V6 resumes exclusive control.

## 11. Demo guidance
Suitable for client demos to showcase conversational naturalness, provided the stochastic nature is transparently disclosed.

## 12. Performance characteristics
Adds ~0.8ms latency and ~35MB VRAM. Fully sustains 25 FPS.

## 13. Privacy considerations
No raw audio, transcripts, or user payloads are logged by the residual observability metrics.
