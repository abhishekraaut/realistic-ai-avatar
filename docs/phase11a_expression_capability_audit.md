# Phase 11A: Advanced Facial Expression Capability Audit

## Objective
Audit the frozen V6 + V9 `v0.2.0-avatar-rc` production system to characterize its capacity to reconstruct diverse observable facial expressions, identify the root causes of expression limitations (data, representation, or audio-conditioning), and recommend the next architectural experiment.

## Findings
1. **Viseme/Lip-Sync**: `CURRENT V6 SUFFICIENT`. Basic phonemic articulation is strongly audio-correlated and V6 models it with high fidelity.
2. **Asymmetry & Extremes**: The deterministic L1 loss heavily penalizes incorrect guesses on ambiguous features (e.g., asymmetric smirks), forcing the 15D prediction to "regress to the mean" (a symmetric, neutral output).
3. **Silent Expressions**: V6 is fundamentally **AUDIO-LIMITED**. During silent periods where spontaneous expressions (smiles, brow raises) naturally occur, the audio-conditioning vector is empty. The deterministic network inevitably predicts a neutral resting state.

## Classification
- Neutral: `SUPPORTED`
- Smile (Audio-driven): `SUPPORTED`
- Asymmetric Mouth: `REPRESENTATION-LIMITED` (Averaged out by L1)
- Silent Smile / Brow Raise: `AUDIO-LIMITED` (No driving signal)
- Anger / Shouting: `DATA-LIMITED` (Sparse examples)

## Recommendation
To solve the "blank stare" during silence and restore high-frequency asymmetric expressions, the deterministic V6 must be augmented. The data overwhelmingly points to **Option D: A learned stochastic expression process**. A probabilistic model (e.g., Diffusion or VAE latent sampling) would allow the avatar to sample spontaneous, non-audio-correlated expressions during silence rather than collapsing to the mean.
