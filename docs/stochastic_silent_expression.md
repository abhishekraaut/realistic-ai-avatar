# Stochastic Silent Expression

## 1. Purpose
Generates plausible silent/listening behavior during conversational pauses.

## 2. Architecture
Residual latent sampling. `M_final = V6 + G_silence * R_stochastic`

## 3. What it does NOT do
It does **NOT**:
- understand emotional state
- infer user intent from silence
- semantically interpret the user's words
- reconstruct the exact ground-truth silent expression

## 4. Configuration & Rollout
Supported tiers: `OFF`, `CANARY_10`, `CANARY_25`, `CANARY_50`, `FULL`.
Default production mode for v0.3.0 is `FULL`.

## 5. Rollback
Emergency rollback is `expression_residual_mode = "disabled"`, restoring V6-exclusive motion.
