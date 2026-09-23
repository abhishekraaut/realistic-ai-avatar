# Architecture

## Production System (v0.3.0-avatar-rc)
- **V6**: Authoritative deterministic audio-driven speech motion.
- **Stochastic Silent-Expression**: Latent VAE generating plausible 15D auxiliary motion during silence, gated by audio RMS.
- **Eye Dynamics**: Independent procedural eye saccade and blink scheduler.
- **V9 Spatial Neural Renderer**: Neural rendering pipeline utilizing runtime-derived spatial masks.
- **WebRTC**: LiveKit-based canonical media transport.

## Limitations
- Silent expression remains synthetic.
- The stochastic residual does not infer semantic/emotional intent.
- Exact GT reconstruction during ambiguous silent periods is not the optimization target.
- Extreme facial-expression fidelity remains bounded by the learned motion representation/data distribution.
- RTX 3050 production capacity remains single-session under the validated capacity contract.
