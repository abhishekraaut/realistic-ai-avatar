# Benchmarking Methodology
- Model Boundary: `T_motion_input_ready` -> `T_renderer_output_ready`
- Browser Boundary: LiveKit publish -> browser receipt -> video presentation -> audio playback.
- Rule: NEVER combine these timing domains.
- Rule: Ensure `measured_vs_estimated` is strictly logged. Fabricated concurrency (testing N sessions on an N-1 capacity worker) is actively rejected by the concurrency harness.
