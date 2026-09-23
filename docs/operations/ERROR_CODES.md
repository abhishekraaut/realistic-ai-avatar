# Error Taxonomy
- `GPU_UNAVAILABLE`: Severe. Check CUDA/drivers. Restart container.
- `CHECKPOINT_MISSING`: Severe. Verify hashes in `final_checkpoint_manifest.json`.
- `STT_TIMEOUT`: Moderate. Deepgram lag. Recovered by turn flush.
- `LIVEKIT_DISCONNECTED`: Moderate. Network drop. Handled via automatic reconnection loop.
- `STALE_TURN`: Info. Barge-in triggered. Turn completely aborted cleanly.
