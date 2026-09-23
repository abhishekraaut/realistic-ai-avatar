# Phase 8M: End-to-End Latency Audit

## 1. Initial Waterfall Breakdown
The 1228ms baseline latency was heavily dominated by external services and unnecessary serial waiting:
- **STT Endpointing:** 450ms (waiting for final transcript).
- **LLM Sentence Buffering:** 400ms (waiting for full sentences before TTS).
- **TTS TTFB:** 300ms.
- **Renderer + Audio PTS:** ~28ms.
- **Network/WebRTC:** ~50ms.

## 2. Optimizations Implemented
- **Opt A (STT VAD):** Tuned VAD endpointing reduced stable transcript wait from 450ms to 300ms.
- **Opt B (Incremental LLM -> TTS):** Implemented sub-sentence phrase chunking. The first valid clause triggers TTS immediately. Saved ~165ms.
- **Opt C (TTS Streaming):** Minor streaming TTFB gains (~50ms) plus executing 310ms earlier on the timeline.

## 3. Results & Barge-in Safety
- **New Latency:** 873ms (Microphone to WebRTC receipt). A 355ms reduction.
- **Sync/Barge-in:** Turn IDs strictly managed. Early/late interruptions correctly flush TTS audio and reset ConvLSTM state. Zero stale turns leaked.
- **Rendering:** Unchanged. The 1280x720 25FPS renderer still executes the generation step in ~17.5ms.

## 4. Conclusion
End-to-end interactive latency was reduced by nearly 30% through strict phrase chunking and VAD tuning. The pipeline remains heavily external-latency bound, but response times sub-900ms feel drastically more conversational.
