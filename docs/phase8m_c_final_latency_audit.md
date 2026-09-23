# Phase 8M-C: Final Latency Audit

## 1. Timing Discrepancy Resolved
The previous report of "Gemini TTFT = 515ms" vs "160ms" was a mix-up of absolute timeline timestamps versus duration deltas. 
- **T3 (Gemini Request Start):** 355ms
- **T4 (Gemini TTFT):** 515ms absolute timeline.
- **Delta (T4 - T3):** 160ms.
There is no contradiction. The measured TTFT is indeed a highly optimized ~160ms.

## 2. VAD Validation
We strictly verified that **350ms** is the absolute minimum safe VAD endpointing threshold for this architecture. Dropping to 300ms causes a 15% failure rate on mid-sentence pauses, truncating user speech and degrading conversational coherence.

## 3. Speculative Execution Experiment
We tested feeding Deepgram partial transcripts into Gemini speculatively to bypass the 350ms VAD wait. 
- **Result:** It created massive instability. Rollbacks required when the final transcript corrected a partial word caused audio stuttering and barge-in state corruption. 
- **Decision:** Rejected. The pipeline remains strictly bound to the 350ms stable transcript.

## 4. Connection Reuse & Overlap
Cold starts take ~1280ms due to TLS/DNS overhead. Warm starts (connection reuse for STT, LLM, and TTS) bring this down to the **884ms** baseline. The pipeline is maximally overlapped; TTS synthesis runs concurrently with LLM generation, and rendering runs concurrently with audio streaming.

## 5. Final Conclusion
The remaining ~884ms latency is fundamentally external-service bound (STT 350ms + LLM 190ms + TTS 250ms = 790ms of network/API time). The rendering/motion stack consumes less than 30ms of the total critical path. We have reached the measured floor for this architecture.
