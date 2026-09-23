# Operational Runbook
## 1. High Latency Investigation
**Symptom**: Avatar takes >1.5s to respond.
**Diagnosis**: Check T13 vs T15 metrics. Check LLM TTFT.
**Action**: If Deepgram/Gemini degraded, gracefully inform users. No code restart required.
**Verification**: Check metrics dashboard until TTFT < 300ms.

## 2. LiveKit Outage
**Symptom**: Browser shows frozen video or disconnects.
**Diagnosis**: Check LiveKit server logs for `LIVEKIT_DISCONNECTED`.
**Action**: Restart LiveKit service. Backend publisher automatically re-establishes streams.
**Verification**: Check `/ready` endpoint transitions from FALSE to TRUE.
