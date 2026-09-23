# Interactive Avatar System (v0.1.0-rc)
## What it does
End-to-end interactive conversational avatar using Deepgram (STT), Gemini (LLM), ElevenLabs (TTS), and a proprietary V6+Neural Renderer generating 1280x720 video at 25FPS over LiveKit WebRTC.

## Requirements
- OS: Windows/Linux
- GPU: NVIDIA RTX 3050 6GB minimum
- Python 3.10+, Node 18+

## Startup
1. `npm install && pip install -r artifacts/release/python_dependencies.txt`
2. `cp .env.example .env` (fill secrets)
3. `python backend/server.py`
4. `npm run dev`

## Known Limitations
See LIMITATIONS.md
