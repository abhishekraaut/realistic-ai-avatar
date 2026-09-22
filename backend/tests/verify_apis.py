import os
import sys
import asyncio
import aiohttp
from dotenv import load_dotenv
from livekit.plugins import deepgram, elevenlabs, google
from livekit.agents.llm import ChatContext

load_dotenv()

async def verify_apis():
    print("\n--- API VALIDATION ---")

    # Gemini - Real Request
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        async with aiohttp.ClientSession() as session:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            payload = {"contents": [{"parts":[{"text": "Hello"}]}]}
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    print("Gemini: PASS - authenticated request succeeded")
                else:
                    text = await resp.text()
                    print(f"Gemini: FAILED - status {resp.status}, {text}")
    except Exception as e:
        print(f"Gemini: FAILED - {type(e).__name__}: {str(e)}")

    # Deepgram - Real minimal authenticated request
    try:
        dg_key = os.getenv("DEEPGRAM_API_KEY")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {dg_key}"}
            ) as resp:
                if resp.status == 200:
                    print("Deepgram: PASS - authenticated request succeeded")
                else:
                    text = await resp.text()
                    print(f"Deepgram: FAILED - status {resp.status}, {text}")
    except Exception as e:
        print(f"Deepgram: FAILED - {type(e).__name__}: {str(e)}")

    # ElevenLabs - Real minimal authenticated request
    try:
        el_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": el_key}
            ) as resp:
                if resp.status == 200:
                    print("ElevenLabs: PASS - authenticated request succeeded")
                else:
                    text = await resp.text()
                    print(f"ElevenLabs: FAILED - status {resp.status}, {text}")
    except Exception as e:
        print(f"ElevenLabs: FAILED - {type(e).__name__}: {str(e)}")

    # LiveKit - Real Auth
    try:
        from livekit import api
        lkapi = api.LiveKitAPI("ws://localhost:7880", os.getenv("LIVEKIT_API_KEY", "devkey"), os.getenv("LIVEKIT_API_SECRET", "secret"))
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        print(f"LiveKit: PASS - authenticated request succeeded")
        await lkapi.aclose()
    except Exception as e:
        print(f"LiveKit: FAILED - {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(verify_apis())
