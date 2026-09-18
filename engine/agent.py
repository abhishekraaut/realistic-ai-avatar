import asyncio
import logging
import multiprocessing.context
if not hasattr(multiprocessing.context, 'ForkServerContext'):
    multiprocessing.context.ForkServerContext = type('ForkServerContext', (object,), {})

from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, JobRequest, WorkerOptions, cli, llm
from livekit.agents.job import AutoSubscribe
from livekit.agents.voice import Agent
from livekit.plugins import deepgram, openai, elevenlabs, silero, google

load_dotenv()
logger = logging.getLogger("avatar-engine")
logger.setLevel(logging.INFO)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext()

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the user to join the room
    logger.info(f"existing participants: {ctx.room.remote_participants}")
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    import os
    
    agent = Agent(
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")),
        instructions="You are a helpful AI advisor. You are conversing over audio, so keep your responses concise, conversational, and natural."
    )

    from livekit.agents.voice import AgentSession
    session = AgentSession(
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        tts=elevenlabs.TTS(api_key=os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")),
    )

    await session.start(
        agent=agent,
        room=ctx.room
    )
    # The agent introduces itself when the user connects
    await session.say("Hello! I am your AI advisor. How can I help you today?", allow_interruptions=True)

async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
        ),
    )
