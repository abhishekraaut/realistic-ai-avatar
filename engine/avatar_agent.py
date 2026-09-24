
import torch
import os
print("WORKER IDENTITY CHECK:")
print("Worker Name: " + os.getenv("WORKER_NAME", "default-worker"))
print("CUDA_VISIBLE_DEVICES: " + str(os.getenv("CUDA_VISIBLE_DEVICES")))
print("Visible CUDA devices = " + str(torch.cuda.device_count()))
if torch.cuda.is_available():
    print("Physical target = GPU " + str(os.getenv("CUDA_VISIBLE_DEVICES")) + " (" + torch.cuda.get_device_name(0) + ")")
else:
    print("No GPU")
import asyncio
import logging
import multiprocessing.context
import time
import os
import sys

if not hasattr(multiprocessing.context, 'ForkServerContext'):
    multiprocessing.context.ForkServerContext = type('ForkServerContext', (object,), {})

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, JobRequest, WorkerOptions, cli, tts
from livekit.agents.job import AutoSubscribe
from livekit.agents.voice import Agent
from livekit.plugins import deepgram, elevenlabs, silero, google

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.shared.media_types import AudioChunk, MediaTimestamp
from backend.engine.media_timeline import CanonicalMediaTimeline, VideoFrameScheduler
from backend.engine.audio_features import StreamingAudioFeatureExtractor
from backend.engine.motion_timeline import StreamingMotionPredictor
from backend.engine.neural_renderer import DiagnosticRenderer

load_dotenv()
logger = logging.getLogger("avatar-engine")
logger.setLevel(logging.INFO)

class SyncStreamWrapper(tts.SynthesizeStream):
    """Wraps the inner TTS stream to extract actual audio frames for the media clock."""
    def __init__(
        self, 
        inner: tts.SynthesizeStream, 
        timeline: CanonicalMediaTimeline,
        feature_extractor: StreamingAudioFeatureExtractor,
        motion_predictor: StreamingMotionPredictor,
        scheduler: VideoFrameScheduler
    ):
        super().__init__(inner._tts)
        self.inner = inner
        self.timeline = timeline
        self.feature_extractor = feature_extractor
        self.motion_predictor = motion_predictor
        self.scheduler = scheduler
        
    def push_text(self, token: str | str):
        self.inner.push_text(token)
        
    def flush(self):
        self.inner.flush()
        
    def end_input(self):
        self.inner.end_input()
        
    async def aclose(self):
        await self.inner.aclose()
        
    async def __anext__(self) -> tts.SynthesizedAudio:
        try:
            chunk = await self.inner.__anext__()
            if getattr(chunk, 'frame', None) is not None:
                frame: rtc.AudioFrame = chunk.frame
                
                # 1. Update Media Timeline Clock
                await self.timeline.push_audio(
                    audio_data=frame.data,
                    num_channels=frame.num_channels,
                    sample_count=frame.samples_per_channel,
                    is_final=chunk.is_final
                )
                
                # 2. Extract Features -> Motion -> Frame Scheduler
                ac = AudioChunk(
                    timestamp=MediaTimestamp(
                        turn_id=self.timeline.current_turn_id,
                        sample_position=self.timeline.current_sample_position - frame.samples_per_channel,
                        sample_rate=self.timeline.sample_rate
                    ),
                    audio_data=frame.data,
                    num_channels=frame.num_channels,
                    sample_count=frame.samples_per_channel,
                    is_final=chunk.is_final
                )
                
                # Data Pipeline processing
                features = self.feature_extractor.append(ac)
                if features:
                    motion_windows = self.motion_predictor.append_features(features)
                    for mw in motion_windows:
                        await self.scheduler.push_motion(mw)
                
            return chunk
        except StopAsyncIteration:
            # Flush pipeline on stop
            features = self.feature_extractor.flush()
            motion_windows = self.motion_predictor.append_features(features)
            motion_windows += self.motion_predictor.flush()
            for mw in motion_windows:
                await self.scheduler.push_motion(mw)
            raise

class SyncTTSWrapper(tts.TTS):
    def __init__(
        self, 
        inner: elevenlabs.TTS, 
        timeline: CanonicalMediaTimeline,
        feature_extractor: StreamingAudioFeatureExtractor,
        motion_predictor: StreamingMotionPredictor,
        scheduler: VideoFrameScheduler
    ):
        super().__init__(
            capabilities=inner.capabilities,
            sample_rate=inner.sample_rate,
            num_channels=inner.num_channels
        )
        self.inner = inner
        self.timeline = timeline
        self.feature_extractor = feature_extractor
        self.motion_predictor = motion_predictor
        self.scheduler = scheduler
        
    def stream(self) -> tts.SynthesizeStream:
        inner_stream = self.inner.stream()
        return SyncStreamWrapper(
            inner_stream, 
            self.timeline, 
            self.feature_extractor, 
            self.motion_predictor, 
            self.scheduler
        )
        
    def synthesize(self, text: str) -> tts.ChunkedStream:
        return self.inner.synthesize(text)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 1. Pipeline Initialization
    timeline = CanonicalMediaTimeline(sample_rate=24000)
    renderer = DiagnosticRenderer()
    video_source = rtc.VideoSource(1920, 1080)
    
    feature_extractor = StreamingAudioFeatureExtractor(sample_rate=24000)
    motion_predictor = StreamingMotionPredictor(target_fps=30.0)
    scheduler = VideoFrameScheduler(renderer, video_source)
    
    video_track = rtc.LocalVideoTrack.create_video_track("avatar-video", video_source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(video_track, options)
    
    scheduler.start()

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting Avatar for participant {participant.identity}")
    
    agent = Agent(
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")),
        instructions="You are a helpful AI advisor."
    )
    
    def on_state_changed(agent: Agent, state):
        state_name = state.name if hasattr(state, 'name') else str(state).split('.')[-1]
        logger.info(f"Avatar State Transition -> {state_name}")
        if state_name == "LISTENING":
            # Barge-in / Interruption
            asyncio.create_task(timeline.increment_turn())
            feature_extractor.cancel_turn(timeline.current_turn_id)
            motion_predictor.cancel_turn(timeline.current_turn_id)
            scheduler.cancel_turn(timeline.current_turn_id)
            
    agent.on("agent_state_changed", on_state_changed)

    # Wire TTS
    base_tts = elevenlabs.TTS(api_key=os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY"))
    sync_tts = SyncTTSWrapper(base_tts, timeline, feature_extractor, motion_predictor, scheduler)

    from livekit.agents.voice import AgentSession
    session = AgentSession(
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        tts=sync_tts,
    )

    await session.start(
        agent=agent,
        room=ctx.room
    )
    
    await session.say("Hello! I am your AI advisor. How can I help you today?", allow_interruptions=True)

async def request_fnc(req: JobRequest) -> None:
    await req.accept()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("WORKER_NAME", "default-worker"),
            port=int(os.getenv("WORKER_PORT", 8081)),
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
        ),
    )
