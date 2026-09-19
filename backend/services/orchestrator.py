import asyncio
import numpy as np
import torch
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, JobProcess
import logging
import os
import sys

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.audio_tokenizer import ExpressVoiceEngine
from engine.gesture_diffusion import ExpressAnimateEngine
from engine.neural_renderer import ExpressRenderEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthesia-clone-orchestrator")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    logger.warning("CRITICAL: Dedicated NVIDIA GPU not found. Running on CPU will be extremely slow.")

class SynthesiaCloneAgent:
    def __init__(self):
        logger.info("Initializing Express-2 Neural Engine cluster...")
        self.voice_eng = ExpressVoiceEngine(device=device)
        self.animate_eng = ExpressAnimateEngine(device=device)
        self.render_eng = ExpressRenderEngine(device=device)

    async def run(self, ctx: JobContext):
        logger.info(f"Connecting to WebRTC room: {ctx.room.name}")
        await ctx.connect()

        # Create the custom WebRTC Video Track Source (1080p, 30fps)
        video_source = rtc.VideoSource(1920, 1080)
        video_track = rtc.LocalVideoTrack.create_video_track("avatar-video", video_source)
        
        # Publish the uncompromised stream to the frontend
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        publication = await ctx.room.local_participant.publish_track(video_track, options)
        logger.info(f"Neural video track successfully published: {publication.sid}")

        input_script = "Hello, I am Tym. I am powered by a custom Neural Video Synthesis Engine."

        # Mock tokenizing text
        text_tokens = torch.randint(0, 50000, (1, 20))

        # 1. Voice Engine: Autoregressively predict audio tokens and waveform
        # Real-time streaming would yield tokens sequentially
        audio_waveform = self.voice_eng.generate_audio(text_tokens)
        
        # Dummy audio tokens representing the RVQ tokens for the animation conditioning
        audio_rvq_tokens = torch.randn(1, 10, 512)

        # The core streaming loop
        for i in range(10): # Mocking 10 frames
            # 2. Animate Engine: Predict precise skeletal and facial movements
            motion_latents = self.animate_eng.predict_motion(audio_rvq_tokens)
            
            # Extract a single frame's motion slice
            motion_frame_latent = motion_latents[:, 0, :] # (B, 156)

            # 3. Render Engine: Render photorealistic frames via Diffusion Transformer
            # Run in executor to prevent blocking the async loop
            loop = asyncio.get_running_loop()
            
            with torch.amp.autocast('cuda'):
                rendered_tensor = await loop.run_in_executor(
                    None, self.render_eng.render_frame, motion_frame_latent
                )
            
            # Post-process tensor to uint8 numpy array (H, W, C)
            rendered_tensor = rendered_tensor.squeeze(0).permute(1, 2, 0)
            rendered_tensor = ((rendered_tensor.clamp(-1, 1) + 1) * 127.5).byte()
            rendered_frame = rendered_tensor.cpu().numpy()

            # 4. Convert frame to WebRTC compatible VideoFrame layout (RGBA)
            rgba_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2RGBA) if 'cv2' in sys.modules else np.dstack((rendered_frame, np.full((1080, 1920), 255, dtype=np.uint8)))
            
            lk_frame = rtc.VideoFrame(
                1920, 1080, 
                rtc.VideoBufferType.RGBA, 
                rgba_frame.tobytes()
            )
            
            # Pushes the frame to the user's browser with sub-500ms delivery latency
            video_source.capture_frame(lk_frame)
            
            # Simulate 30 FPS pacing
            await asyncio.sleep(1/30.0)

async def entry_point(ctx: JobContext):
    agent = SynthesiaCloneAgent()
    await agent.run(ctx)

if __name__ == "__main__":
    # Start the local LiveKit worker daemon listening for execution tokens from the frontend
    JobProcess.listen_to_worker(WorkerOptions(entry_point_fnc=entry_point))
