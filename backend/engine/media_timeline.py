import asyncio
import time
import logging
from typing import Optional, List, Tuple
import numpy as np
from livekit import rtc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.media_types import AudioChunk, MediaTimestamp, TurnContext, MotionWindow
from engine.neural_renderer import RendererProtocol

logger = logging.getLogger("media-timeline")

class CanonicalMediaTimeline:
    """
    Maintains a monotonically increasing media clock based on actual audio samples.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.current_turn_id = 0
        self.current_sample_position = 0
        
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._lock = asyncio.Lock()
        
    async def increment_turn(self):
        """Barge-in / Interruption handler"""
        async with self._lock:
            self.current_turn_id += 1
            self.current_sample_position = 0
            # Flush the queue to prevent stale audio frames from advancing the timeline
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.info(f"[Timeline] Advanced to turn {self.current_turn_id} and flushed stale audio.")

    async def push_audio(self, audio_data: bytes, num_channels: int, sample_count: int, is_final: bool):
        async with self._lock:
            # We assume pushed audio always belongs to the current turn when intercepted
            timestamp = MediaTimestamp(
                turn_id=self.current_turn_id,
                sample_position=self.current_sample_position,
                sample_rate=self.sample_rate
            )
            chunk = AudioChunk(
                timestamp=timestamp,
                audio_data=audio_data,
                num_channels=num_channels,
                sample_count=sample_count,
                is_final=is_final
            )
            self.current_sample_position += sample_count
            self.audio_queue.put_nowait(chunk)
            
    def get_current_time(self) -> float:
        return self.current_sample_position / self.sample_rate if self.sample_rate > 0 else 0.0

class VideoFrameScheduler:
    """
    Consumes MotionWindows from an async queue, interpolates latents for the 
    canonical media PTS, and produces video frames using RendererProtocol.
    """
    def __init__(self, renderer: RendererProtocol, video_source: rtc.VideoSource):
        self.renderer = renderer
        self.video_source = video_source
        self.fps = 30.0
        self._running = False
        self._task = None
        
        self.motion_queue: asyncio.Queue[MotionWindow] = asyncio.Queue(maxsize=300) # Bounded queue (10s at 30fps)
        self.current_turn_id = -1
        
        # State for interpolation and continuous motion
        self.last_motion_window: Optional[MotionWindow] = None
        self.frame_sequence = 0
        
    def start(self):
        self._running = True
        self.frame_sequence = 0
        self._task = asyncio.create_task(self._run_loop())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            
    def cancel_turn(self, turn_id: int):
        if self.current_turn_id == turn_id:
            logger.info(f"[Scheduler] Cancelling turn {turn_id}")
            # Flush queue
            while not self.motion_queue.empty():
                try:
                    self.motion_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.last_motion_window = None
            self.current_turn_id = -1
            self.renderer.cancel_turn(turn_id)
            self.frame_sequence = 0

    async def push_motion(self, mw: MotionWindow):
        if mw.start_timestamp.turn_id != self.current_turn_id:
            self.cancel_turn(self.current_turn_id)
            self.current_turn_id = mw.start_timestamp.turn_id
            self.renderer.initialize_sequence(TurnContext(turn_id=self.current_turn_id))
            
        try:
            await asyncio.wait_for(self.motion_queue.put(mw), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("[Scheduler] Motion queue full! Dropping frame (Backpressure)")

    def _interpolate_latents(self, pts: float) -> list:
        """Linear interpolation of motion latents based on previous state and target PTS."""
        if not self.last_motion_window:
            return [0.0, 0.0, 0.0, 0.0]
            
        # For simplicity in this dummy, if we are slightly ahead of the last window, 
        # we just decay to neutral or hold the last state. Let's hold state for continuity.
        return self.last_motion_window.motion_latents

    async def _run_loop(self):
        logger.info("[Scheduler] Starting video frame scheduler at 30 FPS.")
        frame_duration = 1.0 / self.fps
        
        while self._running:
            # --- SCHEDULER CLOCK ---
            # Wall-clock is ONLY used for physical WebRTC loop pace, NEVER for media PTS
            start_t = time.time()
            
            try:
                # Wait for the next motion window. 
                # In a real environment, we'd buffer to ensure smooth playback.
                mw = await asyncio.wait_for(self.motion_queue.get(), timeout=frame_duration)
                self.last_motion_window = mw
                
                # --- CANONICAL MEDIA CLOCK ---
                # The exact audio-derived media time for this frame
                pts = mw.start_timestamp.time_seconds
                turn_id = mw.start_timestamp.turn_id
                
                # 1. Verify active turn
                if turn_id != self.current_turn_id:
                    continue
                
                # 2. Interpolate motion state (holding last window's exact latents for now)
                latents = self._interpolate_latents(pts)
                
                # 3. Render
                frame = self.renderer.render_motion_window(latents, pts, turn_id)
                
                # 4. Verify turn again before publish
                if turn_id == self.current_turn_id:
                    # Publish to LiveKit
                    lk_frame = rtc.VideoFrame(
                        frame.shape[1], frame.shape[0], 
                        rtc.VideoBufferType.RGBA, 
                        frame.tobytes()
                    )
                    self.video_source.capture_frame(lk_frame)
                    
                    self.frame_sequence += 1
                
            except asyncio.TimeoutError:
                # No motion arrived in time, emit a neutral/idle frame to keep stream alive
                if self.current_turn_id != -1:
                    # Decay latents to neutral
                    latents = [0.0, 0.0, 0.0, 0.0]
                    # Generate a dummy PTS continuing from last known (or just skip)
                    pts = self.frame_sequence / self.fps
                    frame = self.renderer.render_motion_window(latents, pts, self.current_turn_id)
                    
                    lk_frame = rtc.VideoFrame(
                        frame.shape[1], frame.shape[0], 
                        rtc.VideoBufferType.RGBA, 
                        frame.tobytes()
                    )
                    self.video_source.capture_frame(lk_frame)
                    self.frame_sequence += 1

            # Pacing
            elapsed = time.time() - start_t
            await asyncio.sleep(max(0, frame_duration - elapsed))
