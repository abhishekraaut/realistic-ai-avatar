import time
import numpy as np
import sys
import os
import unittest
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from engine.motion_timeline import StreamingMotionPredictor
from engine.neural_renderer import DiagnosticRenderer
from shared.media_types import AudioChunk, MediaTimestamp, TurnContext

class DummyVideoSource:
    def __init__(self):
        self.frames_captured = 0
        self.timestamps = []
        
    def capture_frame(self, lk_frame):
        self.frames_captured += 1

# Mock the Scheduler to bypass LiveKit `asyncio` locks in a simple synchronous loop for testing
class OfflineVideoFrameScheduler:
    def __init__(self, renderer):
        self.renderer = renderer
        self.fps = 30.0
        self.current_turn_id = -1
        self.frame_sequence = 0
        self.frames_rendered = 0
        
    def push_motion(self, mw):
        if mw.start_timestamp.turn_id != self.current_turn_id:
            self.current_turn_id = mw.start_timestamp.turn_id
            self.renderer.initialize_sequence(TurnContext(turn_id=self.current_turn_id))
            self.frame_sequence = 0
            
        pts = mw.start_timestamp.time_seconds
        
        # Render Frame
        frame = self.renderer.render_motion_window(mw.motion_latents, pts, mw.start_timestamp.turn_id)
        self.frames_rendered += 1
        self.frame_sequence += 1

class TestE2EPipeline(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.feature_extractor = StreamingAudioFeatureExtractor(sample_rate=self.sr)
        self.motion_predictor = StreamingMotionPredictor(target_fps=30.0)
        self.renderer = DiagnosticRenderer(width=640, height=480) # smaller for fast test
        self.scheduler = OfflineVideoFrameScheduler(self.renderer)
        
    def generate_sine_wave(self, freq, duration_sec, amplitude=0.5):
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return waveform.astype(np.float32)

    def process_audio(self, duration_sec, turn_id=1):
        audio = self.generate_sine_wave(120, duration_sec)
        chunk_size = int(self.sr * 0.05)
        
        start_t = time.time()
        
        first_audio_t = None
        first_motion_t = None
        first_frame_t = None
        
        for i in range(0, len(audio), chunk_size):
            chunk_data = audio[i:i+chunk_size]
            ac = AudioChunk(
                timestamp=MediaTimestamp(turn_id=turn_id, sample_position=i, sample_rate=self.sr),
                audio_data=chunk_data,
                num_channels=1,
                sample_count=len(chunk_data),
                is_final=False
            )
            
            if first_audio_t is None:
                first_audio_t = time.time()
            
            features = self.feature_extractor.append(ac)
            if features:
                motion_windows = self.motion_predictor.append_features(features)
                if motion_windows and first_motion_t is None:
                    first_motion_t = time.time()
                    
                for mw in motion_windows:
                    self.scheduler.push_motion(mw)
                    if first_frame_t is None:
                        first_frame_t = time.time()
        
        # Flush
        features = self.feature_extractor.flush()
        motion_windows = self.motion_predictor.append_features(features)
        motion_windows += self.motion_predictor.flush()
        for mw in motion_windows:
            self.scheduler.push_motion(mw)
            
        total_time = time.time() - start_t
        return {
            "duration": duration_sec,
            "processing_time": total_time,
            "frames": self.scheduler.frames_rendered,
            "first_audio": first_audio_t,
            "first_motion": first_motion_t,
            "first_frame": first_frame_t
        }

    def test_2_second_stream(self):
        metrics = self.process_audio(2.0)
        self.assertTrue(58 <= metrics["frames"] <= 62) # ~60 frames
        
        print("\n--- 2s STREAM METRICS ---")
        print(f"Frames Rendered: {metrics['frames']}")
        print(f"Processing Time: {metrics['processing_time']:.4f}s")
        print(f"Effective FPS: {metrics['frames'] / metrics['processing_time']:.2f}")
        print(f"Audio->Motion Latency: {(metrics['first_motion'] - metrics['first_audio'])*1000:.1f}ms")
        print(f"Audio->Frame Latency: {(metrics['first_frame'] - metrics['first_audio'])*1000:.1f}ms")

    def test_5_second_stream(self):
        metrics = self.process_audio(5.0)
        self.assertTrue(148 <= metrics["frames"] <= 152) # ~150 frames
        print("\n--- 5s STREAM METRICS ---")
        print(f"Frames Rendered: {metrics['frames']}")
        print(f"Processing Time: {metrics['processing_time']:.4f}s")

    def test_15_second_stream(self):
        metrics = self.process_audio(15.0)
        self.assertTrue(448 <= metrics["frames"] <= 452) # ~450 frames
        print("\n--- 15s STREAM METRICS ---")
        print(f"Frames Rendered: {metrics['frames']}")
        print(f"Processing Time: {metrics['processing_time']:.4f}s")

if __name__ == '__main__':
    unittest.main()
