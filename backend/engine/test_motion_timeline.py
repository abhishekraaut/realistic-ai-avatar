import time
import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from engine.motion_timeline import StreamingMotionPredictor
from shared.media_types import AudioChunk, MediaTimestamp

class TestMotionTimeline(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.audio_extractor = StreamingAudioFeatureExtractor(sample_rate=self.sr)
        self.motion_predictor = StreamingMotionPredictor(target_fps=30.0)
        
    def generate_sine_wave(self, freq, duration_sec, amplitude=0.5):
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return waveform.astype(np.float32)
        
    def test_motion_alignment(self):
        duration_sec = 2.0
        audio = self.generate_sine_wave(120, duration_sec) # Voiced speech proxy
        
        # 1. Push through Audio Extractor
        chunk = AudioChunk(
            timestamp=MediaTimestamp(turn_id=1, sample_position=0, sample_rate=self.sr),
            audio_data=audio,
            num_channels=1,
            sample_count=len(audio),
            is_final=True
        )
        audio_features = self.audio_extractor.append(chunk)
        audio_features += self.audio_extractor.flush()
        
        # 2. Push through Motion Predictor
        motion_windows = self.motion_predictor.append_features(audio_features)
        motion_windows += self.motion_predictor.flush()
        
        # Expected frames at 30 FPS for 2.0 seconds = 60 frames
        # Might be 59-61 due to exact float truncation and flush
        self.assertTrue(59 <= len(motion_windows) <= 61)
        
        # Verify deterministic latency and timestamps
        first_frame = motion_windows[0]
        self.assertEqual(first_frame.start_timestamp.turn_id, 1)
        self.assertAlmostEqual(first_frame.start_timestamp.time_seconds, 0.0, places=2)
        self.assertAlmostEqual(first_frame.end_timestamp.time_seconds, 1.0/30.0, places=2)
        
        # Since it was a loud 120Hz sine wave, it should be marked as speaking
        # and jaw_open latent should be > 0
        self.assertTrue(first_frame.is_speaking)
        self.assertTrue(first_frame.motion_latents[0] > 0.0)

    def test_barge_in_cancellation(self):
        # Simulate initial turn
        audio1 = self.generate_sine_wave(120, 0.1)
        chunk1 = AudioChunk(
            timestamp=MediaTimestamp(turn_id=1, sample_position=0, sample_rate=self.sr),
            audio_data=audio1, num_channels=1, sample_count=len(audio1), is_final=False
        )
        f1 = self.audio_extractor.append(chunk1)
        m1 = self.motion_predictor.append_features(f1)
        
        # Cancel turn 1
        self.audio_extractor.cancel_turn(1)
        self.motion_predictor.cancel_turn(1)
        
        self.assertEqual(len(self.motion_predictor.feature_buffer), 0)
        
        # Simulate incoming barge-in audio (turn 2)
        audio2 = self.generate_sine_wave(150, 0.5)
        chunk2 = AudioChunk(
            timestamp=MediaTimestamp(turn_id=2, sample_position=0, sample_rate=self.sr),
            audio_data=audio2, num_channels=1, sample_count=len(audio2), is_final=False
        )
        f2 = self.audio_extractor.append(chunk2)
        m2 = self.motion_predictor.append_features(f2)
        
        # Ensure outputs are strictly bound to turn 2
        for w in m2:
            self.assertEqual(w.start_timestamp.turn_id, 2)

if __name__ == '__main__':
    unittest.main()
