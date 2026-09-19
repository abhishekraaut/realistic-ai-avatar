import time
import numpy as np
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from shared.media_types import AudioChunk, MediaTimestamp

class TestAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.extractor = StreamingAudioFeatureExtractor(sample_rate=self.sr)
        
    def generate_sine_wave(self, freq, duration_sec, amplitude=0.5):
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        return waveform.astype(np.float32)
        
    def test_sine_pitch_detection(self):
        # 120 Hz sine wave
        freq = 120.0
        audio = self.generate_sine_wave(freq, 0.5) # 500ms
        
        chunk = AudioChunk(
            timestamp=MediaTimestamp(turn_id=1, sample_position=0, sample_rate=self.sr),
            audio_data=audio,
            num_channels=1,
            sample_count=len(audio),
            is_final=True
        )
        
        windows = self.extractor.append(chunk)
        self.assertTrue(len(windows) > 0)
        
        # Test pitch is within tolerance for ZCR proxy
        middle_window = windows[len(windows)//2]
        self.assertAlmostEqual(middle_window.pitch, freq, delta=10.0)
        self.assertTrue(middle_window.voicing)
        self.assertTrue(middle_window.rms_energy > 0.1)

    def test_turn_cancellation(self):
        # Start turn 1
        chunk1 = AudioChunk(
            timestamp=MediaTimestamp(turn_id=1, sample_position=0, sample_rate=self.sr),
            audio_data=self.generate_sine_wave(100, 0.1),
            num_channels=1, sample_count=int(self.sr * 0.1), is_final=False
        )
        w1 = self.extractor.append(chunk1)
        self.assertTrue(len(w1) > 0)
        
        # Cancel turn 1
        self.extractor.cancel_turn(1)
        self.assertEqual(len(self.extractor.audio_buffer), 0)
        
        # Start turn 2
        chunk2 = AudioChunk(
            timestamp=MediaTimestamp(turn_id=2, sample_position=0, sample_rate=self.sr),
            audio_data=self.generate_sine_wave(200, 0.1),
            num_channels=1, sample_count=int(self.sr * 0.1), is_final=False
        )
        w2 = self.extractor.append(chunk2)
        self.assertTrue(len(w2) > 0)
        self.assertEqual(w2[0].turn_id, 2)
        
    def test_realtime_factor(self):
        duration_sec = 2.0
        audio = self.generate_sine_wave(150, duration_sec)
        
        # Break into 50ms chunks (standard WebRTC size roughly)
        chunk_size = int(self.sr * 0.05)
        
        start_time = time.time()
        
        total_windows = 0
        for i in range(0, len(audio), chunk_size):
            chunk_data = audio[i:i+chunk_size]
            chunk = AudioChunk(
                timestamp=MediaTimestamp(turn_id=3, sample_position=i, sample_rate=self.sr),
                audio_data=chunk_data,
                num_channels=1,
                sample_count=len(chunk_data),
                is_final=False
            )
            windows = self.extractor.append(chunk)
            total_windows += len(windows)
            
        windows += self.extractor.flush()
        
        elapsed_time = time.time() - start_time
        rtf = elapsed_time / duration_sec
        
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Audio Duration: {duration_sec}s")
        print(f"Processing Time: {elapsed_time:.4f}s")
        print(f"Real-Time Factor (RTF): {rtf:.4f} (lower is better, <1.0 means realtime)")
        print(f"Total Feature Windows: {total_windows}")
        
        self.assertTrue(rtf < 1.0, "Feature extraction is not keeping up with realtime!")

if __name__ == '__main__':
    unittest.main()
