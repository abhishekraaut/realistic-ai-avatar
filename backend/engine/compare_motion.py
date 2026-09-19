import os
import sys
import time
import soundfile as sf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from engine.motion_timeline import StreamingMotionPredictor
from engine.learned_motion_timeline import LearnedStreamingMotionPredictor
from shared.media_types import AudioChunk, MediaTimestamp

def compare_predictors(audio_path, ckpt_path, duration_sec):
    print(f"\n--- Running Comparison for {duration_sec}s sequence ---")
    audio_data, sr = sf.read(audio_path, dtype='float32')
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    num_samples = int(duration_sec * sr)
    audio_data = audio_data[:num_samples]
    
    # 1. Setup extractors and predictors
    extractor = StreamingAudioFeatureExtractor(sample_rate=sr, window_size=1024, hop_size=256)
    heuristic = StreamingMotionPredictor(target_fps=30.0)
    learned = LearnedStreamingMotionPredictor(checkpoint_path=ckpt_path, target_fps=30.0)
    
    chunk_size = 4000
    heuristic_outputs = []
    learned_outputs = []
    
    start_t = time.time()
    
    # Simulate streaming
    for i in range(0, len(audio_data), chunk_size):
        chunk_data = audio_data[i:i+chunk_size]
        ts = MediaTimestamp(turn_id=1, sample_position=i, sample_rate=sr)
        chunk = AudioChunk(timestamp=ts, audio_data=chunk_data, num_channels=1, sample_count=len(chunk_data), is_final=False)
        
        features = extractor.append(chunk)
        
        h_windows = heuristic.append_features(features)
        heuristic_outputs.extend(h_windows)
        
        l_windows = learned.append_features(features)
        learned_outputs.extend(l_windows)
        
    # Flush
    ts = MediaTimestamp(turn_id=1, sample_position=len(audio_data), sample_rate=sr)
    chunk = AudioChunk(timestamp=ts, audio_data=np.array([], dtype=np.float32), num_channels=1, sample_count=0, is_final=True)
    features = extractor.append(chunk)
    features.extend(extractor.flush())
    
    heuristic_outputs.extend(heuristic.append_features(features))
    heuristic_outputs.extend(heuristic.flush())
    
    learned_outputs.extend(learned.append_features(features))
    learned_outputs.extend(learned.flush())
    
    print(f"Total processing time: {time.time() - start_t:.3f}s")
    
    # Analyze Heuristic
    h_jaw = np.array([w.motion_latents[0] for w in heuristic_outputs])
    print(f"HEURISTIC -> Frames: {len(h_jaw)}, Jaw Range: [{np.min(h_jaw):.3f}, {np.max(h_jaw):.3f}], Variance: {np.var(h_jaw):.5f}")
    
    # Analyze Learned
    l_jaw = np.array([w.motion_latents[0] for w in learned_outputs])
    print(f"LEARNED   -> Frames: {len(l_jaw)}, Jaw Range: [{np.min(l_jaw):.3f}, {np.max(l_jaw):.3f}], Variance: {np.var(l_jaw):.5f}")
    
    # Smoothness (mean absolute difference between consecutive frames)
    if len(h_jaw) > 1:
        h_diff = np.mean(np.abs(np.diff(h_jaw)))
        l_diff = np.mean(np.abs(np.diff(l_jaw)))
        print(f"Temporal Smoothness (Mean Diff, lower is smoother):")
        print(f"  Heuristic: {h_diff:.5f}")
        print(f"  Learned  : {l_diff:.5f}")

if __name__ == "__main__":
    audio_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\audio.wav"
    ckpt_path = r"C:\Users\iabhi\Downloads\Avtar-Project\backend\training\checkpoints\learned_motion_v1.pt"
    
    compare_predictors(audio_path, ckpt_path, 2.0)
    compare_predictors(audio_path, ckpt_path, 5.0)
    # the audio is ~8.3s long, so test on the full length
    compare_predictors(audio_path, ckpt_path, 8.3)
