import os
import sys
import json
import torch
import soundfile as sf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from shared.media_types import AudioChunk, MediaTimestamp

def extract_aligned_features(audio_path, targets_path, output_dir):
    with open(targets_path, "r") as f:
        targets_json = json.load(f)
        
    audio_data, sr = sf.read(audio_path, dtype='float32')
    # If stereo, take left channel
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    print(f"Loaded audio: {audio_data.shape}, sr={sr}")
    
    extractor = StreamingAudioFeatureExtractor(sample_rate=sr, window_size=1024, hop_size=256)
    
    # Process audio through extractor
    # Simulate streaming chunk by chunk
    chunk_size = 4000
    features = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk_data = audio_data[i:i+chunk_size]
        ts = MediaTimestamp(turn_id=1, sample_position=i, sample_rate=sr)
        chunk = AudioChunk(timestamp=ts, audio_data=chunk_data, num_channels=1, sample_count=len(chunk_data), is_final=False)
        windows = extractor.append(chunk)
        features.extend(windows)
        
    ts = MediaTimestamp(turn_id=1, sample_position=len(audio_data), sample_rate=sr)
    chunk = AudioChunk(timestamp=ts, audio_data=np.array([], dtype=np.float32), num_channels=1, sample_count=0, is_final=True)
    features.extend(extractor.append(chunk))
    features.extend(extractor.flush())
    
    print(f"Extracted {len(features)} audio feature windows.")
    
    # Align to 30 FPS
    target_fps = 30.0
    frame_duration = 1.0 / target_fps
    
    X = []
    Y = []
    
    for t_data in targets_json:
        frame_idx = t_data["frame_index"]
        start_t = frame_idx * frame_duration
        end_t = start_t + frame_duration
        
        # Find audio features overlapping this video frame
        window_features = [
            f for f in features
            if start_t <= f.start_time < end_t
        ]
        
        if not window_features:
            # Maybe the video is slightly longer than the audio
            continue
            
        # Average the log_mels for this frame
        mels = np.array([f.log_mel for f in window_features]) # Shape: (N, 80)
        avg_mel = np.mean(mels, axis=0)
        
        # Append target
        y_vec = [
            t_data["jaw_open"],
            t_data["lip_pucker"],
            t_data["head_pitch"],
            t_data["head_yaw"]
        ]
        
        X.append(avg_mel)
        Y.append(y_vec)
        
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    
    print(f"Aligned dataset size: X={X_tensor.shape}, Y={Y_tensor.shape}")
    
    torch.save({"X": X_tensor, "Y": Y_tensor}, os.path.join(output_dir, "dataset.pt"))
    print(f"Dataset saved to {output_dir}/dataset.pt")

if __name__ == "__main__":
    audio_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\audio.wav"
    targets_path = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data\training_targets.json"
    output_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    extract_aligned_features(audio_path, targets_path, output_dir)
