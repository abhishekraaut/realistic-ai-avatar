import os
import sys
import json
import cv2
import math
import torch
import numpy as np
import soundfile as sf
import mediapipe as mp
from moviepy import VideoFileClip
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from shared.media_types import AudioChunk, MediaTimestamp

CONTEXT_SIZE = 16 # 16 hops * 10.6ms = 170.6ms of context

def extract_pose(transform_matrix):
    sy = math.sqrt(transform_matrix[0,0] * transform_matrix[0,0] +  transform_matrix[1,0] * transform_matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(transform_matrix[2,1], transform_matrix[2,2])
        y = math.atan2(-transform_matrix[2,0], sy)
        z = math.atan2(transform_matrix[1,0], transform_matrix[0,0])
    else:
        x = math.atan2(-transform_matrix[1,2], transform_matrix[1,1])
        y = math.atan2(-transform_matrix[2,0], sy)
        z = 0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_video_targets(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    targets = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_resized = cv2.resize(frame, (1920, 1080))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)
        
        jaw_open, lip_pucker, head_pitch, head_yaw = 0.0, 0.0, 0.0, 0.0
        
        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            for b in blendshapes:
                if b.category_name == "jawOpen": jaw_open = b.score
                elif b.category_name == "mouthPucker": lip_pucker = b.score
        
        if detection_result.facial_transformation_matrixes:
            matrix = detection_result.facial_transformation_matrixes[0]
            pitch, yaw, roll = extract_pose(matrix)
            head_pitch = max(min(pitch / 45.0, 1.0), -1.0)
            head_yaw = max(min(yaw / 45.0, 1.0), -1.0)
            
        targets.append({
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps,
            "jaw_open": jaw_open,
            "lip_pucker": lip_pucker,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw
        })
        frame_idx += 1
    
    cap.release()
    return targets, fps

def extract_audio_features(video_path, tmp_audio_path, sr=24000):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(tmp_audio_path, fps=sr, logger=None)
    clip.close()
    
    audio_data, file_sr = sf.read(tmp_audio_path, dtype='float32')
    if len(audio_data.shape) > 1: audio_data = audio_data[:, 0]
    
    extractor = StreamingAudioFeatureExtractor(sample_rate=file_sr, window_size=1024, hop_size=256)
    chunk_size = 4000
    features = []
    
    for i in range(0, len(audio_data), chunk_size):
        chunk_data = audio_data[i:i+chunk_size]
        ts = MediaTimestamp(turn_id=1, sample_position=i, sample_rate=file_sr)
        chunk = AudioChunk(timestamp=ts, audio_data=chunk_data, num_channels=1, sample_count=len(chunk_data), is_final=False)
        features.extend(extractor.append(chunk))
        
    ts = MediaTimestamp(turn_id=1, sample_position=len(audio_data), sample_rate=file_sr)
    chunk = AudioChunk(timestamp=ts, audio_data=np.array([], dtype=np.float32), num_channels=1, sample_count=0, is_final=True)
    features.extend(extractor.append(chunk))
    features.extend(extractor.flush())
    
    return features, len(audio_data) / file_sr

def align_contextual_features(targets, features, fps, context_size=CONTEXT_SIZE):
    X, Y = [], []
    frame_duration = 1.0 / fps
    dropped = 0
    
    # We want a causal context: The audio ending at t_end.
    for t_data in targets:
        end_t = t_data["timestamp"] + frame_duration
        
        # Find the index of the last feature that starts before end_t
        valid_indices = [i for i, f in enumerate(features) if f.start_time < end_t]
        
        if not valid_indices:
            dropped += 1
            continue
            
        last_idx = valid_indices[-1]
        start_idx = last_idx - context_size + 1
        
        context_mels = []
        # Policy: Edge Replication if we don't have enough past context
        for i in range(start_idx, last_idx + 1):
            clamped_i = max(0, i)
            context_mels.append(features[clamped_i].log_mel)
            
        context_mels = np.array(context_mels) # (C, 80)
        
        y_vec = [t_data["jaw_open"], t_data["lip_pucker"], t_data["head_pitch"], t_data["head_yaw"]]
        
        X.append(context_mels)
        Y.append(y_vec)
        
    return np.array(X), np.array(Y), dropped

def build_dataset_v3():
    asset_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets"
    out_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    os.makedirs(out_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(asset_dir) if f.endswith(".mp4")]
    detector = get_face_landmarker()
    
    # Same Splits as V2
    videos.sort()
    splits = {
        "train": videos[:4],
        "val": [videos[4]] if len(videos) > 4 else [],
        "test": [videos[5]] if len(videos) > 5 else []
    }
    
    manifest = []
    dataset = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}
    
    for split_name, files in splits.items():
        for f in files:
            seq_id = os.path.splitext(f)[0]
            v_path = os.path.join(asset_dir, f)
            tmp_audio = os.path.join(out_dir, f"{seq_id}_audio_v3.wav")
            
            print(f"Processing {seq_id} for {split_name}...")
            targets, fps = extract_video_targets(v_path, detector)
            features, audio_dur = extract_audio_features(v_path, tmp_audio)
            
            X_arr, Y_arr, dropped = align_contextual_features(targets, features, fps, CONTEXT_SIZE)
            
            if X_arr.shape[0] > 0:
                dataset[split_name]["X"].append(torch.tensor(X_arr, dtype=torch.float32))
                dataset[split_name]["Y"].append(torch.tensor(Y_arr, dtype=torch.float32))
            
            manifest.append({
                "sequence_id": seq_id,
                "actor_id": "actor_1",
                "source_path": f"public/assets/{f}",
                "frame_count": len(targets),
                "fps": fps,
                "duration": len(targets) / fps,
                "audio_sample_rate": 24000,
                "audio_duration": audio_dur,
                "split": split_name,
                "valid_sample_count": X_arr.shape[0],
                "dropped_samples": dropped,
                "context_size": CONTEXT_SIZE,
                "context_policy": "CAUSAL_EDGE_REPLICATE"
            })
            
            if os.path.exists(tmp_audio):
                os.remove(tmp_audio)
                
    # Combine tensors per split
    final_dataset = {}
    for k in ["train", "val", "test"]:
        if dataset[k]["X"]:
            final_dataset[k] = {
                "X": torch.cat(dataset[k]["X"], dim=0),
                "Y": torch.cat(dataset[k]["Y"], dim=0),
                "sequences": len(dataset[k]["X"])
            }
        else:
            final_dataset[k] = {"X": torch.empty((0, CONTEXT_SIZE, 80)), "Y": torch.empty((0, 4)), "sequences": 0}
            
    dataset_path = os.path.join(out_dir, "dataset_v3.pt")
    torch.save(final_dataset, dataset_path)
    
    with open(os.path.join(out_dir, "dataset_v3_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Dataset V3 generated at {dataset_path}")
    for split in ["train", "val", "test"]:
        print(f"Split {split}: X={final_dataset[split]['X'].shape}, Y={final_dataset[split]['Y'].shape}")

if __name__ == "__main__":
    build_dataset_v3()
