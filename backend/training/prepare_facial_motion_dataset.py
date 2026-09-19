import os
import sys
import json
import cv2
import math
import torch
import pickle
import numpy as np
import soundfile as sf
import mediapipe as mp
from moviepy import VideoFileClip
from sklearn.decomposition import PCA
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.audio_features import StreamingAudioFeatureExtractor
from shared.media_types import AudioChunk, MediaTimestamp

CONTEXT_SIZE = 16
N_PCA_COMPONENTS = 8

def extract_pose(transform_matrix):
    sy = math.sqrt(transform_matrix[0,0] * transform_matrix[0,0] +  transform_matrix[1,0] * transform_matrix[1,0])
    if sy > 1e-6:
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
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_raw_targets(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    targets = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (1920, 1080))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        res = detector.detect(mp_image)
        
        if res.face_landmarks and res.facial_transformation_matrixes:
            lmks = res.face_landmarks[0]
            pts_2d = np.array([[l.x, l.y] for l in lmks]) # (478, 2)
            
            matrix = res.facial_transformation_matrixes[0]
            pose = extract_pose(matrix) # pitch, yaw, roll
            
            targets.append({"pose": pose, "pts_2d": pts_2d})
    cap.release()
    return targets, fps

def align_landmarks(pts_array):
    # pts_array: [N, 478, 2]
    nose_tips = pts_array[:, 1:2, :]
    translated = pts_array - nose_tips
    
    left_eye = translated[:, 33, :]
    right_eye = translated[:, 263, :]
    eye_dist = np.linalg.norm(right_eye - left_eye, axis=1, keepdims=True)
    
    scaled = translated / (eye_dist[:, np.newaxis] + 1e-6)
    return scaled

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

def align_contextual_features(n_frames, features, fps, context_size=CONTEXT_SIZE):
    X = []
    frame_duration = 1.0 / fps
    valid_indices_mask = []
    
    for frame_idx in range(n_frames):
        end_t = (frame_idx / fps) + frame_duration
        valid_indices = [i for i, f in enumerate(features) if f.start_time < end_t]
        
        if not valid_indices:
            valid_indices_mask.append(False)
            continue
            
        last_idx = valid_indices[-1]
        start_idx = last_idx - context_size + 1
        
        context_mels = []
        for i in range(start_idx, last_idx + 1):
            clamped_i = max(0, i)
            context_mels.append(features[clamped_i].log_mel)
            
        X.append(np.array(context_mels))
        valid_indices_mask.append(True)
        
    return np.array(X), valid_indices_mask

def build_dataset_v4():
    asset_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets"
    out_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    os.makedirs(out_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(asset_dir) if f.endswith(".mp4")]
    detector = get_face_landmarker()
    
    videos.sort()
    splits = {
        "train": videos[:4],
        "val": [videos[4]] if len(videos) > 4 else [],
        "test": [videos[5]] if len(videos) > 5 else []
    }
    
    raw_data = {"train": [], "val": [], "test": []}
    
    # Extract ALL raw features first
    for split_name, files in splits.items():
        for f in files:
            v_path = os.path.join(asset_dir, f)
            tmp_audio = os.path.join(out_dir, f"tmp_audio.wav")
            print(f"Extracting {f}...")
            
            targets, fps = extract_raw_targets(v_path, detector)
            features, _ = extract_audio_features(v_path, tmp_audio)
            
            X_audio, valid_mask = align_contextual_features(len(targets), features, fps)
            
            valid_targets = [targets[i] for i in range(len(targets)) if valid_mask[i]]
            
            if X_audio.shape[0] > 0:
                raw_data[split_name].append({
                    "seq_id": f,
                    "X": X_audio,
                    "targets": valid_targets
                })
                
            if os.path.exists(tmp_audio): os.remove(tmp_audio)
            
    # Process Train Targets for PCA
    train_pts = []
    for seq in raw_data["train"]:
        for t in seq["targets"]:
            train_pts.append(t["pts_2d"])
    
    train_pts = np.array(train_pts)
    train_aligned = align_landmarks(train_pts).reshape(train_pts.shape[0], -1) # (N, 478*2)
    
    # Fit PCA
    print(f"Fitting PCA (k={N_PCA_COMPONENTS}) on {train_aligned.shape[0]} train frames...")
    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(train_aligned)
    print(f"Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Save PCA Model
    with open(os.path.join(out_dir, "pca_v4.pkl"), "wb") as f:
        pickle.dump(pca, f)
        
    dataset = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}
    
    # Transform everything
    for split in ["train", "val", "test"]:
        for seq in raw_data[split]:
            pts_array = np.array([t["pts_2d"] for t in seq["targets"]])
            aligned = align_landmarks(pts_array).reshape(pts_array.shape[0], -1)
            
            pca_latents = pca.transform(aligned) # (N, 8)
            
            # Combine Pose + Latents
            # pose is (pitch, yaw, roll) -> roughly scale by 1/45.0
            poses = np.array([t["pose"] for t in seq["targets"]]) / 45.0
            poses = np.clip(poses, -1.0, 1.0)
            
            Y = np.concatenate([poses, pca_latents], axis=1) # (N, 11)
            
            dataset[split]["X"].append(torch.tensor(seq["X"], dtype=torch.float32))
            dataset[split]["Y"].append(torch.tensor(Y, dtype=torch.float32))
            
    final_dataset = {}
    for k in ["train", "val", "test"]:
        if dataset[k]["X"]:
            final_dataset[k] = {
                "X": torch.cat(dataset[k]["X"], dim=0),
                "Y": torch.cat(dataset[k]["Y"], dim=0),
            }
        else:
            final_dataset[k] = {"X": torch.empty((0, CONTEXT_SIZE, 80)), "Y": torch.empty((0, 3 + N_PCA_COMPONENTS))}
            
    torch.save(final_dataset, os.path.join(out_dir, "dataset_v4.pt"))
    print("Dataset V4 saved.")

if __name__ == "__main__":
    build_dataset_v4()
