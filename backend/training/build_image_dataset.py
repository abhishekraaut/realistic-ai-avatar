import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from backend.engine.audio_features import StreamingAudioFeatureExtractor
from backend.shared.media_types import MediaTimestamp, AudioChunk
import soundfile as sf
from moviepy import VideoFileClip

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract_valid_indices(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    valid_indices = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (1920, 1080))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        res = detector.detect(mp_image)
        if res.face_landmarks and res.facial_transformation_matrixes:
            valid_indices.append(idx)
        idx += 1
    cap.release()
    return valid_indices, fps

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

def align_contextual_features(n_frames, features, fps, context_size=16):
    valid_indices_mask = []
    frame_duration = 1.0 / fps
    for frame_idx in range(n_frames):
        end_t = (frame_idx / fps) + frame_duration
        valid_indices = [i for i, f in enumerate(features) if f.start_time < end_t]
        if not valid_indices:
            valid_indices_mask.append(False)
        else:
            valid_indices_mask.append(True)
    return valid_indices_mask

def load_frame(video_path, frame_idx, resolution=(512, 512)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None
    frame = cv2.resize(frame, resolution)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (frame / 127.5) - 1.0
    return torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

def build_image_dataset():
    asset_dir = r"public\assets"
    videos = [f for f in os.listdir(asset_dir) if f.endswith(".mp4")]
    videos.sort()
    
    splits = {
        "train": videos[:4],
        "val": [videos[4]] if len(videos) > 4 else [],
        "test": [videos[5]] if len(videos) > 5 else []
    }
    
    detector = get_face_landmarker()
    images = {"train": [], "val": [], "test": []}
    
    for split_name, files in splits.items():
        for f in files:
            v_path = os.path.join(asset_dir, f)
            print(f"Processing {f} for {split_name}...")
            
            valid_indices, fps = extract_valid_indices(v_path, detector)
            tmp_audio = "tmp_audio.wav"
            features, _ = extract_audio_features(v_path, tmp_audio)
            
            valid_mask = align_contextual_features(len(valid_indices), features, fps)
            
            final_indices = [valid_indices[i] for i in range(len(valid_indices)) if valid_mask[i]]
            print(f"{f}: {len(final_indices)} valid frames.")
            
            if len(final_indices) > 0:
                # We also need the identity reference frame for this video.
                # Usually it's just the first valid frame of the video.
                identity_img = load_frame(v_path, final_indices[0])
                
                for idx in final_indices:
                    target_img = load_frame(v_path, idx)
                    images[split_name].append({
                        "identity_frame": identity_img,
                        "target_frame": target_img
                    })
                    
            if os.path.exists(tmp_audio): os.remove(tmp_audio)

    # Save to disk
    torch.save(images, "synthesia_training_data/dataset_v4_images.pt")
    print("Saved dataset_v4_images.pt")
    
    # Print lengths to verify
    for k in ["train", "val", "test"]:
        print(f"{k} images: {len(images[k])}")

if __name__ == "__main__":
    build_image_dataset()
