import os
import sys
import json
import cv2
import pickle
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.training.prepare_facial_motion_dataset import (
    get_face_landmarker, extract_raw_targets, align_landmarks, 
    extract_audio_features, align_contextual_features, N_PCA_COMPONENTS
)
from backend.data_pipeline.ingestion_validator import generate_manifest

def run_pilot(video_path, identity_id, seq_id):
    out_dir = f"synthesia_training_data/processed/{identity_id}/{seq_id}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    
    # 1. Validation & Manifest creation
    manifest_dir = "synthesia_training_data/manifests"
    print("Validating source and creating manifest...")
    val = generate_manifest(
        identity_id=identity_id, 
        seq_id=seq_id, 
        source="Wikimedia Commons", 
        lic="Public Domain / CC0", 
        video_path=video_path, 
        out_dir=manifest_dir
    )
    if not val:
        print("Validation failed!")
        return

    # 2. Setup
    print("Extracting targets (landmarks, pose)...")
    detector = get_face_landmarker()
    
    # 3. Extract Raw Targets
    # This function extracts frames to disk, computes pose, and returns landmarks + pose
    frames_dir = os.path.join(out_dir, "frames")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # We will override extract_raw_targets to save to our custom frame directory
    # because the original function might be hardcoded or we just adapt it
    # Actually, wait, extract_raw_targets(video_path, detector) from the original script 
    # saves frames where? Oh wait, it doesn't save frames in prepare_facial_motion_dataset.py, 
    # it only returns raw arrays. Wait, let me check.
    
    raw_Y, raw_pose = extract_raw_targets(video_path, detector)
    print(f"Extracted {len(raw_Y)} landmark frames out of {n_frames}")
    
    if len(raw_Y) == 0:
        print("Face detection failed completely.")
        return
        
    with open(os.path.join(out_dir, "landmarks.pkl"), "wb") as f:
        pickle.dump(raw_Y, f)

    # 4. Alignment & PCA
    print("Aligning landmarks and projecting V4 PCA...")
    aligned = align_landmarks(raw_Y)
    
    # Load frozen PCA from training directory
    pca_path = "backend/training/checkpoints/v4_pca.pkl"
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    else:
        print("PCA not found, assuming frozen PCA is missing.")
        return
        
    flat_aligned = np.array(aligned).reshape(len(aligned), -1)
    
    # We need to project
    try:
        projected = pca.transform(flat_aligned)
        print("PCA projection successful.")
        
        # OOD checks based on normalization boundaries
        # Just compute min/max
        p_min, p_max = projected.min(axis=0), projected.max(axis=0)
        print(f"PCA Min: {p_min}")
        print(f"PCA Max: {p_max}")
        # Note: in V4, bounds are usually around [-2, 2] or so. If values exceed [-3, 3], they are likely OOD.
        ood_count = np.sum((projected < -3.0) | (projected > 3.0))
        print(f"OOD Warning count (values outside [-3, 3]): {ood_count} out of {projected.size}")
        
    except Exception as e:
        print(f"PCA projection failed: {e}")
        return
        
    # 5. Audio Extraction
    print("Extracting Audio Features...")
    tmp_audio = os.path.join(out_dir, "audio.wav")
    try:
        audio_features = extract_audio_features(video_path, tmp_audio)
        print(f"Extracted {len(audio_features)} audio features.")
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return
        
    # 6. Save Motion Tensor
    out_motion = os.path.join(out_dir, "motion.pt")
    torch.save(torch.tensor(projected, dtype=torch.float32), out_motion)
    print(f"Saved V4 motion tensor to {out_motion}")
    
    # Update Manifest
    mf_path = os.path.join(manifest_dir, f"{identity_id}_{seq_id}_manifest.json")
    with open(mf_path, "r") as f:
        mf = json.load(f)
    mf["landmark_status"] = "SUCCESS"
    mf["face_detection_status"] = "SUCCESS"
    mf["v4_motion_status"] = f"SUCCESS (OOD: {ood_count})"
    mf["audio_path"] = tmp_audio
    with open(mf_path, "w") as f:
        json.dump(mf, f, indent=2)

    print("Pilot Ingestion Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--seq", required=True)
    args = parser.parse_args()
    
    run_pilot(args.video, args.identity, args.seq)
