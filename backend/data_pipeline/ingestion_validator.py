import os
import json
import cv2
import argparse
from dataclasses import dataclass, asdict

@dataclass
class SequenceManifest:
    identity_id: str
    sequence_id: str
    source: str
    source_license: str
    video_path: str
    audio_path: str
    fps: float
    width: int
    height: int
    duration: float
    frame_count: int
    landmark_status: str
    face_detection_status: str
    v4_motion_status: str
    preprocessing_version: str

def validate_raw_video(video_path: str, expected_fps: float = 25.0) -> dict:
    if not os.path.exists(video_path):
        return {"valid": False, "error": "Video file not found"}
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"valid": False, "error": "Cannot open video file (unreadable or corrupted)"}
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if frames <= 0:
        return {"valid": False, "error": "Missing frames or zero length"}
    
    if w < 512 or h < 512:
        return {"valid": False, "error": f"Invalid dimensions {w}x{h}, minimum 512x512"}
        
    # Note: Using strict FPS validation based on A2 audit policy
    if abs(fps - expected_fps) > 0.01:
        return {"valid": False, "error": f"Invalid FPS: {fps:.2f}. Expected: {expected_fps:.2f}"}
        
    return {
        "valid": True,
        "fps": fps,
        "frames": frames,
        "width": w,
        "height": h,
        "duration": frames / fps
    }

def generate_manifest(identity_id: str, seq_id: str, source: str, lic: str, video_path: str, out_dir: str):
    val = validate_raw_video(video_path)
    if not val["valid"]:
        print(f"Validation FAILED for {video_path}: {val['error']}")
        return False
        
    manifest = SequenceManifest(
        identity_id=identity_id,
        sequence_id=seq_id,
        source=source,
        source_license=lic,
        video_path=video_path,
        audio_path="",  # Filled later by audio extractor
        fps=val["fps"],
        width=val["width"],
        height=val["height"],
        duration=val["duration"],
        frame_count=val["frames"],
        landmark_status="PENDING",
        face_detection_status="PENDING",
        v4_motion_status="PENDING",
        preprocessing_version="1.0"
    )
    
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"{identity_id}_{seq_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
        
    print(f"Manifest created successfully at {manifest_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Identity Ingestion Validator")
    parser.add_argument("--video", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--out_dir", default="../../synthesia_training_data/manifests")
    args = parser.parse_args()
    
    generate_manifest(args.identity, args.sequence, args.source, args.license, args.video, args.out_dir)
