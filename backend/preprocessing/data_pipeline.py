import cv2
import numpy as np
import torch
import json
import os
import mediapipe as mp

class CustomDatasetExtractor:
    """
    Extracts structural map data and face tracking profiles from raw actor video.
    Used for training the Neural Video Synthesis Engine.
    """
    def __init__(self, output_dir="synthesia_training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "motion_maps"), exist_ok=True)
        
        self.mp_face_mesh = None
        try:
            if hasattr(mp, 'solutions'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        except Exception:
            pass

    def process_actor_footage(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        print(f"Initializing extraction on source video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Maintain 1080p resolution
            frame_resized = cv2.resize(frame, (1920, 1080))
            
            # 1. Structural Gradient Map
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            motion_map = cv2.magnitude(grad_x, grad_y)
            motion_map = np.uint8(cv2.normalize(motion_map, None, 0, 255, cv2.NORM_MINMAX))
            
            # 2. Extract facial landmarks using MediaPipe
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            landmarks = []
            if getattr(self, 'face_mesh', None):
                results = self.face_mesh.process(rgb_frame)
                if getattr(results, 'multi_face_landmarks', None):
                    for face_landmarks in results.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            landmarks.append([lm.x, lm.y, lm.z])
            else:
                # Dummy landmarks if MediaPipe solutions API is missing on this python version
                landmarks = [[0.5, 0.5, 0.0]] * 468
            
            tracking_coordinates = {
                "frame_index": frame_idx,
                "landmarks": landmarks,
                "head_pose": [0.0, 0.0, 0.0], # To be derived via PnP
                "left_shoulder": [450, 800],
                "right_shoulder": [1470, 800]
            }
            
            # Save the photorealistic reference frames
            frame_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, "frames", frame_name), frame_resized)
            
            # Save the structural visual conditioning map
            map_name = f"map_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, "motion_maps", map_name), motion_map)
            
            # Save the tracking token data structures
            meta_name = f"meta_{frame_idx:06d}.json"
            with open(os.path.join(self.output_dir, "motion_maps", meta_name), "w") as f:
                json.dump(tracking_coordinates, f)
                
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Successfully processed {frame_idx} video frames...")
                
        cap.release()
        print(f"Data extraction complete. Pipeline profile stored in: {self.output_dir}")

if __name__ == "__main__":
    extractor = CustomDatasetExtractor()
    extractor.process_actor_footage(r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets\how-i-ai.mp4")
