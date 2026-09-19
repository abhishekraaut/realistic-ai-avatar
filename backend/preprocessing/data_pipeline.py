import cv2
import numpy as np
import json
import os
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class CustomDatasetExtractor:
    def __init__(self, output_dir="synthesia_training_data", task_path="face_landmarker.task"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "motion_maps"), exist_ok=True)
        
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_pose(self, transform_matrix):
        # transform_matrix is 4x4. We can decompose it to Euler angles.
        # R is 3x3 top-left
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
        return [math.degrees(x), math.degrees(y), math.degrees(z)] # pitch, yaw, roll

    def process_actor_footage(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        print(f"Initializing extraction on source video: {video_path}")
        
        targets = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_resized = cv2.resize(frame, (1920, 1080))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = self.detector.detect(mp_image)
            
            jaw_open = 0.0
            lip_pucker = 0.0
            head_pitch = 0.0
            head_yaw = 0.0
            
            if detection_result.face_blendshapes:
                blendshapes = detection_result.face_blendshapes[0]
                for b in blendshapes:
                    if b.category_name == "jawOpen":
                        jaw_open = b.score
                    elif b.category_name == "mouthPucker":
                        lip_pucker = b.score
            
            if detection_result.facial_transformation_matrixes:
                matrix = detection_result.facial_transformation_matrixes[0]
                pitch, yaw, roll = self.extract_pose(matrix)
                # Normalize angles roughly to [-1, 1] for typical ranges (-45 to 45)
                head_pitch = max(min(pitch / 45.0, 1.0), -1.0)
                head_yaw = max(min(yaw / 45.0, 1.0), -1.0)
            
            target_data = {
                "frame_index": frame_idx,
                "jaw_open": jaw_open,
                "lip_pucker": lip_pucker,
                "head_pitch": head_pitch,
                "head_yaw": head_yaw
            }
            targets.append(target_data)
            
            # Save frame
            frame_name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, "frames", frame_name), frame_resized)
            
            # Save meta
            meta_name = f"meta_{frame_idx:06d}.json"
            with open(os.path.join(self.output_dir, "motion_maps", meta_name), "w") as f:
                json.dump(target_data, f)
                
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Successfully processed {frame_idx} video frames...")
                
        # Write all targets to a single file for easy dataset loading
        with open(os.path.join(self.output_dir, "training_targets.json"), "w") as f:
            json.dump(targets, f)
            
        cap.release()
        print(f"Data extraction complete. Processed {frame_idx} frames. Pipeline profile stored in: {self.output_dir}")

if __name__ == "__main__":
    extractor = CustomDatasetExtractor(task_path="face_landmarker.task")
    extractor.process_actor_footage(r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets\how-i-ai.mp4")
