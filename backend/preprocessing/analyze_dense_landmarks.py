import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import matplotlib.pyplot as plt

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

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

def analyze_video(video_path):
    print(f"Analyzing {video_path}...")
    detector = get_face_landmarker()
    cap = cv2.VideoCapture(video_path)
    
    landmarks_3d = []
    poses = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = detector.detect(mp_image)
        
        if res.face_landmarks and res.facial_transformation_matrixes:
            lmks = res.face_landmarks[0]
            # Store raw (x, y, z)
            arr = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lmks])
            landmarks_3d.append(arr)
            
            matrix = res.facial_transformation_matrixes[0]
            poses.append(extract_pose(matrix))
            
    cap.release()
    return np.array(landmarks_3d), np.array(poses)

def align_faces(landmarks):
    # landmarks: [T, 478, 3]
    # Simple alignment: Translate nose (index 1) to origin
    nose_tips = landmarks[:, 1:2, :]
    translated = landmarks - nose_tips
    
    # Scale normalization: Distance between outer eyes (33 and 263)
    left_eye = translated[:, 33, :]
    right_eye = translated[:, 263, :]
    eye_dist = np.linalg.norm(right_eye - left_eye, axis=1, keepdims=True)
    
    scaled = translated / (eye_dist[:, np.newaxis] + 1e-6)
    
    # 2D representation (drop z)
    scaled_2d = scaled[:, :, :2]
    return scaled, scaled_2d

def main():
    # Analyze Train Set (for PCA / Variance fitting)
    train_videos = [
        "emotional-rollercoaster.mp4",
        "guess-the-animal.mp4",
        "guess-where.mp4",
        "how-i-ai.mp4"
    ]
    
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets"
    
    all_3d = []
    for v in train_videos:
        l3d, _ = analyze_video(os.path.join(base_dir, v))
        all_3d.append(l3d)
        
    all_3d = np.concatenate(all_3d, axis=0)
    print(f"Extracted {all_3d.shape[0]} training frames.")
    
    aligned_3d, aligned_2d = align_faces(all_3d)
    
    # Flatten
    flat_3d = aligned_3d.reshape(aligned_3d.shape[0], -1)
    flat_2d = aligned_2d.reshape(aligned_2d.shape[0], -1)
    
    print(f"3D Shape: {flat_3d.shape}")
    print(f"2D Shape: {flat_2d.shape}")
    
    # Measure global variance
    var_3d = np.var(flat_3d, axis=0).sum()
    var_2d = np.var(flat_2d, axis=0).sum()
    print(f"Total 3D Variance: {var_3d:.6f}")
    print(f"Total 2D Variance: {var_2d:.6f}")
    
    # Run PCA
    from sklearn.decomposition import PCA
    
    pca_2d = PCA().fit(flat_2d)
    cum_var_2d = np.cumsum(pca_2d.explained_variance_ratio_)
    
    k_90 = np.argmax(cum_var_2d >= 0.90) + 1
    k_95 = np.argmax(cum_var_2d >= 0.95) + 1
    k_99 = np.argmax(cum_var_2d >= 0.99) + 1
    
    print("\nPCA on 2D Aligned Landmarks:")
    print(f"Components for 90% var: {k_90}")
    print(f"Components for 95% var: {k_95}")
    print(f"Components for 99% var: {k_99}")
    
    # Mouth specific variance
    MOUTH_INDICES = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
    mouth_2d = aligned_2d[:, MOUTH_INDICES, :].reshape(aligned_2d.shape[0], -1)
    mouth_var = np.var(mouth_2d, axis=0).sum()
    print(f"\nMouth 2D Variance: {mouth_var:.6f} ({(mouth_var/var_2d)*100:.1f}% of total facial variance)")
    
if __name__ == "__main__":
    main()
