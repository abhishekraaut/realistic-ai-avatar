import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import torch
import argparse

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def load_image_tensor(frame, resolution, cx, cy, crop_size):
    half = crop_size // 2
    h, w = frame.shape[:2]
    
    top = max(0, half - cy)
    bottom = max(0, cy + half - h)
    left = max(0, half - cx)
    right = max(0, cx + half - w)
    
    if top > 0 or bottom > 0 or left > 0 or right > 0:
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        cx += left
        cy += top
        
    crop = frame[cy-half:cy+half, cx-half:cx+half]
    img = cv2.resize(crop, (resolution, resolution))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 127.5) - 1.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()
    
    # Check if task file exists, if not download it
    if not os.path.exists('face_landmarker.task'):
        import urllib.request
        print('Downloading face_landmarker.task...')
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')

    detector = get_face_landmarker()
    v_path = 'synthesia_training_data/raw/ID_002/seq_01/source.mp4'
    cap = cv2.VideoCapture(v_path)
    images = []
    
    print(f'Extracting aligned frames at {args.resolution}x{args.resolution}...')
    
    fixed_cx, fixed_cy = None, None
    crop_size = 1080 
    
    displacements = []
    clipped_frames = 0
    total_frames = 0
    identity_img = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        
        if res.face_blendshapes and res.face_landmarks:
            nose = res.face_landmarks[0][1]
            nx = int(nose.x * frame.shape[1])
            ny = int(nose.y * frame.shape[0])
            
            if fixed_cx is None:
                fixed_cx = nx
                fixed_cy = ny
            
            dx = abs(nx - fixed_cx)
            dy = abs(ny - fixed_cy)
            dist = (dx**2 + dy**2)**0.5
            displacements.append(dist)
            
            # Check if bounding box is outside crop_size
            half = crop_size // 2
            if nx < fixed_cx - half + 100 or nx > fixed_cx + half - 100 or ny < fixed_cy - half + 100 or ny > fixed_cy + half - 100:
                clipped_frames += 1
                
            total_frames += 1
            
            tensor_img = load_image_tensor(frame, args.resolution, fixed_cx, fixed_cy, crop_size)
            if identity_img is None:
                identity_img = tensor_img
            images.append({'identity_frame': identity_img, 'target_frame': tensor_img})
            
    cap.release()
    train_split = int(0.8 * len(images))
    dataset_images = {'train': images[:train_split], 'val': images[train_split:], 'test': []}
    
    out_path = f'synthesia_training_data/dataset_v5_{args.resolution}_images.pt'
    torch.save(dataset_images, out_path)
    
    min_disp = min(displacements)
    max_disp = max(displacements)
    print(f'Saved frames: {len(images)} to {out_path}')
    print(f'Nose Displacement - Min: {min_disp:.2f}px, Max: {max_disp:.2f}px')
    print(f'Potentially clipped frames: {clipped_frames}')
    print(f'Safely contained: {(total_frames - clipped_frames)/total_frames*100:.1f}%')

if __name__ == '__main__':
    build()
