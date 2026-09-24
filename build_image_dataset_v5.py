import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import torch

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def load_image_tensor(frame, resolution=(256, 256)):
    frame = cv2.resize(frame, resolution)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (frame / 127.5) - 1.0
    return torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

def build():
    detector = get_face_landmarker()
    v_path = 'synthesia_training_data/raw/ID_002/seq_01/source.mp4'
    cap = cv2.VideoCapture(v_path)
    images = []
    identity_img = None
    print('Extracting aligned frames...')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        if res.face_blendshapes:
            tensor_img = load_image_tensor(frame)
            if identity_img is None:
                identity_img = tensor_img
            images.append({'identity_frame': identity_img, 'target_frame': tensor_img})
    cap.release()
    train_split = int(0.8 * len(images))
    dataset_images = {'train': images[:train_split], 'val': images[train_split:], 'test': []}
    torch.save(dataset_images, 'synthesia_training_data/dataset_v5_images.pt')
    print('Saved frames:', len(images))

if __name__ == '__main__':
    build()
