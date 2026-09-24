import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pickle
import glob
import soundfile as sf
from moviepy import VideoFileClip

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def extract():
    detector = get_face_landmarker()
    videos = glob.glob('synthesia_training_data/raw/**/*.mp4', recursive=True)
    
    dataset = []
    
    for v_path in videos:
        print(f'Processing {v_path}...')
        seq_id = os.path.basename(os.path.dirname(os.path.dirname(v_path))) + '_' + os.path.basename(os.path.dirname(v_path))
        
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        blendshapes = []
        feature_names = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)
            if res.face_blendshapes:
                if feature_names is None:
                    feature_names = [c.category_name for c in res.face_blendshapes[0]]
                scores = [c.score for c in res.face_blendshapes[0]]
                blendshapes.append(scores)
        
        blendshapes = np.array(blendshapes)
        dataset.append({'seq_id': seq_id, 'blendshapes': blendshapes, 'fps': fps})
        print(f'Extracted {blendshapes.shape} from {v_path}')
    
    with open('dataset_v5_raw.pkl', 'wb') as f:
        pickle.dump({'data': dataset, 'feature_names': feature_names}, f)
    print('Extraction saved to dataset_v5_raw.pkl')

if __name__ == '__main__':
    extract()
