import os
import urllib.request

task_path = "face_landmarker.task"
if not os.path.exists(task_path):
    print("Downloading face_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, task_path)
    print("Downloaded.")
