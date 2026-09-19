import os
import cv2
import json

def get_media_info():
    asset_dir = r"C:\Users\iabhi\Downloads\Avtar-Project\public\assets"
    files = [f for f in os.listdir(asset_dir) if f.endswith(".mp4")]
    
    inventory = []
    
    for f in files:
        path = os.path.join(asset_dir, f)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open {f}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        inventory.append({
            "sequence_id": os.path.splitext(f)[0],
            "filename": f,
            "path": f"public/assets/{f}",
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        })
        
    print(json.dumps(inventory, indent=2))
    
if __name__ == "__main__":
    get_media_info()
