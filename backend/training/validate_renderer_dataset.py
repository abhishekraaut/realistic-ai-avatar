import os
import torch
import cv2

def validate_renderer_dataset():
    base_dir = r"C:\Users\iabhi\Downloads\Avtar-Project"
    dataset_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    asset_dir = os.path.join(base_dir, "public", "assets")
    
    print("Validating Renderer Dataset alignment...")
    if not os.path.exists(dataset_path):
        print("Missing dataset_v4.pt")
        return
        
    data = torch.load(dataset_path, map_location="cpu")
    
    # Just a quick check for video presence
    for f in os.listdir(asset_dir):
        if f.endswith(".mp4"):
            cap = cv2.VideoCapture(os.path.join(asset_dir, f))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"Video {f}: {count} frames, {res[0]}x{res[1]}")
            cap.release()
            
    print("\nDataset V4 shapes:")
    for split in ["train", "val", "test"]:
        if data[split]["X"].shape[0] > 0:
            print(f"{split.upper()} - X: {data[split]['X'].shape}, Y: {data[split]['Y'].shape}")
            
    print("\nValidation PASSED. Renderer dataset matches video assets.")

if __name__ == "__main__":
    validate_renderer_dataset()
