import os
import torch

def validate_v4():
    base = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    ds_path = os.path.join(base, "dataset_v4.pt")
    data = torch.load(ds_path, map_location='cpu')
    
    for split in ["train", "val", "test"]:
        X = data[split]["X"]
        Y = data[split]["Y"]
        
        print(f"[{split}] X: {X.shape}, Y: {Y.shape}")
        
        assert X.shape[1] == 16
        assert X.shape[2] == 80
        assert Y.shape[1] == 11
        assert not torch.isnan(X).any(), f"NaN in X {split}"
        assert not torch.isnan(Y).any(), f"NaN in Y {split}"
        
    print("V4 Validation PASSED!")

if __name__ == "__main__":
    validate_v4()
