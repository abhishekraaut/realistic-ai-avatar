import os
import json
import torch
import math

def validate_dataset(manifest_path, dataset_path):
    print("Validating Dataset V3...")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    data = torch.load(dataset_path, map_location='cpu')
    
    report = {
        "status": "PASS",
        "total_sequences": len(manifest),
        "total_frames": 0,
        "total_duration": 0.0,
        "dropped_samples": 0,
        "issues": [],
        "splits": {
            "train": 0,
            "val": 0,
            "test": 0
        }
    }
    
    expected_samples = {"train": 0, "val": 0, "test": 0}
    context_size = 16
    
    for seq in manifest:
        report["total_frames"] += seq["frame_count"]
        report["total_duration"] += seq["duration"]
        report["dropped_samples"] += seq["dropped_samples"]
        expected_samples[seq["split"]] += seq["valid_sample_count"]
        
        # Duration mismatch logic (5% tolerance)
        v_dur = seq["duration"]
        a_dur = seq["audio_duration"]
        if abs(v_dur - a_dur) > max(v_dur, a_dur) * 0.05:
            report["issues"].append(f"Seq {seq['sequence_id']}: Audio/Video duration mismatch! (V: {v_dur:.2f}, A: {a_dur:.2f})")
            report["status"] = "FAIL"
            
    for split in ["train", "val", "test"]:
        if split not in data:
            report["issues"].append(f"Missing split {split} in dataset")
            report["status"] = "FAIL"
            continue
            
        X = data[split]["X"]
        Y = data[split]["Y"]
        
        report["splits"][split] = X.shape[0]
        
        # Check counts
        if X.shape[0] != expected_samples[split]:
            report["issues"].append(f"Split {split} count mismatch! Manifest says {expected_samples[split]}, Tensor has {X.shape[0]}")
            report["status"] = "FAIL"
            
        if X.shape[0] > 0:
            # Check shape
            if X.dim() != 3 or X.shape[1] != context_size or X.shape[2] != 80:
                report["issues"].append(f"Split {split} X shape invalid! Expected (N, {context_size}, 80), got {X.shape}")
                report["status"] = "FAIL"
                
            # Check NaN/Inf
            if torch.isnan(X).any() or torch.isnan(Y).any():
                report["issues"].append(f"Split {split} contains NaN values!")
                report["status"] = "FAIL"
                
            if torch.isinf(X).any() or torch.isinf(Y).any():
                report["issues"].append(f"Split {split} contains Inf values!")
                report["status"] = "FAIL"
                
    report_path = os.path.join(os.path.dirname(manifest_path), "validation_v3_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(json.dumps(report, indent=2))
    if report["status"] == "FAIL":
        print("VALIDATION FAILED!")
        sys.exit(1)
    else:
        print("VALIDATION PASSED!")

if __name__ == "__main__":
    base = r"C:\Users\iabhi\Downloads\Avtar-Project\synthesia_training_data"
    validate_dataset(os.path.join(base, "dataset_v3_manifest.json"), os.path.join(base, "dataset_v3.pt"))
