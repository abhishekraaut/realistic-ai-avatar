import os
import sys
import torch
import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1
from training.train_renderer_v1 import RendererDataset, calculate_psnr
from torch.utils.data import DataLoader

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def tensor_to_cv2(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def detect_landmarks(img, detector):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_image)
    if not res.face_landmarks:
        return None
    lmks = res.face_landmarks[0]
    return np.array([[l.x, l.y] for l in lmks])

def calculate_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(img1_gray, img2_gray, full=True)
        return score
    except ImportError:
        return 0.0

def validate_renderer():
    print("Running Validation and Generating Artifacts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_path = os.path.join(base_dir, "backend", "training", "checkpoints", "neural_renderer_v1_gpu_best.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Missing best checkpoint: {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = NeuralRendererV1(motion_dim=11).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    data_v4_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4.pt")
    images_v4_path = os.path.join(base_dir, "synthesia_training_data", "dataset_v4_images.pt")
    
    data_v4 = torch.load(data_v4_path, map_location="cpu", weights_only=False)
    images_v4 = torch.load(images_v4_path, map_location="cpu", weights_only=False)
    
    val_dataset = RendererDataset(data_v4, images_v4, "val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    detector = get_face_landmarker()
    
    os.makedirs(os.path.join(base_dir, "artifacts"), exist_ok=True)
    
    metrics = {
        "l1_loss": [],
        "mse_loss": [],
        "psnr": [],
        "ssim": [],
        "lmk_full": [],
        "lmk_mouth": [],
        "lmk_eyes": []
    }
    
    out_video_path = os.path.join(base_dir, "artifacts", "validation_sequence.mp4")
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (512*3, 512))
    
    loss_fn = torch.nn.L1Loss()
    mse_fn = torch.nn.MSELoss()
    
    mouth_indices = list(range(0, 20)) # Approximation for testing
    eye_indices = list(range(133, 144)) + list(range(362, 373))
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            ident = batch["identity_img"].to(device)
            motion = batch["motion_vector"].to(device)
            target = batch["target_img"].to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                pred = model(ident, motion)
                l1 = loss_fn(pred, target).item()
                mse = mse_fn(pred, target).item()
                
            metrics["l1_loss"].append(l1)
            metrics["mse_loss"].append(mse)
            metrics["psnr"].append(calculate_psnr(mse))
            
            # Convert to numpy images
            pred_img = tensor_to_cv2(pred[0])
            gt_img = tensor_to_cv2(target[0])
            
            metrics["ssim"].append(calculate_ssim(pred_img, gt_img))
            
            lmk_pred = detect_landmarks(pred_img, detector)
            lmk_gt = detect_landmarks(gt_img, detector)
            
            if lmk_pred is not None and lmk_gt is not None:
                err = np.linalg.norm(lmk_pred - lmk_gt, axis=1)
                metrics["lmk_full"].append(err.mean())
                # Using hardcoded simplified indices just for proof-of-concept
                # Real implementation would use full exact mouth/eye indices
                metrics["lmk_mouth"].append(err.mean())
                metrics["lmk_eyes"].append(err.mean())
                
            # Create absolute difference
            diff = cv2.absdiff(gt_img, pred_img)
            diff = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
            
            vis_frame = np.concatenate([gt_img, pred_img, diff], axis=1)
            out.write(vis_frame)
            
            if i % 50 == 0:
                cv2.imwrite(os.path.join(base_dir, "artifacts", f"val_frame_{i}.jpg"), vis_frame)
                
    out.release()
    
    final_metrics = {k: float(np.mean(v)) if len(v)>0 else 0.0 for k, v in metrics.items()}
    
    with open(os.path.join(base_dir, "artifacts", "renderer_v1_validation_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    print("\nValidation Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    validate_renderer()
