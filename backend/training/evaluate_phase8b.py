import os
import sys
import math
import time
import json
import torch
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1
from models.motion_model_v4 import LearnedTemporalMotionModelV4
from training.train_renderer_v1 import calculate_psnr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def tensor_to_cv2(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def detect_landmarks_and_pose(img, detector):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_image)
    if not res.face_landmarks:
        return None, None
    lmks = np.array([[l.x, l.y] for l in res.face_landmarks[0]])
    mat = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else np.eye(4)
    # decompose projection matrix to approximate head pose (Euler angles)
    # simplified to just the rotation matrix part
    R = mat[:3, :3]
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    pose = np.array([x, y, z])
    return lmks, pose

def calculate_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = structural_similarity(img1_gray, img2_gray, full=True)
        return score
    except ImportError:
        return 0.0

# Precise landmark indices
MOUTH_IDX = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 415]
EYE_IDX = [33, 133, 159, 145, 362, 263, 386, 374, 7, 163, 144, 153, 154, 155, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Load V4 Motion Model
    motion_ckpt = torch.load(os.path.join(base_dir, "backend/training/checkpoints/learned_motion_v4_best.pt"), map_location=device, weights_only=False)
    motion_model = LearnedTemporalMotionModelV4(input_dim=80, context_size=16, hidden_dim=64, output_dim=11).to(device)
    motion_model.load_state_dict(motion_ckpt["model_state"])
    motion_model.eval()
    
    # Load NeuralRendererV1
    render_ckpt = torch.load(os.path.join(base_dir, "backend/training/checkpoints/neural_renderer_v1_gpu_best.pt"), map_location=device, weights_only=False)
    renderer = NeuralRendererV1(motion_dim=11).to(device)
    renderer.load_state_dict(render_ckpt["model_state_dict"])
    renderer.eval()
    
    # Load dataset
    data_v4 = torch.load(os.path.join(base_dir, "synthesia_training_data/dataset_v4.pt"), map_location="cpu", weights_only=False)
    images_v4 = torch.load(os.path.join(base_dir, "synthesia_training_data/dataset_v4_images.pt"), map_location="cpu", weights_only=False)
    
    detector = get_face_landmarker()
    loss_l1 = torch.nn.L1Loss()
    loss_mse = torch.nn.MSELoss()
    
    out_dir = os.path.join(base_dir, "artifacts", "phase8b_eval")
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for split in ["val", "test"]:
        print(f"\n--- EVALUATING {split.upper()} SPLIT ---")
        X = data_v4[split]["X"].to(device) # [N, 16, 80]
        Y_gt = data_v4[split]["Y"].to(device) # [N, 11]
        images = images_v4[split]
        N = len(images)
        
        metrics = {
            "motion_mse": [], "motion_mae": [],
            "gt_l1": [], "gt_mse": [], "gt_psnr": [], "gt_ssim": [],
            "gt_lmk_full": [], "gt_lmk_mouth": [], "gt_lmk_eyes": [], "gt_pose_err": [],
            "pred_l1": [], "pred_mse": [], "pred_psnr": [], "pred_ssim": [],
            "pred_lmk_full": [], "pred_lmk_mouth": [], "pred_lmk_eyes": [], "pred_pose_err": []
        }
        
        video_out = cv2.VideoWriter(os.path.join(out_dir, f"{split}_sequence.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 25, (512*4, 512))
        
        with torch.no_grad():
            for i in range(N):
                # Predict Motion
                audio_ctx = X[i].unsqueeze(0).unsqueeze(0) # [1, 1, 16, 80]
                with torch.amp.autocast('cuda'):
                    pred_motion = motion_model(audio_ctx).squeeze(0) # [1, 11]
                    
                gt_motion = Y_gt[i].unsqueeze(0) # [1, 11]
                
                metrics["motion_mse"].append(loss_mse(pred_motion, gt_motion).item())
                metrics["motion_mae"].append(loss_l1(pred_motion, gt_motion).item())
                
                # Rendering
                ident = images[i]["identity_frame"].unsqueeze(0).to(device)
                target = images[i]["target_frame"].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    gt_render = renderer(ident, gt_motion)
                    pred_render = renderer(ident, pred_motion)
                    
                    gt_l1 = loss_l1(gt_render, target).item()
                    gt_mse_v = loss_mse(gt_render, target).item()
                    
                    pred_l1 = loss_l1(pred_render, target).item()
                    pred_mse_v = loss_mse(pred_render, target).item()
                    
                metrics["gt_l1"].append(gt_l1)
                metrics["gt_mse"].append(gt_mse_v)
                metrics["gt_psnr"].append(calculate_psnr(gt_mse_v))
                
                metrics["pred_l1"].append(pred_l1)
                metrics["pred_mse"].append(pred_mse_v)
                metrics["pred_psnr"].append(calculate_psnr(pred_mse_v))
                
                # Image Conversion
                t_img = tensor_to_cv2(target[0])
                gt_img = tensor_to_cv2(gt_render[0])
                pred_img = tensor_to_cv2(pred_render[0])
                
                metrics["gt_ssim"].append(calculate_ssim(gt_img, t_img))
                metrics["pred_ssim"].append(calculate_ssim(pred_img, t_img))
                
                # Landmarks
                lmk_t, pose_t = detect_landmarks_and_pose(t_img, detector)
                lmk_g, pose_g = detect_landmarks_and_pose(gt_img, detector)
                lmk_p, pose_p = detect_landmarks_and_pose(pred_img, detector)
                
                if lmk_t is not None:
                    if lmk_g is not None:
                        err = np.linalg.norm(lmk_g - lmk_t, axis=1)
                        metrics["gt_lmk_full"].append(err.mean())
                        metrics["gt_lmk_mouth"].append(err[MOUTH_IDX].mean())
                        metrics["gt_lmk_eyes"].append(err[EYE_IDX].mean())
                        metrics["gt_pose_err"].append(np.linalg.norm(pose_g - pose_t))
                    
                    if lmk_p is not None:
                        err = np.linalg.norm(lmk_p - lmk_t, axis=1)
                        metrics["pred_lmk_full"].append(err.mean())
                        metrics["pred_lmk_mouth"].append(err[MOUTH_IDX].mean())
                        metrics["pred_lmk_eyes"].append(err[EYE_IDX].mean())
                        metrics["pred_pose_err"].append(np.linalg.norm(pose_p - pose_t))
                        
                # Visual output: Target | GT-Render | Pred-Render | Pred-Diff
                diff = cv2.applyColorMap(cv2.absdiff(t_img, pred_img), cv2.COLORMAP_HOT)
                vis = np.concatenate([t_img, gt_img, pred_img, diff], axis=1)
                video_out.write(vis)
                
                if i % 50 == 0:
                    cv2.imwrite(os.path.join(out_dir, f"{split}_frame_{i}.jpg"), vis)
                    
        video_out.release()
        
        final_metrics = {k: float(np.mean(v)) if len(v)>0 else 0.0 for k, v in metrics.items()}
        results[split] = final_metrics
        
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
            
    with open(os.path.join(out_dir, "phase8b_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- PERFORMANCE BENCHMARK ---")
    ident = torch.randn(1, 3, 512, 512, device=device)
    motion = torch.randn(1, 11, device=device)
    audio = torch.randn(1, 1, 16, 80, device=device)
    
    # Warmup
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(10):
            m = motion_model(audio).squeeze(0)
            r = renderer(ident, m)
            
    torch.cuda.synchronize()
    
    m_lat = []
    r_lat = []
    e_lat = []
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(100):
            # End to end
            torch.cuda.synchronize()
            t0 = time.time()
            m = motion_model(audio).squeeze(0)
            r = renderer(ident, m)
            torch.cuda.synchronize()
            e_lat.append((time.time()-t0)*1000)
            
            # Motion only
            torch.cuda.synchronize()
            t0 = time.time()
            m = motion_model(audio)
            torch.cuda.synchronize()
            m_lat.append((time.time()-t0)*1000)
            
            # Render only
            torch.cuda.synchronize()
            t0 = time.time()
            r = renderer(ident, motion)
            torch.cuda.synchronize()
            r_lat.append((time.time()-t0)*1000)
            
    m_lat.sort()
    r_lat.sort()
    e_lat.sort()
    
    print(f"Motion Model (V4) inference: {sum(m_lat)/len(m_lat):.2f} ms")
    print(f"Renderer inference (p50): {r_lat[int(len(r_lat)*0.5)]:.2f} ms")
    print(f"Renderer inference (p95): {r_lat[int(len(r_lat)*0.95)]:.2f} ms")
    print(f"End-to-End inference (p50): {e_lat[int(len(e_lat)*0.5)]:.2f} ms")
    print(f"Peak VRAM Pipeline: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")

if __name__ == "__main__":
    evaluate()
