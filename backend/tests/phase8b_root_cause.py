import os
import sys
import torch
import cv2
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_renderer_v1 import NeuralRendererV1
from models.motion_model_v4 import LearnedTemporalMotionModelV4
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(_base, "backend", "training"))
from evaluate_phase8b import get_face_landmarker, tensor_to_cv2, detect_landmarks_and_pose, MOUTH_IDX, EYE_IDX, calculate_psnr, calculate_ssim

def run_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_dir, "artifacts", "phase8b_root_cause")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Check Identity Split
    print("--- 1. IDENTIFYING SPLIT IDENTITIES ---")
    data_v4 = torch.load(os.path.join(base_dir, "synthesia_training_data/dataset_v4.pt"), map_location="cpu", weights_only=False)
    images_v4 = torch.load(os.path.join(base_dir, "synthesia_training_data/dataset_v4_images.pt"), map_location="cpu", weights_only=False)
    
    detector = get_face_landmarker()
    
    # Save a frame from train, val, test to inspect visually
    train_ident = tensor_to_cv2(images_v4["train"][0]["identity_frame"])
    val_ident = tensor_to_cv2(images_v4["val"][0]["identity_frame"])
    test_ident = tensor_to_cv2(images_v4["test"][0]["identity_frame"])
    
    cv2.imwrite(os.path.join(out_dir, "train_identity.jpg"), train_ident)
    cv2.imwrite(os.path.join(out_dir, "val_identity.jpg"), val_ident)
    cv2.imwrite(os.path.join(out_dir, "test_identity.jpg"), test_ident)
    print("Saved train_identity.jpg, val_identity.jpg, test_identity.jpg for visual inspection.")

    # 2. Verify TEST Pipeline (MediaPipe on GT Test frames)
    print("\n--- 2. VERIFYING TEST GT FRAMES WITH MEDIAPIPE ---")
    detected = 0
    for i in range(len(images_v4["test"])):
        gt = tensor_to_cv2(images_v4["test"][i]["target_frame"])
        lmks, _ = detect_landmarks_and_pose(gt, detector)
        if lmks is not None:
            detected += 1
        if i == 0:
            cv2.imwrite(os.path.join(out_dir, "test_gt_frame_0.jpg"), gt)
    print(f"GT test face detection = {detected} / {len(images_v4['test'])}")
    
    # Load Models
    motion_ckpt = torch.load(os.path.join(base_dir, "backend/training/checkpoints/learned_motion_v4_best.pt"), map_location=device, weights_only=False)
    motion_model = LearnedTemporalMotionModelV4(input_dim=80, context_size=16, hidden_dim=64, output_dim=11).to(device)
    motion_model.load_state_dict(motion_ckpt["model_state"])
    motion_model.eval()
    
    render_ckpt = torch.load(os.path.join(base_dir, "backend/training/checkpoints/neural_renderer_v1_gpu_best.pt"), map_location=device, weights_only=False)
    renderer = NeuralRendererV1(motion_dim=11).to(device)
    renderer.load_state_dict(render_ckpt["model_state_dict"])
    renderer.eval()

    # Helpers
    def predict_m(split, i):
        audio = data_v4[split]["X"][i].unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'), torch.no_grad():
            return motion_model(audio).squeeze(0)
    
    def get_gt_m(split, i):
        return data_v4[split]["Y"][i].unsqueeze(0).to(device)

    def get_ident(split, i):
        return images_v4[split][i]["identity_frame"].unsqueeze(0).to(device)
        
    def get_target(split, i):
        return images_v4[split][i]["target_frame"].unsqueeze(0).to(device)

    def eval_condition(name, ident, motion, target):
        with torch.amp.autocast('cuda'), torch.no_grad():
            render = renderer(ident, motion)
            l1 = F.l1_loss(render, target).item()
            mse = F.mse_loss(render, target).item()
        
        t_img = tensor_to_cv2(target[0])
        r_img = tensor_to_cv2(render[0])
        psnr = calculate_psnr(mse)
        ssim = calculate_ssim(r_img, t_img)
        
        lmk_t, pose_t = detect_landmarks_and_pose(t_img, detector)
        lmk_r, pose_r = detect_landmarks_and_pose(r_img, detector)
        
        errs = {'full':0, 'mouth':0, 'eyes':0, 'pose':0}
        if lmk_t is not None and lmk_r is not None:
            err = np.linalg.norm(lmk_r - lmk_t, axis=1)
            errs['full'] = err.mean()
            errs['mouth'] = err[MOUTH_IDX].mean()
            errs['eyes'] = err[EYE_IDX].mean()
            errs['pose'] = np.linalg.norm(pose_r - pose_t)
            
        return r_img, t_img, l1, mse, psnr, ssim, errs
    
    import torch.nn.functional as F

    print("\n--- 3, 4, 5, 6. ISOLATION EXPERIMENTS (Frame 0) ---")
    
    # Conditions A, B
    val_ident = get_ident("val", 0)
    val_target = get_target("val", 0)
    val_gt_m = get_gt_m("val", 0)
    val_pred_m = predict_m("val", 0)
    
    img_A, tgt_A, l1_A, mse_A, psnr_A, ssim_A, err_A = eval_condition("A (Val GT)", val_ident, val_gt_m, val_target)
    img_B, tgt_B, l1_B, mse_B, psnr_B, ssim_B, err_B = eval_condition("B (Val Pred)", val_ident, val_pred_m, val_target)
    
    # Conditions C, D
    test_ident = get_ident("test", 0)
    test_target = get_target("test", 0)
    test_gt_m = get_gt_m("test", 0)
    test_pred_m = predict_m("test", 0)
    
    img_C, tgt_C, l1_C, mse_C, psnr_C, ssim_C, err_C = eval_condition("C (Test GT)", test_ident, test_gt_m, test_target)
    img_D, tgt_D, l1_D, mse_D, psnr_D, ssim_D, err_D = eval_condition("D (Test Pred)", test_ident, test_pred_m, test_target)
    
    # Conditions E, F
    img_E, tgt_E, l1_E, mse_E, psnr_E, ssim_E, err_E = eval_condition("E (Val ID + Test GT)", val_ident, test_gt_m, test_target)
    img_F, tgt_F, l1_F, mse_F, psnr_F, ssim_F, err_F = eval_condition("F (Test ID + Val GT)", test_ident, val_gt_m, val_target)
    
    cv2.imwrite(os.path.join(out_dir, "CondA_ValGT.jpg"), img_A)
    cv2.imwrite(os.path.join(out_dir, "CondB_ValPred.jpg"), img_B)
    cv2.imwrite(os.path.join(out_dir, "CondC_TestGT.jpg"), img_C)
    cv2.imwrite(os.path.join(out_dir, "CondD_TestPred.jpg"), img_D)
    cv2.imwrite(os.path.join(out_dir, "CondE_ValID_TestMotion.jpg"), img_E)
    cv2.imwrite(os.path.join(out_dir, "CondF_TestID_ValMotion.jpg"), img_F)
    
    print(f"Cond A (Val GT): PSNR {psnr_A:.2f}, L1 {l1_A:.4f}, Lmk {err_A['full']:.4f}")
    print(f"Cond B (Val Pred): PSNR {psnr_B:.2f}, L1 {l1_B:.4f}, Lmk {err_B['full']:.4f}")
    print(f"Cond C (Test GT): PSNR {psnr_C:.2f}, L1 {l1_C:.4f}, Lmk {err_C['full']:.4f}")
    print(f"Cond D (Test Pred): PSNR {psnr_D:.2f}, L1 {l1_D:.4f}, Lmk {err_D['full']:.4f}")
    print(f"Cond E (Val ID + Test Motion -> vs Test Target): PSNR {psnr_E:.2f}, L1 {l1_E:.4f}")
    print(f"Cond F (Test ID + Val Motion -> vs Val Target): PSNR {psnr_F:.2f}, L1 {l1_F:.4f}")
    
    # 7. Motion Space Audit on Test
    print("\n--- 7. MOTION SPACE AUDIT (TEST SPLIT) ---")
    X_test = data_v4["test"]["X"].to(device)
    Y_gt_test = data_v4["test"]["Y"].cpu().numpy()
    
    Y_pred_test = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(len(X_test)):
            m = motion_model(X_test[i].unsqueeze(0).unsqueeze(0)).squeeze(0).cpu().numpy()
            Y_pred_test.append(m[0])
    Y_pred_test = np.array(Y_pred_test)
    
    mae_per_dim = np.abs(Y_gt_test - Y_pred_test).mean(axis=0)
    std_gt = Y_gt_test.std(axis=0)
    std_pred = Y_pred_test.std(axis=0)
    
    print("Per-Dim MAE:", mae_per_dim)
    print("GT STD:", std_gt)
    print("Pred STD:", std_pred)
    
    # 11. Performance
    print("\n--- 11. PERFORMANCE AUDIT ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # warmup
    for _ in range(10):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = renderer(val_ident, val_gt_m)
            
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = renderer(val_ident, val_gt_m)
        torch.cuda.synchronize()
        times.append((time.time() - t0)*1000)
    
    times = np.array(times)
    print(f"Renderer inference Mean: {times.mean():.2f} ms")
    print(f"Renderer inference p50:  {np.percentile(times, 50):.2f} ms")
    print(f"Renderer inference p95:  {np.percentile(times, 95):.2f} ms")
    print(f"Renderer inference Max:  {times.max():.2f} ms")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    print(f"Peak VRAM reserved:  {torch.cuda.max_memory_reserved()/1024**3:.3f} GB")

if __name__ == "__main__":
    run_audit()
