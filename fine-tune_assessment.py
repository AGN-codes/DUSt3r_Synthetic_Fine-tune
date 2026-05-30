import os
import glob
import cv2
import torch
import numpy as np
import re
from tabulate import tabulate

# DUSt3R imports
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images

# ==========================================
# 1. PARSING & UTILITY FUNCTIONS
# ==========================================

def load_pfm_file(file_path):
    """Reads a PFM depth map safely (matching your preprocessing snippet)."""
    with open(file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').strip()

        if header == 'PF':
            is_color = True
        elif header == 'Pf':
            is_color = False
        else:
            raise ValueError('The provided file is not a valid PFM file.')

        dimensions = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError('Invalid PFM header format.')

        endian_scale = float(file.readline().decode('UTF-8').strip())
        if endian_scale < 0:
            dtype = '<f'  # little-endian
        else:
            dtype = '>f'  # big-endian

        data_buffer = file.read()
        img_data = np.frombuffer(data_buffer, dtype=dtype)

        if is_color:
            img_data = np.reshape(img_data, (img_height, img_width, 3))
        else:
            img_data = np.reshape(img_data, (img_height, img_width))

        img_data = cv2.flip(img_data, 0)
    return img_data

def read_cam_txt(path):
    """Parses standard K matrix matching your _load_pose format."""
    with open(path) as f:
        # We need the K matrix located at the bottom lines
        _ = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32) # Skip RT
        K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
    return K

def depth_to_pointmap(depth, K):
    """Converts a dense depth map into a 3D pointmap using camera intrinsics."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.stack((x, y, z), axis=-1)

# ==========================================
# 2. EVALUATION PIPELINE
# ==========================================

def evaluate_model(model_path, data_pairs, device="cuda"):
    print(f"\n[+] Initializing model: {os.path.basename(model_path)}")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    
    all_epes, all_abs_rels, all_rmses, all_deltas = [], [], [], []

    for idx, pair in enumerate(data_pairs):
        try:
            # DUSt3R load_images handles the scaling to the max side specified (512)
            images = load_images([pair['img1'], pair['img2']], size=512)
            pair_input = (((images[0], images[1])),)
            
            with torch.no_grad():
                output = inference(pair_input, model, device, batch_size=1)
            
            # --- ROBUST API UNWRAPPING ---
            pred_pts = None
            if isinstance(output, dict):
                if 'pts3d' in output:
                    pred_pts = output['pts3d']
                else:
                    for val in output.values():
                        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                            if 'pts3d' in val[0]:
                                pred_pts = val[0]['pts3d']
                                break
                        elif isinstance(val, dict) and 'pts3d' in val:
                            pred_pts = val['pts3d']
                            break
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                if isinstance(output[0], dict):
                    if 'pts3d' in output[0]:
                        pred_pts = output[0]['pts3d']
                    elif 'mappings' in output[0]:
                        pred_pts = output[0]['mappings'][0]['pts3d']
            
            if pred_pts is None:
                def find_key_recursive(d, key_name='pts3d'):
                    if isinstance(d, dict):
                        if key_name in d: return d[key_name]
                        for v in d.values():
                            res = find_key_recursive(v, key_name)
                            if res is not None: return res
                    elif isinstance(d, (list, tuple)):
                        for item in d:
                            res = find_key_recursive(item, key_name)
                            if res is not None: return res
                    return None
                pred_pts = find_key_recursive(output, 'pts3d')

            if pred_pts is None:
                raise KeyError("Unable to find 'pts3d' tensor.")
                
            if isinstance(pred_pts, torch.Tensor):
                pred_pts = pred_pts.detach().cpu().numpy()
                
            if pred_pts.ndim == 4:
                pred_pts = pred_pts[0]
            
            # --- DYNAMIC RESOLUTION ADAPTATION ---
            # Get the actual H and W assigned by DUSt3R (e.g., H=384, W=512)
            H, W = pred_pts.shape[:2]
            
            gt_depth = load_pfm_file(pair['depth1'])
            gt_K = read_cam_txt(pair['cam1'])
            
            # Resize GT Depth map to match DUSt3R's actual output frame aspect ratio
            gt_depth_res = cv2.resize(gt_depth, (W, H), interpolation=cv2.INTER_NEAREST)
            
            orig_H, orig_W = gt_depth.shape
            scale_x, scale_y = float(W) / orig_W, float(H) / orig_H
            gt_K_res = gt_K.copy()
            gt_K_res[0, 0] *= scale_x  # fx
            gt_K_res[0, 2] *= scale_x  # cx
            gt_K_res[1, 1] *= scale_y  # fy
            gt_K_res[1, 2] *= scale_y  # cy
            
            gt_pts = depth_to_pointmap(gt_depth_res, gt_K_res)
            
            # Mask generation now matches (H, W) perfectly
            valid_mask = (gt_depth_res > 0.1) & (~np.isnan(gt_depth_res)) & (~np.isinf(gt_depth_res))
            if not np.any(valid_mask): continue
                
            pred_pts_v = pred_pts[valid_mask]
            gt_pts_v = gt_pts[valid_mask]
            
            scale_factor = np.sum(pred_pts_v * gt_pts_v) / np.sum(pred_pts_v * pred_pts_v)
            aligned_pred_pts = scale_factor * pred_pts_v
            
            # Metrics Calculation
            epe = np.mean(np.linalg.norm(gt_pts_v - aligned_pred_pts, axis=-1))
            
            gt_z = gt_pts_v[..., 2]
            pred_z = np.clip(aligned_pred_pts[..., 2], a_min=1e-3, a_max=None)
            
            abs_rel = np.mean(np.abs(gt_z - pred_z) / gt_z)
            rmse = np.sqrt(np.mean((gt_z - pred_z) ** 2))
            
            thresh = np.maximum((gt_z / pred_z), (pred_z / gt_z))
            delta_1 = np.mean(thresh < 1.25) * 100.0
            
            all_epes.append(epe)
            all_abs_rels.append(abs_rel)
            all_rmses.append(rmse)
            all_deltas.append(delta_1)
            
        except Exception as e:
            print(f"[!] Error processing pair {idx}: {e}")
            continue
        
    del model
    torch.cuda.empty_cache()
    
    if len(all_epes) == 0:
        return {"EPE ↓": 0.0, "Abs Rel ↓": 0.0, "RMSE ↓": 0.0, "Delta 1.25 ↑ (%)": 0.0}
    
    return {
        "EPE ↓": np.mean(all_epes),
        "Abs Rel ↓": np.mean(all_abs_rels),
        "RMSE ↓": np.mean(all_rmses),
        "Delta 1.25 ↑ (%)": np.mean(all_deltas)
    }

# ==========================================
# 3. ROBUST DATASET PARSING SETUP
# ==========================================

def gather_dataset_pairs(data_root):
    """Parses paths intelligently matching both raw BlendedMVS structures."""
    data_pairs = []
    # Search for all scene subdirectories (supports hashes or 'scene*' names)
    scenes = [d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)]
    
    for scene in scenes:
        # Determine internal naming scheme variations dynamically
        img_dir = os.path.join(scene, "blended_images") if os.path.exists(os.path.join(scene, "blended_images")) else os.path.join(scene, "image")
        depth_dir = os.path.join(scene, "rendered_depth_maps") if os.path.exists(os.path.join(scene, "rendered_depth_maps")) else os.path.join(scene, "depth_map")
        cam_dir = os.path.join(scene, "cams")
        
        if not os.path.exists(cam_dir) or not os.path.exists(img_dir):
            continue

        pair_txt_path = os.path.join(cam_dir, "pair.txt")
        
        if os.path.exists(pair_txt_path):
            with open(pair_txt_path, 'r') as f:
                lines = f.readlines()
            if not lines: continue
            
            try:
                num_pairs = int(lines[0].strip())
                idx = 1
                for _ in range(num_pairs):
                    if idx >= len(lines): break
                    ref_img = lines[idx].strip()
                    parts = lines[idx+1].strip().split()
                    if not parts: break
                    num_neighbors = int(parts[0])
                    
                    if num_neighbors > 0 and (idx + 2) < len(lines):
                        src_img = lines[idx+2].strip().split()[0]
                        
                        # Support checking both integer padding or literal string extractions
                        ref_id = f"{int(ref_img):08d}" if ref_img.isdigit() else ref_img
                        src_id = f"{int(src_img):08d}" if src_img.isdigit() else src_img
                        
                        pair_dict = {
                            "img1": os.path.join(img_dir, f"{ref_id}.jpg"),
                            "img2": os.path.join(img_dir, f"{src_id}.jpg"),
                            "depth1": os.path.join(depth_dir, f"{ref_id}.pfm"),
                            "cam1": os.path.join(cam_dir, f"{ref_id}_cam.txt")
                        }
                        if os.path.exists(pair_dict['img1']) and os.path.exists(pair_dict['depth1']):
                            data_pairs.append(pair_dict)
                    idx += 2 + num_neighbors
            except Exception:
                pass # Fallback to alternate sequence matching if parsing throws an indexing error
                
        # Sequence Fallback matching if pair.txt logic misses files
        if len(data_pairs) == 0:
            images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
            for i in range(len(images) - 1):
                base_name = os.path.basename(images[i]).replace(".jpg", "")
                next_name = os.path.basename(images[i+1]).replace(".jpg", "")
                
                pair_dict = {
                    "img1": images[i],
                    "img2": images[i+1],
                    "depth1": os.path.join(depth_dir, f"{base_name}.pfm"),
                    "cam1": os.path.join(cam_dir, f"{base_name}_cam.txt")
                }
                if os.path.exists(pair_dict['depth1']) and os.path.exists(pair_dict['cam1']):
                    data_pairs.append(pair_dict)
                    
    print(f"[+] Successfully structured {len(data_pairs)} evaluation sample pairs.")
    return data_pairs

# ==========================================
# 4. RUN EXECUTION ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    DATASET_ROOT = "data/RainScene" 
    BASE_MODEL_PATH = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"
    FINE_TUNED_MODEL_PATH = "checkpoints/dust3r_linear_512_SyScenes3_2T1V30EP/checkpoint-best.pth"
    #FINE_TUNED_MODEL_PATH = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    #FINE_TUNED_MODEL_PATH = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    eval_pairs = gather_dataset_pairs(DATASET_ROOT)
    
    if len(eval_pairs) == 0:
        print("[!] Error: No valid image/depth pairs found. Double check directory paths.")
        exit()
        
    base_metrics = evaluate_model(BASE_MODEL_PATH, eval_pairs, device=DEVICE)
    ft_metrics = evaluate_model(FINE_TUNED_MODEL_PATH, eval_pairs, device=DEVICE)
    
    table_data = []
    for metric in base_metrics.keys():
        table_data.append([metric, f"{base_metrics[metric]:.4f}", f"{ft_metrics[metric]:.4f}"])
        
    print("\n" + "="*50)
    print("           DUST3R BENCHMARK COMPARISON")
    print("="*50)
    print(tabulate(table_data, headers=["Metric", "Base Model", "Fine-Tuned (Ours)"], tablefmt="grid"))