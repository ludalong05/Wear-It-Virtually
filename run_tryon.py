#!/usr/bin/env python3
"""
End-to-end Virtual Try-On runner.
Handles the full pipeline: human parsing -> pose estimation -> stage1 -> stage2.
Can run in demo mode (no model weights needed) using OpenCV.
"""

import os
import sys
import argparse
import subprocess
import shutil
import warnings
import numpy as np
import cv2
import scipy.io

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(PROJECT_ROOT, 'inputs')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
VITON_DIR = os.path.join(PROJECT_ROOT, 'try-on', 'VITON')

# ─── Demo mode: OpenCV-based pipeline (no model weights needed) ────────────────

def resize_keep_aspect(img, target_size=(192, 256)):
    """Resize image keeping aspect ratio, pad to target_size."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to target_size
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded, scale, (x_offset, y_offset)


def demo_human_parsing(person_path, output_dir):
    """Fast body segmentation using GrabCut in demo mode."""
    img = cv2.imread(person_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {person_path}")
    h, w = img.shape[:2]

    # Create mask for body (center region as probable foreground)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = w // 2, h // 2
    rh, rw = h // 3, w // 3
    mask[cy-rh:cy+rh, cx-rw:cx+rw] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (cx-rw, cy-rh, 2*rw, 2*rh),
                bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

    # Refine: keep definitely/probably foreground
    fg_mask = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    # Dilate to fill gaps
    fg_mask = cv2.dilate(fg_mask, None, iterations=3)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)

    name = os.path.splitext(os.path.basename(person_path))[0]
    out_path = os.path.join(output_dir, f'{name}.mat')
    scipy.io.savemat(out_path, {'segment': fg_mask})
    print(f"  [demo] Body seg saved: {out_path}")
    return out_path


def demo_pose_estimation(person_path, output_dir):
    """Extract upper-body keypoints using OpenCV body pose detection."""
    img = cv2.imread(person_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {person_path}")
    h, w = img.shape[:2]

    # Use OpenCV DNN body pose (if available) or simple heuristic
    protoFile = os.path.join(VITON_DIR, "pose_estimation", "openpose_body_pose_model", "pose_deploy_linevec.prototxt")
    weightsFile = os.path.join(VITON_DIR, "pose_estimation", "openpose_body_pose_model", "pose_iter_160000.caffemodel")

    keypoints = np.zeros((18, 3), dtype=np.float32)
    has_openpose = os.path.exists(weightsFile)

    if has_openpose:
        print("  [demo] Using OpenCV OpenPose model...")
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        blob = cv2.dnn.blobFromImage(img, 1.0/255, (w, h), (0, 0, 0), swapRB=False)
        net.setInput(blob)
        out = net.forward()
        # Extract keypoints from heatmap
        n_points = 18
        points = {}
        for i in range(n_points):
            heatmap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            px = int(point[0] * w / out.shape[3])
            py = int(point[1] * h / out.shape[2])
            if conf > 0.1:
                keypoints[i] = [px, py, 1.0]
    else:
        print("  [demo] Using heuristic keypoints (OpenPose weights not found)...")
        # Heuristic keypoints for upper body
        cx, cy = w // 2, h // 2
        s = min(h, w) * 0.4
        kps = [
            (cx, cy - int(s*0.3), 1.0),   # 0: nose
            (cx, cy - int(s*0.15), 1.0),   # 1: neck
            (cx - int(s*0.25), cy - int(s*0.05), 1.0),  # 2: r shoulder
            (cx - int(s*0.4), cy + int(s*0.1), 1.0),   # 3: r elbow
            (cx - int(s*0.5), cy + int(s*0.3), 1.0),   # 4: r wrist
            (cx + int(s*0.25), cy - int(s*0.05), 1.0), # 5: l shoulder
            (cx + int(s*0.4), cy + int(s*0.1), 1.0),   # 6: l elbow
            (cx + int(s*0.5), cy + int(s*0.3), 1.0),   # 7: l wrist
            (cx - int(s*0.15), cy + int(s*0.3), 1.0),  # 8: r hip
            (cx - int(s*0.15), cy + int(s*0.6), 0.0),  # 9: r knee (occluded)
            (cx - int(s*0.15), cy + int(s*0.9), 0.0),  # 10: r ankle (occluded)
            (cx + int(s*0.15), cy + int(s*0.3), 1.0),  # 11: l hip
            (cx + int(s*0.15), cy + int(s*0.6), 0.0),  # 12: l knee (occluded)
            (cx + int(s*0.15), cy + int(s*0.9), 0.0),  # 13: l ankle (occluded)
            (cx - int(s*0.08), cy - int(s*0.4), 1.0), # 14: r eye
            (cx + int(s*0.08), cy - int(s*0.4), 1.0), # 15: l eye
            (cx - int(s*0.15), cy - int(s*0.4), 0.0), # 16: r ear
            (cx + int(s*0.15), cy - int(s*0.4), 0.0), # 17: l ear
        ]
        for i, (x, y, conf) in enumerate(kps):
            keypoints[i] = [x, y, conf]

    name = os.path.splitext(os.path.basename(person_path))[0]
    pose_path = os.path.join(output_dir, f'{name}.mat')
    pose_pkl = os.path.join(output_dir, 'pose.pkl')
    scipy.io.savemat(pose_path, {'pose_keypoints': keypoints})
    import pickle
    with open(pose_pkl, 'wb') as f:
        pickle.dump({
            'pose_keypoints_2d': keypoints.flatten().tolist(),
        }, f)
    print(f"  [demo] Pose saved: {pose_path}")
    return pose_path, pose_pkl


def demo_tryon(person_path, clothing_path, output_dir):
    """Demo try-on: geometric blend of clothing onto person."""
    person = cv2.imread(person_path)
    clothing = cv2.imread(clothing_path)
    if person is None or clothing is None:
        raise FileNotFoundError(f"Cannot read input images")

    ph, pw = person.shape[:2]
    # Resize clothing to similar size as person
    clothing_resized = cv2.resize(clothing, (pw, ph), interpolation=cv2.INTER_LINEAR)

    # Simple geometric blend: paste clothing in upper body region
    result = person.copy()
    body_top = ph // 4
    body_bottom = 3 * ph // 4
    body_left = pw // 6
    body_right = 5 * pw // 6

    # Resize clothing to fit upper body region
    region_h = body_bottom - body_top
    region_w = body_right - body_left
    clothing_fit = cv2.resize(clothing_resized, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    # Create a mask for smooth blending
    mask = cv2.GaussianBlur(clothing_fit, (15, 15), 0) / 255.0
    mask_3d = np.stack([mask]*3, axis=-1)

    # Blend
    region = result[body_top:body_bottom, body_left:body_right]
    blended = (region * (1 - mask_3d) + clothing_fit * mask_3d).astype(np.uint8)
    result[body_top:body_bottom, body_left:body_right] = blended

    out_path = os.path.join(output_dir, 'final.png')
    cv2.imwrite(out_path, result)
    print(f"  [demo] Try-on result saved: {out_path}")
    return out_path


# ─── Real pipeline (requires model weights) ────────────────────────────────────

def run_real_pipeline(person_path, clothing_path):
    """Run the full VITON pipeline using model weights."""
    project_root = os.environ.get('PROJECT_ROOT', PROJECT_ROOT)

    # Clean dirs
    for d in [
        os.path.join(project_root, 'human_parsing', 'output'),
        os.path.join(project_root, 'pose_estimation', 'output'),
        os.path.join(VITON_DIR, 'data', 'segment'),
        os.path.join(VITON_DIR, 'data', 'pose'),
        os.path.join(VITON_DIR, 'data', 'women_top'),
        os.path.join(VITON_DIR, 'results', 'stage1', 'images'),
        os.path.join(VITON_DIR, 'results', 'stage2', 'images'),
    ]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            path = os.path.join(d, f)
            if os.path.isfile(path):
                os.remove(path)

    person_name = os.path.splitext(os.path.basename(person_path))[0]

    # Step 1: Human parsing
    print("\n==> [1/3] Human parsing...")
    import subprocess
    r = subprocess.run(
        [sys.executable, './run_human_parsing.py', person_path],
        cwd=os.path.join(project_root, 'human_parsing'),
    )
    if r.returncode != 0:
        print(f"  WARNING: Human parsing failed (exit {r.returncode}), using demo")
        demo_human_parsing(person_path, os.path.join(project_root, 'human_parsing', 'output'))

    # Step 2: Pose estimation
    print("\n==> [2/3] Pose estimation...")
    r = subprocess.run(
        [sys.executable, './run_pose_estimation.py', person_path,
         '--output', os.path.join(project_root, 'pose_estimation', 'output')],
        cwd=os.path.join(project_root, 'pose_estimation'),
    )
    if r.returncode != 0:
        print(f"  WARNING: Pose estimation failed (exit {r.returncode}), using demo")
        demo_pose_estimation(person_path, os.path.join(project_root, 'pose_estimation', 'output'))

    # Copy to VITON data dir
    seg_src = os.path.join(project_root, 'human_parsing', 'output', f'{person_name}.mat')
    pose_src = os.path.join(project_root, 'pose_estimation', 'output', 'pose.pkl')
    if os.path.exists(seg_src):
        shutil.copy(seg_src, os.path.join(VITON_DIR, 'data', 'segment', f'{person_name}.mat'))
    if os.path.exists(pose_src):
        shutil.copy(pose_src, os.path.join(VITON_DIR, 'data', 'pose.pkl'))

    # Copy clothing
    clothing_name = os.path.splitext(os.path.basename(clothing_path))[0]
    ext = os.path.splitext(clothing_path)[1]
    clothing_dest = os.path.join(VITON_DIR, 'data', 'women_top', f'women_top_1{ext}')
    shutil.copy(clothing_path, clothing_dest)

    pair_name = f'{person_name}_women_top_1'

    # Step 3: Stage 1
    print("\n==> [3/3a] VITON Stage 1 (geometric matching)...")
    stage1_checkpoint = os.path.join(VITON_DIR, 'model', 'stage1', 'model-15000')
    if not os.path.exists(stage1_checkpoint + '.index'):
        print(f"  WARNING: Stage1 checkpoint not found at {stage1_checkpoint}")
        print("  Running demo try-on instead...")
        demo_out = os.path.join(project_root, 'output')
        os.makedirs(demo_out, exist_ok=True)
        out = demo_tryon(person_path, clothing_path, demo_out)
        return out

    r = subprocess.run([
        sys.executable, './model_zalando_mask_content_test.py',
        '--checkpoint', stage1_checkpoint,
        '--person_img', f'../data/women_top/{person_name}{os.path.splitext(person_path)[1]}',
        '--clothing_img', f'../data/women_top/women_top_1{ext}',
        '--result_dir', 'results/stage1/',
        '--pair_name', pair_name,
    ], cwd=VITON_DIR)
    if r.returncode != 0:
        print(f"  WARNING: Stage1 failed (exit {r.returncode}), using demo")
        demo_out = os.path.join(project_root, 'output')
        os.makedirs(demo_out, exist_ok=True)
        out = demo_tryon(person_path, clothing_path, demo_out)
        return out

    # Step 4: Stage 2
    print("\n==> [3/3b] VITON Stage 2 (refinement)...")
    stage2_checkpoint = os.path.join(VITON_DIR, 'model', 'stage2', 'model-6000')
    if not os.path.exists(stage2_checkpoint + '.index'):
        print(f"  WARNING: Stage2 checkpoint not found, using stage1 output")
        final_src = os.path.join(VITON_DIR, 'results', 'stage1', 'images', f'{pair_name}.png')
        final_out = os.path.join(project_root, 'output', 'output.png')
        if os.path.exists(final_src):
            shutil.copy(final_src, final_out)
            print(f"  Result: {final_out}")
            return final_out

    r = subprocess.run([
        sys.executable, './model_zalando_refine_test.py',
        '--checkpoint', stage2_checkpoint,
        '--coarse_result_dir', 'results/stage1/',
        '--result_dir', 'results/stage2/',
        '--pair_name', pair_name,
    ], cwd=VITON_DIR)
    if r.returncode != 0:
        print(f"  WARNING: Stage2 failed (exit {r.returncode})")
        final_src = os.path.join(VITON_DIR, 'results', 'stage1', 'images', f'{pair_name}.png')
        final_out = os.path.join(project_root, 'output', 'output.png')
        if os.path.exists(final_src):
            shutil.copy(final_src, final_out)
            return final_out

    # Copy final output
    final_src = os.path.join(VITON_DIR, 'results', 'stage2', 'images', 'final.png')
    final_out = os.path.join(project_root, 'output', 'output.png')
    if os.path.exists(final_src):
        shutil.copy(final_src, final_out)
        print(f"\n✅ Done: {final_out}")
        return final_out
    else:
        raise RuntimeError("Stage2 did not produce final.png")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Wear-It-Virtually: Virtual Try-On')
    parser.add_argument('person', help='Path to person image')
    parser.add_argument('clothing', help='Path to clothing image')
    parser.add_argument('--demo', action='store_true',
                        help='Force demo mode (no model weights needed)')
    args = parser.parse_args()

    person_path = os.path.abspath(args.person)
    clothing_path = os.path.abspath(args.clothing)

    for p in [person_path, clothing_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("Wear-It-Virtually — Virtual Try-On Pipeline")
    print("=" * 60)
    print(f"Person   : {person_path}")
    print(f"Clothing : {clothing_path}")
    print(f"Mode     : {'DEMO (OpenCV)' if args.demo else 'AUTO (model weights checked)'}")
    print("=" * 60)

    # Check for model weights
    stage1_ckpt = os.path.join(VITON_DIR, 'model', 'stage1', 'model-15000')
    stage2_ckpt = os.path.join(VITON_DIR, 'model', 'stage2', 'model-6000')

    has_weights = (
        os.path.exists(stage1_ckpt + '.index') and
        os.path.exists(stage2_ckpt + '.index')
    )

    if args.demo or not has_weights:
        if not args.demo:
            print("\n⚠️  Model weights not found — running in DEMO mode")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Clean VITON data dirs
        for subdir in ['segment', 'pose', 'women_top']:
            d = os.path.join(VITON_DIR, 'data', subdir)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

        person_name = os.path.splitext(os.path.basename(person_path))[0]
        ext = os.path.splitext(clothing_path)[1]

        # Demo human parsing
        print("\n==> [1/3] Human parsing (demo)...")
        demo_human_parsing(person_path, os.path.join(PROJECT_ROOT, 'human_parsing', 'output'))

        # Demo pose estimation
        print("\n==> [2/3] Pose estimation (demo)...")
        demo_pose_estimation(person_path, os.path.join(PROJECT_ROOT, 'pose_estimation', 'output'))

        # Demo try-on
        print("\n==> [3/3] Try-on (demo blend)...")
        out = demo_tryon(person_path, clothing_path, OUTPUT_DIR)

        print(f"\n✅ Demo complete: {out}")
    else:
        out = run_real_pipeline(person_path, clothing_path)
        print(f"\n✅ Done: {out}")


if __name__ == '__main__':
    main()
