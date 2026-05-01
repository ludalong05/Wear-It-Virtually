
import cv2
import numpy as np
import os
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image

MODEL_PATH = "/tmp/pose_landmarker.task"
PYT = "/tmp/wear-venv/bin/python"

def detect_pose(img_path):
    """Return pose keypoints dict using MediaPipe."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    cv2.imwrite("/tmp/_pose_tmp.jpg", img)
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_poses=1,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    mp_image = Image.create_from_file("/tmp/_pose_tmp.jpg")
    result = landmarker.detect(mp_image)
    landmarker.close()
    
    lm = result.pose_landmarks[0] if result.pose_landmarks else None
    if not lm:
        raise RuntimeError(f"No pose detected in {img_path}")
    
    ls = lm[11]  # left shoulder
    rs = lm[12]  # right shoulder
    nose = lm[0]
    
    return {
        "lm": lm,
        "img_h": h, "img_w": w,
        "ls": (ls.x * w, ls.y * h),
        "rs": (rs.x * w, rs.y * h),
        "nose": (nose.x * w, nose.y * h),
        "sw": abs(rs.x - ls.x) * w,  # shoulder width in px
        "neck_y": (ls.y + rs.y) / 2 * h,
    }


def extract_clothing(img_path):
    """Extract clothing from white/light background using GrabCut + color heuristic."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    
    # Sample 4 corners to detect background color
    corners = np.array([img[2,2], img[2,w-2], img[h-2,2], img[h-2,w-2]], dtype=np.float32)
    bg_color = corners.mean(axis=0)
    
    # GrabCut with rectangle around clothing
    mask = np.zeros((h, w), np.uint8)
    margin = max(5, min(w, h)//10)
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    fg_mask = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    
    # Remove background-colored pixels
    img_f = img.astype(float) / 255.0
    bg_f = bg_color / 255.0
    diff = np.linalg.norm(img_f - bg_f, axis=2)
    color_mask = (diff > 0.20).astype(np.uint8) * 255
    fg_mask = cv2.min(fg_mask, color_mask)
    
    # Clean up
    fg_mask = cv2.erode(fg_mask, np.ones((3,3), np.uint8), iterations=2)
    fg_mask = cv2.dilate(fg_mask, np.ones((3,3), np.uint8), iterations=3)
    
    # Find bounding box of clothing
    rows = np.any(fg_mask > 0, axis=1)
    cols = np.any(fg_mask > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        clothing = img[rmin:rmax+1, cmin:cmax+1]
        alpha = fg_mask[rmin:rmax+1, cmin:cmax+1]
    else:
        clothing = img
        alpha = fg_mask
    
    return cv2.cvtColor(np.dstack([clothing, alpha]), cv2.COLOR_BGR2BGRA), bg_color


def resize_clothing_to_shoulders(clothing_4ch, target_w, target_h):
    """Resize clothing to target dimensions, preserving aspect ratio."""
    ch, cw = clothing_4ch.shape[:2]
    scale = min(target_w / cw, target_h / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    clothing_rs = cv2.resize(clothing_4ch[:,:,:3], (new_w, new_h))
    alpha_rs = cv2.resize(clothing_4ch[:,:,3], (new_w, new_h))
    return clothing_rs, alpha_rs, new_w, new_h


def apply_clothing(person, clothing_img, alpha, target_top, target_left, blend_w, blend_h):
    """Blend clothing onto person at specified position using Poisson blend."""
    h, w = person.shape[:2]
    
    # Crop/pad target region
    x1 = max(0, target_left)
    y1 = max(0, target_top)
    x2 = min(w, target_left + blend_w)
    y2 = min(h, target_top + blend_h)
    
    # Crop clothing to fit
    cx1 = x1 - target_left
    cy1 = y1 - target_top
    cx2 = cx1 + (x2 - x1)
    cy2 = cy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1 or cx2 <= cx1 or cy2 <= cy1:
        return person
    
    cloth_crop = clothing_img[cy1:cy2, cx1:cx2]
    mask_crop = alpha[cy1:cy2, cx1:cx2]
    
    if cloth_crop.size == 0 or mask_crop.size == 0:
        return person
    
    # Poisson blend at center
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    try:
        result = cv2.seamlessClone(
            cloth_crop, person, mask_crop,
            (center_x, center_y), cv2.NORMAL_CLONE
        )
    except Exception:
        # Fallback: alpha blend
        mask_f = (mask_crop.astype(float) / 255.0)[:,:,None]
        result = person.copy()
        region = (cloth_crop.astype(float) * mask_f + 
                  result[y1:y2, x1:x2].astype(float) * (1 - mask_f)).astype(np.uint8)
        result[y1:y2, x1:x2] = region
    
    return result


def run_tryon(person_path, clothing_path, output_dir):
    """End-to-end try-on using MediaPipe pose + GrabCut + Poisson blend."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Person: {person_path}")
    print(f"  Clothing: {clothing_path}")
    
    # Step 1: Detect pose
    print("  [1] Detecting pose...")
    pose = detect_pose(person_path)
    lm = pose["lm"]
    h, w = pose["img_h"], pose["img_w"]
    ls_x, ls_y = pose["ls"]
    rs_x, rs_y = pose["rs"]
    nose_x, nose_y = pose["nose"]
    sw = pose["sw"]
    neck_y = pose["neck_y"]
    
    print(f"      Shoulders: L({ls_x:.0f},{ls_y:.0f}) R({rs_x:.0f},{rs_y:.0f}) SW={sw:.0f}px")
    print(f"      Neck y={neck_y:.0f}, Nose y={nose_y:.0f}")
    
    # Step 2: Extract clothing
    print("  [2] Extracting clothing...")
    clothing_4ch, bg_color = extract_clothing(clothing_path)
    ch, cw = clothing_4ch.shape[:2]
    print(f"      Clothing size: {cw}x{ch}, bg={bg_color}")
    
    # Step 3: Determine target region based on pose
    # Clothing should start just below neck, extend down
    clothing_top = int(neck_y - sw * 0.05)  # just below neck
    clothing_bottom = int(clothing_top + sw * 1.1)  # slightly longer than shoulder width
    clothing_target_h = clothing_bottom - clothing_top
    clothing_target_w = int(sw * 1.2)
    
    # Horizontal center: midpoint of shoulders
    clothing_center_x = int((ls_x + rs_x) / 2)
    clothing_left = int(clothing_center_x - clothing_target_w / 2)
    
    print(f"  [3] Target region: top={clothing_top} h={clothing_target_h} left={clothing_left} w={clothing_target_w}")
    
    # Step 4: Resize clothing
    print("  [4] Resizing clothing...")
    clothing_rs, alpha_rs, rw, rh = resize_clothing_to_shoulders(
        clothing_4ch, clothing_target_w, clothing_target_h)
    
    # Step 5: Apply clothing
    print("  [5] Applying clothing with Poisson blend...")
    person = cv2.imread(person_path)
    result = apply_clothing(
        person, clothing_rs, alpha_rs,
        target_top=clothing_top,
        target_left=clothing_left,
        blend_w=clothing_target_w,
        blend_h=clothing_target_h
    )
    
    # Step 6: Save intermediate
    vis = person.copy()
    for idx in [11, 12, 0]:
        x = int(lm[idx].x * w)
        y = int(lm[idx].y * h)
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(idx), (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    cv2.line(vis, (int(ls_x), int(ls_y)), (int(rs_x), int(rs_y)), (0,200,200), 2)
    cv2.rectangle(vis, (clothing_left, clothing_top), 
                  (clothing_left+clothing_target_w, clothing_top+clothing_target_h),
                  (0,0,255), 2)
    
    out_final = os.path.join(output_dir, "final.png")
    out_vis = os.path.join(output_dir, "pose_overlay.png")
    out_clothing = os.path.join(output_dir, "clothing_extracted.png")
    
    cv2.imwrite(out_final, result)
    cv2.imwrite(out_vis, vis)
    cv2.imwrite(out_clothing, cv2.cvtColor(clothing_4ch, cv2.COLOR_BGRA2BGR))
    
    print(f"  [OK] Saved: {out_final}")
    return out_final


# Run on example images
person = "/Users/wl/Wear-It-Virtually/inputs/example_person.jpg"
clothing = "/Users/wl/Wear-It-Virtually/inputs/example_clothing.jpg"
output = "/tmp/tryon_output"

result = run_tryon(person, clothing, output)
print(f"\nFinal: {result}")
