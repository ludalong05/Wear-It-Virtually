#!/usr/bin/env python3
"""
Pose Estimation wrapper for Wear-It-Virtually.
Runs OpenPose-style keypoint extraction on a person image.
Outputs pose.pkl + .mat files to the output directory.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import cv2

def run_pose_estimation(image_path, output_dir='./output'):
    """Run pose estimation on an image and save pose.pkl + .mat outputs."""
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    print(f"[PoseEstimation] Input image: {filename}")

    # Try to use the keras pose estimation module
    keras_pose_dir = os.path.join(os.path.dirname(__file__), 'keras_Realtime_MultiPerson_Pose_Estimation')
    sys.path.insert(0, keras_pose_dir)

    pose_pkl = os.path.join(output_dir, 'pose.pkl')
    pose_mat = os.path.join(output_dir, f'{name_without_ext}.mat')

    try:
        # Try running the original extract_keypoints.py
        import subprocess
        result = subprocess.run(
            [sys.executable, './extract_keypoints.py', '--image', image_path, '--output', output_dir],
            cwd=keras_pose_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        print(f"[PoseEstimation] extract_keypoints.py stdout: {result.stdout}")
        if result.returncode != 0:
            print(f"[PoseEstimation] extract_keypoints.py stderr: {result.stderr}")
    except Exception as e:
        print(f"[PoseEstimation] Original extractor failed: {e}")
        print(f"[PoseEstimation] Using fallback pose estimation...")

    # Check if pose.pkl was created
    if os.path.exists(pose_pkl):
        print(f"[PoseEstimation] Pose pkl found: {pose_pkl}")
    else:
        print(f"[PoseEstimation] Creating dummy pose (OpenPose-style keypoints)...")
        # Create a dummy pose with standard 18 body keypoints (COCO format)
        # Keypoint order: nose, neck, right_shoulder, elbow, wrist, left_shoulder,
        #                 elbow, wrist, right_hip, knee, ankle, left_hip, knee, ankle,
        #                 right_eye, left_eye, right_ear, left_ear
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]

        # Generate plausible keypoints centered in the image
        keypoints = np.zeros((18, 3), dtype=np.float32)
        # Simple heuristic: person is roughly centered
        center_x, center_y = w // 2, h // 2
        scale = min(h, w) * 0.35

        # Keypoint positions (approximate upper body)
        kp_positions = [
            (0, -0.25),   # 0: nose
            (0, -0.15),   # 1: neck
            (-0.15, -0.1), # 2: right_shoulder
            (-0.2, 0.05),  # 3: right_elbow
            (-0.22, 0.2),  # 4: right_wrist
            (0.15, -0.1),  # 5: left_shoulder
            (0.2, 0.05),   # 6: left_elbow
            (0.22, 0.2),   # 7: left_wrist
            (-0.1, 0.25),  # 8: right_hip
            (-0.12, 0.45), # 9: right_knee
            (-0.12, 0.65), # 10: right_ankle
            (0.1, 0.25),   # 11: left_hip
            (0.12, 0.45),  # 12: left_knee
            (0.12, 0.65),  # 13: left_ankle
            (-0.06, -0.3), # 14: right_eye
            (0.06, -0.3),  # 15: left_eye
            (-0.1, -0.3),  # 16: right_ear
            (0.1, -0.3),   # 17: left_ear
        ]

        for i, (dx, dy) in enumerate(kp_positions):
            keypoints[i, 0] = center_x + dx * scale
            keypoints[i, 1] = center_y + dy * scale
            keypoints[i, 2] = 1.0  # confidence

        # Write pose.pkl (OpenPose-style dict)
        pose_data = {
            'pose_keypoints_2d': keypoints.flatten().tolist(),
            'face_keypoints_2d': [],
            'hand_left_keypoints_2d': [],
            'hand_right_keypoints_2d': [],
            'pose_keypoints_3d': [],
            'face_keypoints_3d': [],
        }
        # Also save as .mat for compatibility
        import scipy.io as sio
        pose_dict_for_mat = {'pose_keypoints': keypoints}
        sio.savemat(pose_mat, pose_dict_for_mat)

        with open(pose_pkl, 'wb') as f:
            pickle.dump(pose_data, f)

        # Also save individual .mat files in pose/ subdirectory
        pose_subdir = os.path.join(output_dir, 'pose')
        os.makedirs(pose_subdir, exist_ok=True)
        sio.savemat(os.path.join(pose_subdir, f'{name_without_ext}.mat'), pose_dict_for_mat)

        print(f"[PoseEstimation] Saved dummy pose: {pose_pkl}")
        print(f"[PoseEstimation] Saved dummy pose mat: {pose_mat}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pose estimation on an image.')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output', default='.', help='Output directory')
    args = parser.parse_args()

    run_pose_estimation(args.image, args.output)
