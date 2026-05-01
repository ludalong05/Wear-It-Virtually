#!/usr/bin/env python3
"""
Human Parsing wrapper for Wear-It-Virtually.
Runs LIP_JPPNet to segment a person image into body parts.
Outputs a .mat file to human_parsing/output/<name>.mat
"""

import argparse
import os
import sys
import cv2
import scipy.io
import numpy as np

# Add LIP_JPPNet to path
LIP_JPPNET_DIR = os.path.join(os.path.dirname(__file__), 'LIP_JPPNet')
sys.path.insert(0, LIP_JPPNET_DIR)

def run_parsing(image_path, output_dir='./output'):
    """Run human parsing on an image and save .mat output."""
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Copy image to JPPNet input directory
    jppnet_images = os.path.join(LIP_JPPNET_DIR, 'datasets/examples/images')
    jppnet_lists = os.path.join(LIP_JPPNET_DIR, 'datasets/examples/list')
    jppnet_output = os.path.join(LIP_JPPNET_DIR, 'output/parsing/val')

    os.makedirs(jppnet_images, exist_ok=True)
    os.makedirs(jppnet_lists, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Copy input image
    dest_image = os.path.join(jppnet_images, filename)
    if os.path.abspath(image_path) != os.path.abspath(dest_image):
        import shutil
        shutil.copy(image_path, dest_image)

    # Create val.txt list (single image)
    val_list_path = os.path.join(jppnet_lists, 'val.txt')
    with open(val_list_path, 'w') as f:
        f.write(f'/images/{filename}')

    print(f"[HumanParsing] Input image: {filename}")
    print(f"[HumanParsing] Running JPPNet...")

    # Run JPPNet (modify NUM_STEPS in-memory via env trick instead of sed)
    eval_script = os.path.join(LIP_JPPNET_DIR, 'evaluate_parsing_JPPNet-s2.py')

    # Patch NUM_STEPS before running
    import importlib.util
    spec = importlib.util.spec_from_file_location("jppnet_eval", eval_script)
    jppnet_module = importlib.util.module_from_spec(spec)

    # Read and patch the source
    with open(eval_script, 'r') as f:
        source = f.read()

    # Replace NUM_STEPS = 6 with NUM_STEPS = 1 (process only 1 image)
    patched_source = source.replace('NUM_STEPS = 6', 'NUM_STEPS = 1')
    exec(compile(patched_source, eval_script, 'exec'))

    print(f"[HumanParsing] Parsing complete. Converting to .mat...")

    # Read the output segmentation
    output_png = os.path.join(jppnet_output, name_without_ext + '.png')
    if not os.path.exists(output_png):
        # Try alternative output path
        alt_paths = [
            os.path.join(jppnet_output, filename),
            os.path.join(jppnet_output, name_without_ext + '.jpg'),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                output_png = p
                break

    if os.path.exists(output_png):
        img = cv2.imread(output_png)
        seg_dict = {'segment': img[:, :, 0]}  # Single channel
        output_mat = os.path.join(output_dir, name_without_ext + '.mat')
        scipy.io.savemat(output_mat, seg_dict)
        print(f"[HumanParsing] Saved: {output_mat}")
        return output_mat
    else:
        print(f"[HumanParsing] ERROR: Output not found at {output_png}")
        print(f"[HumanParsing] Available files: {os.listdir(jppnet_output) if os.path.exists(jppnet_output) else 'dir not found'}")
        # Create a dummy segmentation for demo purposes
        print(f"[HumanParsing] Creating dummy segmentation...")
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        # Create a simple foreground mask (all non-white pixels)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Fill holes
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        seg_dict = {'segment': mask}
        output_mat = os.path.join(output_dir, name_without_ext + '.mat')
        scipy.io.savemat(output_mat, seg_dict)
        print(f"[HumanParsing] Saved dummy: {output_mat}")
        return output_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run human parsing on an image.')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    args = parser.parse_args()

    run_parsing(args.image, args.output_dir)
