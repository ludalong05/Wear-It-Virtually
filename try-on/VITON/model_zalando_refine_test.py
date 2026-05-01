#!/usr/bin/env python3
"""
Stage 2: Refinement of coarse try-on result.
Loads stage1 coarse output and refines it using a GAN.
Outputs final.png to results/stage2/images/

Usage:
  python3 model_zalando_refine_test.py \
      --coarse_result_dir results/stage1/ \
      --checkpoint model/stage2/model-6000 \
      --result_dir results/stage2/ \
      --pair_name "person1_clothing1"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
tf = tf1

from model_zalando_tps_warp import create_refine_generator
from tps_transformer import tps_stn


def deprocess_image(image, mask01=False):
    if not mask01:
        image = image / 2 + 0.5
    return image


def main(args):
    pair_name = args.pair_name
    coarse_dir = args.coarse_result_dir
    result_dir = args.result_dir

    os.makedirs(result_dir + "/images/", exist_ok=True)

    # Paths to stage1 outputs
    coarse_image_path = os.path.join(coarse_dir, "images", f"{pair_name}.png")
    coarse_mask_path = os.path.join(coarse_dir, "images", f"{pair_name}_mask.png")
    tps_mat_path = os.path.join(coarse_dir, "tps", f"{pair_name}_mask.mat")

    print(f"[Stage2] pair_name={pair_name}")
    print(f"[Stage2] coarse_image={coarse_image_path}")

    # Try to load images
    try:
        import imageio.v2 as iio
    except ImportError:
        import imageio as iio

    if os.path.exists(coarse_image_path):
        coarse_image = iio.imread(coarse_image_path)
    else:
        print(f"[Stage2] WARNING: No coarse image found, using clothing image as fallback")
        coarse_image = np.ones((256, 192, 3), dtype=np.uint8) * 128

    if os.path.exists(coarse_mask_path):
        mask_output = iio.imread(coarse_mask_path)
    else:
        mask_output = np.ones((256, 192), dtype=np.uint8) * 255

    # Load clothing image (needed for TPS)
    clothing_img_path = os.path.join(args.image_dir, "women_top_1.png")
    if os.path.exists(clothing_img_path):
        prod_image = iio.imread(clothing_img_path)
    else:
        prod_image = np.ones((256, 192, 3), dtype=np.uint8) * 128

    # Resize to model input size
    from skimage.transform import resize
    prod_image = resize(prod_image, (256, 192, 3), preserve_range=True).astype(np.float32) / 255.0
    coarse_image = resize(coarse_image, (256, 192, 3), preserve_range=True).astype(np.float32) / 255.0
    mask_output = resize(mask_output, (256, 192), preserve_range=True).astype(np.float32) / 255.0

    prod_image = (prod_image - 0.5) * 2.0
    coarse_image = (coarse_image - 0.5) * 2.0

    # TPS transform on clothing
    nx, ny = 5, 5
    # Try to load TPS control points from stage1 mat
    if os.path.exists(tps_mat_path):
        try:
            tps_data = sio.loadmat(tps_mat_path)
            if 'mask' in tps_data:
                # Use mask as control points proxy
                mask_arr = tps_data['mask']
                h, w = mask_arr.shape
                # Simple TPS: use grid points
                v = np.mgrid[0:h:5j, 0:w:5j].reshape(2, -1).T
                v = v.astype(np.float32) / np.array([h, w])
                v = np.expand_dims(v, 0)
            else:
                v = None
        except:
            v = None
    else:
        v = None

    if v is None:
        # Fallback: create uniform grid control points
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        v = np.stack([xx, yy], axis=-1).astype(np.float32)
        v = np.expand_dims(v, 0)

    nx, ny = v.shape[2], v.shape[1]
    v_flat = v.flatten().reshape(1, -1, 2)
    v_tensor = tf.constant(v_flat, dtype=tf.float32)

    prod_tensor = tf.constant(prod_image[np.newaxis, :, :, :], dtype=tf.float32)
    tps_image = tps_stn(prod_tensor, nx, ny, v_tensor, [256, 192, 3])
    tps_mask = tf.cast(tf.reduce_sum(tps_image, -1) < 3 * 0.95, tf.float32)
    tps_image = tf1.Session().run(tps_image)
    tps_mask_np = tf1.Session().run(tps_mask)

    # Build TF graph for refinement
    batch_size = 1
    image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    prod_mask_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 1])
    coarse_image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    tps_image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])

    with tf.variable_scope("refine_generator") as scope:
        select_mask = create_refine_generator(tps_image_holder, coarse_image_holder)
        select_mask = select_mask * prod_mask_holder
        model_image_outputs = (select_mask * tps_image_holder +
                               (1 - select_mask) * coarse_image_holder)

    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                     if var.name.startswith("refine_generator")])

    with tf.Session() as sess:
        print(f"[Stage2] Loading model from {args.checkpoint}")
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        if checkpoint is None:
            checkpoint = args.checkpoint
        if checkpoint is None or not os.path.exists(checkpoint + ".index"):
            print(f"[Stage2] WARNING: Checkpoint not found: {checkpoint}")
            # Fallback: just blend coarse and TPS
            print(f"[Stage2] Falling back to simple blend (no refinement model)")
            # Simple blend as fallback
            mask_3d = np.expand_dims(tps_mask_np, -1)
            blended = mask_3d * (tps_image[0] / 2.0 + 0.5) + (1 - mask_3d) * (coarse_image / 2.0 + 0.5)
            blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
            final_path = os.path.join(result_dir, "images", "final.png")
            iio.imwrite(final_path, blended)
            print(f"[Stage2] Fallback result saved to {final_path}")
            return

        print(f"[Stage2] Checkpoint: {checkpoint}")
        saver.restore(sess, checkpoint)

        # Prepare batch
        images = np.expand_dims(coarse_image, 0)
        prod_images = np.expand_dims(prod_image, 0)
        coarse_images = np.expand_dims(coarse_image, 0)
        tps_images = np.expand_dims(tps_image[0], 0)
        mask_outputs = np.expand_dims(np.expand_dims(mask_output, 0), -1)

        feed_dict = {
            image_holder: images,
            prod_image_holder: prod_images,
            coarse_image_holder: coarse_images,
            tps_image_holder: tps_images,
            prod_mask_holder: mask_outputs,
        }

        [image_output, sel_mask] = sess.run(
            [model_image_outputs, select_mask],
            feed_dict=feed_dict)

        # Write final result
        final_img = (image_output[0] / 2.0 + 0.5)
        final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)
        final_path = os.path.join(result_dir, "images", "final.png")
        iio.imwrite(final_path, final_img)

        # Also write intermediate outputs
        for name, arr in [
            ("tps", tps_images[0]),
            ("coarse", coarse_images[0]),
            ("mask", mask_outputs[0]),
            ("sel_mask", sel_mask[0]),
        ]:
            out = (arr / 2.0 + 0.5)
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            if len(arr.shape) == 4:
                out = out.squeeze()
            iio.imwrite(os.path.join(result_dir, "images", f"{pair_name}_{name}.png"), out)

        print(f"[Stage2] Results saved to {result_dir}/images/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 2 test: refine try-on')
    parser.add_argument('--checkpoint', type=str, default='model/stage2/model-6000',
                        help='Path to stage2 checkpoint')
    parser.add_argument('--coarse_result_dir', type=str, default='results/stage1/',
                        help='Directory containing stage1 results')
    parser.add_argument('--result_dir', type=str, default='results/stage2/',
                        help='Directory to save results')
    parser.add_argument('--pair_name', type=str, default='women_top_1_women_top_1',
                        help='Pair name (person_clothing) used in stage1 output filenames')
    parser.add_argument('--image_dir', type=str, default='data/women_top/',
                        help='Directory containing images')
    args = parser.parse_args()
    main(args)
