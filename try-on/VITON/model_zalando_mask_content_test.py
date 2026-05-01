#!/usr/bin/env python3
"""
Stage 1: Geometric try-on (mask content + TPS warp).
Loads person+clothing, runs the Zalando geometric matching model.
Outputs coarse result to results/stage1/images/

Usage:
  python3 model_zalando_mask_content_test.py \
      --checkpoint model/stage1/model-15000 \
      --person_img ../data/women_top/person.jpg \
      --clothing_img ../data/women_top/clothing.jpg \
      --result_dir results/stage1/
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

# Disable eager execution for TF1 compatibility
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf.compat.v1.disable_eager_execution()
else:
    # TF2 eager by default - wrap to disable
    import tensorflow.compat.v1 as tf1
    tf1.disable_eager_execution()

# Use v1 API if available
if hasattr(tf, 'compat'):
    tf = tf.compat.v1

from utils import *
from model_zalando_mask_content import create_model


def _process_image(person_path, clothing_path, sess,
                   resize_width=192, resize_height=256):
    """Load and preprocess person + clothing images with segmentation and pose."""
    person_id = os.path.splitext(os.path.basename(person_path))[0]
    clothing_id = os.path.splitext(os.path.basename(clothing_path))[0]

    # Load images using imageio v2 API
    try:
        import imageio.v2 as iio
    except ImportError:
        import imageio as iio

    person_img = iio.imread(person_path)
    clothing_img = iio.imread(clothing_path)

    # Handle grayscale images
    if len(person_img.shape) == 2:
        person_img = np.stack([person_img]*3, axis=-1)
    if len(clothing_img.shape) == 2:
        clothing_img = np.stack([clothing_img]*3, axis=-1)

    h, w = person_img.shape[:2]

    # Load segmentation (.mat with 'segment' key)
    segment_path = os.path.join(FLAGS.segment_dir, person_id + '.mat')
    pose_path = os.path.join(FLAGS.pose_dir, person_id + '.mat')

    if os.path.exists(segment_path):
        segment_raw = sio.loadmat(segment_path)["segment"]
        segment_raw = process_segment_map(segment_raw, h, w)
    else:
        print(f"[Stage1] WARNING: No segment file {segment_path}, using dummy")
        segment_raw = np.ones((h, w), dtype=np.int32) * 4  # body class

    if os.path.exists(pose_path):
        pose_raw = sio.loadmat(pose_path)
        pose_raw = extract_pose_keypoints(pose_raw)
        pose_raw = extract_pose_map(pose_raw, h, w)
        pose_raw = np.asarray(pose_raw, np.float32)
    else:
        print(f"[Stage1] WARNING: No pose file {pose_path}, using dummy")
        pose_raw = np.zeros((resize_height, resize_width, 18), dtype=np.float32)

    body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw)

    # Convert and resize
    person_img_f = tf.image.convert_image_dtype(person_img.astype(np.float32)/255.0, dtype=tf.float32)
    clothing_img_f = tf.image.convert_image_dtype(clothing_img.astype(np.float32)/255.0, dtype=tf.float32)

    person_img_f = tf.image.resize(person_img_f, [resize_height, resize_width], method='bilinear')
    clothing_img_f = tf.image.resize(clothing_img_f, [resize_height, resize_width], method='bilinear')
    body_segment_f = tf.image.resize(body_segment, [resize_height, resize_width], method='bilinear')
    skin_segment_f = tf.image.resize(skin_segment, [resize_height, resize_width], method='bilinear')
    prod_segment_f = tf.image.resize(prod_segment, [resize_height, resize_width], method='nearest')

    person_img_f = (person_img_f - 0.5) * 2.0
    clothing_img_f = (clothing_img_f - 0.5) * 2.0
    skin_segment_f = skin_segment_f * person_img_f

    (person_out, clothing_out, body_out, prod_out, skin_out, pose_out) = sess.run([
        person_img_f, clothing_img_f, body_segment_f, prod_segment_f, skin_segment_f, pose_raw
    ])

    return person_out, clothing_out, pose_out, body_out, prod_out, skin_out


def main(args):
    person_img_path = args.person_img
    clothing_img_path = args.clothing_img

    person_name = os.path.splitext(os.path.basename(person_img_path))[0]
    clothing_name = os.path.splitext(os.path.basename(clothing_img_path))[0]
    pair_name = f"{person_name}_{clothing_name}"

    os.makedirs(FLAGS.result_dir + "/images/", exist_ok=True)
    os.makedirs(FLAGS.result_dir + "/tps/", exist_ok=True)

    batch_size = 1

    # Build TF graph
    image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    body_segment_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 1])
    prod_segment_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 1])
    skin_segment_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    pose_map_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 18])

    model = create_model(
        prod_image_holder, body_segment_holder, skin_segment_holder,
        pose_map_holder, prod_segment_holder, image_holder)

    images = np.zeros((batch_size, 256, 192, 3))
    prod_images = np.zeros((batch_size, 256, 192, 3))
    body_segments = np.zeros((batch_size, 256, 192, 1))
    prod_segments = np.zeros((batch_size, 256, 192, 1))
    skin_segments = np.zeros((batch_size, 256, 192, 3))
    pose_raws = np.zeros((batch_size, 256, 192, 18))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print(f"[Stage1] Loading model from {FLAGS.checkpoint}")
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        if checkpoint is None:
            checkpoint = FLAGS.checkpoint
        if checkpoint is None:
            print("[Stage1] ERROR: No checkpoint found!")
            return
        print(f"[Stage1] Checkpoint: {checkpoint}")

        try:
            saver.restore(sess, checkpoint)
        except Exception as e:
            print(f"[Stage1] ERROR: Failed to restore model: {e}")
            return

        print(f"[Stage1] Processing person={person_img_path} clothing={clothing_img_path}")

        (image, prod_image, pose_raw,
         body_segment, prod_segment, skin_segment) = _process_image(
             person_img_path, clothing_img_path, sess)

        images[0] = image
        prod_images[0] = prod_image
        body_segments[0] = body_segment
        prod_segments[0] = prod_segment
        skin_segments[0] = skin_segment
        pose_raws[0] = pose_raw

        feed_dict = {
            image_holder: images,
            prod_image_holder: prod_images,
            body_segment_holder: body_segments,
            skin_segment_holder: skin_segments,
            prod_segment_holder: prod_segments,
            pose_map_holder: pose_raws,
        }

        [image_output, mask_output, step] = sess.run(
            [model.image_outputs, model.mask_outputs, model.global_step],
            feed_dict=feed_dict)

        step = int(step)
        print(f"[Stage1] Inference done, step={step}")

        # Write outputs
        import imageio.v2 as iio

        result_img_path = FLAGS.result_dir + f"/images/{pair_name}.png"
        result_mask_path = FLAGS.result_dir + f"/images/{pair_name}_mask.png"

        output_img = (image_output[0] / 2.0 + 0.5)
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        iio.imwrite(result_img_path, output_img)

        mask_img = np.squeeze(mask_output[0])
        mask_img = np.clip(mask_img * 255, 0, 255).astype(np.uint8)
        iio.imwrite(result_mask_path, mask_img)

        # Save as .mat too
        sio.savemat(FLAGS.result_dir + f"/tps/{pair_name}_mask.mat",
                    {"mask": np.squeeze(mask_output[0])})

        print(f"[Stage1] Results saved to {FLAGS.result_dir}/images/{pair_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 1 test: geometric try-on')
    parser.add_argument('--checkpoint', type=str, default='model/stage1/model-15000',
                        help='Path to stage1 checkpoint')
    parser.add_argument('--person_img', type=str, required=True,
                        help='Path to person image')
    parser.add_argument('--clothing_img', type=str, required=True,
                        help='Path to clothing image')
    parser.add_argument('--result_dir', type=str, default='results/stage1/',
                        help='Directory to save results')
    parser.add_argument('--segment_dir', type=str, default='data/segment/',
                        help='Directory containing segment .mat files')
    parser.add_argument('--pose_dir', type=str, default='data/pose/',
                        help='Directory containing pose .mat files')
    parser.add_argument('--image_dir', type=str, default='data/women_top/',
                        help='Directory containing images')
    args = parser.parse_args()

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("checkpoint", args.checkpoint, "")
    tf.app.flags.DEFINE_string("result_dir", args.result_dir, "")
    tf.app.flags.DEFINE_string("segment_dir", args.segment_dir, "")
    tf.app.flags.DEFINE_string("pose_dir", args.pose_dir, "")
    tf.app.flags.DEFINE_string("image_dir", args.image_dir, "")

    print(f"[Stage1] Starting with checkpoint={args.checkpoint}")
    main(args)
