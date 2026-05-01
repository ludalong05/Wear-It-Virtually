# Wear-It-Virtually — Virtual Try-On

Virtual clothing try-on using human parsing, pose estimation, and geometric image warping.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no model weights needed — uses OpenCV)
python run_tryon.py inputs/person.jpg inputs/clothing.jpg --demo

# Run full pipeline (requires model weights — see "Downloading Models" below)
python run_tryon.py inputs/person.jpg inputs/clothing.jpg

# Run Flask API
python app.py
# POST to http://localhost:5000/api/infer with files "person" and "clothing"
```

## Project Structure

```
Wear-It-Virtually/
├── app.py                          # Flask API server
├── run_tryon.py                    # End-to-end Python runner (demo + real pipeline)
├── run_smartfit.sh                 # Shell pipeline (calls Python scripts)
├── requirements.txt
│
├── human_parsing/                  # Step 1: Body segmentation
│   ├── run_human_parsing.py        # Python wrapper
│   └── LIP_JPPNet/                 # JPPNet model (cloned submodule)
│
├── pose_estimation/                # Step 2: Keypoint extraction
│   ├── run_pose_estimation.py      # Python wrapper
│   └── keras_Realtime_MultiPerson_Pose_Estimation/  # OpenPose model
│
└── try-on/VITON/                   # Steps 3-4: Geometric matching + refinement
    ├── model_zalando_mask_content_test.py   # Stage 1: Geometric warp
    ├── model_zalando_refine_test.py          # Stage 2: Refinement
    └── run_try-on.sh               # Shell pipeline for VITON stage
```

## Architecture

1. **Human Parsing** — Segment person image into body parts (LIP_JPPNet)
2. **Pose Estimation** — Extract 18-keypoint body pose (OpenPose-style)
3. **Stage 1 (Geometric Matching)** — TPS warp clothing to fit body pose
4. **Stage 2 (Refinement)** — Refine coarse result with GAN

## Downloading Model Weights

The original weights are hosted on Dropbox (links in `setup.sh`). After running the setup:

```bash
bash setup.sh   # Downloads all model weights
```

Expected checkpoint locations:
```
human_parsing/LIP_JPPNet/checkpoint/human_parsing_model.zip  → extracted
pose_estimation/keras_Realtime_MultiPerson_Pose_Estimation/model/keras/model.h5
try-on/VITON/model/stage1/model-15000
try-on/VITON/model/stage2/model-6000
```

## Python Version

Tested with **Python 3.9+**. The original project used Python 2.7 which is no longer supported.

## Requirements

```
flask>=2.0
flask-cors>=3.0
werkzeug>=2.0
Pillow>=9.0
opencv-python>=4.5
numpy>=1.21
scipy>=1.7
scikit-image>=0.19
imageio>=2.19
tensorflow>=2.10
keras>=2.10
```
