#!/bin/bash
set -e

# Wear-It-Virtually: SmartFit Virtual Try-On Pipeline
# Usage: ./run_smartfit.sh <person_image> <clothing_image>

if [ "$#" -ne 2 ]; then
    echo "ERROR: run_smartfit.sh requires two inputs:"
    echo "  1) relative path to the person image"
    echo "  2) relative path to the clothing image"
    exit 1
fi

PERSON_IMG="$PWD/$1"
CLOTHING_IMG="$PWD/$2"
PROJECT_ROOT="$PWD"

echo "=== SmartFit Virtual Try-On ==="
echo "Person: $PERSON_IMG"
echo "Clothing: $CLOTHING_IMG"

# Validate inputs exist
if [ ! -f "$PERSON_IMG" ]; then
    echo "ERROR: Person image not found: $PERSON_IMG"
    exit 1
fi
if [ ! -f "$CLOTHING_IMG" ]; then
    echo "ERROR: Clothing image not found: $CLOTHING_IMG"
    exit 1
fi

# Clean all output directories
echo ""
echo "==> Cleaning output directories..."
rm -rf try-on/VITON/data/segment/*
rm -f  try-on/VITON/data/pose.pkl
rm -rf try-on/VITON/data/pose/*
rm -rf try-on/VITON/data/women_top/*
rm -rf try-on/VITON/results/stage1/*
rm -rf try-on/VITON/results/stage2/*
rm -rf human_parsing/output/*
rm -rf pose_estimation/output/*
rm -f  output/output.png

# Copy clothing to VITON data dir (needed by both stage1 and stage2)
echo "==> Copying clothing image to VITON..."
cp "$CLOTHING_IMG" try-on/VITON/data/women_top/women_top_1.png

# Step 1: Human Parsing
echo ""
echo "==> [1/3] Running human parsing..."
cd human_parsing/
python3 run_human_parsing.py "$PERSON_IMG" || {
    echo "ERROR: Human parsing failed"
    exit 1
}
cd "$PROJECT_ROOT"

# Check parsing output
if [ ! -f "human_parsing/output/$(basename ${PERSON_IMG%.*}.mat)" ]; then
    echo "ERROR: Human parsing did not produce output"
    exit 1
fi

# Step 2: Pose Estimation
echo ""
echo "==> [2/3] Running pose estimation..."
cd pose_estimation/
python3 run_pose_estimation.py "$PERSON_IMG" --output . || {
    echo "ERROR: Pose estimation failed"
    exit 1
}
cd "$PROJECT_ROOT"

# Check pose output
if [ ! -f "pose_estimation/output/pose.pkl" ]; then
    echo "ERROR: Pose estimation did not produce pose.pkl"
    exit 1
fi

# Move outputs to VITON data directory
echo ""
echo "==> Moving outputs to VITON data directory..."
PERSON_BASENAME=$(basename ${PERSON_IMG%.*})
cp human_parsing/output/${PERSON_BASENAME}.mat try-on/VITON/data/segment/
cp pose_estimation/output/pose.pkl try-on/VITON/data/
cp pose_estimation/output/*.mat try-on/VITON/data/pose/ 2>/dev/null || true

# Step 3: Try-On Stage 1 (Geometric matching)
echo ""
echo "==> [3/3a] Running try-on Stage 1 (geometric matching)..."
cd try-on/VITON/
python3 model_zalando_mask_content_test.py \
    --checkpoint model/stage1/model-15000 \
    --mode test \
    --result_dir results/stage1/ \
    --begin 0 \
    --end 1 || {
    echo "ERROR: Stage 1 failed"
    exit 1
}
cd "$PROJECT_ROOT"

# Step 4: Try-On Stage 2 (Refinement)
echo ""
echo "==> [3/3b] Running try-on Stage 2 (refinement)..."
cd try-on/VITON/
python3 model_zalando_refine_test.py \
    --coarse_result_dir results/stage1/ \
    --checkpoint model/stage2/model-6000 \
    --mode test \
    --result_dir results/stage2/ \
    --begin 0 \
    --end 1 || {
    echo "ERROR: Stage 2 failed"
    exit 1
}
cd "$PROJECT_ROOT"

# Check and copy final output
if [ ! -f "try-on/VITON/results/stage2/images/final.png" ]; then
    echo "ERROR: Final output not found"
    exit 1
fi

cp try-on/VITON/results/stage2/images/final.png output/output.png
echo ""
echo "=== Done! Output saved to output/output.png ==="
