#!/bin/bash
# VITON try-on pipeline (Stage 1 + Stage 2)
# Usage: ./run_try-on.sh <person_img> <clothing_img>

set -e

PERSON_IMG="$1"
CLOTHING_IMG="$2"

if [ -z "$PERSON_IMG" ] || [ -z "$CLOTHING_IMG" ]; then
    echo "Usage: $0 <person_image> <clothing_image>"
    exit 1
fi

PERSON_NAME=$(basename "${PERSON_IMG%.*}")
PAIR_NAME="${PERSON_NAME}_women_top_1"

echo "==> VITON Stage 1: Geometric matching..."
python3 model_zalando_mask_content_test.py \
    --checkpoint model/stage1/model-15000 \
    --person_img ../data/women_top/${PERSON_NAME}.jpg \
    --clothing_img ../data/women_top/women_top_1.png \
    --result_dir results/stage1/ \
    --pair_name "$PAIR_NAME"

echo "==> VITON Stage 2: Refinement..."
python3 model_zalando_refine_test.py \
    --checkpoint model/stage2/model-6000 \
    --coarse_result_dir results/stage1/ \
    --result_dir results/stage2/ \
    --pair_name "$PAIR_NAME"

echo "==> Done. Final output: results/stage2/images/final.png"
