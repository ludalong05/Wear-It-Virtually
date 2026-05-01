#!/bin/bash
set -e
python3 model_zalando_refine_test.py \
  --coarse_result_dir results/stage1/ \
  --checkpoint model/stage2/model-6000 \
  --mode test \
  --result_dir results/stage2/ \
  --begin 0 \
  --end 1
echo "Stage 2 completed."
