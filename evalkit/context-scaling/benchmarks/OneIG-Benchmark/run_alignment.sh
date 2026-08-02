#!/bin/bash

# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN

# image_root_dir
IMAGE_DIR=""

# model list
MODEL_NAMES=("gpt-4o" "imagen4")
# model_names=("gpt-4o" "imagen4")

# image grid
# IMAGE_GRIDS=("2,2")
IMAGE_GRIDS=("2,2" "1,4")

pip install transformers==4.50.0

# Alignment Score

echo "It's alignment time."

python -m scripts.alignment.alignment_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \
  --class_items "anime" "human" "object" \

# In ZH mode, the class_items list can be extended to include "multilingualism".

rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."