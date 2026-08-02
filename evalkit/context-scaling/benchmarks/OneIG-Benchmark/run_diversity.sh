#!/bin/bash

# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN

# image_root_dir
IMAGE_DIR=""

# model list
MODEL_NAMES=("gpt-4o")
# model_names=("gpt-4o" "imagen4")

# image grid
IMAGE_GRIDS=("2,2")
# IMAGE_GRIDS=("2,2" "1,4")

pip install transformers==4.50.0

# Diversity Score

echo "It's diversity time."

python -m scripts.diversity.diversity_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \
  --class_items "anime" "human" "object" "text" "reasoning" \

# In ZH mode, the class_items list can be extended to include "multilingualism".

rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."