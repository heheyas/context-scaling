#!/bin/bash

# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN

# image_root_dir
IMAGE_DIR=""

# model list
MODEL_NAMES=("janus-pro")
# model_names=("gpt-4o" "imagen4")

# image grid
IMAGE_GRIDS=("2,2")
# IMAGE_GRIDS=("2,2" "1,4")

echo "Running all evaluation scripts"

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

# Text Score

echo "It's text time."

python -m scripts.text.text_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR/text" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \

# Diversity Score

echo "It's diversity time."

python -m scripts.diversity.diversity_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \
  --class_items "anime" "human" "object" "text" "reasoning" \

# Style Score

echo "It's style time."

python -m scripts.style.style_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR/anime" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \

# Reasoning Score

echo "It's reasoning time."

python -m scripts.reasoning.reasoning_score \
  --mode "$MODE" \
  --image_dirname "${IMAGE_DIR}/reasoning" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \


rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."