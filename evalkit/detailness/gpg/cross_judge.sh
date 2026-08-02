#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Cross-judge GPG robustness sweep: re-run the v6_dropmore recipe with
# multiple VLM judges so we can report robustness in the paper appendix.
#
# Small judges (≤16GB):  8-shard parallel, one shard per GPU.
# Large MoE judges:      device_map=auto across all 8 GPUs, sequential
#                        shards (16 shards × 8 GPUs since each shard
#                        already uses all GPUs).
#
# Usage:
#   POOL_DIR=/tmp/detailness_real_n100 ./cross_judge.sh           # default judges
#   POOL_DIR=/tmp/detailness_real_n100 JUDGES=qwen2_5_vl,internvl3 ./cross_judge.sh
set -e

POOL_DIR=${POOL_DIR:-/tmp/detailness_real_n100}
JUDGES=${JUDGES:-qwen2_vl,qwen2_5_vl,qwen3_vl,qwen3_5_moe_35b,qwen3_5_moe_122b,internvl3}

# Repo root, so we can `python -m detailness.gpg.score`
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$REPO"

LOG=$POOL_DIR/cross_judge.log
echo "=== cross_judge start $(date) ===" > $LOG
echo "POOL_DIR=$POOL_DIR  JUDGES=$JUDGES" | tee -a $LOG

# ------------------------------------------------------------------ #
# Judge registry: tag → (model_path, size_class)
#   size_class = "small" → 8 shards × 1 GPU each
#   size_class = "moe"   → 1 shard × all 8 GPUs (device_map=auto)
# ------------------------------------------------------------------ #
WEIGHTS_HDFS=<HDFS_ROOT>/weights
WEIGHTS_BN=<LOCAL_ROOT>/weights

declare -A JUDGE_PATH JUDGE_CLASS
JUDGE_PATH[qwen2_vl]=$WEIGHTS_HDFS/Qwen/Qwen2-VL-7B-Instruct/origin/raw_data ; JUDGE_CLASS[qwen2_vl]=small
JUDGE_PATH[qwen2_5_vl]=$WEIGHTS_HDFS/Qwen2.5-VL-7B-Instruct        ; JUDGE_CLASS[qwen2_5_vl]=small
JUDGE_PATH[qwen3_vl]=$WEIGHTS_HDFS/Qwen3-VL-8B-Instruct            ; JUDGE_CLASS[qwen3_vl]=small
JUDGE_PATH[qwen3_5_moe_35b]=$WEIGHTS_HDFS/Qwen3.5-35B-A3B         ; JUDGE_CLASS[qwen3_5_moe_35b]=moe
JUDGE_PATH[qwen3_5_moe_122b]=$WEIGHTS_HDFS/Qwen3.5-122B-A10B      ; JUDGE_CLASS[qwen3_5_moe_122b]=moe
JUDGE_PATH[internvl3]=$WEIGHTS_BN/InternVL3-8B                     ; JUDGE_CLASS[internvl3]=small

# v6_dropmore recipe (must match paper)
DROP_KEYS="atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect"
KINDS="structured,dense,abl_json_full,abl_json_no_depth,abl_json_no_atmosphere_lighting,abl_json_no_relationships,abl_json_no_scene,abl_json_no_bbox,abl_json_no_element_photography,spatial_coarse,spatial_fine,spatial_finer"

wait_for_shards() {
  local dir=$1 n=$2
  while [ "$(ls $dir/shard*.jsonl 2>/dev/null | wc -l)" -lt "$n" ] || \
        [ "$(grep -l 'done\]' $dir/shard*.log 2>/dev/null | wc -l)" -lt "$n" ]; do
    sleep 30
  done
}

run_small_judge() {
  # 8 parallel shards, one per GPU
  local tag=$1 model=$2 out_dir=$3
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i nohup python -m detailness.gpg.score \
      --pool $POOL_DIR/pool.jsonl --pool-dir $POOL_DIR \
      --model "$model" \
      --out $out_dir/shard${i}.jsonl \
      --shard $i --num-shards 8 \
      --content-only --canonicalize-json \
      --extra-drop-keys "$DROP_KEYS" \
      --kinds-filter "$KINDS" \
      > $out_dir/shard${i}.log 2>&1 &
  done
  wait_for_shards $out_dir 8
}

run_moe_judge() {
  # 1 shard at a time, device_map=auto over all visible GPUs.
  # Run sequentially: 8 shards × ~10 min/shard ≈ 80 min total for 35B,
  # ≈ 3 hr for 122B.
  local tag=$1 model=$2 out_dir=$3
  for i in 0 1 2 3 4 5 6 7; do
    nohup python -m detailness.gpg.score \
      --pool $POOL_DIR/pool.jsonl --pool-dir $POOL_DIR \
      --model "$model" \
      --out $out_dir/shard${i}.jsonl \
      --shard $i --num-shards 8 \
      --device auto --max-memory-per-gpu 70GiB \
      --content-only --canonicalize-json \
      --extra-drop-keys "$DROP_KEYS" \
      --kinds-filter "$KINDS" \
      > $out_dir/shard${i}.log 2>&1
  done
}

# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #
IFS=',' read -ra JUDGE_LIST <<< "$JUDGES"
for tag in "${JUDGE_LIST[@]}"; do
  model=${JUDGE_PATH[$tag]}
  class=${JUDGE_CLASS[$tag]}
  out=$POOL_DIR/gpg_v6_${tag}
  if [ -z "$model" ]; then
    echo "[skip] unknown judge tag: $tag" | tee -a $LOG
    continue
  fi
  if [ ! -d "$model" ]; then
    echo "[skip] $tag: model path $model does not exist" | tee -a $LOG
    continue
  fi
  mkdir -p $out

  done_recs=$(cat $out/shard*.jsonl 2>/dev/null | wc -l)
  if [ "$done_recs" -ge 3500 ]; then
    echo "[skip] $tag: already has $done_recs records in $out" | tee -a $LOG
    continue
  fi

  echo "=== $tag ($class) start $(date) ===" | tee -a $LOG
  if [ "$class" == "small" ]; then
    run_small_judge $tag "$model" $out
  else
    run_moe_judge $tag "$model" $out
  fi
  echo "=== $tag done $(date) ===" | tee -a $LOG
done

echo "=== ALL DONE $(date) ===" | tee -a $LOG
