#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Sweep 4 GPG settings (v5 schema, v6 dropmore, v7 schema+dropmore, v8 super)
# against the same paired pool, 8-shard parallel on 8 GPUs.
#
# Override defaults via environment:
#   POOL_DIR=/path/to/detailness ./overnight.sh
#   MODEL=/path/to/qwen3vl       ./overnight.sh
set -e
export HOME=${HOME:-<RAY_SERVE_HOME>}

POOL_DIR=${POOL_DIR:-/tmp/detailness_real_n100}
MODEL=${MODEL:-<HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct}

# Repo root, so we can call `python -m detailness.gpg.score`
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$REPO"

LOG=$POOL_DIR/overnight.log
echo "=== overnight start $(date) ===" > $LOG
echo "POOL_DIR=$POOL_DIR  MODEL=$MODEL" >> $LOG

wait_for_shards() {
  local dir=$1
  while [ "$(ls $dir/shard*.jsonl 2>/dev/null | wc -l)" -lt 8 ] || \
        [ "$(grep -l 'done\]' $dir/shard*.log 2>/dev/null | wc -l)" -lt 8 ]; do
    sleep 30
  done
}

KINDS_V6="structured,dense,abl_json_full,abl_json_no_depth,abl_json_no_atmosphere_lighting,abl_json_no_relationships,abl_json_no_scene,abl_json_no_bbox,abl_json_no_element_photography,spatial_coarse,spatial_fine,spatial_finer"
# v7/v8 add a system prompt that only applies to structured + abl_* kinds (no spatial)
SP_KINDS="structured,abl_json_full,abl_json_no_depth,abl_json_no_atmosphere_lighting,abl_json_no_relationships,abl_json_no_scene,abl_json_no_bbox,abl_json_no_element_photography,spatial_coarse,spatial_fine,spatial_finer"

run_sweep() {
  local out_dir=$1; shift
  mkdir -p "$out_dir"
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i nohup python -m detailness.gpg.score \
      --pool $POOL_DIR/pool.jsonl \
      --pool-dir $POOL_DIR \
      --model $MODEL \
      --out $out_dir/shard${i}.jsonl \
      --shard $i --num-shards 8 \
      --content-only --canonicalize-json \
      --kinds-filter "$KINDS_V6" \
      "$@" \
      > $out_dir/shard${i}.log 2>&1 &
  done
  wait_for_shards "$out_dir"
}

SCHEMA_PROMPT=evalkit/detailness/gpg/prompts/structured_schema.txt
DROP_V6="atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect"
DROP_V8="atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect,visibility,composition,perspective"

# v5 schema system prompt only (no extra canonicalize)
echo "=== EXP1 v5_schema ===" >> $LOG
run_sweep $POOL_DIR/gpg_v5_schema \
  --system-prompt-file $SCHEMA_PROMPT \
  --system-prompt-kinds "$SP_KINDS"
echo "v5 done at $(date)" >> $LOG

# v6 extra canonicalize (no schema prompt) — paper version
echo "=== EXP2 v6_dropmore ===" >> $LOG
run_sweep $POOL_DIR/gpg_v6_dropmore \
  --extra-drop-keys "$DROP_V6"
echo "v6 done at $(date)" >> $LOG

# v7 schema + extra canonicalize
echo "=== EXP3 v7_combined ===" >> $LOG
run_sweep $POOL_DIR/gpg_v7_combined \
  --extra-drop-keys "$DROP_V6" \
  --system-prompt-file $SCHEMA_PROMPT \
  --system-prompt-kinds "$SP_KINDS"
echo "v7 done at $(date)" >> $LOG

# v8 super aggressive
echo "=== EXP4 v8_super ===" >> $LOG
run_sweep $POOL_DIR/gpg_v8_super \
  --extra-drop-keys "$DROP_V8" \
  --system-prompt-file $SCHEMA_PROMPT \
  --system-prompt-kinds "$SP_KINDS"
echo "v8 done at $(date)" >> $LOG

echo "=== ALL DONE $(date) ===" >> $LOG
