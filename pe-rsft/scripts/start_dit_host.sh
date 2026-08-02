#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Start DiT host: serve native → health check → run gen_images.
#
# Usage:
#   bash scripts/start_dit_host.sh --config configs/iter0.yaml \
#       --serve_ckpt /path/to/model.safetensors --serve_port 8091
#
#   # With specific GPUs:
#   bash scripts/start_dit_host.sh --config configs/iter0.yaml \
#       --serve_ckpt /path/to/model.safetensors --serve_port 8091 \
#       --serve_gpus 0,1,2,3

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# ── Defaults ──
CONFIG=""
SERVE_CKPT=""
SERVE_PORT=8091
SERVE_GPUS=""
SERVE_MAX_PER_GPU=4
CKPT_ROOT=""
SERVE_TRIAL=""
SERVE_STEP=""
SERVE_EMA=""
RANK=""

# HDFS paths.
TRIALS_ROOT="${TRIALS_ROOT:-<HDFS_ROOT>/<TRIALS_ROOT>}"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)            CONFIG="$2"; shift 2 ;;
        --serve_ckpt)        SERVE_CKPT="$2"; shift 2 ;;
        --serve_port)        SERVE_PORT="$2"; shift 2 ;;
        --serve_gpus)        SERVE_GPUS="$2"; shift 2 ;;
        --serve_max_per_gpu) SERVE_MAX_PER_GPU="$2"; shift 2 ;;
        --ckpt_root)         CKPT_ROOT="$2"; shift 2 ;;
        --serve_trial)       SERVE_TRIAL="$2"; shift 2 ;;
        --serve_step)        SERVE_STEP="$2"; shift 2 ;;
        --serve_ema)         SERVE_EMA="true"; shift ;;
        --rank)              RANK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/start_dit_host.sh --config <yaml> [options]"
    echo ""
    echo "  --config              Path to iter config YAML"
    echo "  --serve_ckpt          Path to merged model.safetensors"
    echo "  --serve_port          Serve port (default: 8091)"
    echo "  --serve_gpus          GPU ids (e.g., 0,1,2,3; default: all)"
    echo "  --serve_max_per_gpu   Max concurrent requests per GPU (default: 4)"
    echo "  --ckpt_root           QwenImage base weights path"
    echo "  --serve_trial         Trial name (for checkpoint merge)"
    echo "  --serve_step          Step number (for checkpoint merge)"
    echo "  --serve_ema           Use EMA weights"
    exit 1
fi

URL="http://localhost:${SERVE_PORT}"
mkdir -p logs

# ── HDFS helpers ──
to_hdfs_url() {
    echo "$1" | sed 's|^<HDFS_ROOT>/user/|<HDFS_NATIVE>/home/data|'
}

hdfs_cache() {
    local src="$1" dst="$2"
    if [ ! -f "$dst" ]; then
        echo "[DiT] Caching $(to_hdfs_url "$src") -> $dst..."
        hdfs dfs -get "$(to_hdfs_url "$src")" "$dst"
    fi
}

hdfs_cache_dir() {
    local src="$1" dst="$2"
    if [ ! -d "$dst" ]; then
        echo "[DiT] Caching $(to_hdfs_url "$src") -> $dst..."
        hdfs dfs -get "$(to_hdfs_url "$src")" "$dst"
    fi
}

# ──────────────────────────────────────────────
# Kill serving
# ──────────────────────────────────────────────
kill_serving() {
    echo "[DiT] Killing serving..."
    if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
        kill -9 -"$SERVE_PID" 2>/dev/null || true
        kill -9 "$SERVE_PID" 2>/dev/null || true
    fi
    lsof -ti:${SERVE_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

SERVE_PID=""
trap 'kill_serving; exit 1' INT TERM

# ── Step 1: Resolve checkpoint ──
if [ -n "$SERVE_TRIAL" ] && [ -z "$SERVE_CKPT" ]; then
    WEIGHT_NAME="model"
    [ "$SERVE_EMA" = "true" ] && WEIGHT_NAME="ema"
    SERVE_CKPT="${TRIALS_ROOT}/${SERVE_TRIAL}/${SERVE_STEP}/${WEIGHT_NAME}.safetensors"

    if [ ! -f "$SERVE_CKPT" ]; then
        echo "[DiT] ERROR: Checkpoint not found: $SERVE_CKPT"
        echo "[DiT] If checkpoint needs merging, run the merge script first."
        exit 1
    fi
fi

if [ -z "$SERVE_CKPT" ]; then
    echo "[DiT] ERROR: --serve_ckpt or --serve_trial required"
    exit 1
fi

# ── Step 2: Cache checkpoint to local ──
LOCAL_CKPT="/tmp/$(basename "$SERVE_CKPT")"
if [[ "$SERVE_CKPT" == <HDFS_ROOT>/* ]]; then
    hdfs_cache "$SERVE_CKPT" "$LOCAL_CKPT"
else
    LOCAL_CKPT="$SERVE_CKPT"
fi

# ── Step 3: Cache base weights if needed ──
if [ -z "$CKPT_ROOT" ]; then
    CKPT_ROOT_LOCAL="<QWENIMAGE_CKPT>"
    CKPT_ROOT_HDFS="<WEIGHTS_ROOT>/qwenimage"
    hdfs_cache_dir "$CKPT_ROOT_HDFS" "$CKPT_ROOT_LOCAL"
    CKPT_ROOT="${CKPT_ROOT_LOCAL}/origin/raw_data"
fi

# ── Read shared_root from config ──
SHARED_ROOT=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('shared_root', ''))
" 2>/dev/null)
echo "[DiT] shared_root=$SHARED_ROOT"

# ── Step 4: Kill previous serving ──
kill_serving

# ── Step 5: Start native serving ──
GPU_ARGS=""
[ -n "$SERVE_GPUS" ] && GPU_ARGS="--gpus $SERVE_GPUS"

# Use the serve script from the original context-scaling repo (has correct PYTHONPATH for modeling.*).
QWENIMAGE_ROOT="${QWENIMAGE_ROOT:-<LOCAL_ROOT>/repos/context-scaling}"
SERVE_SCRIPT="${QWENIMAGE_ROOT}/scripts/merge/serve_qwenimage_multigpu.py"

SERVE_CMD="python3 $SERVE_SCRIPT \
    --merged_ckpt $LOCAL_CKPT \
    --ckpt_root $CKPT_ROOT \
    --port $SERVE_PORT \
    --max_per_gpu $SERVE_MAX_PER_GPU \
    $GPU_ARGS"

SERVE_LOG="logs/dit_serve_port${SERVE_PORT}.log"
echo "[DiT] Starting native serving..."
echo "[DiT] Command: $SERVE_CMD"
setsid $SERVE_CMD > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
echo "[DiT] Serve PID=$SERVE_PID"

# ── Step 6: Two-phase health check ──
# Phase 1: wait for /health to respond.
echo "[DiT] Phase 1: Waiting for server process..."
for i in $(seq 1 120); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[DiT] ERROR: Serving process died. Check $SERVE_LOG"
        exit 1
    fi
    if curl -s "${URL}/health" --max-time 3 > /dev/null 2>&1; then
        echo "[DiT] Server process started, waiting for workers..."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "[DiT] ERROR: Server timed out (10 min). Check $SERVE_LOG"
        kill_serving; exit 1
    fi
    sleep 5
done

# Phase 2: test with real generation request.
echo "[DiT] Phase 2: Testing generation..."
for i in $(seq 1 120); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[DiT] ERROR: Serving process died. Check $SERVE_LOG"
        exit 1
    fi
    TEST_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${URL}/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"test","height":256,"width":256,"num_steps":1,"seed":0,"cfg_scale":1.0}' \
        --max-time 120 2>/dev/null || echo "000")
    if [ "$TEST_CODE" = "200" ]; then
        echo "[DiT] Serving ready! (test generation returned 200)"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "[DiT] ERROR: Workers not ready (10 min). Check $SERVE_LOG"
        kill_serving; exit 1
    fi
    sleep 5
done

# ── Step 7: Run gen_images ──
# gen_images handles burn_gpu internally: when idle (no shards to claim),
# it starts burn_gpu.py as a subprocess to hold GPUs; when a shard appears,
# it kills burn_gpu and processes the shard. No serve restart needed.
RANK_ARG=""
[ -n "$RANK" ] && RANK_ARG="--rank $RANK"
echo "[DiT] Starting gen_images consumer (rank=${RANK:-auto})..."
python3 -m src.dit.gen_images --config "$CONFIG" $RANK_ARG

GEN_EXIT=$?

# ── Step 8: Cleanup ──
echo "[DiT] gen_images finished (exit=$GEN_EXIT). Stopping serving..."
kill_serving
exit $GEN_EXIT
