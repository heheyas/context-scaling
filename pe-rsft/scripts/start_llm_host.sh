#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Start LLM host: serve vLLM → health check → run rollout.
#
# Usage:
#   # Single node:
#   bash scripts/start_llm_host.sh --config configs/iter0.yaml \
#       --serve_model /path/to/Qwen3.5-9B --serve_port 8000
#
#   # Multi-node (3 nodes, this is node 1):
#   bash scripts/start_llm_host.sh --config configs/iter0.yaml \
#       --serve_model qwen3.5-9b --serve_port 8000 \
#       --rank 1 --world_size 3

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

# ── Defaults ──
CONFIG=""
SERVE_MODEL=""
SERVE_PORT=8000
RANK=0
WORLD_SIZE=1
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
SERVE_TP=0
SERVE_DP=0
VLLM_EXTRA=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)      CONFIG="$2"; shift 2 ;;
        --serve_model) SERVE_MODEL="$2"; shift 2 ;;
        --serve_port)  SERVE_PORT="$2"; shift 2 ;;
        --rank)        RANK="$2"; shift 2 ;;
        --world_size)  WORLD_SIZE="$2"; shift 2 ;;
        --weights_dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --serve_tp)    SERVE_TP="$2"; shift 2 ;;
        --serve_dp)    SERVE_DP="$2"; shift 2 ;;
        --vllm_extra)  VLLM_EXTRA="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/start_llm_host.sh --config <yaml> [options]"
    echo ""
    echo "  --config        Path to iter config YAML (required)"
    echo "  --serve_model   Model name or path (default: read from config llm.model_path)"
    echo "  --serve_port    vLLM port (default: 8000)"
    echo "  --rank          Node rank for multi-node (default: 0)"
    echo "  --world_size    Total LLM nodes (default: 1)"
    echo "  --weights_dir   Model weights directory"
    echo "  --serve_tp      Tensor parallel size (default: auto)"
    echo "  --serve_dp      Data parallel size (default: auto)"
    echo "  --vllm_extra    Extra args passed to vllm serve (quoted string)"
    exit 1
fi

# If --serve_model not provided, read from config.
if [ -z "$SERVE_MODEL" ]; then
    SERVE_MODEL=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
llm = cfg.get('llm', {})
# Check llm.model_path first, then llm.serve.model_path
path = llm.get('model_path', '') or llm.get('serve', {}).get('model_path', '')
print(path)
" 2>/dev/null)
    if [ -z "$SERVE_MODEL" ]; then
        echo "ERROR: --serve_model not provided and llm.model_path not set in config"
        exit 1
    fi
    echo "[LLM] Model from config: $SERVE_MODEL"
fi

URL="http://localhost:${SERVE_PORT}"
mkdir -p logs

# ──────────────────────────────────────────────
# Kill vLLM processes
# ──────────────────────────────────────────────
kill_serving() {
    echo "[LLM] Killing vLLM processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    ps aux | grep -E "VLLM::|multiprocessing.spawn" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    lsof -ti:${SERVE_PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

trap 'kill_serving; exit 1' INT TERM

# ── Step 1: Kill previous serving ──
kill_serving

# ── Step 2: Start vLLM ──
# Large models (e.g. 35B MoE) can take >10 min to load weights.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
echo "[LLM] Starting vLLM serving (model=$SERVE_MODEL, port=$SERVE_PORT, engine_timeout=${VLLM_ENGINE_READY_TIMEOUT_S}s)..."

# Detect backend from config to pick the right serve script.
BACKEND=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('llm', {}).get('backend', 'vllm'))
" 2>/dev/null || echo "vllm")

SERVE_SCRIPT="${SERVE_SCRIPT:-<PATH_TO_SERVE_SCRIPT>}"   # override via env var

if [ "$BACKEND" = "vllm_mcp" ]; then
    # MCP backend: use FP8 deploy script WITH tool-use flags (default).
    echo "[LLM] MCP backend → $SERVE_SCRIPT (tool calling enabled)"
    MODEL_PATH="$SERVE_MODEL" API_PORT="$SERVE_PORT" \
        bash "$SERVE_SCRIPT" --bf16-online-fp8 \
        < /dev/null > "logs/vllm_llm_rank${RANK}.log" 2>&1 &
else
    # Plain vllm backend: same FP8 deploy script, but DISABLE tool calling
    # via env vars (so the model doesn't emit <tool_call> tags it can't act on).
    echo "[LLM] vllm backend → $SERVE_SCRIPT (tool calling disabled)"
    MODEL_PATH="$SERVE_MODEL" API_PORT="$SERVE_PORT" \
    TOOL_CALL_PARSER="" ENABLE_AUTO_TOOL_CHOICE=0 \
        bash "$SERVE_SCRIPT" --bf16-online-fp8 \
        < /dev/null > "logs/vllm_llm_rank${RANK}.log" 2>&1 &
fi
SERVE_PID=$!
echo "[LLM] vLLM PID=$SERVE_PID"

# ── Step 3: Health check ──
echo "[LLM] Waiting for vLLM to be ready..."
SERVING_OK=false
for i in $(seq 1 180); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[LLM] ERROR: vLLM process died. Check logs/vllm_llm_rank${RANK}.log"
        exit 1
    fi

    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${URL}/health" 2>/dev/null || echo "000")
    if [ "$HEALTH_CODE" = "200" ]; then
        # Verify model is loaded.
        SERVED_MODEL=$(curl -s "${URL}/v1/models" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['data'][0]['id'])
except:
    print('')
" 2>/dev/null || echo "")
        if [ -n "$SERVED_MODEL" ]; then
            SERVING_OK=true
            echo "[LLM] vLLM ready! Model: $SERVED_MODEL"
            break
        fi
    fi
    sleep 5
done

if [ "$SERVING_OK" != "true" ]; then
    echo "[LLM] ERROR: vLLM failed to start after 15 min."
    kill_serving
    exit 1
fi

# ── Step 4: Run rollout ──
echo "[LLM] Starting rollout (rank=$RANK, world_size=$WORLD_SIZE)..."
python3 -m src.llm.rollout \
    --config "$CONFIG" \
    --rank "$RANK" \
    --world_size "$WORLD_SIZE"

ROLLOUT_EXIT=$?

# ── Step 5: Cleanup ──
echo "[LLM] Rollout finished (exit=$ROLLOUT_EXIT). Stopping vLLM..."
kill_serving
exit $ROLLOUT_EXIT
