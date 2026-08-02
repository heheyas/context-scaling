#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Serve a Qwen3.5 LLM model via vLLM for prompt rewriting.
# Adapted from EvalKit/context-scaling/scripts/serve_qwen_llm.sh.
#
# Usage:
#   bash scripts/serve_qwen_llm.sh --model /path/to/Qwen3.5-9B [--port 8000] [--tp 1] [--dp 0]
#   bash scripts/serve_qwen_llm.sh --model qwen3.5-9b --weights_dir <HDFS_ROOT>/user/.../weights

set -euo pipefail

# ── Defaults ──
MODEL=""
PORT=8000
TP=0
DP=0
EAGER=true
EXTRA_ARGS=""
WEIGHTS_DIR="${WEIGHTS_DIR:-<HDFS_ROOT>/weights}"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        --tp)          TP="$2"; shift 2 ;;
        --dp)          DP="$2"; shift 2 ;;
        --weights_dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --eager)       EAGER=true; shift ;;
        --no-eager)    EAGER=false; shift ;;
        --extra)       EXTRA_ARGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/serve_qwen_llm.sh --model <name_or_path> [options]"
    echo ""
    echo "  --model        Model name (e.g. qwen3.5-9b) or full path"
    echo "  --port PORT    HTTP port (default: 8000)"
    echo "  --tp N         Tensor parallel size (default: auto)"
    echo "  --dp N         Data parallel size (default: auto)"
    echo "  --weights_dir  Base directory for model weights (default: \$WEIGHTS_DIR)"
    exit 1
fi

# ── Resolve model path ──
declare -A MODEL_MAP
MODEL_MAP[qwen3.5-0.8b]="Qwen3.5-0.8B"
MODEL_MAP[qwen3.5-4b]="Qwen3.5-4B-Base"
MODEL_MAP[qwen3.5-9b]="Qwen3.5-9B"
MODEL_MAP[qwen3.5-35b]="Qwen3.5-35B-A3B"
MODEL_MAP[qwen3.5-122b]="Qwen3.5-122B-A10B"

MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if [ -n "${MODEL_MAP[$MODEL_LOWER]+x}" ]; then
    MODEL_PATH="${WEIGHTS_DIR}/${MODEL_MAP[$MODEL_LOWER]}"
elif [ -d "$MODEL" ]; then
    MODEL_PATH="$MODEL"
elif [ -d "${WEIGHTS_DIR}/${MODEL}" ]; then
    MODEL_PATH="${WEIGHTS_DIR}/${MODEL}"
else
    echo "Error: Cannot resolve model: $MODEL"
    echo "Tried: MODEL_MAP, direct path, \${WEIGHTS_DIR}/${MODEL}"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# ── Auto-detect TP/DP based on GPU count ──
TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)

if [ "$TP" -eq 0 ] && [ "$DP" -eq 0 ]; then
    case "$MODEL_LOWER" in
        qwen3.5-0.8b|qwen3.5-4b|qwen3.5-9b)
            TP=1; DP=$TOTAL_GPUS ;;
        qwen3.5-35b)
            TP=2; DP=$((TOTAL_GPUS / 2)) ;;
        qwen3.5-122b)
            TP=4; DP=$((TOTAL_GPUS / 4)) ;;
        *)
            TP=1; DP=$TOTAL_GPUS ;;
    esac
fi

# ── Build vllm args ──
VLLM_ARGS=""
[ "$TP" -gt 1 ] && VLLM_ARGS="${VLLM_ARGS} --tensor-parallel-size ${TP}"
[ "$DP" -gt 1 ] && VLLM_ARGS="${VLLM_ARGS} --data-parallel-size ${DP}"

# MoE models benefit from expert parallelism.
case "$MODEL_LOWER" in
    qwen3.5-35b|qwen3.5-122b) VLLM_ARGS="${VLLM_ARGS} --enable-expert-parallel" ;;
esac

[ "$EAGER" = true ] && VLLM_ARGS="${VLLM_ARGS} --enforce-eager"

VLLM_ARGS="${VLLM_ARGS} --language-model-only"
VLLM_ARGS="${VLLM_ARGS} --enable-prefix-caching"
VLLM_ARGS="${VLLM_ARGS} --max-num-batched-tokens 16384"
VLLM_ARGS="${VLLM_ARGS} --reasoning-parser qwen3"

# MoE models benefit from expert parallelism.
case "$MODEL_LOWER" in
    qwen3.5-35b|qwen3.5-122b) VLLM_ARGS="${VLLM_ARGS} --enable-expert-parallel" ;;
esac

echo "============================================"
echo "vLLM Qwen3.5 Serving"
echo "============================================"
echo "  Model:       $MODEL_PATH"
echo "  Port:        $PORT"
echo "  TP:          $TP"
echo "  DP:          $DP"
echo "  Total GPUs:  $TOTAL_GPUS"
echo "============================================"

# Kill any existing vllm on this port.
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
lsof -ti:${PORT} 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 3

exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    ${VLLM_ARGS} \
    ${EXTRA_ARGS}
