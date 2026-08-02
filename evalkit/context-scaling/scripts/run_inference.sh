#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Run inference for any benchmark with rewritten prompts.
# Supports: external URL, auto merge+serve via vllm-omni, or auto merge+serve via native serving.
#
# Model types:
#   qwenimage  — QwenImage diffusion model (default)
#   bagel      — Bagel (CausalFusion) model
#
# Serving modes (auto-selected based on model type, or override with --serve_mode):
#   vllm       — vllm-omni data-parallel serving (default for qwenimage)
#   native     — Custom multi-GPU FastAPI serving (default for bagel, also available for qwenimage)
#   external   — User-provided URL (--url), no auto merge/serve
#
# Usage:
#   # QwenImage + vllm-omni (default):
#   bash scripts/run_inference.sh --benchmark geneval2 --rewrite gemini \
#       --serve_trial <TRIAL_NAME> \
#       --serve_step 0014500 --trial sft-from7k-14500
#
#   # QwenImage + native serving:
#   bash scripts/run_inference.sh --benchmark geneval2 --rewrite gemini \
#       --model qwenimage --serve_mode native \
#       --serve_trial <TRIAL_NAME> \
#       --serve_step 0014500 --trial sft-from7k-14500
#
#   # Bagel (auto native serving):
#   bash scripts/run_inference.sh --benchmark dpgbench --rewrite gemini \
#       --model bagel \
#       --serve_trial <TRIAL_NAME> \
#       --serve_step 0030000 --trial <TRIAL_NAME>
#
#   # With EMA weights:
#   bash scripts/run_inference.sh --benchmark dpgbench --rewrite gemini \
#       --model bagel --serve_ema \
#       --serve_trial <TRIAL_NAME> \
#       --serve_step 0030000 --trial <TRIAL_NAME>
#
#   # External serving (already running):
#   bash scripts/run_inference.sh --benchmark geneval2 --rewrite gemini \
#       --url http://localhost:8899 --trial my_exp
#
# Required:
#   --benchmark      geneval, geneval2, dpgbench, oneig, sp_l10_200
#   --rewrite        gemini, gpt, qwen3.5-9b, l10, etc. (selects rewritten JSONL)
#   --trial          output directory trial name
#   --url OR --serve_trial   (one of the two)

set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)"

# ── Paths ──
TRIALS_ROOT="<HDFS_ROOT>/<TRIALS_ROOT>"
QWENIMAGE_ROOT="<LOCAL_ROOT>/repos/context-scaling"
BAGEL_ROOT="<LOCAL_ROOT>/repos/<BAGEL_REPO>"
SERVE_VLLM_SCRIPT="${QWENIMAGE_ROOT}/scripts/merge/serve_vllm_omni_dp.sh"

# Bagel dependencies
BAGEL_LLM_PATH="<LLM_PATH>"
BAGEL_LLM_HDFS="<HDFS_NATIVE>/home/data<USER>/model/Qwen2.5-7B-Instruct"
BAGEL_VAE_PATH="<WEIGHTS_ROOT>/vae/ae.safetensors"

# ── Defaults ──
MODEL="qwenimage"
SERVE_MODE=""           # auto-select based on model type
BENCHMARK=""
REWRITE=""
URL=""
TRIAL=""
SERVE_TRIAL=""
SERVE_STEP=""
SERVE_EMA=""
SERVE_PORT=8091
SERVE_REPLICAS=4
SERVE_CFG=2
SERVE_MAX_PER_GPU=4
HEIGHT=""
WIDTH=""
NUM_STEPS=25
CFG_SCALE=""
SEED=42
WORKERS=32
MAX_PROMPTS=0
MODEL_NAME="model"
DISABLE_REWRITTEN=""
FIX_LENS_KEY=""
INFERENCE_MODULE=""
REWRITTEN_JSONL_OVERRIDE=""
MAX_PROMPT_CHARS=0
TIMESTEP_SHIFT=""
SERVE_GPUS=""
SERVE_CKPT=""
RANK=""
WORLD_SIZE=""
PSM=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2"; shift 2 ;;
        --serve_mode)     SERVE_MODE="$2"; shift 2 ;;
        --benchmark)      BENCHMARK="$2"; shift 2 ;;
        --rewrite)        REWRITE="$2"; shift 2 ;;
        --url)            URL="$2"; shift 2 ;;
        --trial)          TRIAL="$2"; shift 2 ;;
        --serve_trial)    SERVE_TRIAL="$2"; shift 2 ;;
        --serve_step)     SERVE_STEP="$2"; shift 2 ;;
        --serve_ema)      SERVE_EMA="--ema"; shift ;;
        --serve_port)     SERVE_PORT="$2"; shift 2 ;;
        --serve_replicas) SERVE_REPLICAS="$2"; shift 2 ;;
        --serve_cfg)      SERVE_CFG="$2"; shift 2 ;;
        --serve_max_per_gpu) SERVE_MAX_PER_GPU="$2"; shift 2 ;;
        --serve_gpus)     SERVE_GPUS="$2"; shift 2 ;;
        --serve_ckpt)     SERVE_CKPT="$2"; shift 2 ;;
        --height)         HEIGHT="$2"; shift 2 ;;
        --width)          WIDTH="$2"; shift 2 ;;
        --num_steps)      NUM_STEPS="$2"; shift 2 ;;
        --cfg_scale)      CFG_SCALE="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --workers)        WORKERS="$2"; shift 2 ;;
        --max_prompts)    MAX_PROMPTS="$2"; shift 2 ;;
        --model_name)     MODEL_NAME="$2"; shift 2 ;;
        --disable_rewritten) DISABLE_REWRITTEN="--disable_rewritten"; shift ;;
        --fix_lens_key)   FIX_LENS_KEY="--fix_lens_key"; shift ;;
        --rewritten_jsonl) REWRITTEN_JSONL_OVERRIDE="$2"; shift 2 ;;
        --max_prompt_chars) MAX_PROMPT_CHARS="$2"; shift 2 ;;
        --timestep_shift) TIMESTEP_SHIFT="$2"; shift 2 ;;
        --rank)           RANK="$2"; shift 2 ;;
        --world_size)     WORLD_SIZE="$2"; shift 2 ;;
        --vae_path)       BAGEL_VAE_PATH="$2"; shift 2 ;;
        --llm_path)       BAGEL_LLM_PATH="$2"; shift 2 ;;
        --psm)            PSM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Model-type defaults ──
if [ "$MODEL" = "bagel" ]; then
    [ -z "$HEIGHT" ] && HEIGHT=512
    [ -z "$WIDTH" ] && WIDTH=512
    [ -z "$CFG_SCALE" ] && CFG_SCALE=8.0
    [ -z "$SERVE_MODE" ] && SERVE_MODE="native"
elif [ "$MODEL" = "qwenimage" ]; then
    [ -z "$HEIGHT" ] && HEIGHT=1024
    [ -z "$WIDTH" ] && WIDTH=1024
    [ -z "$CFG_SCALE" ] && CFG_SCALE=4.0
    [ -z "$SERVE_MODE" ] && SERVE_MODE="vllm"
else
    echo "Error: --model must be 'qwenimage' or 'bagel', got: $MODEL"
    exit 1
fi

# External URL overrides serve_mode
if [ -n "$URL" ] && [ -z "$SERVE_TRIAL" ] && [ -z "$SERVE_CKPT" ] && [ -z "$PSM" ]; then
    SERVE_MODE="external"

    # Try to get model info from /health for output path naming
    if [ -z "$TRIAL" ]; then
        URL_INFO=$(/usr/bin/python3 -c "
import json, requests
try:
    resp = requests.get('${URL}/health', timeout=5, proxies={'http': None, 'https': None})
    info = resp.json()
    print(json.dumps({'trial': info.get('trial',''), 'step': info.get('step',''), 'ema': info.get('ema', False)}))
except: print('{}')
" 2>/dev/null || echo '{}')

        URL_TRIAL=$(echo "$URL_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('trial',''))" 2>/dev/null)
        URL_STEP=$(echo "$URL_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step',''))" 2>/dev/null)
        URL_EMA=$(echo "$URL_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ema',False))" 2>/dev/null)

        if [ -n "$URL_TRIAL" ]; then
            [ -z "$SERVE_TRIAL" ] && SERVE_TRIAL="$URL_TRIAL"
            [ -z "$SERVE_STEP" ] && [ -n "$URL_STEP" ] && SERVE_STEP="$URL_STEP"
            if [ -z "$SERVE_EMA" ] && [ "$URL_EMA" = "True" ]; then
                SERVE_EMA="--ema"
            fi
            echo "  URL model info: trial=$SERVE_TRIAL step=$SERVE_STEP ema=$URL_EMA"
        fi
    fi
fi

# PSM overrides serve_mode (skip local serving entirely)
if [ -n "$PSM" ]; then
    SERVE_MODE="external"
    URL="psm"  # placeholder, actual URL resolved in Python via --psm

    # Query PSM to get current model info for output path naming
    if [ -z "$SERVE_TRIAL" ] || [ -z "$SERVE_STEP" ]; then
        echo "Querying PSM for model info..."
        PSM_INFO=$(HOME=<RAY_SERVE_HOME> /usr/bin/python3 -c "
import sys, json, requests
sys.path.insert(0, '<RAY_SERVE_SITE_PACKAGES>')
import os; os.environ['HOME'] = '<RAY_SERVE_HOME>'
from ray.serve import get_serve_http_client
client = get_serve_http_client(psm='$PSM')
eps = client.endpoint_getter.get_urls()
host, port = eps[0]['Host'], eps[0]['Port']
url = f'http://[{host}]:{port}'
resp = requests.get(f'{url}/health', timeout=10, proxies={'http': None, 'https': None})
info = resp.json()
print(json.dumps({'trial': info.get('trial',''), 'step': info.get('step',''), 'ema': info.get('ema', False)}))
" 2>/dev/null || echo '{}')

        PSM_TRIAL_INFO=$(echo "$PSM_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('trial',''))" 2>/dev/null)
        PSM_STEP_INFO=$(echo "$PSM_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step',''))" 2>/dev/null)
        PSM_EMA_INFO=$(echo "$PSM_INFO" | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ema',False))" 2>/dev/null)

        [ -z "$SERVE_TRIAL" ] && [ -n "$PSM_TRIAL_INFO" ] && SERVE_TRIAL="$PSM_TRIAL_INFO"
        [ -z "$SERVE_STEP" ] && [ -n "$PSM_STEP_INFO" ] && SERVE_STEP="$PSM_STEP_INFO"
        if [ -z "$SERVE_EMA" ] && [ "$PSM_EMA_INFO" = "True" ]; then
            SERVE_EMA="--ema"
        fi

        echo "  PSM model: trial=$SERVE_TRIAL step=$SERVE_STEP ema=$PSM_EMA_INFO"
    fi
fi

# --serve_ckpt implies we need to auto-serve (like serve_trial but skip merge)
if [ -n "$SERVE_CKPT" ] && [ -z "$SERVE_TRIAL" ]; then
    # Use ckpt path directly, set a dummy serve_trial to trigger serve logic
    SERVE_TRIAL="__direct_ckpt__"
fi

# ── Default trial to serve_trial ──
if [ -z "$TRIAL" ] && [ -n "$SERVE_TRIAL" ] && [ "$SERVE_TRIAL" != "__direct_ckpt__" ]; then
    TRIAL="$SERVE_TRIAL"
fi

# Fallback trial name when using external URL/PSM without explicit trial
if [ -z "$TRIAL" ] && { [ -n "$URL" ] || [ -n "$PSM" ]; }; then
    TRIAL="external"
fi

# ── Validate ──
if [ -z "$BENCHMARK" ] || [ -z "$REWRITE" ] || [ -z "$TRIAL" ]; then
    echo "Error: --benchmark, --rewrite, --trial are required."
    echo "Also need: --url OR --serve_trial OR --serve_ckpt OR --psm"
    exit 1
fi
if [ -z "$URL" ] && [ -z "$SERVE_TRIAL" ] && [ -z "$SERVE_CKPT" ] && [ -z "$PSM" ]; then
    echo "Error: provide --url, --serve_trial, --serve_ckpt, or --psm"
    exit 1
fi

# ── Resolve paths ──
if [ -n "$REWRITTEN_JSONL_OVERRIDE" ]; then
    REWRITTEN_JSONL="$REWRITTEN_JSONL_OVERRIDE"
else
    REWRITTEN_JSONL="rewritten/${BENCHMARK}_${REWRITE}.jsonl"
fi

# Resolve step number (auto-detect latest if not specified)
# Skip local HDFS detection when using PSM (remote server handles it)
if [ -z "$SERVE_STEP" ] && [ -n "$SERVE_TRIAL" ] && [ -z "$PSM" ]; then
    SERVE_STEP=$(ls -1d "${TRIALS_ROOT}/${SERVE_TRIAL}"/[0-9]* 2>/dev/null | sort -n | tail -1 | xargs basename 2>/dev/null || echo "unknown")
    echo "Auto-detected latest step: $SERVE_STEP"
fi

# Build output subdir: {step}_{res}_{cfg}_{ema}
OUT_STEP="${SERVE_STEP:-unknown}"
OUT_RES="${WIDTH}x${HEIGHT}"
OUT_CFG=$(echo "$CFG_SCALE" | sed 's/\./_/g')  # 4.0 -> 4_0
OUT_EMA=""
[ -n "$SERVE_EMA" ] && OUT_EMA="_ema"
OUTPUT_DIR="outputs/${BENCHMARK}_${REWRITE}_${TRIAL}/${OUT_STEP}_${OUT_RES}_cfg${OUT_CFG}${OUT_EMA}"

if [ ! -f "$REWRITTEN_JSONL" ]; then
    echo "Error: rewritten JSONL not found: $REWRITTEN_JSONL"
    exit 1
fi

# Benchmark-specific args
EXTRA_ARGS=""
case $BENCHMARK in
    geneval)
        EXTRA_ARGS="--benchmark_data benchmarks/geneval/prompts/evaluation_metadata.jsonl"
        ;;
    oneig|tiif_short|tiif_long)
        EXTRA_ARGS="--model_name $MODEL_NAME"
        ;;
esac

MAX_PROMPTS_ARG=""
if [ "$MAX_PROMPTS" -gt 0 ] 2>/dev/null; then
    MAX_PROMPTS_ARG="--max_prompts $MAX_PROMPTS"
fi

MAX_PROMPT_CHARS_ARG=""
if [ "$MAX_PROMPT_CHARS" -gt 0 ] 2>/dev/null; then
    MAX_PROMPT_CHARS_ARG="--max_prompt_chars $MAX_PROMPT_CHARS"
fi

TIMESTEP_SHIFT_ARG=""
if [ -n "$TIMESTEP_SHIFT" ]; then
    TIMESTEP_SHIFT_ARG="--timestep_shift $TIMESTEP_SHIFT"
fi

# Convert <HDFS_ROOT>/user/... to <HDFS_NATIVE>/user/<USER>
to_hdfs_url() {
    echo "$1" | sed 's|^<HDFS_ROOT>/user/|<HDFS_NATIVE>/home/data|'
}

# Cache file from HDFS to local /tmp
hdfs_cache() {
    local hdfs_local_path="$1" target_path="$2"
    local hdfs_url
    hdfs_url=$(to_hdfs_url "$hdfs_local_path")
    echo "  Caching $hdfs_url -> $target_path..."
    hdfs dfs -get "$hdfs_url" "$target_path"
}

# Cache directory from HDFS to local /tmp
hdfs_cache_dir() {
    local hdfs_local_path="$1" target_path="$2"
    local hdfs_url
    hdfs_url=$(to_hdfs_url "$hdfs_local_path")
    echo "  Caching $hdfs_url -> $target_path..."
    hdfs dfs -get "$hdfs_url" "$target_path"
}

# ──────────────────────────────────────────────
# Kill serving helper
# ──────────────────────────────────────────────
kill_serving() {
    echo "  Killing serving process..."
    # Only kill the serving process group we started (SERVE_PID is set by setsid)
    if [ -n "$SERVE_PID" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
        kill -9 -"$SERVE_PID" 2>/dev/null || true  # kill process group
        kill -9 "$SERVE_PID" 2>/dev/null || true
    fi
    # Kill only our port (the one we allocated), not hardcoded list
    if [ -n "$SERVE_PORT" ]; then
        MY_PID=$$
        lsof -ti:${SERVE_PORT} 2>/dev/null | while read pid; do
            [ "$pid" != "$MY_PID" ] && kill -9 "$pid" 2>/dev/null || true
        done
    fi
    sleep 3
}

# ──────────────────────────────────────────────
# Auto-serve if --serve_trial provided
# ──────────────────────────────────────────────
SERVE_PID=""

trap 'if [ -n "$SERVE_PID" ]; then kill_serving; fi; exit 1' INT TERM

if [ -n "$SERVE_TRIAL" ] && [ -z "$PSM" ]; then
    # Kill any leftover serving
    kill_serving

    # Find a free port (override SERVE_PORT if it's occupied)
    find_free_port() {
        /usr/bin/python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
port = s.getsockname()[1]
s.close()
print(port)
"
    }
    SERVE_PORT=$(find_free_port)
    URL="http://localhost:${SERVE_PORT}"
    echo "  Using port: $SERVE_PORT"

    # ── Step A: Resolve / Merge checkpoint ──
    if [ -n "$SERVE_CKPT" ]; then
        # Direct checkpoint path — skip merge
        SAFETENSORS="$SERVE_CKPT"
        if [ ! -f "$SAFETENSORS" ]; then
            echo "ERROR: --serve_ckpt not found: $SAFETENSORS"
            exit 1
        fi
        echo "  Using direct checkpoint: $SAFETENSORS"
    else
        STEP_DIR="${TRIALS_ROOT}/${SERVE_TRIAL}/${SERVE_STEP}"
        WEIGHT_NAME="model"
        [ -n "$SERVE_EMA" ] && WEIGHT_NAME="ema"
        SAFETENSORS="${STEP_DIR}/${WEIGHT_NAME}.safetensors"
    fi

    if [ ! -f "$SAFETENSORS" ]; then
        echo ""
        echo "=== Merging checkpoint ==="
        echo "  Model:  $MODEL"
        echo "  Trial:  $SERVE_TRIAL"
        echo "  Step:   $SERVE_STEP"
        echo "  Type:   $WEIGHT_NAME"

        if [ "$MODEL" = "bagel" ]; then
            # Cache Qwen2.5-7B-Instruct LLM if needed
            if [ ! -d "$BAGEL_LLM_PATH" ]; then
                echo "  Caching LLM to $BAGEL_LLM_PATH..."
                hdfs dfs -get "$BAGEL_LLM_HDFS" "$BAGEL_LLM_PATH"
            fi
            CKPT_SUBDIR="${STEP_DIR}/${WEIGHT_NAME}"
            python "${BAGEL_ROOT}/pack_fsdp_ckpt_bagel_cpu.py" \
                --ckpt_dir "$CKPT_SUBDIR" \
                --llm_path "$BAGEL_LLM_PATH" \
                --vae_path "$BAGEL_VAE_PATH" \
                --output "$SAFETENSORS" \
                --dtype bf16
        else
            # QwenImage merge
            cd "$QWENIMAGE_ROOT"
            bash "${QWENIMAGE_ROOT}/scripts/merge/merge.sh" "$SERVE_TRIAL" "$SERVE_STEP"
            cd "$(dirname "$0")/.."
        fi

        if [ ! -f "$SAFETENSORS" ]; then
            echo "ERROR: merge failed, $SAFETENSORS not found"
            exit 1
        fi
        echo "  Merge done: $SAFETENSORS"
    else
        echo "  Checkpoint already merged: $SAFETENSORS"
    fi

    # ── Step B: Start serving (background) ──
    mkdir -p logs
    SERVE_LOG="logs/serve_inference_${TRIAL}.log"

    if [ "$SERVE_MODE" = "vllm" ]; then
        # ──────── vllm-omni serving ────────
        SERVE_CMD="bash $SERVE_VLLM_SCRIPT $SERVE_TRIAL --port $SERVE_PORT --replicas $SERVE_REPLICAS --cfg $SERVE_CFG --fp8 --cache-dit"
        [ -n "$SERVE_STEP" ] && SERVE_CMD="$SERVE_CMD --step $SERVE_STEP"
        [ -n "$SERVE_EMA" ] && SERVE_CMD="$SERVE_CMD $SERVE_EMA"

        echo ""
        echo "Starting vllm-omni serving..."
        echo "  Command: $SERVE_CMD"
        setsid $SERVE_CMD > "$SERVE_LOG" 2>&1 &
        SERVE_PID=$!
        echo "  PID=$SERVE_PID"

        # Wait for backends to be ready via test request
        echo "  Waiting for serving (checking backends)..."
        for i in $(seq 1 180); do
            if ! kill -0 "$SERVE_PID" 2>/dev/null; then
                echo "  ERROR: serving process died. Check $SERVE_LOG"
                exit 1
            fi
            TEST_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${URL}/v1/images/generations" \
                -H "Content-Type: application/json" \
                -d '{"prompt":"test","size":"1024x1024","seed":0}' 2>/dev/null || echo "000")
            if [ "$TEST_CODE" = "200" ]; then
                echo "  Serving ready! (test request returned 200)"
                break
            elif [ "$TEST_CODE" != "000" ] && [ "$TEST_CODE" != "502" ]; then
                echo "  Serving ready! (server responded with $TEST_CODE)"
                break
            fi
            if [ "$i" -eq 180 ]; then
                echo "  ERROR: serving timed out (15 min). Check $SERVE_LOG"
                kill_serving
                exit 1
            fi
            sleep 5
        done

        INFERENCE_MODULE="inference.generate_openai"

    elif [ "$SERVE_MODE" = "native" ]; then
        # ──────── Native multi-GPU serving ────────
        # Use direct ckpt path or cache from HDFS to /tmp
        if [ -n "$SERVE_CKPT" ]; then
            LOCAL_CKPT="$SAFETENSORS"
        else
            LOCAL_CKPT="/tmp/${SERVE_TRIAL}_${SERVE_STEP}_${WEIGHT_NAME}.safetensors"
            if [ ! -f "$LOCAL_CKPT" ]; then
                hdfs_cache "$SAFETENSORS" "$LOCAL_CKPT"
            fi
        fi

        GPU_ARGS=""
        [ -n "$SERVE_GPUS" ] && GPU_ARGS="--gpus $SERVE_GPUS"

        if [ "$MODEL" = "bagel" ]; then
            # Cache LLM if needed
            if [ ! -d "$BAGEL_LLM_PATH" ]; then
                echo "  Caching LLM to $BAGEL_LLM_PATH..."
                hdfs dfs -get "$BAGEL_LLM_HDFS" "$BAGEL_LLM_PATH"
            fi

            SERVE_CMD="python ${BAGEL_ROOT}/scripts/merge/serve_bagel.py \
                --merged_ckpt $LOCAL_CKPT \
                --llm_path $BAGEL_LLM_PATH \
                --vae_path $BAGEL_VAE_PATH \
                --port $SERVE_PORT \
                --max_per_gpu $SERVE_MAX_PER_GPU \
                $GPU_ARGS"
        else
            # QwenImage native serving
            CKPT_ROOT_LOCAL="<QWENIMAGE_CKPT>"
            CKPT_ROOT_HDFS="<WEIGHTS_ROOT>/qwenimage"
            if [ ! -d "$CKPT_ROOT_LOCAL" ]; then
                hdfs_cache_dir "$CKPT_ROOT_HDFS" "$CKPT_ROOT_LOCAL"
            fi

            SERVE_CMD="python ${QWENIMAGE_ROOT}/scripts/merge/serve_qwenimage_multigpu.py \
                --merged_ckpt $LOCAL_CKPT \
                --ckpt_root ${CKPT_ROOT_LOCAL}/origin/raw_data \
                --port $SERVE_PORT \
                --max_per_gpu $SERVE_MAX_PER_GPU \
                $GPU_ARGS"
        fi

        echo ""
        echo "Starting native serving ($MODEL)..."
        echo "  Command: $SERVE_CMD"
        setsid $SERVE_CMD > "$SERVE_LOG" 2>&1 &
        SERVE_PID=$!
        echo "  PID=$SERVE_PID"

        # Wait for serving to be truly ready
        # Phase 1: wait for /health to respond (server process started)
        echo "  Waiting for server process to start..."
        for i in $(seq 1 120); do
            if ! kill -0 "$SERVE_PID" 2>/dev/null; then
                echo "  ERROR: serving process died. Check $SERVE_LOG"
                exit 1
            fi
            if curl -s "${URL}/health" --max-time 3 > /dev/null 2>&1; then
                echo "  Server process started, waiting for workers to load..."
                break
            fi
            if [ "$i" -eq 120 ]; then
                echo "  ERROR: server process timed out (10 min). Check $SERVE_LOG"
                kill_serving
                exit 1
            fi
            sleep 5
        done

        # Phase 2: test with real generation request (workers may still be loading)
        for i in $(seq 1 120); do
            if ! kill -0 "$SERVE_PID" 2>/dev/null; then
                echo "  ERROR: serving process died. Check $SERVE_LOG"
                exit 1
            fi
            TEST_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${URL}/generate" \
                -H "Content-Type: application/json" \
                -d '{"prompt":"test","height":256,"width":256,"num_steps":1,"seed":0,"cfg_scale":1.0}' \
                --max-time 120 2>/dev/null || echo "000")
            if [ "$TEST_CODE" = "200" ]; then
                echo "  Serving ready! (test generation returned 200)"
                break
            elif [ "$TEST_CODE" = "403" ] || [ "$TEST_CODE" = "404" ]; then
                # 403/404 likely means port conflict with another service
                echo "  WARNING: got $TEST_CODE, port may be occupied by another service. Retrying..."
            elif [ "$TEST_CODE" != "000" ] && [ "$TEST_CODE" != "502" ] && [ "$TEST_CODE" != "503" ]; then
                echo "  Serving ready! (server responded with $TEST_CODE)"
                break
            fi
            if [ "$i" -eq 120 ]; then
                echo "  ERROR: serving timed out (workers not ready). Check $SERVE_LOG"
                kill_serving
                exit 1
            fi
            sleep 5
        done

        INFERENCE_MODULE="inference.generate"
    fi
else
    # External URL — determine inference module based on model type
    # vllm-omni uses /v1/images/generations (generate_openai)
    # Native serving uses /generate (generate)
    # Default: try /health to detect, or use generate (native) as default for external
    INFERENCE_MODULE="inference.generate"
fi

# Safety: ensure INFERENCE_MODULE is set
if [ -z "$INFERENCE_MODULE" ]; then
    if [ "$SERVE_MODE" = "vllm" ]; then
        INFERENCE_MODULE="inference.generate_openai"
    else
        INFERENCE_MODULE="inference.generate"
    fi
fi

# ── Run ──
echo ""
echo "============================================"
echo "Inference: ${BENCHMARK} + ${REWRITE} rewrite"
echo "============================================"
echo "  Model:       $MODEL"
echo "  Serve mode:  $SERVE_MODE"
echo "  Trial:       $TRIAL"
echo "  URL:         $URL"
echo "  Serve trial: ${SERVE_TRIAL:-external}"
echo "  Inference:   $INFERENCE_MODULE"
echo "  Input:       $REWRITTEN_JSONL"
echo "  Output:      $OUTPUT_DIR"
echo "  Resolution:  ${WIDTH}x${HEIGHT}"
echo "  Steps:       $NUM_STEPS"
echo "  CFG:         $CFG_SCALE"
echo "  Seed:        $SEED"
echo "  Workers:     $WORKERS"
echo "  Max prompts: ${MAX_PROMPTS:-all}"
[ -n "$DISABLE_REWRITTEN" ] && echo "  Mode:        original prompt (rewrite disabled)"
[ "$MAX_PROMPT_CHARS" -gt 0 ] 2>/dev/null && echo "  Max chars:   $MAX_PROMPT_CHARS (truncation enabled)"
[ -n "$TIMESTEP_SHIFT" ] && echo "  TS shift:    $TIMESTEP_SHIFT"
[ -n "$WORLD_SIZE" ] && echo "  Shard:       rank $RANK / $WORLD_SIZE"
[ -n "$PSM" ] && echo "  PSM:         $PSM"
echo ""

# Build optional shard args
RANK_ARG=""
[ -n "$RANK" ] && RANK_ARG="--rank $RANK"
[ -n "$WORLD_SIZE" ] && RANK_ARG="$RANK_ARG --world_size $WORLD_SIZE"

# Build PSM args
PSM_ARGS=""
if [ -n "$PSM" ]; then
    PSM_ARGS="--psm $PSM"
    [ -n "$SERVE_TRIAL" ] && PSM_ARGS="$PSM_ARGS --psm_trial $SERVE_TRIAL"
    [ -n "$SERVE_STEP" ] && PSM_ARGS="$PSM_ARGS --psm_step $SERVE_STEP"
    [ -n "$SERVE_EMA" ] && PSM_ARGS="$PSM_ARGS --psm_ema"
fi

/usr/bin/python3 -m "$INFERENCE_MODULE" \
    --rewritten_jsonl "$REWRITTEN_JSONL" \
    --url "$URL" \
    --output_dir "$OUTPUT_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_steps "$NUM_STEPS" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --cfg_scale "$CFG_SCALE" \
    $EXTRA_ARGS \
    $MAX_PROMPTS_ARG \
    $MAX_PROMPT_CHARS_ARG \
    $TIMESTEP_SHIFT_ARG \
    $RANK_ARG \
    $PSM_ARGS \
    $DISABLE_REWRITTEN \
    $FIX_LENS_KEY

# ── Cleanup serving ──
if [ -n "$SERVE_PID" ]; then
    echo ""
    echo "Stopping serving..."
    kill_serving
fi

echo "Done."
