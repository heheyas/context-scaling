#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Run evaluation/judge for any benchmark.
#
# Usage:
#   bash scripts/run_judge.sh --benchmark <benchmark> --output_dir <path> [options]
#
# Required args:
#   --benchmark    geneval, geneval2, genevalpp, oneig, wise, tiif, textbench, dpgbench, genexam
#   --output_dir   inference output directory (e.g. outputs/geneval2_gemini_my_trial)
#
# Optional args (GenEval):
#   --detector         HF model ID or local path
#   --clip_pretrained  CLIP weights path
#
# Optional args (GenEval2):
#   --method       soft_tifa_gm, soft_tifa_am, tifa, vqascore (default: soft_tifa_gm)
#   --num_gpus     number of GPUs for parallel eval (default: 0 = all available)
#
# Optional args (WISE / TIIF):
#   --api_key      OpenAI API key (required)
#   --api_base     OpenAI API base URL
#   --gpt_model    GPT model name (default: gpt-4o)
#   --max_workers  parallel eval threads (default: 10)
#
# Optional args (TIIF):
#   --model_name   T2I model name used in output dir structure (required)
#
# Optional args (TextBench):
#   --mode         en or zh (default: en)
#   --num_gpus     number of GPUs (default: 8)
#
# Optional args (DPG-Bench):
#   --num_gpus     number of GPUs (default: 8)
#   --pic_num      images per prompt in grid (default: 4)
#   --resolution   image resolution (default: 1024)
#
# Examples:
#   bash scripts/run_judge.sh --benchmark geneval2 --output_dir outputs/geneval2_gemini
#   bash scripts/run_judge.sh --benchmark wise --output_dir outputs/wise_gemini --api_key sk-xxx
#   bash scripts/run_judge.sh --benchmark tiif --output_dir outputs/tiif_gemini --model_name my_model --api_key sk-xxx
#   bash scripts/run_judge.sh --benchmark textbench --output_dir outputs/textbench_gemini --mode en
#   bash scripts/run_judge.sh --benchmark dpgbench --output_dir outputs/dpgbench_gemini
#   bash scripts/run_judge.sh --benchmark genexam --output_dir outputs/genexam_gemini --api_config api_config.json
#   bash scripts/run_judge.sh --benchmark genexam --output_dir outputs/genexam_gemini --score_only

set -e
cd "$(dirname "$0")/.."

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export MS_CACHE_HOME="<LOCAL_ROOT>/modelscope"
export MODELSCOPE_CACHE="<LOCAL_ROOT>/modelscope"

# ── Defaults ──
BENCHMARK=""
OUTPUT_DIR=""
METHOD="soft_tifa_gm"
NUM_GPUS=0
DETECTOR="/tmp/ms_cache/facebook/mask2former-swin-large-coco-instance"
CLIP_PRETRAINED="/tmp/ms_cache/timm/vit_large_patch14_clip_224___openai/open_clip_pytorch_model.bin"
MODEL_NAME="model"
API_KEY=""
API_BASE=""
API_CONFIG=""
GPT_MODEL="gpt-4o"
MAX_WORKERS=10
MODE="en"
PIC_NUM=4
RESOLUTION=1024
PORT=29500
SCORE_ONLY=""
MINI=""

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)        BENCHMARK="$2"; shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --method)           METHOD="$2"; shift 2 ;;
        --num_gpus)         NUM_GPUS="$2"; shift 2 ;;
        --detector)         DETECTOR="$2"; shift 2 ;;
        --clip_pretrained)  CLIP_PRETRAINED="$2"; shift 2 ;;
        --model_name)       MODEL_NAME="$2"; shift 2 ;;
        --api_key)          API_KEY="$2"; shift 2 ;;
        --api_base)         API_BASE="$2"; shift 2 ;;
        --api_config)       API_CONFIG="$2"; shift 2 ;;
        --gpt_model)        GPT_MODEL="$2"; shift 2 ;;
        --max_workers)      MAX_WORKERS="$2"; shift 2 ;;
        --mode)             MODE="$2"; shift 2 ;;
        --pic_num)          PIC_NUM="$2"; shift 2 ;;
        --resolution)       RESOLUTION="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --score_only)       SCORE_ONLY="--score_only"; shift ;;
        --mini)             MINI="--mini"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$BENCHMARK" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --benchmark and --output_dir are required."
    echo "Usage: bash scripts/run_judge.sh --benchmark geneval2 --output_dir outputs/geneval2_gemini_my_trial"
    exit 1
fi

RESULTS_FILE="${OUTPUT_DIR}/results_${BENCHMARK}.jsonl"

echo "============================================"
echo "Judge: ${BENCHMARK}"
echo "============================================"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Results:     $RESULTS_FILE"

# ── GenEval2 ──
if [ "$BENCHMARK" = "geneval2" ]; then
    IMAGE_DATA="${OUTPUT_DIR}/image_filepath_data.json"
    SCORES_FILE="${OUTPUT_DIR}/scores_${METHOD}.json"

    if [ ! -f "$IMAGE_DATA" ]; then
        echo "Error: $IMAGE_DATA not found. Run inference first."
        exit 1
    fi

    N_IMAGES=$(python3 -c "import json; print(len(json.load(open('$IMAGE_DATA'))))")
    ACTUAL_GPUS=${NUM_GPUS:-0}
    if [ "$ACTUAL_GPUS" = "0" ]; then
        ACTUAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
    fi
    echo "  Method:      $METHOD"
    echo "  Images:      $N_IMAGES"
    echo "  GPUs:        $ACTUAL_GPUS"
    echo ""

    python benchmarks/GenEval2/evaluation_multigpu.py \
        --benchmark_data benchmarks/GenEval2/geneval2_data.jsonl \
        --image_filepath_data "$IMAGE_DATA" \
        --method "$METHOD" \
        --output_file "$SCORES_FILE" \
        --num_gpus "$ACTUAL_GPUS"

    echo ""
    echo "Scores saved to: $SCORES_FILE"

# ── GenEval ──
elif [ "$BENCHMARK" = "geneval" ]; then
    N_FOLDERS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name '[0-9]*' | wc -l)
    echo "  Detector:    $DETECTOR"
    echo "  CLIP:        $CLIP_PRETRAINED"
    echo "  Folders:     $N_FOLDERS"
    echo ""

    python benchmarks/geneval/evaluation/evaluate_images_hf.py \
        "$OUTPUT_DIR" \
        --outfile "$RESULTS_FILE" \
        --detector "$DETECTOR" \
        --clip-pretrained "$CLIP_PRETRAINED"

    echo ""
    echo "Results saved to: $RESULTS_FILE"
    echo ""
    echo "Summary:"
    python benchmarks/geneval/evaluation/summary_scores.py "$RESULTS_FILE"

# ── OneIG ──
elif [ "$BENCHMARK" = "oneig" ]; then
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: --model_name is required for oneig benchmark"
        exit 1
    fi
    bash scripts/run_judge_oneig.sh \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME"

# ── WISE ──
elif [ "$BENCHMARK" = "wise" ]; then
    if [ -z "$API_CONFIG" ] && [ -z "$API_KEY" ]; then
        echo "Error: --api_config or --api_key required for WISE"
        exit 1
    fi

    WISE_DIR="benchmarks/WISE"
    WISE_DATA="${WISE_DIR}/data"
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"
    mkdir -p "$EVAL_OUTPUT"

    # Build key args: prefer --api_config (multi-key), fallback to --api_key
    KEY_ARGS=""
    if [ -n "$API_CONFIG" ]; then
        KEY_ARGS="--api_config $API_CONFIG"
    else
        KEY_ARGS="--api_key $API_KEY"
        [ -n "$API_BASE" ] && KEY_ARGS="$KEY_ARGS --api_base $API_BASE"
    fi

    echo "  GPT Model:   $GPT_MODEL"
    echo "  Workers:     $MAX_WORKERS"
    echo "  Key source:  ${API_CONFIG:-single key}"
    echo ""

    SCORE_FILES=""
    for json_file in "$WISE_DATA"/cultural_common_sense.json \
                     "$WISE_DATA"/spatio-temporal_reasoning.json \
                     "$WISE_DATA"/natural_science.json; do
        name=$(basename "$json_file" .json)
        full_name="${name}_full_results.json"
        scores_name="${name}_scores_results.jsonl"

        echo "--- Evaluating: $name ---"
        /usr/bin/python3 "${WISE_DIR}/gpt_eval.py" \
            --json_path "$json_file" \
            --image_dir "$OUTPUT_DIR" \
            --output_dir "$EVAL_OUTPUT" \
            $KEY_ARGS \
            --model "$GPT_MODEL" \
            --result_full "$full_name" \
            --result_scores "$scores_name" \
            --max_workers "$MAX_WORKERS"
        SCORE_FILES="$SCORE_FILES ${EVAL_OUTPUT}/${scores_name}"
        echo ""
    done

    echo "--- Calculating WiScore ---"
    /usr/bin/python3 "${WISE_DIR}/Calculate.py" $SCORE_FILES --category all

# ── TIIF-Bench ──
elif [ "$BENCHMARK" = "tiif" ]; then
    if [ -z "$API_KEY" ] && [ -z "$API_CONFIG" ]; then
        echo "Error: --api_key or --api_config required for TIIF-Bench"
        exit 1
    fi

    TIIF_DIR="benchmarks/TIIF-Bench"
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"
    mkdir -p "$EVAL_OUTPUT"

    BASE_URL_ARG=""
    if [ -n "$API_BASE" ]; then
        BASE_URL_ARG="--base_url $API_BASE"
    fi

    # Support testmini subset via --mini flag
    SAMPLE_IDX_ARG=""
    if [ -n "$MINI" ]; then
        SAMPLE_IDX_FILE="${TIIF_DIR}/data/testmini_sample_idx.json"
        if [ ! -f "$SAMPLE_IDX_FILE" ]; then
            echo "Error: testmini sample index file not found: $SAMPLE_IDX_FILE"
            exit 1
        fi
        SAMPLE_IDX_ARG="--sample_idx_file $SAMPLE_IDX_FILE"
        echo "  Subset:      testmini (277 samples)"
    fi

    # API key args: --api_config for multi-key, --api_key for single key
    KEY_ARGS=""
    if [ -n "$API_CONFIG" ]; then
        KEY_ARGS="--api_config $API_CONFIG"
    else
        KEY_ARGS="--api_key $API_KEY"
    fi

    echo "  Model name:  $MODEL_NAME"
    echo "  GPT Model:   $GPT_MODEL"
    echo "  Workers:     $MAX_WORKERS"
    echo "  Key source:  ${API_CONFIG:-single key}"
    echo ""

    echo "--- Running VLM evaluation ---"
    /usr/bin/python3 "${TIIF_DIR}/eval/eval_with_vlm.py" \
        --jsonl_dir "${TIIF_DIR}/data/test_eval_prompts" \
        --image_dir "$OUTPUT_DIR" \
        --eval_model "$MODEL_NAME" \
        --output_dir "$EVAL_OUTPUT" \
        $KEY_ARGS \
        --model "$GPT_MODEL" \
        --workers "$MAX_WORKERS" \
        $BASE_URL_ARG \
        $SAMPLE_IDX_ARG

    echo ""
    echo "--- Summarizing results ---"
    /usr/bin/python3 "${TIIF_DIR}/eval/summary_results.py" --input_dir "$EVAL_OUTPUT"

    if [ -f "${EVAL_OUTPUT}/result_summary.xlsx" ]; then
        echo ""
        echo "--- Dimension breakdown ---"
        /usr/bin/python3 "${TIIF_DIR}/eval/summary_dimension_results.py" \
            --input_excel "${EVAL_OUTPUT}/result_summary.xlsx" \
            --output_txt "${EVAL_OUTPUT}/result_summary_dimension.txt"
        cat "${EVAL_OUTPUT}/result_summary_dimension.txt" 2>/dev/null || true
    fi

# ── TextBench (X-Omni) ──
elif [ "$BENCHMARK" = "textbench" ]; then
    TB_DIR="benchmarks/X-Omni/textbench"
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"
    mkdir -p "$EVAL_OUTPUT"

    ACTUAL_GPUS=$NUM_GPUS
    if [ "$ACTUAL_GPUS" = "0" ]; then
        ACTUAL_GPUS=$(/usr/bin/python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
    fi

    echo "  Mode:        $MODE"
    echo "  GPUs:        $ACTUAL_GPUS"
    echo ""

    echo "--- Running OCR evaluation ---"
    torchrun --nnodes=1 --node-rank=0 --nproc_per_node="$ACTUAL_GPUS" \
        "${TB_DIR}/evaluate_text_reward.py" \
        --sample_dir "$OUTPUT_DIR" \
        --output_dir "$EVAL_OUTPUT" \
        --mode "$MODE"

    echo "--- Merging chunk results ---"
    cat "${EVAL_OUTPUT}"/results_chunk*.jsonl > "${EVAL_OUTPUT}/results.jsonl"
    rm -f "${EVAL_OUTPUT}"/results_chunk*.jsonl

    echo "--- Calculating Text Score ---"
    /usr/bin/python3 "${TB_DIR}/summary_scores.py" "${EVAL_OUTPUT}/results.jsonl" --mode "$MODE"

# ── DPG-Bench ──
elif [ "$BENCHMARK" = "dpgbench" ]; then
    DPG_DIR="benchmarks/ELLA/dpg_bench"

    ACTUAL_GPUS=$NUM_GPUS
    if [ "$ACTUAL_GPUS" = "0" ]; then
        ACTUAL_GPUS=$(/usr/bin/python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 8)
    fi

    MULTI_GPU_FLAG=""
    if [ "$ACTUAL_GPUS" -gt 1 ]; then
        MULTI_GPU_FLAG="--multi_gpu"
    fi

    echo "  GPUs:        $ACTUAL_GPUS"
    echo "  PIC_NUM:     $PIC_NUM"
    echo "  Resolution:  $RESOLUTION"
    echo ""

    echo "--- Running DPG-Bench evaluation ---"
    http_proxy=http://<INTERNAL_PROXY> https_proxy=http://<INTERNAL_PROXY> no_proxy=<INTERNAL_GIT> accelerate launch --num_machines 1 --num_processes "$ACTUAL_GPUS" $MULTI_GPU_FLAG \
        --mixed_precision "fp16" --main_process_port "$PORT" \
        "${DPG_DIR}/compute_dpg_bench.py" \
        --image-root-path "$OUTPUT_DIR" \
        --resolution "$RESOLUTION" \
        --pic-num "$PIC_NUM" \
        --vqa-model mplug \
        --csv "${DPG_DIR}/dpg_bench.csv"

# ── GenExam ──
elif [ "$BENCHMARK" = "genexam" ]; then
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"
    mkdir -p "$EVAL_OUTPUT"

    if [ -n "$SCORE_ONLY" ]; then
        echo "  Mode:        score only (no API calls)"
        echo ""
        /usr/bin/python3 -m benchmarks.GenExam.run_judge \
            --eval_dir "$EVAL_OUTPUT" \
            --score_only
    else
        if [ -z "$API_CONFIG" ]; then
            echo "Error: --api_config required for GenExam (or use --score_only)"
            exit 1
        fi

        echo "  GPT Model:   $GPT_MODEL"
        echo "  Workers:     $MAX_WORKERS"
        echo "  API config:  $API_CONFIG"
        [ -n "$MINI" ] && echo "  Subset:      mini (251 samples)"
        echo ""

        /usr/bin/python3 -m benchmarks.GenExam.run_judge \
            --img_dir "$OUTPUT_DIR" \
            --eval_dir "$EVAL_OUTPUT" \
            --api_config "$API_CONFIG" \
            --model "$GPT_MODEL" \
            --workers "$MAX_WORKERS" \
            $MINI
    fi

# ── GenEval++ ──
elif [ "$BENCHMARK" = "genevalpp" ]; then
    if [ -z "$API_CONFIG" ] && [ -z "$API_KEY" ]; then
        echo "Error: --api_config or --api_key required for GenEval++"
        exit 1
    fi

    META_PATH="benchmarks/Echo-4o/test_scripts/Geneval++.jsonl"
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_genevalpp.json"

    KEY_ARGS=""
    if [ -n "$API_CONFIG" ]; then
        KEY_ARGS="--api_config $API_CONFIG"
    else
        KEY_ARGS="--api_key $API_KEY"
        [ -n "$API_BASE" ] && KEY_ARGS="$KEY_ARGS --api_base $API_BASE"
    fi

    echo "  GPT Model:   $GPT_MODEL"
    echo "  Workers:     $MAX_WORKERS"
    echo ""

    /usr/bin/python3 benchmarks/Echo-4o/eval_genevalpp.py \
        --meta_path "$META_PATH" \
        --image_dir "$OUTPUT_DIR" \
        --output_path "$EVAL_OUTPUT" \
        $KEY_ARGS \
        --model "$GPT_MODEL" \
        --workers "$MAX_WORKERS"

# ── T2I-CoReBench ──
elif [ "$BENCHMARK" = "corebench" ]; then
    COREBENCH_DIR="benchmarks/T2I-CoReBench"

    # CoReBench evaluate.py expects images at {output_path}/{MODEL}/{TASK}/{ID}.png
    # Our pipeline saves flat {output_dir}/{ID}.png. Create symlinks to match.
    COREBENCH_LOGS="${OUTPUT_DIR}/corebench_logs"
    mkdir -p "$COREBENCH_LOGS"

    # Create the directory structure evaluate.py expects
    COREBENCH_MODEL_NAME="${MODEL_NAME}"
    [ "$COREBENCH_MODEL_NAME" = "model" ] && COREBENCH_MODEL_NAME="$(basename "$OUTPUT_DIR")"
    for TASK in C-MI C-MA C-MR C-TR R-LR R-BR R-HR R-PR R-GR R-AR R-CR R-RR; do
        TASK_DIR="${COREBENCH_LOGS}/${COREBENCH_MODEL_NAME}/${TASK}"
        mkdir -p "$TASK_DIR"
        # Link images matching this task's IDs
        python3 -c "
import json, os
out_dir = '${OUTPUT_DIR}'
task_dir = '${TASK_DIR}'
with open('${COREBENCH_DIR}/data/${TASK}.json') as f:
    data = json.load(f)
linked = 0
for item_id in data:
    src = os.path.join(out_dir, item_id + '.png')
    if not os.path.exists(src):
        src = os.path.join(out_dir, item_id.replace('-', '_') + '.png')
    dst = os.path.join(task_dir, item_id + '.png')
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)
        linked += 1
print(f'  ${TASK}: linked {linked} images')
"
    done

    # Determine eval model
    MLLM="${GPT_MODEL:-Gemini_2_5_Flash}"
    TASKS="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR"

    echo "  MLLM:        $MLLM"
    echo "  Model name:  $COREBENCH_MODEL_NAME"
    echo "  Tasks:       $TASKS"
    echo ""

    # Need proxy for Gemini API
    if [[ "$MLLM" == *"Gemini"* ]]; then
        export http_proxy=http://<INTERNAL_PROXY>
        export https_proxy=http://<INTERNAL_PROXY>
        if [ -z "$GEMINI_API_KEY" ] && [ -n "$API_KEY" ]; then
            export GEMINI_API_KEY="$API_KEY"
        fi
    fi

    # GPT models: set api config
    if [[ "$MLLM" == *"GPT"* ]]; then
        if [ -n "$API_CONFIG" ]; then
            export GPT_API_CONFIG="$API_CONFIG"
        fi
    fi

    cd "$COREBENCH_DIR"
    python evaluate.py \
        --model "$COREBENCH_MODEL_NAME" \
        --mllm "$MLLM" \
        --gen_eval_file "$TASKS" \
        --output_path "$COREBENCH_LOGS" \
        --batch_size "$MAX_WORKERS"
    cd - > /dev/null

    # Print summary
    echo ""
    echo "--- CoReBench Results ---"
    python3 -c "
import json, os, glob
logs_dir = '${COREBENCH_LOGS}/${COREBENCH_MODEL_NAME}'
scores = {}
for f in sorted(glob.glob(os.path.join(logs_dir, '*-${MLLM}.json'))):
    task = os.path.basename(f).replace('-${MLLM}.json', '')
    with open(f) as fh:
        data = json.load(fh)
    ms = data.get('mean_score', '')
    scores[task] = ms
    print(f'  {task}: {ms:.4f}' if isinstance(ms, float) else f'  {task}: {ms}')
if scores:
    valid = [v for v in scores.values() if isinstance(v, (int, float))]
    print(f'  ---')
    print(f'  Overall: {sum(valid)/len(valid):.4f}' if valid else '  Overall: N/A')
"

else
    echo "Error: unsupported benchmark '$BENCHMARK'."
    echo "Supported: geneval, geneval2, genevalpp, oneig, wise, tiif, textbench, dpgbench, genexam, corebench"
    exit 1
fi
