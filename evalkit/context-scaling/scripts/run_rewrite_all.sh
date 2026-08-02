#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Rewrite all 4 benchmarks with both Gemini and GPT in parallel.
# Gemini and GPT run concurrently; within each, the 4 benchmarks run sequentially.
#
# Usage:
#   bash scripts/run_rewrite_all.sh
#   GEMINI_MODEL=gemini-3-pro-preview-new GPT_MODEL=gpt-4o-2024-11-20 bash scripts/run_rewrite_all.sh
#
#   # With suffix (output: geneval2_gemini_v2.jsonl, etc.)
#   SUFFIX=v2 bash scripts/run_rewrite_all.sh
#   SUFFIX=new_prompt SYSTEM_PROMPT=/path/to/new.txt bash scripts/run_rewrite_all.sh

set -e

cd "$(dirname "$0")/.."

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ── Config (override via environment variables) ──
SYSTEM_PROMPT="${SYSTEM_PROMPT:-demo/system_prompts/rft_iter0_with_ratio_v6_synthesis_student.txt}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-<prompt> [width: <width>, height: <height>]}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
WORKERS="${WORKERS:-64}"

GEMINI_KEYS="${GEMINI_KEYS:-<PATH_TO_GEMINI_KEYS_FILE>}"       # one key per line
GPT_KEYS="${GPT_KEYS:-<PATH_TO_GPT_API_CONFIG_JSON>}"          # {"api_key": "...", "api_base": "...", ...}
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-pro-preview-new}"
GPT_MODEL="${GPT_MODEL:-gpt-4o-2024-11-20}"

OUTPUT_DIR="${OUTPUT_DIR:-rewritten}"
SUFFIX="${SUFFIX:-}"
mkdir -p "$OUTPUT_DIR"

# Build suffix string for filenames: "_v2" or "" if empty
if [ -n "$SUFFIX" ]; then
    FSUFFIX="_${SUFFIX}"
else
    FSUFFIX=""
fi

# ── Benchmark configs: name, data_path ──
BENCHMARKS=(
    "geneval2|benchmarks/GenEval2/geneval2_data.jsonl"
    "geneval|benchmarks/geneval/prompts/evaluation_metadata.jsonl"
    "dpgbench|benchmarks/ELLA/dpg_bench/dpg_bench.csv"
    "oneig|benchmarks/OneIG-Benchmark/OneIG-Bench.csv"
    "genevalpp|benchmarks/Echo-4o/test_scripts/Geneval++.jsonl"
)

# ── Gemini: run 4 benchmarks sequentially ──
run_gemini() {
    echo "[Gemini] Starting rewrite (model=$GEMINI_MODEL)..."
    for entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r bench data_path <<< "$entry"
        echo "[Gemini] Rewriting $bench ..."
        /usr/bin/python3 -m rewrite_prompts.rewrite \
            --benchmark "$bench" \
            --data_path "$data_path" \
            --backend gemini \
            --model "$GEMINI_MODEL" \
            --output "${OUTPUT_DIR}/${bench}_gemini${FSUFFIX}.jsonl" \
            --system_prompt "$SYSTEM_PROMPT" \
            --input_template "$INPUT_TEMPLATE" \
            --width "$WIDTH" \
            --height "$HEIGHT" \
            --api_config "$GEMINI_KEYS" \
            --workers "$WORKERS"
        echo "[Gemini] Done: $bench"
    done
    echo "[Gemini] All benchmarks complete."
}

# ── GPT: run 4 benchmarks sequentially ──
run_gpt() {
    echo "[GPT] Starting rewrite (model=$GPT_MODEL)..."
    for entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r bench data_path <<< "$entry"
        echo "[GPT] Rewriting $bench ..."
        /usr/bin/python3 -m rewrite_prompts.rewrite \
            --benchmark "$bench" \
            --data_path "$data_path" \
            --backend gpt \
            --model "$GPT_MODEL" \
            --output "${OUTPUT_DIR}/${bench}_gpt${FSUFFIX}.jsonl" \
            --system_prompt "$SYSTEM_PROMPT" \
            --input_template "$INPUT_TEMPLATE" \
            --width "$WIDTH" \
            --height "$HEIGHT" \
            --api_config "$GPT_KEYS" \
            --workers "$WORKERS"
        echo "[GPT] Done: $bench"
    done
    echo "[GPT] All benchmarks complete."
}

# ── Run Gemini and GPT in parallel ──
run_gemini 2>&1 | sed 's/^/[gemini] /' &
PID_GEMINI=$!

run_gpt 2>&1 | sed 's/^/[gpt]    /' &
PID_GPT=$!

echo "Launched Gemini (PID=$PID_GEMINI) and GPT (PID=$PID_GPT) in parallel."
echo "Output dir: $OUTPUT_DIR/"

# Wait for both
wait $PID_GEMINI
EXIT_GEMINI=$?
wait $PID_GPT
EXIT_GPT=$?

echo ""
echo "========================================"
echo "Results:"
echo "========================================"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no output files found)"
echo ""
echo "Gemini exit code: $EXIT_GEMINI"
echo "GPT exit code:    $EXIT_GPT"

if [ $EXIT_GEMINI -ne 0 ] || [ $EXIT_GPT -ne 0 ]; then
    echo "WARNING: One or more backends failed."
    exit 1
fi

echo "All done."
