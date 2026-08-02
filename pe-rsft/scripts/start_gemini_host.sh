#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Start Gemini host: set proxy → start monitor → run ranker.
#
# Usage:
#   bash scripts/start_gemini_host.sh --config configs/iter0.yaml

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

CONFIG=""
MONITOR_INTERVAL=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)           CONFIG="$2"; shift 2 ;;
        --monitor_interval) MONITOR_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/start_gemini_host.sh --config <yaml>"
    exit 1
fi

mkdir -p logs

# ── Set proxy based on backend ──
BACKEND=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('gemini', {}).get('backend', 'gemini'))
" 2>/dev/null)
echo "[Gemini] Backend: $BACKEND"

if [ "$BACKEND" = "seed" ]; then
    # Seed is internal — no proxy needed (httpx client also uses trust_env=False).
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    echo "[Gemini] Proxy: disabled (seed backend)"
else
    export http_proxy="${http_proxy:-http://<INTERNAL_PROXY>}"
    export https_proxy="${https_proxy:-http://<INTERNAL_PROXY>}"
    export no_proxy="${no_proxy:-<INTERNAL_GIT>}"
    echo "[Gemini] Proxy: $https_proxy"
fi

# ── Start monitor in background ──
echo "[Gemini] Starting monitor (interval=${MONITOR_INTERVAL}s)..."
python3 -m src.ops.monitor \
    --config "$CONFIG" \
    --interval "$MONITOR_INTERVAL" \
    --log-file "logs/monitor.log" &
MONITOR_PID=$!

trap 'kill $MONITOR_PID 2>/dev/null; exit 1' INT TERM

# ── Run ranker ──
echo "[Gemini] Starting ranker..."
python3 -m src.judge.rank_bon --config "$CONFIG"
RANK_EXIT=$?

# ── Cleanup ──
echo "[Gemini] Ranker finished (exit=$RANK_EXIT). Stopping monitor..."
kill $MONITOR_PID 2>/dev/null || true
exit $RANK_EXIT
