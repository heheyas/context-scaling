#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Run OneIG-Benchmark evaluation for a given output directory.
#
# This script patches all hardcoded model paths to use local HDFS weights,
# then runs each evaluation dimension (alignment, text, diversity, style, reasoning).
#
# Usage:
#   bash scripts/run_judge_oneig.sh --output_dir outputs/oneig_gemini_my_trial --model_name qwenimage
#
# Required:
#   --output_dir    inference output directory (contains anime/, human/, object/, etc.)
#   --model_name    model name used in directory structure (e.g. "qwenimage")
#
# Optional:
#   --dimensions    space-separated list of dimensions to run (default: all)
#                   choices: alignment text diversity style reasoning
#   --image_grid    grid size (default: "2,2")
#   --mode          EN or ZH (default: EN)
#   --weights_dir   base weights directory (default: <HDFS_ROOT>/weights)
#
# Examples:
#   bash scripts/run_judge_oneig.sh --output_dir outputs/oneig_gemini_exp1 --model_name qwenimage
#   bash scripts/run_judge_oneig.sh --output_dir outputs/oneig_gemini_exp1 --model_name qwenimage --dimensions "alignment text"

set -e
cd "$(dirname "$0")/.."

# ── Parse args ──
OUTPUT_DIR=""
MODEL_NAME=""
DIMENSIONS="alignment text diversity style reasoning"
IMAGE_GRID="2,2"
MODE="EN"
WEIGHTS_DIR="<HDFS_ROOT>/weights"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --model_name)   MODEL_NAME="$2"; shift 2 ;;
        --dimensions)   DIMENSIONS="$2"; shift 2 ;;
        --image_grid)   IMAGE_GRID="$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --weights_dir)  WEIGHTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: --output_dir and --model_name are required."
    echo "Usage: bash scripts/run_judge_oneig.sh --output_dir outputs/oneig_gemini_exp1 --model_name qwenimage"
    exit 1
fi

# ── Model paths (all on HDFS) ──
QWEN_PATH="${WEIGHTS_DIR}/Qwen2.5-VL-7B-Instruct"
CSD_CKPT="${WEIGHTS_DIR}/oneig_style_models/checkpoint.pth"
CSD_CLIP_PATH="${WEIGHTS_DIR}/oneig_style_models/ViT-L-14.pt"
STYLE_ENCODER_PATH="${WEIGHTS_DIR}/OneIG-StyleEncoder"
CLIP_PROCESSOR_PATH="${WEIGHTS_DIR}/clip-vit-large-patch14-336"
LLM2CLIP_MODEL_PATH="${WEIGHTS_DIR}/LLM2CLIP-Openai-L-14-336"
LLM2CLIP_LLM_PATH="${WEIGHTS_DIR}/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
DREAMSIM_CACHE="${WEIGHTS_DIR}/dreamsim_ensemble"

# Verify key models exist
for p in "$QWEN_PATH" "$CSD_CKPT" "$CSD_CLIP_PATH" "$STYLE_ENCODER_PATH" "$CLIP_PROCESSOR_PATH" "$LLM2CLIP_MODEL_PATH" "$LLM2CLIP_LLM_PATH"; do
    if [ ! -e "$p" ]; then
        echo "WARNING: model not found: $p"
    fi
done

# ── Export as env vars so Python scripts can read them ──
export ONEIG_QWEN_PATH="$QWEN_PATH"
export ONEIG_CSD_CKPT="$CSD_CKPT"
export ONEIG_CSD_CLIP_PATH="$CSD_CLIP_PATH"
export ONEIG_STYLE_ENCODER_PATH="$STYLE_ENCODER_PATH"
export ONEIG_CLIP_PROCESSOR_PATH="$CLIP_PROCESSOR_PATH"
export ONEIG_LLM2CLIP_MODEL_PATH="$LLM2CLIP_MODEL_PATH"
export ONEIG_LLM2CLIP_LLM_PATH="$LLM2CLIP_LLM_PATH"
export ONEIG_DREAMSIM_CACHE="$DREAMSIM_CACHE"

echo "============================================"
echo "OneIG-Benchmark Judge"
echo "============================================"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Model name:  $MODEL_NAME"
echo "  Dimensions:  $DIMENSIONS"
echo "  Grid:        $IMAGE_GRID"
echo "  Mode:        $MODE"
echo "  Weights:     $WEIGHTS_DIR"
echo ""

# ── Change to OneIG-Benchmark dir (scripts use relative imports) ──
ONEIG_DIR="benchmarks/OneIG-Benchmark"
cd "$ONEIG_DIR"

# We need the output_dir as absolute path
ABS_OUTPUT_DIR="$(cd "../../$OUTPUT_DIR" 2>/dev/null && pwd)" || ABS_OUTPUT_DIR="$OUTPUT_DIR"
if [[ "$OUTPUT_DIR" == /* ]]; then
    ABS_OUTPUT_DIR="$OUTPUT_DIR"
fi

# ── Apply model path patches via a Python preamble ──
# This monkey-patches the default model paths before the scoring scripts import them.
PATCH_SCRIPT=$(cat << 'PYEOF'
import os, sys

# Patch inference.py defaults via environment variables
_qwen = os.environ.get("ONEIG_QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
_csd_ckpt = os.environ.get("ONEIG_CSD_CKPT", "scripts/style/models/checkpoint.pth")
_csd_clip = os.environ.get("ONEIG_CSD_CLIP_PATH", "scripts/style/models/ViT-L-14.pt")
_se = os.environ.get("ONEIG_STYLE_ENCODER_PATH", "xingpng/OneIG-StyleEncoder")
_clip_proc = os.environ.get("ONEIG_CLIP_PROCESSOR_PATH", "openai/clip-vit-large-patch14-336")
_llm2clip = os.environ.get("ONEIG_LLM2CLIP_MODEL_PATH", "microsoft/LLM2CLIP-Openai-L-14-336")
_llm2clip_llm = os.environ.get("ONEIG_LLM2CLIP_LLM_PATH", "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")

# Store in a module-level dict for patching
os.environ["_ONEIG_PATHS_READY"] = "1"
PYEOF
)

# Write the patch as a sitecustomize-like module
PATCH_DIR=$(mktemp -d)
cat > "$PATCH_DIR/oneig_patch.py" << 'PYEOF'
"""Monkey-patch OneIG model paths to use local weights."""
import os

def patch_inference():
    """Patch scripts.utils.inference module after it's imported."""
    import scripts.utils.inference as inf

    # Patch Qwen default
    _orig_qwen_init = inf.Qwen2_5VLBatchInferencer.__init__
    _qwen_path = os.environ.get("ONEIG_QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    def _new_qwen_init(self, model_path=_qwen_path, **kwargs):
        _orig_qwen_init(self, model_path=model_path, **kwargs)
    inf.Qwen2_5VLBatchInferencer.__init__ = _new_qwen_init

    # Patch CSD default
    _orig_csd_init = inf.CSDStyleEmbedding.__init__
    _csd_path = os.environ.get("ONEIG_CSD_CKPT", "scripts/style/models/checkpoint.pth")
    def _new_csd_init(self, model_path=_csd_path, **kwargs):
        _orig_csd_init(self, model_path=model_path, **kwargs)
    inf.CSDStyleEmbedding.__init__ = _new_csd_init

    # Patch SE default
    _orig_se_init = inf.SEStyleEmbedding.__init__
    _se_path = os.environ.get("ONEIG_STYLE_ENCODER_PATH", "xingpng/OneIG-StyleEncoder")
    def _new_se_init(self, pretrained_path=_se_path, **kwargs):
        _orig_se_init(self, pretrained_path=pretrained_path, **kwargs)
    inf.SEStyleEmbedding.__init__ = _new_se_init

    # Patch LLM2CLIP defaults
    _orig_llm2clip_init = inf.LLM2CLIP.__init__
    _clip_proc = os.environ.get("ONEIG_CLIP_PROCESSOR_PATH", "openai/clip-vit-large-patch14-336")
    _llm2clip_model = os.environ.get("ONEIG_LLM2CLIP_MODEL_PATH", "microsoft/LLM2CLIP-Openai-L-14-336")
    _llm2clip_llm = os.environ.get("ONEIG_LLM2CLIP_LLM_PATH", "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
    def _new_llm2clip_init(self, processor_model=_clip_proc, model_name=_llm2clip_model,
                           llm_model_name=_llm2clip_llm, device='cuda'):
        _orig_llm2clip_init(self, processor_model=processor_model, model_name=model_name,
                           llm_model_name=llm_model_name, device=device)
    inf.LLM2CLIP.__init__ = _new_llm2clip_init

def patch_csd_config():
    """Patch CSD_config to use local CLIP path."""
    import scripts.utils.CSD_config as csd_cfg
    import clip
    _csd_clip_path = os.environ.get("ONEIG_CSD_CLIP_PATH", "scripts/style/models/ViT-L-14.pt")

    _orig_csd_clip_init = csd_cfg.CSD_CLIP.__init__
    def _new_csd_clip_init(self, name='vit_large', content_proj_head='default', model_path=None):
        if model_path is None:
            model_path = _csd_clip_path
        _orig_csd_clip_init(self, name=name, content_proj_head=content_proj_head, model_path=model_path)
    csd_cfg.CSD_CLIP.__init__ = _new_csd_clip_init
PYEOF

# ── Run each dimension ──
# We use PYTHONPATH=$PATCH_DIR + usercustomize.py to auto-patch before any import
cat > "$PATCH_DIR/usercustomize.py" << 'PYEOF2'
"""Auto-patch OneIG model paths on import."""
import importlib, os, sys

_patched = set()

class _OneIGImportHook:
    def find_module(self, name, path=None):
        if name in ("scripts.utils.inference", "scripts.utils.CSD_config") and name not in _patched:
            return self
        return None

    def load_module(self, name):
        _patched.add(name)
        # Let the real import happen first
        self.__class__.find_module = lambda s, n, p=None: None
        mod = importlib.import_module(name)
        self.__class__.find_module = _OneIGImportHook.find_module

        if name == "scripts.utils.inference":
            # Patch Qwen
            _qwen = os.environ.get("ONEIG_QWEN_PATH")
            if _qwen:
                _orig_qwen = mod.Qwen2_5VLBatchInferencer.__init__
                def _new_qwen(self, model_path=_qwen, _orig=_orig_qwen, **kw): _orig(self, model_path=model_path, **kw)
                mod.Qwen2_5VLBatchInferencer.__init__ = _new_qwen

            # Patch CSD
            _csd = os.environ.get("ONEIG_CSD_CKPT")
            if _csd:
                _orig_csd = mod.CSDStyleEmbedding.__init__
                def _new_csd(self, model_path=_csd, _orig=_orig_csd, **kw): _orig(self, model_path=model_path, **kw)
                mod.CSDStyleEmbedding.__init__ = _new_csd

            # Patch SE
            _se = os.environ.get("ONEIG_STYLE_ENCODER_PATH")
            if _se:
                _orig_se = mod.SEStyleEmbedding.__init__
                def _new_se(self, pretrained_path=_se, _orig=_orig_se, **kw): _orig(self, pretrained_path=pretrained_path, **kw)
                mod.SEStyleEmbedding.__init__ = _new_se

            # Patch LLM2CLIP
            _cp = os.environ.get("ONEIG_CLIP_PROCESSOR_PATH")
            _lm = os.environ.get("ONEIG_LLM2CLIP_MODEL_PATH")
            _ll = os.environ.get("ONEIG_LLM2CLIP_LLM_PATH")
            if _cp and _lm and _ll:
                _orig_l2c = mod.LLM2CLIP.__init__
                def _new_l2c(self, processor_model=_cp, model_name=_lm, llm_model_name=_ll, device='cuda', _orig=_orig_l2c):
                    _orig(self, processor_model=processor_model, model_name=model_name,
                          llm_model_name=llm_model_name, device=device)
                mod.LLM2CLIP.__init__ = _new_l2c

        elif name == "scripts.utils.CSD_config":
            _clip_path = os.environ.get("ONEIG_CSD_CLIP_PATH")
            if _clip_path:
                _orig_csd_clip = mod.CSD_CLIP.__init__
                def _new_csd_clip(self, name='vit_large', content_proj_head='default', model_path=_clip_path, _orig=_orig_csd_clip):
                    _orig(self, name=name, content_proj_head=content_proj_head, model_path=model_path)
                mod.CSD_CLIP.__init__ = _new_csd_clip

        return mod

if os.environ.get("_ONEIG_PATHS_READY") == "1":
    sys.meta_path.insert(0, _OneIGImportHook())
PYEOF2

export ENABLE_USER_SITE=1
export _ONEIG_PATHS_READY=1
export PYTHONPATH="$PATCH_DIR:${PYTHONPATH:-}"

run_dimension() {
    local dim=$1
    echo ""
    echo "--- Running: $dim ---"

    RESULTS_DIR="${ABS_OUTPUT_DIR}/oneig_results"

    case $dim in
        alignment)
            python -m scripts.alignment.alignment_score \
                --mode "$MODE" \
                --image_dirname "$ABS_OUTPUT_DIR" \
                --model_names "$MODEL_NAME" \
                --image_grid "$IMAGE_GRID" \
                --class_items "anime" "human" "object" \
                --results_dir "$RESULTS_DIR"
            ;;
        text)
            python -m scripts.text.text_score \
                --mode "$MODE" \
                --image_dirname "${ABS_OUTPUT_DIR}/text" \
                --model_names "$MODEL_NAME" \
                --image_grid "$IMAGE_GRID" \
                --results_dir "$RESULTS_DIR"
            ;;
        diversity)
            python -m scripts.diversity.diversity_score \
                --mode "$MODE" \
                --image_dirname "$ABS_OUTPUT_DIR" \
                --model_names "$MODEL_NAME" \
                --image_grid "$IMAGE_GRID" \
                --class_items "anime" "human" "object" "text" "reasoning" \
                --results_dir "$RESULTS_DIR"
            ;;
        style)
            python -m scripts.style.style_score \
                --mode "$MODE" \
                --image_dirname "${ABS_OUTPUT_DIR}/anime" \
                --model_names "$MODEL_NAME" \
                --image_grid "$IMAGE_GRID" \
                --results_dir "$RESULTS_DIR"
            ;;
        reasoning)
            python -m scripts.reasoning.reasoning_score \
                --mode "$MODE" \
                --image_dirname "${ABS_OUTPUT_DIR}/reasoning" \
                --model_names "$MODEL_NAME" \
                --image_grid "$IMAGE_GRID" \
                --results_dir "$RESULTS_DIR"
            ;;
        *)
            echo "Unknown dimension: $dim"
            ;;
    esac
}

for dim in $DIMENSIONS; do
    run_dimension "$dim"
done

# Clean up
rm -rf "$PATCH_DIR"

echo ""
echo "============================================"
echo "OneIG-Benchmark Judge Complete"
echo "Results: ${ABS_OUTPUT_DIR}/oneig_results"
echo "============================================"
