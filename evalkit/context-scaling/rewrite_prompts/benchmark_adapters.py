# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Load prompts from each benchmark into a unified format.

Unified format: {"id": str, "benchmark": str, "category": str|None, "prompt": str}
"""

import csv
import glob
import json
import logging
import os
from typing import List

log = logging.getLogger(__name__)

BENCHMARKS = (
    "geneval", "geneval2", "genevalpp", "dpgbench", "oneig", "genexam",
    "sp_l10_200", "sp_sft_200", "sp_rl_200",
    "wise", "tiif_short", "tiif_long",
    "tiif_testmini_short", "tiif_testmini_long",
    "textbench", "alignment_v5", "alignment_v5_long", "infograph", "corebench",
)

# Default data paths relative to the context-scaling directory
DEFAULT_DATA_PATHS = {
    "geneval": "benchmarks/geneval/prompts/evaluation_metadata.jsonl",
    "geneval2": "benchmarks/GenEval2/geneval2_data.jsonl",
    "dpgbench": "benchmarks/ELLA/dpg_bench/dpg_bench.csv",
    "oneig": "benchmarks/OneIG-Benchmark/OneIG-Bench.csv",
    "genexam": "benchmarks/GenExam/genexam_data.jsonl",
    "sp_l10_200": "benchmarks/sp_l10_200/sp_l10_200.jsonl",
    "sp_sft_200": "benchmarks/sp_sft_200/sp_sft_200.jsonl",
    "sp_rl_200": "benchmarks/sp_rl_200/sp_rl_200.jsonl",
    "alignment_v5": "benchmarks/alignment_v5/prompts_t2i_align_v5.1_en.prompt",
    "alignment_v5_long": "benchmarks/alignment_v5_long/prompts_t2i_align_v5.1_long_en.prompt",
    "infograph": "benchmarks/infograph/infograph_data.jsonl",
    "corebench": "benchmarks/T2I-CoReBench/corebench_all.jsonl",
}


def load_prompts(benchmark: str, data_path: str) -> List[dict]:
    """Load prompts from a benchmark data file into unified format.

    Args:
        benchmark: One of "geneval", "geneval2", "dpgbench", "oneig".
        data_path: Path to the benchmark data file.

    Returns:
        List of dicts with keys: id, benchmark, category, prompt.
    """
    loader = {
        "geneval": _load_geneval,
        "geneval2": _load_geneval2,
        "genevalpp": _load_genevalpp,
        "dpgbench": _load_dpgbench,
        "oneig": _load_oneig,
        "sp_l10_200": _load_sp_l10_200,
        "sp_sft_200": _load_sp_l10_200,  # same format
        "sp_rl_200": _load_sp_l10_200,   # same format
        "genexam": _load_genexam,
        "wise": _load_wise,
        "tiif_short": lambda p: _load_tiif(p, "short"),
        "tiif_long": lambda p: _load_tiif(p, "long"),
        "tiif_testmini_short": lambda p: _load_tiif_testmini(p, "short"),
        "tiif_testmini_long": lambda p: _load_tiif_testmini(p, "long"),
        "textbench": _load_textbench,
        "alignment_v5": _load_alignment_v5,
        "alignment_v5_long": _load_alignment_v5_long,
        "infograph": _load_infograph,
        "corebench": _load_corebench,
    }
    if benchmark not in loader:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of {BENCHMARKS}")
    prompts = loader[benchmark](data_path)
    log.info("Loaded %d prompts from %s (%s)", len(prompts), benchmark, data_path)
    return prompts


def _load_geneval2(data_path: str) -> List[dict]:
    """GenEval2: JSONL with 'prompt' field. Use line index as id."""
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": str(idx),
                "benchmark": "geneval2",
                "category": None,
                "prompt": item["prompt"],
            })
    return prompts


def _load_genevalpp(data_path: str) -> List[dict]:
    """GenEval++: JSONL with 'prompt' and 'tag' fields. 280 prompts, 1-indexed IDs."""
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": str(idx + 1),  # 1-indexed to match official image naming
                "benchmark": "genevalpp",
                "category": item.get("tag"),
                "prompt": item["prompt"],
            })
    return prompts


def _load_geneval(data_path: str) -> List[dict]:
    """GenEval: JSONL (evaluation_metadata.jsonl) with 'prompt' and 'tag' fields."""
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": str(idx).zfill(5),
                "benchmark": "geneval",
                "category": item.get("tag"),
                "prompt": item["prompt"],
            })
    return prompts


def _load_dpgbench(data_path: str) -> List[dict]:
    """DPG-Bench: CSV with 'item_id' and 'text' columns. Deduplicate by item_id."""
    seen = {}
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"]
            if item_id not in seen:
                seen[item_id] = {
                    "id": item_id,
                    "benchmark": "dpgbench",
                    "category": None,
                    "prompt": row["text"],
                }
    return list(seen.values())


def _load_oneig(data_path: str) -> List[dict]:
    """OneIG-Bench: CSV with 'id', 'prompt_en', 'category' columns.

    IDs in the CSV are NOT unique — the same id appears in multiple categories.
    We prefix with category to make them unique: "{category}/{id}".
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "")
            prompts.append({
                "id": f"{category}/{row['id']}",
                "benchmark": "oneig",
                "category": category,
                "prompt": row["prompt_en"],
            })
    return prompts


def _load_genexam(data_path: str) -> List[dict]:
    """GenExam: JSONL with 'id', 'prompt', 'taxonomy' fields."""
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Use top-level subject from taxonomy (e.g. "Mathematics" from "Mathematics/Analytic_Geometry/...")
            taxonomy = item.get("taxonomy", "")
            category = taxonomy.split("/")[0] if taxonomy else None
            prompts.append({
                "id": item["id"],
                "benchmark": "genexam",
                "category": category,
                "prompt": item["prompt"],
            })
    return prompts


def _load_sp_l10_200(data_path: str) -> List[dict]:
    """SP-L10-200: JSONL with 'id', 'original_prompt', 'rewritten_prompt' fields.

    This is a custom benchmark sampled from training data with l10 structured prompts.
    The rewritten_prompt already contains the full structured prompt JSON.
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": item["id"],
                "benchmark": "sp_l10_200",
                "category": item.get("category"),
                "prompt": item["original_prompt"],
            })
    return prompts


def _load_wise(data_path: str) -> List[dict]:
    """WISE: 3 JSON files in data/ directory, each an array of objects.

    data_path can be the data/ directory (loads all non-rewrite files) or a single JSON file.
    Fields: Prompt, Explanation, Category, Subcategory, prompt_id (1-1000).
    """
    if os.path.isdir(data_path):
        json_files = sorted(glob.glob(os.path.join(data_path, "*.json")))
        json_files = [f for f in json_files if "rewrite" not in os.path.basename(f)]
    else:
        json_files = [data_path]

    prompts = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            prompts.append({
                "id": str(item["prompt_id"]),
                "benchmark": "wise",
                "category": item.get("Subcategory") or item.get("Category"),
                "prompt": item["Prompt"],
            })
    return prompts


def _load_tiif(data_path: str, desc_type: str) -> List[dict]:
    """TIIF-Bench: 39 JSONL files in test_prompts/ directory.

    data_path should be the test_prompts/ directory.
    desc_type: "short" or "long" — selects short_description or long_description.
    ID format: "{type}/{idx}" for unique identification and formatter path reconstruction.
    """
    benchmark = f"tiif_{desc_type}"
    field = f"{desc_type}_description"

    if os.path.isdir(data_path):
        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*_prompts.jsonl")))
    else:
        jsonl_files = [data_path]

    prompts = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                dim_type = item["type"]
                prompts.append({
                    "id": f"{dim_type}/{idx}",
                    "benchmark": benchmark,
                    "category": dim_type,
                    "prompt": item[field],
                })
    return prompts


def _load_tiif_testmini(data_path: str, desc_type: str) -> List[dict]:
    """TIIF-Bench testmini subset: 277 samples from the full 2538.

    data_path should be the TIIF-Bench data/ directory (containing test_prompts/
    and testmini_sample_idx.json).
    """
    benchmark = f"tiif_testmini_{desc_type}"

    # Load testmini index
    idx_file = os.path.join(data_path, "testmini_sample_idx.json")
    with open(idx_file, "r", encoding="utf-8") as f:
        testmini_idx = json.load(f)  # {attr_type: [indices]}

    # Build set of (attr_type, idx) pairs for fast lookup
    testmini_set = set()
    for attr_type, indices in testmini_idx.items():
        for idx in indices:
            testmini_set.add((attr_type, idx))

    # Load full TIIF and filter
    full_prompts = _load_tiif(os.path.join(data_path, "test_prompts"), desc_type)
    prompts = []
    for item in full_prompts:
        # id format is "{type}/{idx}"
        parts = item["id"].split("/", 1)
        attr_type, idx = parts[0], int(parts[1])
        if (attr_type, idx) in testmini_set:
            item["benchmark"] = benchmark
            prompts.append(item)

    return prompts


def _load_textbench(data_path: str) -> List[dict]:
    """X-Omni TextBench: JSONL with prompt_id, prompt, text, category, length.

    data_path: path to text_prompts.jsonl (EN) or text_prompts_zh.jsonl (ZH).
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": str(item["prompt_id"]),
                "benchmark": "textbench",
                "category": item.get("length"),  # "short" or "long" (text length category)
                "prompt": item["prompt"],
            })
    return prompts


def _load_alignment_v5(data_path: str) -> List[dict]:
    """Alignment V5: tab-separated file, each line is '{id}\\t{prompt}'.

    data_path: path to .prompt file.
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            prompts.append({
                "id": parts[0],
                "benchmark": "alignment_v5",
                "category": None,
                "prompt": parts[1],
            })
    return prompts


def _load_alignment_v5_long(data_path: str) -> List[dict]:
    """Alignment V5 (long): same TSV format as alignment_v5; prompts are pre-expanded
    detailed natural-language descriptions. IDs match the alignment_v5 benchmark.
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            prompts.append({
                "id": parts[0],
                "benchmark": "alignment_v5_long",
                "category": None,
                "prompt": parts[1],
            })
    return prompts


def _load_infograph(data_path: str) -> List[dict]:
    """Infograph: JSONL with per-prompt dimensions.

    Format: {id, prompt, width, height, ref_image}
    """
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts.append({
                "id": str(item["id"]),
                "benchmark": "infograph",
                "category": None,
                "prompt": item["prompt"],
                "width": item.get("width"),
                "height": item.get("height"),
            })
    return prompts


def _load_corebench(data_path: str) -> List[dict]:
    """T2I-CoReBench: JSONL with Checklist-based evaluation.

    Format: {ID, Main Class, Sub Class, Prompt, Checklist, Remark}
    Can be a single merged file or a directory of split files.
    """
    if os.path.isdir(data_path):
        jsonl_files = sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
    else:
        jsonl_files = [data_path]

    prompts = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompts.append({
                    "id": item["ID"],
                    "benchmark": "corebench",
                    "category": f"{item.get('Main Class', '')}/{item.get('Sub Class', '')}",
                    "prompt": item["Prompt"],
                })
    return prompts
