#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Assemble filtered synthetic reasoning into SFT training data.

Reads filtered.jsonl, selects the best reasoning per (index, mode),
and formats into the final training JSONL.

Usage:
    python -m src.synth.assemble --config configs/synth_reasoning.yaml
    python -m src.synth.assemble --config configs/synth_reasoning.yaml --top-k 2
"""

import os
import json
import logging
import argparse
from collections import defaultdict

log = logging.getLogger(__name__)


def load_filtered(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, "filtered.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    log.info("Loaded %d filtered records", len(records))
    return records


def score_reasoning(rec: dict) -> float:
    """Heuristic score for reasoning quality (higher = better).

    Used to rank multiple samples for the same (index, mode).
    """
    thinking = rec.get("thinking", "")
    score = 0.0

    # Prefer medium length (not too short, not too long)
    length = len(thinking)
    if 800 <= length <= 4000:
        score += 2.0
    elif 500 <= length <= 6000:
        score += 1.0

    # Reward specificity markers
    import re
    # Bbox references
    score += min(len(re.findall(r'bbox|position|coord', thinking.lower())), 5) * 0.3
    # Numerical values (shows specific reasoning)
    score += min(len(re.findall(r'\d{2,}', thinking)), 10) * 0.2
    # Decision language ("I need to", "because", "since")
    decision_markers = re.findall(
        r'\b(need to|should|because|since|therefore|considering|given that)\b',
        thinking.lower(),
    )
    score += min(len(decision_markers), 8) * 0.3

    # Penalize repetitive patterns
    sentences = thinking.split(". ")
    if len(sentences) > 3:
        unique_starts = len(set(s[:30].lower() for s in sentences if len(s) > 30))
        diversity = unique_starts / len(sentences)
        score += diversity * 2.0

    return score


def select_top_k(records: list[dict], top_k: int = 1) -> list[dict]:
    """Select top-K reasoning traces per (index, mode) by heuristic score."""
    grouped = defaultdict(list)
    for rec in records:
        key = (rec["index"], rec["mode"])
        grouped[key].append(rec)

    selected = []
    for key, recs in grouped.items():
        scored = [(score_reasoning(r), r) for r in recs]
        scored.sort(key=lambda x: -x[0])
        for _, rec in scored[:top_k]:
            selected.append(rec)

    log.info("Selected %d records (top-%d per index-mode, from %d groups)",
             len(selected), top_k, len(grouped))
    return selected


def format_sft_record(rec: dict) -> dict:
    """Format a record for SFT training.

    Output format matches what Qwen3.5 currently produces:
    input: prompt [width: W, height: H]
    output: <think>reasoning</think>{json_sp}

    For forward mode: if bbox IoU with reference is too low, use generated SP
    (which is spatially consistent with the thinking) instead of reference SP.
    """
    prompt = rec["prompt"]
    w = rec.get("width", 1024)
    h = rec.get("height", 1024)
    thinking = rec["thinking"]

    # Choose SP source
    if rec.get("_use_generated_sp") and rec.get("generated_sp"):
        sp = rec["generated_sp"]
        sp_source = "generated"
    else:
        sp = rec.get("reference_sp", rec.get("structured_prompt", ""))
        sp_source = "reference"

    return {
        "index": rec["index"],
        "mode": rec["mode"],
        "sp_source": sp_source,
        "input": f"{prompt} [width: {w}, height: {h}]",
        "output": f"<think>\n{thinking}\n</think>\n{sp}",
        "prompt": prompt,
        "thinking": thinking,
        "structured_prompt": sp,
    }


def run_assemble(cfg: dict, top_k: int = 1):
    """Assemble final SFT dataset."""
    synth_cfg = cfg["synth"]
    output_dir = synth_cfg["output_dir"]

    records = load_filtered(output_dir)
    selected = select_top_k(records, top_k=top_k)

    # Format for SFT
    sft_records = [format_sft_record(r) for r in selected]

    # Write output
    sft_path = os.path.join(output_dir, "sft_data.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    mode_counts = defaultdict(int)
    for r in sft_records:
        mode_counts[r["mode"]] += 1

    log.info("Assembled %d SFT records → %s", len(sft_records), sft_path)
    log.info("By mode: %s", dict(mode_counts))

    # Also write a stats summary
    stats = {
        "total_records": len(sft_records),
        "by_mode": dict(mode_counts),
        "top_k": top_k,
        "unique_indices": len(set(r["index"] for r in sft_records)),
    }
    stats_path = os.path.join(output_dir, "assemble_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Assemble SFT dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Keep top-K per (index, mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_assemble(cfg, top_k=args.top_k)


if __name__ == "__main__":
    main()
