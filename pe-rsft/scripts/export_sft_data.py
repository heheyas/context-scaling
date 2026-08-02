#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Export agreed-upon best samples as ms-swift SFT training data.

Given N judge ranking directories, finds indices where all judges agree
on the best image, then extracts the corresponding LLM messages from
llm_out shards and writes ms-swift JSONL format.

Usage:
    python scripts/export_sft_data.py \
        --judge "Seed GRM" /path/to/rankings/ \
        --judge "Gemini v2" /path/to/rankings_gemini_v2/ \
        --judge "HPSv3" /path/to/rankings_hpsv3/ \
        --llm-out /path/to/llm_out/done/ \
        --output sft_train.jsonl

    # Filter by minimum score threshold (all judges must exceed):
    python scripts/export_sft_data.py \
        --judge "Seed GRM" /path/to/rankings/ \
        --judge "HPSv3" /path/to/rankings_hpsv3/ \
        --llm-out /path/to/llm_out/done/ \
        --output sft_train.jsonl \
        --min-score 7.0
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


class JudgeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        judges = getattr(namespace, self.dest, None) or []
        judges.append((values[0], values[1]))
        setattr(namespace, self.dest, judges)


def load_rankings(dirpath: str) -> Dict[int, dict]:
    """Load all results_*.jsonl from a directory."""
    pattern = os.path.join(dirpath, "results_*.jsonl")
    files = sorted(glob.glob(pattern))
    groups = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = item["index"]
                if idx not in groups:
                    groups[idx] = item
    return groups


def load_llm_shards(llm_out_dir: str) -> Dict[Tuple[int, int], dict]:
    """Load all shard_*.jsonl and index by (index, image_idx)."""
    pattern = os.path.join(llm_out_dir, "shard_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"WARNING: No shard files found at {pattern}")
        return {}

    records = {}
    total = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = (item["index"], item["image_idx"])
                if key not in records:
                    records[key] = item
                    total += 1

    print(f"Loaded {total} LLM records from {len(files)} shard files")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Export agreed-upon best samples as ms-swift SFT data"
    )
    parser.add_argument(
        "--judge", nargs=2, action=JudgeAction, dest="judges",
        metavar=("LABEL", "DIR"),
        help="Judge label and rankings dir (repeat for each judge)"
    )
    parser.add_argument(
        "--llm-out", required=True,
        help="Path to llm_out/done/ directory containing shard_*.jsonl"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path (ms-swift format)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0,
        help="Min score threshold: best image must score >= this in ALL judges (0=no filter)"
    )
    parser.add_argument(
        "--require-pass", action="store_true",
        help="Require pass=True in all judges that have a pass field"
    )
    parser.add_argument(
        "--max-waits", type=int, default=0,
        help="Max 'wait' count in thinking (0=no filter). Recommended: 10"
    )
    parser.add_argument(
        "--max-think-len", type=int, default=0,
        help="Max thinking char length (0=no filter). Recommended: 10000"
    )
    args = parser.parse_args()

    if not args.judges or len(args.judges) < 1:
        parser.error("Need at least 1 --judge arg")

    judge_names = [j[0] for j in args.judges]
    print("=" * 60)
    print(f"Export SFT Data — Judges: {', '.join(judge_names)}")
    print("=" * 60)

    # Load all judges
    judges = []
    for label, dirpath in args.judges:
        data = load_rankings(dirpath)
        print(f"  [{label}]: {len(data)} indices")
        judges.append((label, data))

    # Find overlap
    all_sets = [set(jd.keys()) for _, jd in judges]
    overlap = all_sets[0]
    for s in all_sets[1:]:
        overlap &= s
    print(f"Overlap: {len(overlap)} indices")

    # Find all-agree indices
    agree_indices = {}  # idx -> best_image_idx
    for idx in sorted(overlap):
        bests = [jd[idx]["best_image_idx"] for _, jd in judges]
        if len(set(bests)) == 1:
            best_img = bests[0]

            # Score filter
            if args.min_score > 0:
                skip = False
                for _, jd in judges:
                    scores = jd[idx].get("scores", {})
                    sc = scores.get(str(best_img), scores.get(best_img, {}))
                    score = sc.get("score", 0) if isinstance(sc, dict) else 0
                    rating = jd[idx].get("ratings", {}).get(str(best_img), 0)
                    final_score = score or float(rating)
                    if final_score < args.min_score:
                        skip = True
                        break
                if skip:
                    continue

            # Pass filter
            if args.require_pass:
                skip = False
                for _, jd in judges:
                    scores = jd[idx].get("scores", {})
                    sc = scores.get(str(best_img), scores.get(best_img, {}))
                    if isinstance(sc, dict) and "pass" in sc and not sc["pass"]:
                        skip = True
                        break
                if skip:
                    continue

            agree_indices[idx] = best_img

    print(f"All-agree indices: {len(agree_indices)}")
    if args.min_score > 0:
        print(f"  (after min_score >= {args.min_score} filter)")
    if args.require_pass:
        print(f"  (after require_pass filter)")

    # Load LLM shards
    llm_records = load_llm_shards(args.llm_out)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    missing = 0
    filtered_waits = 0
    filtered_think_len = 0

    def _extract_thinking(record):
        """Extract thinking text from assistant message or llm_raw_response."""
        raw = record.get("llm_raw_response", "")
        if not raw:
            # Try from messages
            for m in record.get("messages", []):
                if m["role"] == "assistant":
                    raw = m["content"]
                    break
        if not raw:
            return ""
        m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        return m.group(1) if m else ""

    with open(args.output, "w", encoding="utf-8") as f:
        for idx in sorted(agree_indices.keys()):
            best_img = agree_indices[idx]
            key = (idx, best_img)
            record = llm_records.get(key)

            if record is None:
                missing += 1
                continue

            messages = record.get("messages")
            if not messages:
                missing += 1
                continue

            # Thinking quality filters
            if args.max_waits > 0 or args.max_think_len > 0:
                thinking = _extract_thinking(record)

                if args.max_think_len > 0 and len(thinking) > args.max_think_len:
                    filtered_think_len += 1
                    continue

                if args.max_waits > 0:
                    wait_count = len(re.findall(r"\bwait\b", thinking, re.IGNORECASE))
                    if wait_count > args.max_waits:
                        filtered_waits += 1
                        continue

            # Write ms-swift format
            out = {"messages": messages}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print()
    print(f"Written: {written} samples to {args.output}")
    if missing:
        print(f"Missing: {missing} (not found in llm_out shards)")
    if filtered_waits:
        print(f"Filtered (waits > {args.max_waits}): {filtered_waits}")
    if filtered_think_len:
        print(f"Filtered (think_len > {args.max_think_len}): {filtered_think_len}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
