#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Export SFT training data from a single ranking file + llm_out shards.

Differs from the existing scripts/export_sft_data.py (which expects multiple
judges with cross-judge agreement); this one fits the iter1 single-judge
verifier setup where each ranking record carries its own verification block.

Filters supported:
  --min-score       overall_score >= this (default 0 = no filter)
  --require-pass    overall_pass must be True (default off)
  --all-facets-pass every facet (alignment/thinking/aesthetic/structure)
                    must have pass=True
  --max-waits N     skip rollouts whose <think> contains > N occurrences
                    of the word "wait" (overthinking heuristic)
  --max-think-len N skip rollouts whose <think> exceeds N chars

Handles two ranking formats automatically:
  A) verification top-level (manual_cot): rk["verification"]
  B) per-image scores (single_detail):    rk["scores"][str(best_image_idx)]

Output: JSONL with one ms-swift training row per kept sample. Each row is
{"messages": <full multi-turn trajectory from llm_out>}.

Usage:
    python scripts/export_sft_from_ranking.py \\
        --ranking-jsonl /path/to/results_*.jsonl \\
        --llm-out-globs /path/to/llm_out/done/* \\
        --output sft_iter1_ge9.jsonl \\
        --require-pass --min-score 9.0
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter


def get_verification(d: dict) -> dict:
    """Best-image verification block, regardless of ranking format."""
    v = d.get("verification")
    if isinstance(v, dict) and v:
        return v
    scores = d.get("scores") or {}
    best = d.get("best_image_idx")
    if best is None or not isinstance(scores, dict):
        return {}
    return scores.get(str(best)) or scores.get(best) or {}


# `thinking` and `aesthetic` both excluded from clean overall and from
# facet-pass check — both observed unreliable by the user for this verifier.
SCORED_FACETS = ("alignment", "structure")


def clean_overall(v: dict):
    vals = []
    for k in SCORED_FACETS:
        sub = v.get(k)
        if isinstance(sub, dict):
            s = sub.get("score")
            if s is not None:
                vals.append(float(s))
    if not vals:
        return None
    return sum(vals) / len(vals)


def all_facets_pass(v: dict) -> bool:
    """All SCORED_FACETS (excluding thinking) must have pass=True."""
    for facet in SCORED_FACETS:
        sub = v.get(facet)
        if not isinstance(sub, dict):
            return False
        if not sub.get("pass"):
            return False
    return True


def load_llm_out(globs: list) -> dict:
    """(idx, image_idx) -> llm_out record."""
    out = {}
    for pat in globs:
        for fp in glob.glob(pat):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    key = (d.get("index"), d.get("image_idx"))
                    if key[0] is not None and key[1] is not None:
                        out[key] = d
    return out


def extract_thinking(record: dict) -> str:
    """<think>...</think> content from llm_raw_response or assistant message."""
    raw = record.get("llm_raw_response") or ""
    if not raw:
        for m in record.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "assistant":
                raw = m.get("content") or ""
                break
    if not raw:
        return ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return m.group(1) if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-jsonl", nargs="+", required=True,
                   help="One or more ranking JSONL files (concatenated)")
    p.add_argument("--llm-out-globs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-score", type=float, default=0,
                   help="Min CLEAN overall_score = mean of alignment/structure (thinking + aesthetic dropped)")
    p.add_argument("--min-structure", type=float, default=0)
    p.add_argument("--min-alignment", type=float, default=0)
    p.add_argument("--require-pass", action="store_true",
                   help="Require overall_pass=True (verifier's own; includes thinking + aesthetic)")
    p.add_argument("--all-facets-pass", action="store_true",
                   help="Require alignment & structure both pass=True (thinking + aesthetic ignored)")
    p.add_argument("--max-waits", type=int, default=0,
                   help="Drop if <think> has > N 'wait' occurrences (0 off)")
    p.add_argument("--max-think-len", type=int, default=0,
                   help="Drop if <think> exceeds N chars (0 off)")
    p.add_argument("--include-tools-field", action="store_true",
                   help="Also write 'tools' field (full schema). Off by default")
    args = p.parse_args()

    print(f"Reading {len(args.ranking_jsonl)} ranking file(s)…")
    rankings = []
    seen_idx = set()
    for fp in args.ranking_jsonl:
        n_file = 0
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                idx = d.get("index")
                if idx in seen_idx:
                    continue  # de-dup across files (last-write-wins not needed here)
                seen_idx.add(idx)
                rankings.append(d)
                n_file += 1
        print(f"  {fp}: {n_file} unique indices")
    print(f"  total: {len(rankings)} ranking records")

    print(f"Reading llm_out shards…")
    llm_idx = load_llm_out(args.llm_out_globs)
    print(f"  {len(llm_idx)} (idx,img) llm_out records")

    n_pass = n_score = n_facets = n_waits = n_think_len = 0
    n_no_llm = n_no_msgs = 0
    n_tool = n_no_tool = 0
    score_dist = Counter()
    written = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as out_f:
        for rk in rankings:
            idx = rk["index"]
            best = rk.get("best_image_idx")
            if best is None:
                continue

            v = get_verification(rk)

            if args.require_pass and not v.get("overall_pass"):
                continue
            n_pass += 1

            if args.all_facets_pass and not all_facets_pass(v):
                continue
            n_facets += 1

            os_ = clean_overall(v)
            if args.min_score > 0:
                if os_ is None or os_ < args.min_score:
                    continue

            # Per-facet score filters (thinking excluded by SCORED_FACETS)
            facet_skip = False
            for facet_name, threshold in [
                ("structure", args.min_structure),
                ("alignment", args.min_alignment),
            ]:
                if threshold <= 0:
                    continue
                sub = v.get(facet_name) or {}
                sc = sub.get("score") if isinstance(sub, dict) else None
                if sc is None or sc < threshold:
                    facet_skip = True
                    break
            if facet_skip:
                continue
            n_score += 1

            llm_rec = llm_idx.get((idx, best))
            if llm_rec is None:
                n_no_llm += 1
                continue
            messages = llm_rec.get("messages")
            if not messages:
                n_no_msgs += 1
                continue

            thinking = extract_thinking(llm_rec)
            if args.max_think_len > 0 and len(thinking) > args.max_think_len:
                n_think_len += 1
                continue
            if args.max_waits > 0:
                w = len(re.findall(r"\bwait\b", thinking, re.IGNORECASE))
                if w > args.max_waits:
                    n_waits += 1
                    continue

            tools_used = llm_rec.get("tool_calls_used") or []
            if tools_used:
                n_tool += 1
            else:
                n_no_tool += 1

            row = {"messages": messages}
            if args.include_tools_field and llm_rec.get("tools"):
                row["tools"] = llm_rec["tools"]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if os_ is not None:
                score_dist[round(os_, 1)] += 1

    print()
    print(f"Filtering summary:")
    print(f"  passed overall_pass:       {n_pass}")
    print(f"  passed all-facets-pass:    {n_facets}")
    print(f"  passed score >= {args.min_score}:  {n_score}")
    print(f"  dropped (no llm_out):      {n_no_llm}")
    print(f"  dropped (no messages):     {n_no_msgs}")
    if args.max_think_len:
        print(f"  dropped (think_len > {args.max_think_len}):  {n_think_len}")
    if args.max_waits:
        print(f"  dropped (waits > {args.max_waits}):  {n_waits}")
    print()
    print(f"Written: {written} rows to {args.output}")
    print(f"  with tool : {n_tool}  ({100*n_tool/max(written,1):.1f}%)")
    print(f"  no tool   : {n_no_tool}  ({100*n_no_tool/max(written,1):.1f}%)")
    if score_dist:
        print(f"\nScore distribution (CLEAN overall = mean A/S, thinking + aesthetic dropped):")
        for k in sorted(score_dist.keys(), reverse=True):
            print(f"  {k}: {score_dist[k]}")
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nFile size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
