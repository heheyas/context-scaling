# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Aspect-ratio cleaning for a teacher-rollout JSONL.

Reads the teacher's `rollout_text` output, extracts the `ratio` field from
its JSON body, compares against the source image's aspect ratio, and drops
rows where the teacher's chosen ratio flips orientation (portrait ↔
landscape).

Usage:
    python -m pe-rsft.teacher_distill.ratio_clean \\
        --in  /path/to/rollout_v6_structured_full.jsonl \\
        --out /path/to/rollout_v6_structured_full_ratioclean.jsonl
"""
import argparse
import json
import os
import re
from collections import Counter

from PIL import Image


RATIO_VAL = {
    "1:1": 1.0, "4:3": 4/3, "3:4": 3/4, "16:9": 16/9, "9:16": 9/16,
    "3:2": 3/2, "2:3": 2/3, "4:5": 4/5, "21:9": 21/9,
}


def parse_ratio(rollout_text: str):
    """Extract the `ratio` field from the JSON body after </analysis>."""
    if not rollout_text:
        return None
    mm = re.search(r"</analysis>\s*(.*)", rollout_text, re.DOTALL)
    body = (mm.group(1) if mm else rollout_text).strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.M)
    s = body.find("{")
    if s < 0:
        return None
    m = re.search(r'"ratio"\s*:\s*"([0-9]+:[0-9]+)"', body[s:])
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True,
                   help="Teacher rollout JSONL (output of teacher_rollout.py)")
    p.add_argument("--out", required=True,
                   help="Cleaned JSONL — orientation-flipped rows dropped")
    p.add_argument("--dataset-dir", default=None,
                   help="If a row lacks orig_width/height, read them from "
                        "{dataset_dir}/{image}. Optional; only needed for "
                        "old manifests that didn't record image size.")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    print(f"total rows: {len(rows)}", flush=True)

    cats = Counter()
    ratio_dist = Counter()
    keep = []
    for r in rows:
        rt = r.get("rollout_text", "")
        if not rt or rt.startswith("__ERROR__"):
            cats["err"] += 1
            continue
        chosen = parse_ratio(rt)
        if chosen not in RATIO_VAL:
            cats["noratio"] += 1
            continue
        ratio_dist[chosen] += 1
        W, H = r.get("orig_width"), r.get("orig_height")
        if not (W and H) and args.dataset_dir and r.get("image"):
            with Image.open(os.path.join(args.dataset_dir, r["image"])) as im:
                W, H = im.size
        if not (W and H):
            cats["nosize"] += 1
            continue
        cv, av = RATIO_VAL[chosen], W / H
        if (av > 1.05 and cv < 0.95) or (av < 0.95 and cv > 1.05):
            cats["flip"] += 1
            continue
        r["ratio_chosen"] = chosen
        keep.append(r)

    with open(args.out, "w", encoding="utf-8") as fo:
        for r in keep:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept:    {len(keep)}", flush=True)
    print(f"dropped: {sum(cats.values())}  {dict(cats)}", flush=True)
    print(f"ratio distribution (kept): {dict(ratio_dist)}", flush=True)


if __name__ == "__main__":
    main()
