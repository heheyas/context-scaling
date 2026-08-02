#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Filter synthetic reasoning traces for quality.

Reads raw generation output (WAL files), applies filters, and writes
filtered results to a clean JSONL.

Usage:
    python -m src.synth.filter --config configs/synth_reasoning.yaml
"""

import os
import re
import json
import logging
import argparse
from typing import Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Boilerplate stripping
# ──────────────────────────────────────────────

# Patterns that Gemini likes to prepend — meta-commentary about the task
# itself rather than actual reasoning. These get stripped before filtering.
_BOILERPLATE_PATTERNS = [
    # Backward mode: "Here is..." / "This is..." meta openers about the task itself
    re.compile(
        r"^(?:Here is|This is) [^\n]*?(?:reasoning|trace|thinking|blueprint|process"
        r"|monologue|scene director|visual planner|walking through|breaking down"
        r"|creation of)[^\n]*[.:]\s*",
        re.IGNORECASE,
    ),
    # Forward mode: echoing our prompt template headers (with optional ### prefix)
    re.compile(
        r"^(?:#{1,4}\s*)?(?:\*{0,2})?Part \d+\s*[-—–:]\s*Reasoning\s*"
        r"(?:\(MANDATORY\))?\s*(?:\*{0,2})?\s*:?\s*",
        re.IGNORECASE,
    ),
    # Chinese boilerplate: "好的，我将根据您的要求..." / "接下来我将..."
    re.compile(
        r"^(?:好的[，,]?\s*)?(?:我将|接下来我将|下面我将)[^\n]*?(?:构思|蓝图|JSON|推理|分析)[^\n]*[。.]\s*",
    ),
    # "My thinking process for..." meta opener (various endings)
    re.compile(
        r"^My thinking process for [^\n]*?(?:is as follows|goes as follows"
        r"|goes like this|is as below)[^\n]*:?\s*",
        re.IGNORECASE,
    ),
    # Horizontal rules / decorative separators after the boilerplate
    re.compile(r"^[-*_]{3,}\s*"),
    # Stray bold markers left after stripping (e.g. "**" on its own line)
    re.compile(r"^\*{2,}\s*$", re.MULTILINE),
]


def strip_boilerplate(thinking: str) -> str:
    """Remove meta-commentary boilerplate from the start of thinking text."""
    text = thinking.strip()

    # Apply patterns repeatedly (boilerplate + separator can stack)
    changed = True
    max_rounds = 5
    while changed and max_rounds > 0:
        changed = False
        max_rounds -= 1
        for pat in _BOILERPLATE_PATTERNS:
            new_text = pat.sub("", text, count=1).strip()
            if new_text != text:
                text = new_text
                changed = True

    return text


# ──────────────────────────────────────────────
# Filter functions
# ──────────────────────────────────────────────

def check_length(thinking: str, min_len: int = 200, max_len: int = 15000) -> bool:
    """Check that thinking text is within acceptable length range."""
    return min_len <= len(thinking) <= max_len


def check_no_leakage(thinking: str) -> bool:
    """Check that backward-mode reasoning doesn't passively copy the reference.

    We want to catch PASSIVE DESCRIPTION of the reference answer:
      "According to the blueprint, the lighting is warm"  ← bad, just copying
      "As specified in the reference, use eye-level angle" ← bad, no reasoning

    But NOT penalize ACTIVE REASONING that discusses design values:
      "I'll set depth to 56 because the earring is closer" ← good, explaining why
      "The blueprint has depth 56, which makes sense because..." ← good, justifying

    Strategy: flag sentences that INTRODUCE information with passive framing
    ("according to", "as shown in", "the reference says") — these signal the
    model is reading the answer rather than reasoning toward it.
    """
    # Passive citation patterns — the model is telling you what the reference says
    # rather than reasoning about what it should be
    passive_patterns = [
        r"according to the (?:reference|blueprint|given|provided)",
        r"as (?:shown|specified|stated|given|indicated|defined) in the (?:reference|blueprint|json)",
        r"the (?:reference|provided|given) (?:blueprint|json|output|answer) (?:shows|says|states|indicates|tells|specifies) that",
        r"looking at the (?:reference|given|provided) (?:blueprint|json|output)[,.]?\s*(?:i can see|we can see|it shows|we see|it has|there)",
        r"(?:from|per|based on) the (?:reference|given|provided) (?:blueprint|json|answer)",
        r"the reference (?:shows|says|indicates|tells|specifies) ",
        r"copying (?:from|the) (?:reference|blueprint)",
    ]
    text_lower = thinking.lower()
    for pattern in passive_patterns:
        if re.search(pattern, text_lower):
            return False

    # Also check for excessive verbatim JSON quoting — if the thinking contains
    # long raw JSON fragments (>200 chars), it's probably copy-pasting
    json_blocks = re.findall(r'\{[^}]{200,}\}', thinking)
    if len(json_blocks) >= 2:
        return False

    return True


def check_coherence(thinking: str, prompt: str) -> bool:
    """Check that reasoning mentions key subjects from the prompt.

    Extracts nouns/subjects from prompt and checks that at least some
    appear in the thinking. Very lightweight heuristic.
    """
    # Extract significant words (>4 chars, not common words)
    stop_words = {
        "with", "that", "this", "from", "their", "which", "about",
        "would", "there", "these", "other", "into", "more", "some",
        "been", "have", "will", "each", "make", "like", "than",
        "them", "then", "when", "were", "what", "your", "also",
        "very", "being", "image", "prompt", "scene",
    }
    prompt_words = set()
    for word in re.findall(r'[a-zA-Z]+', prompt.lower()):
        if len(word) > 4 and word not in stop_words:
            prompt_words.add(word)

    if not prompt_words:
        return True  # can't check if prompt has no significant words

    thinking_lower = thinking.lower()
    matches = sum(1 for w in prompt_words if w in thinking_lower)
    coverage = matches / len(prompt_words) if prompt_words else 1.0

    # At least 30% of significant prompt words should appear
    return coverage >= 0.3


def check_not_templated(thinking: str) -> bool:
    """Check that reasoning isn't a generic template with no specifics.

    Flags responses that are too generic / boilerplate.
    """
    # Check for diversity of content — a good reasoning trace should have
    # specific numbers, colors, materials, etc.
    has_numbers = bool(re.search(r'\d{2,}', thinking))  # multi-digit numbers
    has_specifics = bool(re.search(
        r'(bbox|pixel|coord|position|depth|foreground|background|'
        r'lighting|shadow|texture|material|color|composition)',
        thinking.lower(),
    ))
    # At least one of these should be present
    return has_numbers or has_specifics


def _parse_sp_json(sp_str: str) -> Optional[dict]:
    """Try to parse a structured prompt JSON string."""
    if not sp_str:
        return None
    try:
        import json_repair
        obj = json_repair.loads(sp_str)
    except Exception:
        try:
            obj = json.loads(sp_str)
        except (json.JSONDecodeError, TypeError):
            return None
    return obj if isinstance(obj, dict) else None


def check_forward_json(generated_sp: Optional[str]) -> bool:
    """For forward mode: check that generated SP is valid JSON with key fields."""
    obj = _parse_sp_json(generated_sp)
    if obj is None:
        return False
    required = {"intent", "elements"}
    return required.issubset(obj.keys())


def _extract_element_nouns(sp_obj: dict) -> set[str]:
    """Extract key noun phrases from SP elements' captions.

    Pulls out short lowercase tokens from element captions to build a
    rough bag-of-concepts for comparison. Not precise, but catches big
    mismatches (e.g. Gemini describes a beach scene but reference is a kitchen).
    """
    stop = {
        "with", "that", "this", "from", "their", "which", "about",
        "would", "there", "these", "other", "into", "more", "some",
        "been", "have", "will", "each", "make", "like", "than",
        "them", "then", "when", "were", "what", "your", "also",
        "very", "the", "and", "for", "are", "but", "not", "you",
        "all", "can", "its", "his", "her", "was", "one", "our",
        "has",
    }
    nouns = set()
    elements = sp_obj.get("elements", [])
    if isinstance(sp_obj.get("scene"), dict):
        elements = elements + sp_obj["scene"].get("elements", [])
    for elem in elements:
        caption = elem.get("caption", "")
        for word in re.findall(r'[a-zA-Z]+', caption.lower()):
            if len(word) > 3 and word not in stop:
                nouns.add(word)
    # Also grab intent
    intent = sp_obj.get("intent", "")
    for word in re.findall(r'[a-zA-Z]+', intent.lower()):
        if len(word) > 3 and word not in stop:
            nouns.add(word)
    return nouns


def check_forward_consistency(generated_sp: Optional[str], reference_sp: str) -> bool:
    """Check that forward-mode generated SP is reasonably consistent with reference.

    Compares element-level bag-of-concepts between generated and reference SP.
    If overlap is too low, the thinking was about a different scene and shouldn't
    be paired with the reference SP.

    Returns True if overlap >= 25% (generous threshold — we just want to catch
    completely divergent scenes, not require exact match).
    """
    gen_obj = _parse_sp_json(generated_sp)
    ref_obj = _parse_sp_json(reference_sp)

    if gen_obj is None or ref_obj is None:
        return False

    gen_nouns = _extract_element_nouns(gen_obj)
    ref_nouns = _extract_element_nouns(ref_obj)

    if not ref_nouns:
        return True  # can't compare

    overlap = len(gen_nouns & ref_nouns)
    coverage = overlap / len(ref_nouns)

    return coverage >= 0.25


def check_element_count(generated_sp: Optional[str], reference_sp: str,
                        tolerance: float = 0.5) -> bool:
    """Check that element counts are in the same ballpark.

    If reference has 8 elements and generated has 2, the scene structure
    is probably very different — the thinking won't match.
    """
    gen_obj = _parse_sp_json(generated_sp)
    ref_obj = _parse_sp_json(reference_sp)

    if gen_obj is None or ref_obj is None:
        return False

    gen_count = len(gen_obj.get("elements", []))
    ref_count = len(ref_obj.get("elements", []))

    if ref_count == 0:
        return True

    ratio = gen_count / ref_count
    # Allow 50% deviation: e.g. ref=8, gen must be in [4, 12]
    return (1 - tolerance) <= ratio <= (1 + tolerance)


def check_bbox_validity(sp_str: Optional[str]) -> bool:
    """Check that all bboxes in an SP are well-formed.

    Validates:
    - All coords in [0, 999]
    - x_min < x_max, y_min < y_max
    - No degenerate boxes (area > 0)
    - Main elements are not full-frame (only scene bg should be 0 0 999 999)
    """
    obj = _parse_sp_json(sp_str)
    if obj is None:
        return False

    elements = obj.get("elements", [])
    for elem in elements:
        pos = elem.get("position", "")
        m = re.search(r"<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>", str(pos))
        if not m:
            return False
        x1, y1, x2, y2 = (int(x) for x in m.groups())
        # Basic validity
        if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
            return False
        # No degenerate (too tiny) boxes for main elements
        area = (x2 - x1) * (y2 - y1)
        if area < 100:  # less than 0.01% of frame
            return False

    return len(elements) > 0  # must have at least one element


def check_bbox_iou(generated_sp: Optional[str], reference_sp: str,
                   min_avg_iou: float = 0.15) -> bool:
    """Check bbox overlap between generated and reference SP.

    Matches elements by caption word overlap, then computes IoU.
    Low avg IoU means the spatial layouts are very different.
    """
    gen_obj = _parse_sp_json(generated_sp)
    ref_obj = _parse_sp_json(reference_sp)
    if gen_obj is None or ref_obj is None:
        return False

    def _get_elems(obj):
        result = []
        for elem in obj.get("elements", []):
            pos = elem.get("position", "")
            m = re.search(r"<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>", str(pos))
            if m:
                words = set(re.findall(r"[a-z]+", elem.get("caption", "").lower()))
                result.append((tuple(int(x) for x in m.groups()), words))
        return result

    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    gen_elems = _get_elems(gen_obj)
    ref_elems = _get_elems(ref_obj)

    if not gen_elems or not ref_elems:
        return False

    ious = []
    for g_bbox, g_words in gen_elems:
        best_iou = 0
        best_overlap = 0
        for r_bbox, r_words in ref_elems:
            word_overlap = len(g_words & r_words)
            if word_overlap > best_overlap:
                best_overlap = word_overlap
                best_iou = _iou(g_bbox, r_bbox)
        if best_overlap >= 3:
            ious.append(best_iou)

    if not ious:
        return False

    return sum(ious) / len(ious) >= min_avg_iou


# ──────────────────────────────────────────────
# Main filter pipeline
# ──────────────────────────────────────────────

def filter_record(rec: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Apply all filters to a single record.

    Returns (passed, list_of_failed_filter_names).
    """
    filter_cfg = cfg.get("filter", {})
    min_len = filter_cfg.get("min_length", 200)
    max_len = filter_cfg.get("max_length", 15000)

    thinking = rec.get("thinking")
    if not thinking:
        return False, ["no_thinking"]

    # Strip boilerplate before all checks
    thinking = strip_boilerplate(thinking)
    rec["thinking"] = thinking

    failures = []

    if not check_length(thinking, min_len, max_len):
        failures.append("length")

    if rec.get("mode") == "backward":
        if not check_no_leakage(thinking):
            failures.append("leakage")

    if not check_coherence(thinking, rec.get("prompt", "")):
        failures.append("coherence")

    if not check_not_templated(thinking):
        failures.append("templated")

    if rec.get("mode") == "forward":
        gen_sp = rec.get("generated_sp")
        ref_sp = rec.get("reference_sp", "")
        if not check_forward_json(gen_sp):
            failures.append("invalid_json")
        elif not check_bbox_validity(gen_sp):
            failures.append("invalid_bbox")
        else:
            # JSON + bboxes valid — check consistency with reference
            if not check_forward_consistency(gen_sp, ref_sp):
                failures.append("scene_mismatch")
            if not check_element_count(gen_sp, ref_sp):
                failures.append("element_count")
            # Check spatial layout match — determines which SP to use
            bbox_match = check_bbox_iou(gen_sp, ref_sp, min_avg_iou=0.15)
            rec["_bbox_match"] = bbox_match
            if not bbox_match:
                # Bbox layout diverges from reference. The thinking is
                # consistent with generated_sp, not reference_sp.
                # Mark it so assemble uses generated_sp instead.
                rec["_use_generated_sp"] = True

    return len(failures) == 0, failures


def load_raw_results(output_dir: str) -> list[dict]:
    """Load all results from WAL files."""
    results = []
    wal_dir = os.path.join(output_dir, "_wal")
    if not os.path.isdir(wal_dir):
        return results

    for fname in sorted(os.listdir(wal_dir)):
        if not fname.endswith(".partial.jsonl"):
            continue
        fpath = os.path.join(wal_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    log.info("Loaded %d raw results from WAL", len(results))
    return results


def run_filter(cfg: dict):
    """Run the filter pipeline."""
    synth_cfg = cfg["synth"]
    output_dir = synth_cfg["output_dir"]

    results = load_raw_results(output_dir)
    if not results:
        log.warning("No results found in %s", output_dir)
        return

    # Filter
    passed = []
    stats = {"total": 0, "passed": 0, "failed": 0, "fail_reasons": {}}

    for rec in results:
        if not rec.get("success"):
            continue
        stats["total"] += 1

        ok, failures = filter_record(rec, synth_cfg)
        if ok:
            stats["passed"] += 1
            passed.append(rec)
        else:
            stats["failed"] += 1
            for f in failures:
                stats["fail_reasons"][f] = stats["fail_reasons"].get(f, 0) + 1

    # Write filtered results
    filtered_path = os.path.join(output_dir, "filtered.jsonl")
    with open(filtered_path, "w", encoding="utf-8") as f:
        for rec in passed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Filter results: %d/%d passed (%.1f%%)",
             stats["passed"], stats["total"],
             100 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0)
    log.info("Failure breakdown: %s", json.dumps(stats["fail_reasons"], indent=2))
    log.info("Filtered output: %s", filtered_path)

    # Write stats
    stats_path = os.path.join(output_dir, "filter_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Filter synthetic reasoning")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_filter(cfg)


if __name__ == "__main__":
    main()
