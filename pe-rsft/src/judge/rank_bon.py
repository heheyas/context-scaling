# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: Gemini pairwise + Bradley-Terry ranking daemon — file-queue consumer.

Polls rank_in/pending/ for ranking tasks emitted by Stage 2, runs all
C(N,2)×2 pairwise Gemini comparisons per index, fits a Bradley-Terry
model, and appends the result to a per-host-per-pid output file.

Usage:
    python -m src.judge.rank_bon --config configs/iter0.yaml

Legacy dependency:
    Pairwise comparison: legacy.gemini_bon.compare_pair
    Bradley-Terry MLE:   legacy.gemini_bon.bradley_terry_mle
    Key management:      legacy.gemini_bon.KeyPool, load_api_keys, load_system_prompt

    We do NOT use legacy rank_single_index because it hardcodes
    bradley_terry_mle(reg=1e-3). Instead, we port a thin wrapper that
    calls the same compare_pair + symmetrisation logic but passes
    reg=1e-2.

    # TODO: Gemini API traffic is ~80GB for iter-0 at 1024² images.
    # Downscale to 512² before base64 encoding to cut traffic 4×.
    # Don't implement until Stage 3 is validated end-to-end.
"""

import json
import logging
import os
import socket
import time
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from src.ops.file_queue import claim, claim_batch, release_to_done
from src.ops.config import load_config, resolve_paths

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Worker ID
# ──────────────────────────────────────────────

def make_worker_id() -> str:
    hostname = socket.gethostname().replace("_", "-")
    return f"{hostname}-{os.getpid()}"


# ──────────────────────────────────────────────
# Task parsing
# ──────────────────────────────────────────────

def parse_task(task_path: str) -> dict:
    """Parse a ranking task JSON file.

    Expected format (from Stage 2):
        {"index": 42, "query": "...", "root": "/mnt/shared/rft_iter0",
         "images": [{"image_idx": 0, "path": "images/42/0.png", "success": true}, ...]}

    Raises ValueError on malformed JSON or missing required fields.
    """
    with open(task_path, "r", encoding="utf-8") as f:
        task = json.loads(f.read())

    for key in ("index", "query", "root", "images"):
        if key not in task:
            raise ValueError(f"Task missing required field '{key}': {task_path}")
    if not isinstance(task["images"], list):
        raise ValueError(f"Task 'images' field is not a list: {task_path}")

    return task


def resolve_image_paths(task: dict) -> list[dict]:
    """Resolve relative image paths to absolute using task root.

    Returns list of sample dicts with absolute image_path, suitable for
    passing to compare_pair.
    """
    root = task["root"]
    samples = []
    for img in task["images"]:
        abs_path = os.path.join(root, img["path"])
        samples.append({
            "image_idx": img["image_idx"],
            "image_path": abs_path,
            "success": img.get("success", True),
        })
    return samples


def filter_valid_samples(samples: list[dict]) -> list[dict]:
    """Filter samples to those with success=True and an existing image file."""
    return [
        s for s in samples
        if s.get("success", True)
        and s.get("image_path")
        and os.path.exists(s["image_path"])
    ]


def compare_pair_with_retry(compare_pair_fn, key_pool, prompt, path_a, path_b,
                            max_retries=500, **kwargs):
    """Wrap compare_pair with outer retry loop.

    Legacy compare_pair has its own internal 8-retry loop, but returns
    {winner: None} on exhaustion. This wrapper retries until we get a
    valid winner, up to max_retries total outer attempts.
    """
    for attempt in range(max_retries):
        cmp = compare_pair_fn(key_pool, prompt, path_a, path_b, **kwargs)
        if cmp.get("winner") is not None:
            return cmp
        wait = min(1.01 ** attempt, 60)
        log.warning("compare_pair returned no winner (outer attempt %d/%d), "
                    "retrying in %.1fs...", attempt + 1, max_retries, wait)
        time.sleep(wait)
    log.error("compare_pair exhausted %d outer retries, returning None", max_retries)
    return {"winner": None, "score_a": None, "score_b": None, "y": None}


def score_single_with_retry(score_fn, key_pool, prompt, image_path,
                            max_retries=500, **kwargs):
    """Wrap score_single with outer retry loop.

    Returns {"score": float} on success, {"score": None} after exhaustion.
    """
    for attempt in range(max_retries):
        result = score_fn(key_pool, prompt, image_path, **kwargs)
        if result.get("score") is not None:
            return result
        wait = min(1.01 ** attempt, 60)
        log.warning("score_single returned no score (outer attempt %d/%d), "
                    "retrying in %.1fs...", attempt + 1, max_retries, wait)
        time.sleep(wait)
    log.error("score_single exhausted %d outer retries, returning None", max_retries)
    return {"score": None, "alignment_score": None, "structural_score": None,
            "aesthetic_score": None, "passed": None}


# ──────────────────────────────────────────────
# LLM trace lookup (for mix mode verification)
# ──────────────────────────────────────────────

def fetch_llm_record(llm_out_dir: str, index: int, image_idx: int,
                     shard_size: int) -> dict | None:
    """Fetch the full LLM record for a specific (index, image_idx) from shard files.

    Used by mix mode to retrieve the complete message trace (including
    thinking, tool calls) for Gemini verification of the best image.

    Args:
        llm_out_dir: Path to llm_out directory (contains pending/, done/).
        index: Query index.
        image_idx: Image variant index.
        shard_size: Shard size used during rollout (from config).

    Returns:
        Full record dict with 'messages', 'llm_raw_response', etc., or None.
    """
    shard_id = index // shard_size
    shard_name = f"shard_{shard_id:04d}.jsonl"

    # Search done/, pending/, and claimed/ (claimed files have a worker_id prefix).
    candidates = []
    for subdir in ("done", "pending", "claimed"):
        subpath = os.path.join(llm_out_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for fname in os.listdir(subpath):
            # Exact match or worker_id prefixed match (e.g. "host-pid_shard_0001.jsonl")
            if fname == shard_name or fname.endswith("_" + shard_name):
                candidates.append(os.path.join(subpath, fname))

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("index") == index and rec.get("image_idx") == image_idx:
                        return rec
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Error reading shard %s: %s", path, e)
    return None


# ──────────────────────────────────────────────
# Ranking logic (thin wrapper around legacy compare_pair + BT)
# ──────────────────────────────────────────────

def rank_index(
    idx: int,
    samples: list[dict],
    prompt: str,
    key_pool,
    gemini_model: str,
    gemini_system_prompt: str | None,
    gemini_input_template: str | None,
    gemini_workers: int,
    bt_reg: float = 1e-2,
    compare_pair_fn=None,
    bradley_terry_fn=None,
) -> dict:
    """Run pairwise comparisons + Bradley-Terry for one index.

    This is a thin wrapper around legacy compare_pair and bradley_terry_mle.
    We don't use legacy rank_single_index because it hardcodes reg=1e-3;
    this wrapper passes the configurable bt_reg (default 1e-2).

    Args:
        idx: Query index.
        samples: List of sample dicts with image_idx, image_path, success.
        prompt: Original query text.
        key_pool: Gemini API key pool (legacy KeyPool instance).
        gemini_model: Model name.
        gemini_system_prompt: System prompt text or None.
        gemini_input_template: User message template or None.
        gemini_workers: Max parallel comparison threads for this task.
        bt_reg: Bradley-Terry L2 regularization strength.
        compare_pair_fn: Override for compare_pair (for testing).
        bradley_terry_fn: Override for bradley_terry_mle (for testing).

    Returns:
        Result dict matching legacy schema (index, original_prompt,
        n_samples, n_valid, rankings, ratings, pairwise_results,
        best_image_idx, best_image_path).
    """
    if compare_pair_fn is None:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "legacy"))
        from gemini_bon import compare_pair
        compare_pair_fn = compare_pair
    if bradley_terry_fn is None:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "legacy"))
        from gemini_bon import bradley_terry_mle
        bradley_terry_fn = bradley_terry_mle

    valid_samples = filter_valid_samples(samples)

    if len(valid_samples) < 2:
        rankings = ([s["image_idx"] for s in valid_samples]
                    + [s["image_idx"] for s in samples if s not in valid_samples])
        return {
            "index": idx,
            "original_prompt": prompt,
            "n_samples": len(samples),
            "n_valid": len(valid_samples),
            "rankings": rankings,
            "ratings": {str(s["image_idx"]): 0.0 for s in samples},
            "pairwise_results": [],
            "best_image_idx": rankings[0] if rankings else 0,
            "best_image_path": valid_samples[0]["image_path"] if valid_samples else "",
        }

    n_valid = len(valid_samples)
    pairs = list(combinations(range(n_valid), 2))

    # Build comparison tasks: forward + reverse for position-bias correction.
    tasks = []
    for li, lj in pairs:
        sa, sb = valid_samples[li], valid_samples[lj]
        tasks.append((li, lj, "fwd", sa["image_path"], sb["image_path"]))
        tasks.append((li, lj, "rev", sb["image_path"], sa["image_path"]))

    # Run comparisons in parallel.
    raw_results = []
    actual_workers = min(gemini_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        def _compare(task):
            li, lj, direction, path_a, path_b = task
            cmp = compare_pair_with_retry(
                compare_pair_fn, key_pool, prompt, path_a, path_b,
                model=gemini_model,
                system_prompt=gemini_system_prompt,
                input_template=gemini_input_template,
            )
            return (li, lj, direction, cmp)

        futures = {pool.submit(_compare, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                raw_results.append(future.result())
            except Exception as e:
                task = futures[future]
                log.error("[Gemini %d] Comparison failed %s: %s", idx, task[:3], e)

    # Aggregate: symmetrise forward + reverse.
    pair_results = {}
    for li, lj, direction, cmp in raw_results:
        key = (li, lj)
        if key not in pair_results:
            pair_results[key] = {}
        pair_results[key][direction] = cmp

    pairwise_results = []
    comparisons = []

    for (li, lj), dirs in sorted(pair_results.items()):
        sa, sb = valid_samples[li], valid_samples[lj]
        cmp_fwd = dirs.get("fwd", {"winner": None, "score_a": None, "score_b": None, "y": None})
        cmp_rev = dirs.get("rev", {"winner": None, "score_a": None, "score_b": None, "y": None})

        y_fwd = cmp_fwd["y"] if cmp_fwd["y"] is not None else 0.5
        y_rev = cmp_rev["y"] if cmp_rev["y"] is not None else 0.5
        y_sym = (y_fwd + (1.0 - y_rev)) / 2.0

        if y_sym > 0.5:
            winner_sym = "A"
        elif y_sym < 0.5:
            winner_sym = "B"
        else:
            winner_sym = "TIE"

        comparisons.append((li, lj, y_sym))
        pr = {
            "a": sa["image_idx"], "b": sb["image_idx"],
            "winner": winner_sym, "y_sym": round(y_sym, 4),
            "fwd": {"winner": cmp_fwd["winner"] or "UNKNOWN"},
            "rev": {"winner": cmp_rev["winner"] or "UNKNOWN"},
        }
        if cmp_fwd.get("score_a") is not None:
            pr["fwd"]["score_a"] = cmp_fwd["score_a"]
            pr["fwd"]["score_b"] = cmp_fwd["score_b"]
        if cmp_rev.get("score_a") is not None:
            pr["rev"]["score_a"] = cmp_rev["score_a"]
            pr["rev"]["score_b"] = cmp_rev["score_b"]
        pairwise_results.append(pr)

    # Fit Bradley-Terry with the configured regularization.
    ratings = bradley_terry_fn(n_valid, comparisons, reg=bt_reg)
    rating_map = {}
    for i, s in enumerate(valid_samples):
        rating_map[s["image_idx"]] = float(ratings[i])
    for s in samples:
        if s["image_idx"] not in rating_map:
            rating_map[s["image_idx"]] = float("-inf")

    ranked = sorted(rating_map.items(), key=lambda x: -x[1])
    rankings = [img_idx for img_idx, _ in ranked]

    return {
        "index": idx,
        "original_prompt": prompt,
        "n_samples": len(samples),
        "n_valid": n_valid,
        "rankings": rankings,
        "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
        "pairwise_results": pairwise_results,
        "best_image_idx": rankings[0],
        "best_image_path": next(
            (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""
        ),
    }


# ──────────────────────────────────────────────
# Single-image scoring logic
# ──────────────────────────────────────────────

def score_index(
    idx: int,
    samples: list[dict],
    prompt: str,
    key_pool,
    model: str,
    system_prompt: str | None,
    input_template: str | None,
    workers: int,
    score_fn=None,
) -> dict:
    """Score each image independently for one index.

    Unlike rank_index (pairwise + BT), this calls the VLM once per image
    and uses the returned score directly as the rating.

    Returns result dict with the same schema as rank_index for downstream
    compatibility (rankings, ratings, best_image_idx, etc.).
    """
    valid_samples = filter_valid_samples(samples)

    if not valid_samples:
        rankings = [s["image_idx"] for s in samples]
        return {
            "index": idx,
            "original_prompt": prompt,
            "n_samples": len(samples),
            "n_valid": 0,
            "rankings": rankings,
            "ratings": {str(s["image_idx"]): 0.0 for s in samples},
            "pairwise_results": [],
            "scores": {},
            "best_image_idx": rankings[0] if rankings else 0,
            "best_image_path": "",
        }

    # Score each valid sample in parallel.
    score_results = {}  # image_idx -> full result dict
    actual_workers = min(workers, len(valid_samples))

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        def _score(sample):
            result = score_single_with_retry(
                score_fn, key_pool, prompt, sample["image_path"],
                model=model,
                system_prompt=system_prompt,
                input_template=input_template,
            )
            return sample["image_idx"], result

        futures = {pool.submit(_score, s): s for s in valid_samples}
        for future in as_completed(futures):
            try:
                img_idx, result = future.result()
                score_results[img_idx] = result
            except Exception as e:
                s = futures[future]
                log.error("[Score %d] Scoring failed img_%d: %s", idx, s["image_idx"], e)
                score_results[s["image_idx"]] = {"score": 0.0}

    # Build rating map (valid samples get their score, invalid get -inf).
    rating_map = {}
    for s in valid_samples:
        r = score_results.get(s["image_idx"], {})
        rating_map[s["image_idx"]] = r.get("score", 0.0) if r.get("score") is not None else 0.0
    for s in samples:
        if s["image_idx"] not in rating_map:
            rating_map[s["image_idx"]] = float("-inf")

    ranked = sorted(rating_map.items(), key=lambda x: -x[1])
    rankings = [img_idx for img_idx, _ in ranked]

    # Build per-image detail scores.
    detail_scores = {}
    for img_idx, r in score_results.items():
        entry = {"score": r.get("score")}
        if r.get("alignment_score") is not None:
            entry["alignment_score"] = r["alignment_score"]
        if r.get("structural_score") is not None:
            entry["structural_score"] = r["structural_score"]
        if r.get("aesthetic_score") is not None:
            entry["aesthetic_score"] = r["aesthetic_score"]
        if r.get("passed") is not None:
            entry["pass"] = r["passed"]
        detail_scores[str(img_idx)] = entry

    return {
        "index": idx,
        "original_prompt": prompt,
        "n_samples": len(samples),
        "n_valid": len(valid_samples),
        "rankings": rankings,
        "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
        "pairwise_results": [],
        "scores": detail_scores,
        "best_image_idx": rankings[0],
        "best_image_path": next(
            (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""
        ),
    }


# ──────────────────────────────────────────────
# Result I/O
# ──────────────────────────────────────────────

def append_result(result: dict, results_path: str) -> None:
    """Append a ranking result to the per-host-per-pid results file."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Task processing
# ──────────────────────────────────────────────

def process_task(
    task_path: str,
    rankings_dir: str,
    results_path: str,
    key_pool,
    gemini_model: str,
    gemini_system_prompt: str | None,
    gemini_input_template: str | None,
    gemini_workers: int,
    bt_reg: float,
    compare_pair_fn=None,
    bradley_terry_fn=None,
) -> dict:
    """Process a single ranking task.

    Returns the ranking result dict.
    """
    task = parse_task(task_path)
    samples = resolve_image_paths(task)

    result = rank_index(
        idx=task["index"],
        samples=samples,
        prompt=task["query"],
        key_pool=key_pool,
        gemini_model=gemini_model,
        gemini_system_prompt=gemini_system_prompt,
        gemini_input_template=gemini_input_template,
        gemini_workers=gemini_workers,
        bt_reg=bt_reg,
        compare_pair_fn=compare_pair_fn,
        bradley_terry_fn=bradley_terry_fn,
    )

    append_result(result, results_path)
    log.info(
        "[Rank %d] Done. n_valid=%d, best=img_%d, ratings=%s",
        task["index"], result["n_valid"], result["best_image_idx"],
        result["ratings"],
    )
    return result


def quarantine_task(task_path: str, quarantine_dir: str, error: str) -> None:
    """Move a malformed task to the quarantine directory.

    Strips any worker_id prefix from the claimed filename to restore the
    original task filename in quarantine.
    """
    os.makedirs(quarantine_dir, exist_ok=True)
    # Strip worker_id prefix (same convention as file_queue._strip_worker_prefix).
    claimed_name = os.path.basename(task_path)
    idx = claimed_name.find("_")
    original_name = claimed_name[idx + 1:] if idx >= 0 else claimed_name
    dst = os.path.join(quarantine_dir, original_name)
    shutil.move(task_path, dst)

    # Write an error sidecar file.
    err_path = dst + ".error"
    with open(err_path, "w") as f:
        f.write(error + "\n")
    log.error("Quarantined malformed task: %s -> %s (%s)", task_path, dst, error)


# ──────────────────────────────────────────────
# Main daemon loop
# ──────────────────────────────────────────────

_UNSET = object()


def run_ranker(
    cfg: dict,
    key_pool=_UNSET,
    compare_pair_fn=None,
    bradley_terry_fn=None,
    score_fn=None,
    single_pass: bool = False,
):
    """Run the Gemini ranking/scoring consumer loop.

    Supports two judge modes (gemini.judge_mode config):
      - "pairwise" (default): C(N,2)×2 pairwise comparisons + Bradley-Terry.
      - "single": Score each image independently (1 API call per image).

    Both modes use flat parallelism for maximum throughput.

    Args:
        cfg: Config dict (from load_config + resolve_paths).
        key_pool: Gemini KeyPool / SeedPool instance. If _UNSET, builds from config.
        compare_pair_fn: Override for compare_pair (pairwise mode, for testing).
        bradley_terry_fn: Override for bradley_terry_mle (pairwise mode, for testing).
        score_fn: Override for score_single (single mode, for testing).
        single_pass: If True, process available tasks and return.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "legacy"))

    gemini_cfg = cfg["gemini"]
    backend = gemini_cfg.get("backend", "gemini")
    judge_mode = gemini_cfg.get("judge_mode", "pairwise")

    # HPSv3 backend forces single mode (local model, no pairwise API).
    if backend == "hpsv3" and judge_mode != "single":
        log.info("HPSv3 backend only supports single mode, overriding judge_mode='%s' -> 'single'",
                 judge_mode)
        judge_mode = "single"

    if judge_mode not in ("pairwise", "single", "mix", "single_detail"):
        raise ValueError(f"Unknown judge_mode: {judge_mode!r} (expected 'pairwise', 'single', 'mix', or 'single_detail')")

    if judge_mode in ("pairwise", "mix") and bradley_terry_fn is None:
        from gemini_bon import bradley_terry_mle
        bradley_terry_fn = bradley_terry_mle

    if backend == "hpsv3":
        # ── HPSv3 local model backend ──
        # Runs on local GPU, no API keys needed.
        # Multi-GPU: set hpsv3_device="0,1,2,3" and num_workers >= n_gpus.
        hpsv3_device = gemini_cfg.get("hpsv3_device", "cuda")
        if score_fn is None:
            from src.judge.hpsv3_backend import HPSv3Scorer, _parse_devices
            score_fn = HPSv3Scorer(
                device=hpsv3_device,
                threshold=gemini_cfg.get("hpsv3_threshold"),
                checkpoint=gemini_cfg.get("hpsv3_checkpoint"),
            )
            # Default num_workers to number of GPUs (each GPU has its own lock).
            n_gpus = len(_parse_devices(hpsv3_device))
            if gemini_cfg.get("num_workers") is None:
                gemini_cfg["num_workers"] = n_gpus
        if key_pool is _UNSET:
            key_pool = None  # No API key pool needed
    elif backend == "seed":
        # ── Seed VLM backend ──
        image_max_size = gemini_cfg.get("image_max_size", 1024)
        import functools
        if judge_mode in ("pairwise", "mix") and compare_pair_fn is None:
            from gemini_bon import compare_pair_seed
            compare_pair_fn = functools.partial(compare_pair_seed,
                                                image_max_size=image_max_size)
        if judge_mode == "single" and score_fn is None:
            from gemini_bon import score_single_seed
            score_fn = functools.partial(score_single_seed,
                                         image_max_size=image_max_size)
        if key_pool is _UNSET:
            from gemini_bon import build_seed_pool
            key_pool = build_seed_pool(
                psm=gemini_cfg.get("seed_psm"),
                url=gemini_cfg.get("seed_url"),
            )
    else:
        # ── Gemini API backend (default) ──
        if judge_mode in ("pairwise", "mix") and compare_pair_fn is None:
            from gemini_bon import compare_pair
            compare_pair_fn = compare_pair
        if judge_mode == "single" and score_fn is None:
            from gemini_bon import score_single
            score_fn = score_single
        if key_pool is _UNSET:
            from gemini_bon import KeyPool, load_api_keys
            api_keys = load_api_keys(gemini_cfg.get("api_keys_path"))
            key_pool = KeyPool(api_keys)
            log.info("Key pool: %d key(s)", len(api_keys))

    gemini_model = gemini_cfg.get("model", "default" if backend == "seed" else "gemini-3-pro-preview-new")
    bt_reg = gemini_cfg.get("bt_reg", 1e-2)
    gemini_workers = gemini_cfg.get("num_workers", 32)
    poll_interval = gemini_cfg.get("poll_interval_sec", 10)
    batch_size = gemini_cfg.get("batch_size", 0)  # 0 = claim all available

    # Load system prompt.
    sys_prompt_path = gemini_cfg.get("system_prompt_path")
    if sys_prompt_path and os.path.exists(sys_prompt_path):
        with open(sys_prompt_path, "r", encoding="utf-8") as f:
            gemini_system_prompt = f.read()
    else:
        gemini_system_prompt = sys_prompt_path

    gemini_input_template = gemini_cfg.get("input_template")

    # Mix mode: load verification prompts for 4 dimensions (Phase 2).
    verify_prompts = {}  # {dimension: system_prompt_text}
    verify_fns = {}      # {dimension: callable}
    if judge_mode in ("mix", "single_detail"):
        from gemini_bon import verify_best_image, verify_thinking

        def _load_prompt(key, default=None):
            p = gemini_cfg.get(key)
            if p and os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            return p or default

        verify_prompts["thinking"] = _load_prompt("verify_thinking_prompt_path")
        verify_prompts["structure"] = _load_prompt("verify_structure_prompt_path")
        verify_prompts["alignment"] = _load_prompt("verify_alignment_prompt_path")
        verify_prompts["aesthetic"] = _load_prompt("verify_aesthetic_prompt_path")
        # Fallback: single verify prompt for all image dimensions
        fallback_sp = _load_prompt("verify_system_prompt_path")
        for dim in ("structure", "alignment", "aesthetic"):
            if verify_prompts[dim] is None:
                verify_prompts[dim] = fallback_sp

        verify_fns["thinking"] = verify_thinking       # text-only, no image
        verify_fns["structure"] = verify_best_image     # image + prompt
        verify_fns["alignment"] = verify_best_image     # image + prompt
        verify_fns["aesthetic"] = verify_best_image      # image + prompt

    rank_in = cfg["rank_in"]
    rankings_dir = cfg["rankings"]
    rank_pending = os.path.join(rank_in, "pending")
    rank_claimed = os.path.join(rank_in, "claimed")
    rank_done = os.path.join(rank_in, "done")
    quarantine_dir = os.path.join(rank_in, "quarantine")

    for d in (rank_pending, rank_claimed, rank_done, rankings_dir, quarantine_dir):
        os.makedirs(d, exist_ok=True)

    worker_id = make_worker_id()
    results_path = os.path.join(rankings_dir, f"results_{worker_id}.jsonl")

    log.info(
        "Ranker started (worker=%s, backend=%s, judge_mode=%s, model=%s, workers=%d)",
        worker_id, backend, judge_mode, gemini_model, gemini_workers,
    )

    # Default batch_size: claim a reasonable chunk, not all 20k at once.
    if batch_size <= 0:
        batch_size = 100

    while True:
        # ── Claim a batch of tasks (single listdir, much faster on HDFS) ──
        claimed_tasks = []  # list of (task_path, task_dict, samples)
        log.info("Claiming tasks (batch_size=%d) ...", batch_size)
        claimed_paths = claim_batch(rank_pending, rank_claimed, worker_id,
                                    max_count=batch_size)
        for task_path in claimed_paths:
            try:
                task = parse_task(task_path)
                samples = resolve_image_paths(task)
                claimed_tasks.append((task_path, task, samples))
            except (ValueError, json.JSONDecodeError) as e:
                quarantine_task(task_path, quarantine_dir, str(e))

        if not claimed_tasks:
            if single_pass:
                log.info("Single-pass mode: no more tasks, exiting")
                return
            time.sleep(poll_interval)
            continue

        log.info("Claimed %d task(s), building %s plan...",
                 len(claimed_tasks), judge_mode)

        if judge_mode == "single":
            _run_batch_single(
                claimed_tasks, key_pool, score_fn, gemini_model,
                gemini_system_prompt, gemini_input_template,
                gemini_workers, results_path, rank_done, quarantine_dir,
            )
        elif judge_mode == "mix":
            _run_batch_mix(
                claimed_tasks, key_pool, compare_pair_fn, bradley_terry_fn,
                verify_fns, verify_prompts, gemini_model,
                gemini_system_prompt, gemini_input_template,
                gemini_workers, bt_reg, results_path, rank_done, quarantine_dir,
                llm_out_dir=cfg.get("llm_out", os.path.join(cfg["shared_root"], "llm_out")),
                shard_size=cfg.get("shard_size", 64),
            )
        elif judge_mode == "single_detail":
            _run_batch_single_detail(
                claimed_tasks, key_pool, verify_fns, verify_prompts,
                gemini_model, gemini_workers,
                results_path, rank_done, quarantine_dir,
                llm_out_dir=cfg.get("llm_out", os.path.join(cfg["shared_root"], "llm_out")),
                shard_size=cfg.get("shard_size", 64),
            )
        else:
            _run_batch_pairwise(
                claimed_tasks, key_pool, compare_pair_fn, bradley_terry_fn,
                gemini_model, gemini_system_prompt, gemini_input_template,
                gemini_workers, bt_reg, results_path, rank_done, quarantine_dir,
            )

        if single_pass:
            continue


def _run_batch_single(
    claimed_tasks, key_pool, score_fn, model,
    system_prompt, input_template, workers,
    results_path, rank_done, quarantine_dir,
):
    """Process a batch of tasks in single-image scoring mode.

    Flat parallelism: all images across all indices are flattened into
    one ThreadPoolExecutor. Results are aggregated per-index as they
    complete (streaming aggregation).
    """
    from collections import defaultdict
    import threading

    index_meta = {}   # idx -> {prompt, samples, valid_samples, n_valid, expected, task_path}
    all_score_tasks = []  # (idx, image_idx, image_path)
    trivial_results = []

    for task_path, task, samples in claimed_tasks:
        idx = task["index"]
        prompt = task["query"]
        valid_samples = filter_valid_samples(samples)

        if not valid_samples:
            rankings = [s["image_idx"] for s in samples]
            trivial_results.append((task_path, {
                "index": idx, "original_prompt": prompt,
                "n_samples": len(samples), "n_valid": 0,
                "rankings": rankings,
                "ratings": {str(s["image_idx"]): 0.0 for s in samples},
                "pairwise_results": [], "scores": {},
                "best_image_idx": rankings[0] if rankings else 0,
                "best_image_path": "",
            }))
            continue

        index_meta[idx] = {
            "prompt": prompt, "samples": samples,
            "valid_samples": valid_samples, "n_valid": len(valid_samples),
            "expected": len(valid_samples), "task_path": task_path,
        }

        for s in valid_samples:
            all_score_tasks.append((idx, s["image_idx"], s["image_path"]))

    # Write trivial results immediately.
    for task_path, result in trivial_results:
        append_result(result, results_path)
        release_to_done(task_path, rank_done)
        log.info("[Score %d] Trivial (n_valid=%d)", result["index"], result["n_valid"])

    if not all_score_tasks:
        log.info("No scoring tasks needed for this batch.")
        return

    log.info(
        "Running %d scoring calls across %d indices (%d workers)...",
        len(all_score_tasks), len(index_meta), workers,
    )

    acc_lock = threading.Lock()
    acc = defaultdict(dict)        # idx -> {image_idx: full_result_dict}
    acc_count = defaultdict(int)   # idx -> count of completed scores

    def _run_one(score_task):
        idx, image_idx, image_path = score_task
        prompt = index_meta[idx]["prompt"]
        result = score_single_with_retry(
            score_fn, key_pool, prompt, image_path,
            model=model,
            system_prompt=system_prompt,
            input_template=input_template,
        )
        return (idx, image_idx, result)

    def _on_index_complete(idx):
        """Aggregate scores for one index and write output."""
        meta = index_meta[idx]
        valid_samples = meta["valid_samples"]
        samples = meta["samples"]
        score_results = acc[idx]  # {image_idx: full_result_dict}

        rating_map = {}
        for s in valid_samples:
            r = score_results.get(s["image_idx"], {})
            rating_map[s["image_idx"]] = r.get("score", 0.0) if r.get("score") is not None else 0.0
        for s in samples:
            if s["image_idx"] not in rating_map:
                rating_map[s["image_idx"]] = float("-inf")

        ranked = sorted(rating_map.items(), key=lambda x: -x[1])
        rankings = [img_idx for img_idx, _ in ranked]

        # Build per-image detail scores for output.
        detail_scores = {}
        for img_idx, r in score_results.items():
            entry = {"score": r.get("score")}
            if r.get("alignment_score") is not None:
                entry["alignment_score"] = r["alignment_score"]
            if r.get("structural_score") is not None:
                entry["structural_score"] = r["structural_score"]
            if r.get("aesthetic_score") is not None:
                entry["aesthetic_score"] = r["aesthetic_score"]
            if r.get("passed") is not None:
                entry["pass"] = r["passed"]
            detail_scores[str(img_idx)] = entry

        result = {
            "index": idx, "original_prompt": meta["prompt"],
            "n_samples": len(samples), "n_valid": meta["n_valid"],
            "rankings": rankings,
            "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
            "pairwise_results": [],
            "scores": detail_scores,
            "best_image_idx": rankings[0],
            "best_image_path": next(
                (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""),
        }
        append_result(result, results_path)
        release_to_done(meta["task_path"], rank_done)
        del acc[idx]
        del acc_count[idx]

        # Summary log: show total scores and pass/fail.
        score_summary = {str(k): round(r.get("score", 0) or 0, 1) for k, r in score_results.items()}
        pass_summary = {str(k): r.get("passed") for k, r in score_results.items()}
        log.info("[Score %d] Done. scores=%s pass=%s best=img_%d",
                 idx, score_summary, pass_summary, result["best_image_idx"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in all_score_tasks}
        for future in as_completed(futures):
            try:
                idx, image_idx, result = future.result()
                with acc_lock:
                    acc[idx][image_idx] = result
                    acc_count[idx] += 1
                    if acc_count[idx] == index_meta[idx]["expected"]:
                        _on_index_complete(idx)
            except Exception as e:
                score_task = futures[future]
                log.error("Scoring failed %s: %s", score_task[:3], e)


def _run_batch_pairwise(
    claimed_tasks, key_pool, compare_pair_fn, bradley_terry_fn,
    model, system_prompt, input_template,
    workers, bt_reg, results_path, rank_done, quarantine_dir,
):
    """Process a batch of tasks in pairwise comparison mode.

    Flat parallelism: all C(N,2)×2 comparisons across all indices are
    flattened into one ThreadPoolExecutor with streaming aggregation.
    """
    from collections import defaultdict
    import threading

    index_meta = {}   # idx -> {prompt, samples, valid_samples, n_valid, expected, task_path}
    all_comparisons = []  # (idx, li, lj, direction, path_a, path_b)
    trivial_results = []

    for task_path, task, samples in claimed_tasks:
        idx = task["index"]
        prompt = task["query"]
        valid_samples = filter_valid_samples(samples)

        if len(valid_samples) < 2:
            # Trivial: no comparisons needed.
            rankings = ([s["image_idx"] for s in valid_samples]
                        + [s["image_idx"] for s in samples if s not in valid_samples])
            trivial_results.append((task_path, {
                "index": idx, "original_prompt": prompt,
                "n_samples": len(samples), "n_valid": len(valid_samples),
                "rankings": rankings,
                "ratings": {str(s["image_idx"]): 0.0 for s in samples},
                "pairwise_results": [],
                "best_image_idx": rankings[0] if rankings else 0,
                "best_image_path": valid_samples[0]["image_path"] if valid_samples else "",
            }))
            continue

        n_valid = len(valid_samples)
        pairs = list(combinations(range(n_valid), 2))
        index_meta[idx] = {
            "prompt": prompt, "samples": samples,
            "valid_samples": valid_samples, "n_valid": n_valid,
            "expected": len(pairs) * 2, "task_path": task_path,
        }

        for li, lj in pairs:
            sa, sb = valid_samples[li], valid_samples[lj]
            all_comparisons.append((idx, li, lj, "fwd", sa["image_path"], sb["image_path"]))
            all_comparisons.append((idx, li, lj, "rev", sb["image_path"], sa["image_path"]))

    # Write trivial results immediately.
    for task_path, result in trivial_results:
        append_result(result, results_path)
        release_to_done(task_path, rank_done)
        log.info("[Rank %d] Trivial (n_valid=%d)", result["index"], result["n_valid"])

    if not all_comparisons:
        log.info("No comparisons needed for this batch.")
        return

    log.info(
        "Running %d API calls across %d indices (%d workers)...",
        len(all_comparisons), len(index_meta), workers,
    )

    acc_lock = threading.Lock()
    acc = defaultdict(list)       # idx -> [(li, lj, direction, cmp)]
    acc_count = defaultdict(int)  # idx -> count of completed comparisons

    def _run_one(comp_task):
        idx, li, lj, direction, path_a, path_b = comp_task
        prompt = index_meta[idx]["prompt"]
        cmp = compare_pair_with_retry(
            compare_pair_fn, key_pool, prompt, path_a, path_b,
            model=model,
            system_prompt=system_prompt,
            input_template=input_template,
        )
        return (idx, li, lj, direction, cmp)

    def _on_index_complete(idx):
        """Aggregate results for one index and write output."""
        meta = index_meta[idx]
        valid_samples = meta["valid_samples"]
        samples = meta["samples"]
        n_valid = meta["n_valid"]
        raw_results = acc[idx]

        # Symmetrise (same as rank_index).
        pair_results = {}
        for li, lj, direction, cmp in raw_results:
            key = (li, lj)
            if key not in pair_results:
                pair_results[key] = {}
            pair_results[key][direction] = cmp

        pairwise_results = []
        comparisons = []
        for (li, lj), dirs in sorted(pair_results.items()):
            sa, sb = valid_samples[li], valid_samples[lj]
            cmp_fwd = dirs.get("fwd", {"winner": None, "score_a": None, "score_b": None, "y": None})
            cmp_rev = dirs.get("rev", {"winner": None, "score_a": None, "score_b": None, "y": None})
            y_fwd = cmp_fwd["y"] if cmp_fwd["y"] is not None else 0.5
            y_rev = cmp_rev["y"] if cmp_rev["y"] is not None else 0.5
            y_sym = (y_fwd + (1.0 - y_rev)) / 2.0
            winner_sym = "A" if y_sym > 0.5 else ("B" if y_sym < 0.5 else "TIE")
            comparisons.append((li, lj, y_sym))
            pr = {"a": sa["image_idx"], "b": sb["image_idx"],
                  "winner": winner_sym, "y_sym": round(y_sym, 4),
                  "fwd": {"winner": cmp_fwd["winner"] or "UNKNOWN"},
                  "rev": {"winner": cmp_rev["winner"] or "UNKNOWN"}}
            if cmp_fwd.get("score_a") is not None:
                pr["fwd"]["score_a"] = cmp_fwd["score_a"]
                pr["fwd"]["score_b"] = cmp_fwd["score_b"]
            if cmp_rev.get("score_a") is not None:
                pr["rev"]["score_a"] = cmp_rev["score_a"]
                pr["rev"]["score_b"] = cmp_rev["score_b"]
            pairwise_results.append(pr)

        ratings = bradley_terry_fn(n_valid, comparisons, reg=bt_reg)
        rating_map = {}
        for i, s in enumerate(valid_samples):
            rating_map[s["image_idx"]] = float(ratings[i])
        for s in samples:
            if s["image_idx"] not in rating_map:
                rating_map[s["image_idx"]] = float("-inf")

        ranked = sorted(rating_map.items(), key=lambda x: -x[1])
        rankings = [img_idx for img_idx, _ in ranked]

        result = {
            "index": idx, "original_prompt": meta["prompt"],
            "n_samples": len(samples), "n_valid": n_valid,
            "rankings": rankings,
            "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
            "pairwise_results": pairwise_results,
            "best_image_idx": rankings[0],
            "best_image_path": next(
                (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""),
        }
        append_result(result, results_path)
        release_to_done(meta["task_path"], rank_done)
        del acc[idx]
        del acc_count[idx]
        log.info("[Rank %d] Done. best=img_%d", idx, result["best_image_idx"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in all_comparisons}
        for future in as_completed(futures):
            try:
                idx, li, lj, direction, cmp = future.result()
                with acc_lock:
                    acc[idx].append((li, lj, direction, cmp))
                    acc_count[idx] += 1
                    if acc_count[idx] == index_meta[idx]["expected"]:
                        _on_index_complete(idx)
            except Exception as e:
                comp_task = futures[future]
                log.error("Comparison failed %s: %s", comp_task[:4], e)


def _run_batch_mix(
    claimed_tasks, key_pool, compare_pair_fn, bradley_terry_fn,
    verify_fns, verify_prompts, model,
    pairwise_system_prompt, pairwise_input_template,
    workers, bt_reg, results_path, rank_done, quarantine_dir,
    llm_out_dir, shard_size,
):
    """Process a batch of tasks in mix mode: pairwise ranking + verification.

    Phase 1: Gemini pairwise + Bradley-Terry to select best image per index.
    Phase 2: For each best image, fetch LLM trace and run Gemini verification.

    This gives the best of both worlds: pairwise comparison for selection,
    plus deep chain-of-thought inspection for training data quality.
    """
    from collections import defaultdict
    import threading
    from itertools import combinations

    index_meta = {}   # idx -> {prompt, samples, valid_samples, n_valid, expected, task_path}
    all_comparisons = []
    trivial_results = []

    for task_path, task, samples in claimed_tasks:
        idx = task["index"]
        prompt = task["query"]
        valid_samples = filter_valid_samples(samples)

        if len(valid_samples) < 2:
            rankings = ([s["image_idx"] for s in valid_samples]
                        + [s["image_idx"] for s in samples if s not in valid_samples])
            trivial_results.append((task_path, task, {
                "index": idx, "original_prompt": prompt,
                "n_samples": len(samples), "n_valid": len(valid_samples),
                "rankings": rankings,
                "ratings": {str(s["image_idx"]): 0.0 for s in samples},
                "pairwise_results": [],
                "best_image_idx": rankings[0] if rankings else 0,
                "best_image_path": valid_samples[0]["image_path"] if valid_samples else "",
            }))
            continue

        n_valid = len(valid_samples)
        pairs = list(combinations(range(n_valid), 2))
        index_meta[idx] = {
            "prompt": prompt, "samples": samples,
            "valid_samples": valid_samples, "n_valid": n_valid,
            "expected": len(pairs) * 2, "task_path": task_path,
        }

        for li, lj in pairs:
            sa, sb = valid_samples[li], valid_samples[lj]
            all_comparisons.append((idx, li, lj, "fwd", sa["image_path"], sb["image_path"]))
            all_comparisons.append((idx, li, lj, "rev", sb["image_path"], sa["image_path"]))

    # ── Unified pipeline: Phase 1 (pairwise) + Phase 2 (verify) in one pool ──
    # When an index's pairwise comparisons complete, its verification
    # task is immediately submitted to the same pool — no waiting for
    # all indices to finish Phase 1 first.

    def _call_verify_with_retry(fn, kwargs, idx, dim, max_retries=500):
        """Call a verify function with outer retry loop."""
        for attempt in range(max_retries):
            v = fn(**kwargs)
            if v.get("score") is not None:
                return v
            wait = min(1.01 ** attempt, 60)
            log.warning("[Mix %d] %s verify returned no score (attempt %d), retrying...",
                        idx, dim, attempt + 1)
            time.sleep(wait)
        return {"score": None, "pass": None, "issues": []}

    def _verify_one(idx, best_img_idx, best_img_path, prompt, task_path, result):
        """Phase 2: run 4 parallel verifications on best image."""
        llm_record = fetch_llm_record(llm_out_dir, idx, best_img_idx, shard_size)
        messages = llm_record.get("messages") if llm_record else None

        verification = {}
        threads = []
        thread_results = {}
        result_lock = threading.Lock()

        def _run_dim(dim, fn, kwargs):
            v = _call_verify_with_retry(fn, kwargs, idx, dim)
            with result_lock:
                thread_results[dim] = v

        # 1. Thinking (text-only, no image) — only if messages available
        if messages:
            threads.append(threading.Thread(target=_run_dim, args=(
                "thinking", verify_fns["thinking"], {
                    "key_pool": key_pool, "prompt": prompt, "messages": messages,
                    "model": model, "system_prompt": verify_prompts.get("thinking"),
                },
            )))
        else:
            log.warning("[Mix %d] No LLM messages for img_%d, skipping thinking verify",
                        idx, best_img_idx)
            thread_results["thinking"] = {"score": None, "pass": None, "issues": ["no messages found"]}

        # 2. Structure (image + prompt, strict)
        threads.append(threading.Thread(target=_run_dim, args=(
            "structure", verify_fns["structure"], {
                "key_pool": key_pool, "prompt": prompt,
                "image_path": best_img_path, "messages": messages or [],
                "model": model, "system_prompt": verify_prompts.get("structure"),
            },
        )))

        # 3. Alignment (image + prompt)
        threads.append(threading.Thread(target=_run_dim, args=(
            "alignment", verify_fns["alignment"], {
                "key_pool": key_pool, "prompt": prompt,
                "image_path": best_img_path, "messages": messages or [],
                "model": model, "system_prompt": verify_prompts.get("alignment"),
            },
        )))

        # 4. Aesthetic (image only)
        threads.append(threading.Thread(target=_run_dim, args=(
            "aesthetic", verify_fns["aesthetic"], {
                "key_pool": key_pool, "prompt": prompt,
                "image_path": best_img_path, "messages": messages or [],
                "model": model, "system_prompt": verify_prompts.get("aesthetic"),
            },
        )))

        # Run all in parallel.
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Aggregate.
        verification = thread_results
        all_pass = all(
            v.get("pass") is True
            for v in verification.values()
            if v.get("score") is not None
        )
        scores = [v["score"] for v in verification.values() if v.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 4) if scores else None
        verification["overall_score"] = avg_score
        verification["overall_pass"] = all_pass
        result["verification"] = verification

        append_result(result, results_path)
        release_to_done(task_path, rank_done)
        dim_summary = {d: f'{v.get("score")}/{v.get("pass")}' for d, v in verification.items()
                       if d not in ("overall_score", "overall_pass")}
        log.info("[Mix %d] Done. best=img_%d, %s, overall=%s/%s",
                 idx, result["best_image_idx"], dim_summary, avg_score, all_pass)

    # Handle trivial cases first (n_valid < 2): submit verification directly.
    trivial_verify_items = []
    for task_path, task, result in trivial_results:
        idx = result["index"]
        if result["n_valid"] >= 1:
            trivial_verify_items.append(
                (idx, result["best_image_idx"], result["best_image_path"],
                 result["original_prompt"], task_path, result))
        else:
            result["verification"] = None
            append_result(result, results_path)
            release_to_done(task_path, rank_done)
            log.info("[Mix %d] Trivial (n_valid=0)", idx)

    if not all_comparisons and not trivial_verify_items:
        log.info("No tasks to process in this batch.")
        return

    total_pairwise = len(all_comparisons)
    total_indices = len(index_meta) + len(trivial_verify_items)
    log.info(
        "Running mix mode: %d pairwise calls + %d verifications across %d indices (%d workers)...",
        total_pairwise, total_indices, total_indices, workers,
    )

    acc_lock = threading.Lock()
    acc = defaultdict(list)
    acc_count = defaultdict(int)

    # Two separate pools: compare pool for Phase 1, verify threads for Phase 2.
    # This ensures verify tasks start IMMEDIATELY when an index's Phase 1
    # completes, instead of queuing behind remaining compare tasks.
    verify_threads = []

    def _on_index_complete(idx):
        """Aggregate pairwise → BT → spawn verify thread immediately."""
        meta = index_meta[idx]
        valid_samples = meta["valid_samples"]
        samples = meta["samples"]
        n_valid = meta["n_valid"]
        raw_results = acc[idx]

        pair_results = {}
        for li, lj, direction, cmp in raw_results:
            key = (li, lj)
            if key not in pair_results:
                pair_results[key] = {}
            pair_results[key][direction] = cmp

        pairwise_results = []
        comparisons_list = []
        for (li, lj), dirs in sorted(pair_results.items()):
            sa, sb = valid_samples[li], valid_samples[lj]
            cmp_fwd = dirs.get("fwd", {"winner": None, "score_a": None, "score_b": None, "y": None})
            cmp_rev = dirs.get("rev", {"winner": None, "score_a": None, "score_b": None, "y": None})
            y_fwd = cmp_fwd["y"] if cmp_fwd["y"] is not None else 0.5
            y_rev = cmp_rev["y"] if cmp_rev["y"] is not None else 0.5
            y_sym = (y_fwd + (1.0 - y_rev)) / 2.0
            winner_sym = "A" if y_sym > 0.5 else ("B" if y_sym < 0.5 else "TIE")
            comparisons_list.append((li, lj, y_sym))
            pr = {"a": sa["image_idx"], "b": sb["image_idx"],
                  "winner": winner_sym, "y_sym": round(y_sym, 4),
                  "fwd": {"winner": cmp_fwd["winner"] or "UNKNOWN"},
                  "rev": {"winner": cmp_rev["winner"] or "UNKNOWN"}}
            if cmp_fwd.get("score_a") is not None:
                pr["fwd"]["score_a"] = cmp_fwd["score_a"]
                pr["fwd"]["score_b"] = cmp_fwd["score_b"]
            if cmp_rev.get("score_a") is not None:
                pr["rev"]["score_a"] = cmp_rev["score_a"]
                pr["rev"]["score_b"] = cmp_rev["score_b"]
            pairwise_results.append(pr)

        ratings = bradley_terry_fn(n_valid, comparisons_list, reg=bt_reg)
        rating_map = {}
        for i, s in enumerate(valid_samples):
            rating_map[s["image_idx"]] = float(ratings[i])
        for s in samples:
            if s["image_idx"] not in rating_map:
                rating_map[s["image_idx"]] = float("-inf")

        ranked = sorted(rating_map.items(), key=lambda x: -x[1])
        rankings = [img_idx for img_idx, _ in ranked]

        result = {
            "index": idx, "original_prompt": meta["prompt"],
            "n_samples": len(samples), "n_valid": n_valid,
            "rankings": rankings,
            "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
            "pairwise_results": pairwise_results,
            "best_image_idx": rankings[0],
            "best_image_path": next(
                (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""),
        }

        del acc[idx]
        del acc_count[idx]
        log.info("[Mix %d] Phase 1 done. best=img_%d → starting verification",
                 idx, result["best_image_idx"])

        # Start Phase 2 in a NEW thread — not queued in compare pool.
        t = threading.Thread(
            target=_verify_one,
            args=(idx, result["best_image_idx"], result["best_image_path"],
                  meta["prompt"], meta["task_path"], result),
            daemon=True,
        )
        t.start()
        verify_threads.append(t)

    # Phase 1: run all pairwise comparisons.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(
            compare_pair_with_retry, compare_pair_fn, key_pool,
            index_meta[t[0]]["prompt"], t[4], t[5],
            model=model, system_prompt=pairwise_system_prompt,
            input_template=pairwise_input_template,
        ): t for t in all_comparisons}

        for future in as_completed(futures):
            comp_task = futures[future]
            idx, li, lj, direction = comp_task[0], comp_task[1], comp_task[2], comp_task[3]
            try:
                cmp = future.result()
                with acc_lock:
                    acc[idx].append((li, lj, direction, cmp))
                    acc_count[idx] += 1
                    if acc_count[idx] == index_meta[idx]["expected"]:
                        _on_index_complete(idx)
            except Exception as e:
                log.error("Comparison failed idx=%d %s: %s", idx, direction, e)

    # Submit trivial verifications as threads too.
    for item in trivial_verify_items:
        t = threading.Thread(target=_verify_one, args=item, daemon=True)
        t.start()
        verify_threads.append(t)

    # Wait for all verify threads to finish.
    for t in verify_threads:
        t.join()
    log.info("All %d verifications complete.", len(verify_threads))


def _run_batch_single_detail(
    claimed_tasks, key_pool, verify_fns, verify_prompts,
    model, workers,
    results_path, rank_done, quarantine_dir,
    llm_out_dir, shard_size,
):
    """Score each image independently on 4 dimensions (no pairwise).

    For each image: thinking + structure + alignment + aesthetic
    evaluated in parallel. Rankings are by overall_score.
    """
    from collections import defaultdict
    import threading

    index_meta = {}
    all_score_tasks = []  # (idx, image_idx, image_path)
    trivial_results = []

    for task_path, task, samples in claimed_tasks:
        idx = task["index"]
        prompt = task["query"]
        valid_samples = filter_valid_samples(samples)

        if not valid_samples:
            rankings = [s["image_idx"] for s in samples]
            trivial_results.append((task_path, {
                "index": idx, "original_prompt": prompt,
                "n_samples": len(samples), "n_valid": 0,
                "rankings": rankings,
                "ratings": {str(s["image_idx"]): 0.0 for s in samples},
                "pairwise_results": [], "scores": {},
                "best_image_idx": rankings[0] if rankings else 0,
                "best_image_path": "",
            }))
            continue

        index_meta[idx] = {
            "prompt": prompt, "samples": samples,
            "valid_samples": valid_samples, "n_valid": len(valid_samples),
            "expected": len(valid_samples), "task_path": task_path,
        }
        for s in valid_samples:
            all_score_tasks.append((idx, s["image_idx"], s["image_path"]))

    # Write trivial results.
    for task_path, result in trivial_results:
        append_result(result, results_path)
        release_to_done(task_path, rank_done)
        log.info("[SingleDetail %d] Trivial (n_valid=0)", result["index"])

    if not all_score_tasks:
        log.info("No scoring tasks needed for this batch.")
        return

    log.info(
        "Running single_detail: %d images × 4 dimensions across %d indices (%d workers)...",
        len(all_score_tasks), len(index_meta), workers,
    )

    def _call_with_retry(fn, kwargs, idx, img_idx, dim, max_retries=500):
        for attempt in range(max_retries):
            v = fn(**kwargs)
            if v.get("score") is not None:
                return v
            wait = min(1.01 ** attempt, 60)
            log.warning("[SingleDetail %d/img_%d] %s returned no score (attempt %d), retrying...",
                        idx, img_idx, dim, attempt + 1)
            time.sleep(wait)
        return {"score": None, "pass": None, "issues": []}

    def _score_one_image(idx, image_idx, image_path):
        """Run 4 dimension evaluations on one image in parallel threads."""
        prompt = index_meta[idx]["prompt"]

        # Fetch LLM messages for thinking eval.
        llm_record = fetch_llm_record(llm_out_dir, idx, image_idx, shard_size)
        messages = llm_record.get("messages") if llm_record else None

        dim_results = {}
        dim_lock = threading.Lock()

        def _run_dim(dim, fn, kwargs):
            v = _call_with_retry(fn, kwargs, idx, image_idx, dim)
            with dim_lock:
                dim_results[dim] = v

        threads = []

        # Thinking (text-only).
        if messages:
            threads.append(threading.Thread(target=_run_dim, args=(
                "thinking", verify_fns["thinking"], {
                    "key_pool": key_pool, "prompt": prompt, "messages": messages,
                    "model": model, "system_prompt": verify_prompts.get("thinking"),
                },
            )))
        else:
            dim_results["thinking"] = {"score": None, "pass": None, "issues": ["no messages found"]}

        # Structure, alignment, aesthetic (all need image).
        for dim in ("structure", "alignment", "aesthetic"):
            threads.append(threading.Thread(target=_run_dim, args=(
                dim, verify_fns[dim], {
                    "key_pool": key_pool, "prompt": prompt,
                    "image_path": image_path, "messages": messages or [],
                    "model": model, "system_prompt": verify_prompts.get(dim),
                },
            )))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Compute overall.
        scores = [v["score"] for v in dim_results.values() if v.get("score") is not None]
        overall_score = round(sum(scores) / len(scores), 4) if scores else None
        overall_pass = all(
            v.get("pass") is True for v in dim_results.values()
            if v.get("score") is not None
        )
        dim_results["overall_score"] = overall_score
        dim_results["overall_pass"] = overall_pass

        return (idx, image_idx, dim_results)

    # Streaming aggregation: track per-index completion.
    acc_lock = threading.Lock()
    acc = defaultdict(dict)      # idx -> {image_idx: dim_results}
    acc_count = defaultdict(int)

    def _on_image_complete(idx, image_idx, dim_results):
        """Called when one image's 4-dim scoring is done."""
        acc[idx][image_idx] = dim_results
        acc_count[idx] += 1
        if acc_count[idx] == index_meta[idx]["expected"]:
            _on_index_complete(idx)

    def _on_index_complete(idx):
        """All images for this index are scored. Write result."""
        meta = index_meta[idx]
        samples = meta["samples"]
        score_results = acc[idx]

        # Build rating map from overall_score.
        rating_map = {}
        for s in meta["valid_samples"]:
            r = score_results.get(s["image_idx"], {})
            rating_map[s["image_idx"]] = r.get("overall_score", 0.0) or 0.0
        for s in samples:
            if s["image_idx"] not in rating_map:
                rating_map[s["image_idx"]] = float("-inf")

        ranked = sorted(rating_map.items(), key=lambda x: -x[1])
        rankings = [img_idx for img_idx, _ in ranked]

        # Build per-image detail scores.
        detail_scores = {}
        for img_idx, dims in score_results.items():
            detail_scores[str(img_idx)] = dims

        result = {
            "index": idx, "original_prompt": meta["prompt"],
            "n_samples": len(samples), "n_valid": meta["n_valid"],
            "rankings": rankings,
            "ratings": {str(k): round(v, 4) for k, v in rating_map.items()},
            "pairwise_results": [],
            "scores": detail_scores,
            "best_image_idx": rankings[0],
            "best_image_path": next(
                (s["image_path"] for s in samples if s["image_idx"] == rankings[0]), ""),
        }
        append_result(result, results_path)
        release_to_done(meta["task_path"], rank_done)
        del acc[idx]
        del acc_count[idx]

        summary = {str(img): f'{d.get("overall_score")}/{d.get("overall_pass")}'
                   for img, d in score_results.items()}
        log.info("[SingleDetail %d] Done. %s best=img_%d",
                 idx, summary, result["best_image_idx"])

    # Run all images through the thread pool.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task in all_score_tasks:
            f = pool.submit(_score_one_image, *task)
            futures[f] = task

        for future in as_completed(futures):
            try:
                idx, image_idx, dim_results = future.result()
                with acc_lock:
                    _on_image_complete(idx, image_idx, dim_results)
            except Exception as e:
                task = futures[future]
                log.error("SingleDetail scoring failed %s: %s", task, e)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def _apply_rank_suffix(cfg: dict, suffix: str) -> dict:
    """Append a suffix to rank_in and rankings paths for parallel experiments.

    E.g. suffix="seed" -> rank_in_seed/, rankings_seed/
    """
    root = cfg["shared_root"]
    cfg["rank_in"] = os.path.join(root, f"rank_in_{suffix}")
    cfg["rankings"] = os.path.join(root, f"rankings_{suffix}")
    return cfg


def _populate_pending(cfg: dict, source_cfg: dict) -> int:
    """Copy ranking tasks into the new rank_in/pending/ from the source.

    Scans source pending/done/claimed, deduplicates, and copies into
    target pending/ using parallel threads. Task JSON files are tiny
    (~200 bytes) so parallel read+write is fast even through FUSE.

    Returns number of tasks copied.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target_pending = os.path.join(cfg["rank_in"], "pending")
    os.makedirs(target_pending, exist_ok=True)

    # ── Collect source task files ──
    source_root = source_cfg["rank_in"]
    source_files = {}  # original_name -> source_path
    for sub in ("pending", "done", "claimed"):
        d = os.path.join(source_root, sub)
        if not os.path.isdir(d):
            continue
        log.info("Scanning source %s/ ...", sub)
        names = [f for f in os.listdir(d) if f.endswith(".json")]
        log.info("  found %d files in %s/", len(names), sub)
        for fname in names:
            original_name = fname
            if sub in ("claimed", "done"):
                idx = fname.find("_")
                if idx >= 0:
                    original_name = fname[idx + 1:]
            if original_name not in source_files:
                source_files[original_name] = os.path.join(d, fname)

    log.info("Total unique tasks from source: %d", len(source_files))

    # ── Build existing set in target ──
    existing = set()
    for sub in ("pending", "done", "claimed"):
        d = os.path.join(cfg["rank_in"], sub)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                idx = fname.find("_")
                existing.add(fname[idx + 1:] if idx >= 0 else fname)

    to_copy = [(n, p) for n, p in source_files.items() if n not in existing]
    log.info("To copy: %d (skipping %d already in target)", len(to_copy), len(source_files) - len(to_copy))

    if not to_copy:
        return 0

    # ── Parallel read+write (files are ~200 bytes each) ──
    copied = 0
    errors = 0
    lock = __import__("threading").Lock()

    def _copy_one(name_path):
        name, src_path = name_path
        dst = os.path.join(target_pending, name)
        with open(src_path, "r") as f_in:
            content = f_in.read()
        with open(dst, "w") as f_out:
            f_out.write(content)

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(_copy_one, item): item for item in to_copy}
        for fut in as_completed(futures):
            try:
                fut.result()
                copied += 1
                if copied % 2000 == 0:
                    log.info("  copied %d / %d ...", copied, len(to_copy))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.warning("  copy failed: %s", e)

    log.info("Copied %d files (%d errors)", copied, errors)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Gemini ranking daemon")
    parser.add_argument("--config", required=True, help="Path to iter config YAML")
    parser.add_argument("--single-pass", action="store_true",
                        help="Process available tasks once and exit")
    parser.add_argument("--rank-suffix", type=str, default=None,
                        help="Suffix for rank_in/rankings dirs (e.g. 'seed' -> rank_in_seed/). "
                             "Auto-populates pending from the default rank_in.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg)

    if args.rank_suffix:
        source_cfg = dict(cfg)
        cfg = _apply_rank_suffix(cfg, args.rank_suffix)
        log.info("rank_suffix=%s: target rank_in=%s", args.rank_suffix, cfg["rank_in"])
        log.info("Populating pending from source %s ...", source_cfg["rank_in"])
        n = _populate_pending(cfg, source_cfg)
        log.info("Populated %d tasks into %s", n, os.path.join(cfg["rank_in"], "pending"))

    run_ranker(cfg, single_pass=args.single_pass)


if __name__ == "__main__":
    main()
