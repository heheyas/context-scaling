# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Stage 1: LLM rollout daemon — file-queue producer.

Reads queries from {shared_root}/queries.jsonl, generates K rollouts per
query via the LLM backend, and emits completed shards as JSONL files into
llm_out/pending/ for Stage 2 (DiT) to consume.

Supports crash recovery via a write-ahead log (WAL):
  - Each individual rollout is appended to _wal/shard_XXXX.partial.jsonl
    as it completes.
  - On restart, shards in pending/ or done/ are skipped; incomplete shards
    resume from their WAL without redoing completed tasks.
  - Once a shard is fully assembled, the final JSONL is atomically emitted
    to pending/ and the WAL is deleted.

Usage:
    python -m src.llm.rollout --config configs/iter0.yaml

The actual LLM call is delegated to the backend selected in config
(vllm / vllm_mcp). Tests inject a mock via run_llm_fn parameter.
"""

import os
import json
import hashlib
import logging
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from src.ops.file_queue import emit_pending
from src.ops.config import load_config, resolve_paths

log = logging.getLogger(__name__)

ASPECT_RATIOS = ["1:1", "3:4", "4:3", "16:9", "9:16", "1:2", "2:1"]
ASPECT_RATIO_WEIGHTS = [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]


# ──────────────────────────────────────────────
# Deterministic hashing (stdlib only)
# ──────────────────────────────────────────────

def _stable_hash(s: str) -> int:
    """SHA-256 based stable hash. Identical to legacy._stable_hash."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def compute_seed(index: int, image_idx: int) -> int:
    """Deterministic LLM seed for a (index, image_idx) pair.

    Uses _stable_hash so that reruns produce the same seed regardless
    of execution order or shard boundaries.
    """
    return _stable_hash(f"{index}_{image_idx}") % (2**31)


def compute_dit_seed(index: int) -> int:
    """Deterministic DiT seed, shared across all image_idx of a given index.

    The N rollouts for one prompt vary their structured prompt (LLM seed
    differs) but use the SAME DiT noise seed, so BT ranking measures SP
    quality, not DiT-noise lottery.
    """
    return _stable_hash(f"dit_{index}") % (2**31)


# ──────────────────────────────────────────────
# Query loading
# ──────────────────────────────────────────────

def load_queries(queries_path: str) -> list[dict]:
    """Load queries from a JSONL file.

    Expected format: one JSON object per line with at least {index, query}.
    """
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if "index" not in q or "query" not in q:
                raise ValueError(f"Query missing 'index' or 'query': {q}")
            queries.append(q)
    return queries


# ──────────────────────────────────────────────
# Shard planning
# ──────────────────────────────────────────────

def shard_filename(shard_id: int) -> str:
    return f"shard_{shard_id:04d}.jsonl"


def wal_path_for_shard(llm_out_dir: str, shard_id: int) -> str:
    return os.path.join(llm_out_dir, "_wal", f"shard_{shard_id:04d}.partial.jsonl")


def plan_shards(
    queries: list[dict],
    num_images_per_prompt: int,
    shard_size: int,
) -> dict[int, list[dict]]:
    """Plan all tasks grouped by shard_id.

    shard_id = index // shard_size, so shard assignment is deterministic
    and resume is a simple set-difference on shard filenames.

    Returns:
        Dict mapping shard_id -> list of task dicts, each with keys:
        {index, image_idx, query, seed}.
    """
    shards: dict[int, list[dict]] = defaultdict(list)
    for q in queries:
        idx = q["index"]
        shard_id = idx // shard_size
        for img_idx in range(num_images_per_prompt):
            shards[shard_id].append({
                "index": idx,
                "image_idx": img_idx,
                "query": q["query"],
                "seed": compute_seed(idx, img_idx),
            })
    return dict(shards)


def find_completed_shards(llm_out_dir: str) -> set[str]:
    """Return filenames of shards already in pending/ or done/.

    These are skipped on resume — their tasks are fully done.
    """
    completed = set()
    for subdir in ("pending", "done"):
        path = os.path.join(llm_out_dir, subdir)
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.endswith(".jsonl") and not name.startswith("."):
                    completed.add(name)
    # Also check claimed/ — these are in-progress by another worker or
    # stale (janitor will re-park them). Don't redo them.
    claimed_path = os.path.join(llm_out_dir, "claimed")
    if os.path.isdir(claimed_path):
        for name in os.listdir(claimed_path):
            if name.endswith(".jsonl") and not name.startswith("."):
                # Strip worker prefix to get original shard filename.
                orig = name.split("_", 1)[1] if "_" in name else name
                completed.add(orig)
    return completed


# ──────────────────────────────────────────────
# WAL (write-ahead log) for mid-shard crash recovery
# ──────────────────────────────────────────────

def load_wal(wal_path: str) -> dict[tuple[int, int], dict]:
    """Load partial results from a WAL file.

    Returns:
        Dict mapping (index, image_idx) -> record dict.
    """
    results = {}
    if not os.path.exists(wal_path):
        return results
    with open(wal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (record["index"], record["image_idx"])
                results[key] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def append_wal(wal_path: str, record: dict) -> None:
    """Append a single record to the WAL file."""
    os.makedirs(os.path.dirname(wal_path), exist_ok=True)
    with open(wal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Shard assembly + emission
# ──────────────────────────────────────────────

def assemble_shard_payload(results: dict[tuple[int, int], dict]) -> bytes:
    """Assemble a shard JSONL payload from a dict of results.

    Records are sorted by (index, image_idx) for deterministic output.
    """
    records = sorted(results.values(), key=lambda r: (r["index"], r["image_idx"]))
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


# ──────────────────────────────────────────────
# LLM backend wrapper
# ──────────────────────────────────────────────

def make_llm_args(cfg: dict) -> SimpleNamespace:
    """Build the args namespace that run_llm (legacy or vllm) expects."""
    llm_cfg = cfg["llm"]
    return SimpleNamespace(
        input_template=llm_cfg.get("input_template", "<prompt>"),
        llm_backend=llm_cfg.get("backend", "seed"),
        llm_psm=llm_cfg.get("psm", ""),
        llm_temperature=llm_cfg.get("temperature", 0.9),
        llm_top_p=llm_cfg.get("top_p", 0.95),
        thinking=llm_cfg.get("thinking", True),
        system_prompt_role=llm_cfg.get("system_prompt_role", "system"),
        # vLLM-specific fields (ignored by legacy backend).
        vllm_url=llm_cfg.get("vllm_url", "http://localhost:8000"),
        vllm_model=llm_cfg.get("vllm_model", None),
        bucket_size=llm_cfg.get("bucket_size", 2048),
        max_tokens=llm_cfg.get("max_tokens", 65536),
        # MCP-specific fields (used by vllm_mcp backend).
        mcp_server=llm_cfg.get("mcp_server"),
        mcp_env=llm_cfg.get("mcp_env", {}),
        agent_timeout=llm_cfg.get("agent_timeout", 900),
        max_retries=llm_cfg.get("max_retries", 3),
        sampling=llm_cfg.get("sampling", {}),
    )


def resolve_backend(cfg: dict):
    """Return (run_llm_fn, sample_ar_fn) based on config backend setting."""
    backend = cfg["llm"].get("backend", "vllm")

    def _make_ar_sampler():
        from src.llm.backends.vllm import resolve_target_size
        from src.llm.rollout import _stable_hash, ASPECT_RATIOS, ASPECT_RATIO_WEIGHTS
        import random as _random

        def sample_ar(bucket_size, prompt, seed):
            rng = _random.Random(_stable_hash(prompt) ^ seed)
            ratio_str = rng.choices(ASPECT_RATIOS, weights=ASPECT_RATIO_WEIGHTS, k=1)[0]
            width, height = resolve_target_size(
                ratio_spec=ratio_str, bucket_size=bucket_size, factor=32,
            )
            return width, height, ratio_str

        return sample_ar

    if backend == "vllm":
        from src.llm.backends.vllm import run_llm_vllm
        return run_llm_vllm, _make_ar_sampler()
    elif backend == "vllm_mcp":
        from src.llm.backends.vllm_mcp import run_llm_mcp
        return run_llm_mcp, _make_ar_sampler()
    raise ValueError(f"Unknown llm.backend={backend!r}; expected 'vllm' or 'vllm_mcp'")


def load_system_prompts(cfg: dict) -> list[tuple[str | None, float]]:
    """Load system prompts with sampling weights from config.

    Supports two config formats:

    1. Single prompt (legacy):
        system_prompt_path: "/path/to/prompt.txt"
      → returns [("/path/to/prompt.txt content", 1.0)]

    2. Multiple prompts with weights:
        system_prompts:
          "/path/to/prompt_a.txt": 0.6
          "/path/to/prompt_b.txt": 0.3
          null: 0.1
      → returns [(content_a, 0.6), (content_b, 0.3), (None, 0.1)]

    Returns list of (prompt_text_or_None, weight) tuples.
    """
    llm_cfg = cfg["llm"]

    # Check both "system_prompts" and "system_prompt_path" — either can
    # hold the dict format or the single-path string.
    prompts_cfg = llm_cfg.get("system_prompts") or llm_cfg.get("system_prompt_path")

    if prompts_cfg is None:
        return [(None, 1.0)]

    # Dict format: {path: weight, ...}
    if isinstance(prompts_cfg, dict):
        result = []
        for path, weight in prompts_cfg.items():
            if path is None or str(path).lower() == "null":
                result.append((None, float(weight)))
            else:
                with open(str(path), "r", encoding="utf-8") as f:
                    result.append((f.read(), float(weight)))
        return result

    # Single string path (legacy).
    with open(str(prompts_cfg), "r", encoding="utf-8") as f:
        return [(f.read(), 1.0)]


def sample_system_prompt(
    prompts: list[tuple[str | None, float]],
    seed: int,
) -> str | None:
    """Deterministically sample a system prompt based on seed."""
    if len(prompts) == 1:
        return prompts[0][0]
    rng = random.Random(seed)
    texts = [p[0] for p in prompts]
    weights = [p[1] for p in prompts]
    return rng.choices(texts, weights=weights, k=1)[0]


# ──────────────────────────────────────────────
# Main rollout logic
# ──────────────────────────────────────────────

def process_shard(
    shard_id: int,
    tasks: list[dict],
    llm_out_dir: str,
    llm_args: SimpleNamespace,
    system_prompts: list[tuple[str | None, float]],
    bucket_size: int,
    num_workers: int,
    run_llm_fn,
    sample_ar_fn,
):
    """Process a single shard: run LLM for all tasks, emit to pending/.

    Resumes from WAL if a previous run crashed mid-shard.
    """
    wal = wal_path_for_shard(llm_out_dir, shard_id)
    completed = load_wal(wal)
    pending_dir = os.path.join(llm_out_dir, "pending")

    tasks_to_run = [
        t for t in tasks
        if (t["index"], t["image_idx"]) not in completed
    ]

    if completed:
        log.info(
            "Shard %04d: resuming — %d/%d tasks already in WAL",
            shard_id, len(completed), len(tasks),
        )

    if not tasks_to_run:
        log.info("Shard %04d: all tasks already in WAL, assembling", shard_id)
    else:
        log.info(
            "Shard %04d: running %d tasks (%d workers)",
            shard_id, len(tasks_to_run), num_workers,
        )

        def _do_one(task):
            seed = task["seed"]
            # Aspect ratio is fixed per index (not per image_idx) so that
            # all BoN candidates for the same query share the same resolution.
            ar_seed = _stable_hash(str(task["index"])) % (2**31)
            w, h, ratio_str = sample_ar_fn(bucket_size, task["query"], ar_seed)
            # System prompt is fixed per index (same as aspect ratio) so
            # all BoN candidates for the same query use the same prompt.
            sys_prompt = sample_system_prompt(system_prompts, ar_seed)
            result = run_llm_fn(
                task["index"], task["image_idx"], task["query"], seed,
                llm_args, sys_prompt, w, h, ratio_str,
            )
            return result

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_do_one, t): t for t in tasks_to_run}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    key = (result["index"], result["image_idx"])
                    completed[key] = result
                    append_wal(wal, result)
                except Exception as e:
                    log.error(
                        "Shard %04d task (%d, %d) failed: %s",
                        shard_id, task["index"], task["image_idx"], e,
                    )

    # Assemble and emit the final shard.
    payload = assemble_shard_payload(completed)
    fname = shard_filename(shard_id)
    emit_pending(pending_dir, fname, payload, overwrite=True)
    log.info("Shard %04d: emitted %s (%d records)", shard_id, fname, len(completed))

    # Clean up WAL.
    try:
        os.unlink(wal)
    except OSError:
        pass


def run_rollout(
    cfg: dict,
    run_llm_fn=None,
    sample_ar_fn=None,
    rank: int = 0,
    world_size: int = 1,
):
    """Run the full LLM rollout stage.

    Loads queries, plans shards, skips completed shards, and processes
    the rest. Each shard is processed sequentially; tasks within a shard
    run concurrently.

    Args:
        cfg: Config dict (from load_config + resolve_paths).
        run_llm_fn: Callable with signature matching legacy run_llm.
            If None, resolved from config backend setting.
        sample_ar_fn: Callable with signature matching legacy
            sample_aspect_ratio_and_size. If None, resolved from config.
        rank: This node's rank for multi-node shard partitioning (0-indexed).
        world_size: Total number of LLM nodes. Each node processes shards
            where shard_id % world_size == rank.
    """
    if run_llm_fn is None or sample_ar_fn is None:
        _llm_fn, _ar_fn = resolve_backend(cfg)
        if run_llm_fn is None:
            run_llm_fn = _llm_fn
        if sample_ar_fn is None:
            sample_ar_fn = _ar_fn

    llm_out = cfg["llm_out"]
    llm_args = make_llm_args(cfg)
    system_prompts = load_system_prompts(cfg)
    log.info("Loaded %d system prompt(s): weights=%s",
             len(system_prompts),
             [w for _, w in system_prompts])
    bucket_size = cfg["llm"].get("bucket_size", 2048)
    num_workers = cfg["llm"].get("num_workers", 128)
    num_images = cfg["num_images_per_prompt"]
    shard_size = cfg["shard_size"]

    # Ensure queue directories exist.
    for sub in ("pending", "claimed", "done", "_wal"):
        os.makedirs(os.path.join(llm_out, sub), exist_ok=True)

    # Load queries and plan shards.
    queries = load_queries(cfg["queries_path"])
    log.info("Loaded %d queries from %s", len(queries), cfg["queries_path"])

    all_shards = plan_shards(queries, num_images, shard_size)
    log.info("Planned %d shards (shard_size=%d)", len(all_shards), shard_size)

    # Resume: skip completed shards.
    completed_names = find_completed_shards(llm_out)
    shards_to_run = {
        sid: tasks
        for sid, tasks in sorted(all_shards.items())
        if shard_filename(sid) not in completed_names
        and sid % world_size == rank  # Multi-node partition.
    }

    if world_size > 1:
        log.info(
            "Multi-node: rank=%d/%d, %d shards assigned to this node",
            rank, world_size, len(shards_to_run),
        )
    log.info(
        "Resume: %d shards done, %d to process",
        len(all_shards) - len(shards_to_run), len(shards_to_run),
    )

    for shard_id, tasks in sorted(shards_to_run.items()):
        process_shard(
            shard_id=shard_id,
            tasks=tasks,
            llm_out_dir=llm_out,
            llm_args=llm_args,
            system_prompts=system_prompts,
            bucket_size=bucket_size,
            num_workers=num_workers,
            run_llm_fn=run_llm_fn,
            sample_ar_fn=sample_ar_fn,
        )

    log.info("Rollout complete. All shards emitted.")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: LLM rollout daemon")
    parser.add_argument("--config", required=True, help="Path to iter config YAML")
    parser.add_argument("--rank", type=int, default=0, help="Node rank (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1, help="Total LLM nodes")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg)
    run_rollout(cfg, rank=args.rank, world_size=args.world_size)


if __name__ == "__main__":
    main()
