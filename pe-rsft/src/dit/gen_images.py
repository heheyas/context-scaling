# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Stage 2: DiT image generation daemon — file-queue consumer.

Polls llm_out/pending/ for completed LLM shards, renders each record
via the DiT T2I backend, saves PNGs to images/{index}/{image_idx}.png,
appends metadata to a per-host-per-pid file, and emits ranking tasks
to rank_in/pending/ when all images for an index are ready.

Usage:
    python -m src.dit.gen_images --config configs/iter0.yaml

The actual DiT call is delegated to the backend selected in config
(native / psm_direct). Tests inject a mock via run_dit_fn parameter.
"""

import json
import logging
import os
import socket
import time
import argparse
from types import SimpleNamespace

from src.ops.file_queue import claim, claim_batch, emit_pending, release_to_done
from src.ops.config import load_config, resolve_paths

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Worker ID
# ──────────────────────────────────────────────

def make_worker_id() -> str:
    """Return a worker ID in {hostname}-{pid} format (hyphen, no underscore)."""
    hostname = socket.gethostname().replace("_", "-")
    return f"{hostname}-{os.getpid()}"


# ──────────────────────────────────────────────
# Resume: scan existing PNGs
# ──────────────────────────────────────────────

def scan_existing_images(images_dir: str) -> set[tuple[int, int]]:
    """Return set of (index, image_idx) pairs that already have PNGs on disk."""
    existing = set()
    if not os.path.isdir(images_dir):
        return existing
    for index_dir_name in os.listdir(images_dir):
        index_path = os.path.join(images_dir, index_dir_name)
        if not os.path.isdir(index_path):
            continue
        try:
            index = int(index_dir_name)
        except ValueError:
            continue
        for fname in os.listdir(index_path):
            if fname.endswith(".png"):
                try:
                    image_idx = int(fname.replace(".png", ""))
                    existing.add((index, image_idx))
                except ValueError:
                    continue
    return existing


# ──────────────────────────────────────────────
# Shard parsing
# ──────────────────────────────────────────────

def parse_shard(shard_path: str) -> list[dict]:
    """Read a shard JSONL file into a list of record dicts."""
    records = []
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
# Image saving + metadata
# ──────────────────────────────────────────────

def save_image(pil_image, images_dir: str, index: int, image_idx: int) -> str:
    """Save a PIL image to images/{index}/{image_idx}.png.

    Returns the relative path (relative to shared_root).
    """
    index_dir = os.path.join(images_dir, str(index))
    os.makedirs(index_dir, exist_ok=True)
    filename = f"{image_idx}.png"
    abs_path = os.path.join(index_dir, filename)
    pil_image.save(abs_path)
    return abs_path


def build_metadata_record(dit_result: dict, image_path: str | None) -> dict:
    """Build a metadata record matching the legacy schema.

    Preserves the same field names as legacy save_result so downstream
    tools (build_train_data, ranking) don't break.
    """
    record = {
        "index": dit_result["index"],
        "image_idx": dit_result["image_idx"],
        "original_prompt": dit_result["prompt"],
        "seed": dit_result["seed"],
        "dit_seed": dit_result.get("dit_seed"),
        "aspect_ratio": dit_result.get("aspect_ratio"),
        "width": dit_result.get("width"),
        "height": dit_result.get("height"),
        "llm_raw_response": dit_result.get("llm_raw_response"),
        "structured_prompt": dit_result.get("structured_prompt"),
        "image_path": image_path,
        "success": dit_result["success"],
    }
    if "error" in dit_result:
        record["error"] = dit_result["error"]
    return record


def append_metadata(record: dict, metadata_path: str) -> None:
    """Append a metadata record to the per-host-per-pid metadata file."""
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Ranking task emission
# ──────────────────────────────────────────────

def count_images_for_index(images_dir: str, index: int) -> int:
    """Count PNG files in images/{index}/."""
    index_dir = os.path.join(images_dir, str(index))
    if not os.path.isdir(index_dir):
        return 0
    return sum(1 for f in os.listdir(index_dir) if f.endswith(".png"))


def build_ranking_task(
    index: int,
    query: str,
    images_dir: str,
    shared_root: str,
) -> dict:
    """Build a ranking task JSON with relative paths + root field."""
    index_dir = os.path.join(images_dir, str(index))
    images = []
    for fname in sorted(os.listdir(index_dir)):
        if not fname.endswith(".png"):
            continue
        image_idx = int(fname.replace(".png", ""))
        rel_path = os.path.join("images", str(index), fname)
        images.append({
            "image_idx": image_idx,
            "path": rel_path,
            "success": True,
        })
    return {
        "index": index,
        "query": query,
        "root": shared_root,
        "images": images,
    }


def check_and_emit_ranking_task(
    index: int,
    query: str,
    images_dir: str,
    shared_root: str,
    rank_in_pending: str,
    num_images_per_prompt: int,
    index_done_count: dict | None = None,
) -> bool:
    """Emit a ranking task when all tasks for this index are done.

    A task is "done" when it has been processed (successfully rendered a
    PNG, or failed at LLM/DiT level). We emit the ranking task when
    done_count == num_images_per_prompt, regardless of how many actually
    succeeded. This way indices with partial failures still get ranked
    (with fewer valid images) instead of being stuck forever.

    Race-safe: uses emit_pending(overwrite=False).

    Returns True if the task was emitted (or already existed).
    """
    # If we have a done counter, use it to decide.
    if index_done_count is not None:
        if index_done_count.get(index, 0) < num_images_per_prompt:
            return False
    else:
        # Fallback: only check PNG count (legacy behavior).
        count = count_images_for_index(images_dir, index)
        if count < num_images_per_prompt:
            return False

    n_images = count_images_for_index(images_dir, index)
    if n_images < 2:
        log.warning(
            "Index %d: all %d tasks done but only %d valid images, skipping ranking",
            index, num_images_per_prompt, n_images,
        )
        return False

    task = build_ranking_task(index, query, images_dir, shared_root)
    filename = f"idx_{index}.json"
    payload = json.dumps(task, ensure_ascii=False).encode("utf-8")

    try:
        emit_pending(rank_in_pending, filename, payload, overwrite=False)
        log.info("Emitted ranking task for index %d (%d images)", index, n_images)
    except FileExistsError:
        log.debug("Ranking task for index %d already emitted (race OK)", index)

    return True


# ──────────────────────────────────────────────
# DiT backend wrapper
# ──────────────────────────────────────────────

def make_dit_args(cfg: dict) -> SimpleNamespace:
    """Build the args namespace that run_dit (legacy or native) expects."""
    dit_cfg = cfg["dit"]
    return SimpleNamespace(
        dit_backend=dit_cfg.get("backend", "seedream5p0"),
        num_steps=dit_cfg.get("num_steps", 25),
        cfg_scale=dit_cfg.get("cfg_scale", 4.0),
        negative_prompt=dit_cfg.get("negative_prompt", ""),
        timeout=dit_cfg.get("timeout", 600),
        max_retries=dit_cfg.get("max_retries", 5),
        num_dit_workers=dit_cfg.get("num_workers", 32),
    )


def resolve_dit_backend(cfg: dict):
    """Return (run_dit_fn, dit_api_bases) based on config backend setting."""
    backend = cfg["dit"].get("backend", "native")
    if backend == "native":
        from src.dit.backends.native import run_dit_native
        dit_url = cfg["dit"].get("serve_url", "http://localhost:8091")
        return run_dit_native, dit_url
    elif backend == "psm_direct":
        from src.dit.backends.psm import run_dit_psm, PSMPool
        psm = cfg["dit"]["psm"]
        pool = PSMPool(psm)
        return run_dit_psm, pool
    raise ValueError(f"Unknown dit.backend={backend!r}; expected 'native' or 'psm_direct'")


# ──────────────────────────────────────────────
# Shard processing
# ──────────────────────────────────────────────

def process_shard(
    shard_path: str,
    images_dir: str,
    metadata_path: str,
    rank_in_pending: str,
    shared_root: str,
    num_images_per_prompt: int,
    dit_args: SimpleNamespace,
    dit_api_bases,
    existing_images: set[tuple[int, int]],
    run_dit_fn,
) -> dict:
    """Process all records in a shard file.

    Returns summary dict: {"processed": int, "skipped": int, "failed": int,
                           "ranking_tasks_emitted": int}.
    """
    records = parse_shard(shard_path)
    stats = {"processed": 0, "skipped": 0, "failed": 0, "ranking_tasks_emitted": 0}
    num_dit_workers = getattr(dit_args, "num_dit_workers", 32)

    # Track how many tasks are done per index (success + failure).
    # When done_count == num_images_per_prompt, emit ranking task
    # even if some images failed — so partial failures still get ranked.
    from collections import defaultdict
    index_done_count = defaultdict(int)

    # Build per-index query map for ranking task emission.
    index_query = {}
    for record in records:
        index_query[record["index"]] = record["prompt"]

    # Separate records into skip/fail (handle immediately) and to-render.
    to_render = []
    for record in records:
        index = record["index"]
        image_idx = record["image_idx"]

        if (index, image_idx) in existing_images:
            stats["skipped"] += 1
            index_done_count[index] += 1
            if check_and_emit_ranking_task(
                index, record["prompt"], images_dir, shared_root,
                rank_in_pending, num_images_per_prompt,
                index_done_count=index_done_count,
            ):
                stats["ranking_tasks_emitted"] += 1
            continue

        if not record.get("success", True):
            meta = build_metadata_record(
                {**record, "image": None}, image_path=None,
            )
            append_metadata(meta, metadata_path)
            stats["failed"] += 1
            index_done_count[index] += 1
            if check_and_emit_ranking_task(
                index, record["prompt"], images_dir, shared_root,
                rank_in_pending, num_images_per_prompt,
                index_done_count=index_done_count,
            ):
                stats["ranking_tasks_emitted"] += 1
            continue

        to_render.append(record)

    if not to_render:
        return stats

    # Render in parallel to saturate all GPUs on the serve.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _render_one(record):
        return run_dit_fn(record, dit_api_bases, dit_args)

    log.info("Rendering %d images (%d workers)...", len(to_render), num_dit_workers)
    with ThreadPoolExecutor(max_workers=num_dit_workers) as pool:
        futures = {pool.submit(_render_one, r): r for r in to_render}
        for future in as_completed(futures):
            record = futures[future]
            index = record["index"]
            image_idx = record["image_idx"]

            try:
                dit_result = future.result()
            except Exception as e:
                log.error("[DiT %d/%d] Unexpected error: %s", index, image_idx, e)
                dit_result = {**record, "image": None, "success": False,
                              "error": f"DiT error: {e}"}

            image_path = None
            if dit_result.get("image") is not None:
                image_path = save_image(
                    dit_result["image"], images_dir, index, image_idx,
                )
                existing_images.add((index, image_idx))
                log.info(
                    "[DiT %d/%d] OK %dx%d (%s)",
                    index, image_idx,
                    dit_result.get("width", 0), dit_result.get("height", 0),
                    dit_result.get("aspect_ratio", "?"),
                )
            else:
                log.warning(
                    "[DiT %d/%d] FAIL: %s",
                    index, image_idx, dit_result.get("error", "unknown"),
                )

            meta = build_metadata_record(dit_result, image_path)
            append_metadata(meta, metadata_path)

            if dit_result["success"]:
                stats["processed"] += 1
            else:
                stats["failed"] += 1

            index_done_count[index] += 1
            if check_and_emit_ranking_task(
                index, record["prompt"], images_dir, shared_root,
                rank_in_pending, num_images_per_prompt,
                index_done_count=index_done_count,
            ):
                stats["ranking_tasks_emitted"] += 1

    return stats


# ──────────────────────────────────────────────
# Main daemon loop
# ──────────────────────────────────────────────

_UNSET = object()


def _recover_claimed(claimed_dir: str, worker_id: str) -> list[str]:
    """Find files in claimed/ that belong to this worker (from a prior crash).

    Returns list of paths to resume processing, oldest first.
    """
    recovered = []
    if not os.path.isdir(claimed_dir):
        return recovered
    prefix = f"{worker_id}_"
    for name in sorted(os.listdir(claimed_dir)):
        if name.startswith(prefix):
            recovered.append(os.path.join(claimed_dir, name))
    return recovered


def run_dit_consumer(
    cfg: dict,
    run_dit_fn=None,
    dit_api_bases=_UNSET,
    single_pass: bool = False,
    rank: int | None = None,
):
    """Run the DiT consumer loop.

    Polls llm_out/pending/ for shards, processes them, and releases to
    llm_out/done/.

    Args:
        cfg: Config dict (from load_config + resolve_paths).
        run_dit_fn: Callable that renders one image given (prompt, seed,
            resolution, api_base). If None, resolved from cfg.dit.backend.
        dit_api_bases: DiT API base URLs / pool. If _UNSET (default),
            resolved from cfg.dit.backend. Pass any explicit value
            (including None) to skip the resolution.
        single_pass: If True, process available shards once and return
            (for testing). If False, loop with poll interval.
        rank: Fixed worker rank. If set, worker_id is "dit-{rank}" (stable
            across restarts, enables resume of claimed shards). If None,
            falls back to hostname-pid.
    """
    if run_dit_fn is None or dit_api_bases is _UNSET:
        _dit_fn, _api_bases = resolve_dit_backend(cfg)
        if run_dit_fn is None:
            run_dit_fn = _dit_fn
        if dit_api_bases is _UNSET:
            dit_api_bases = _api_bases

    shared_root = cfg["shared_root"]
    llm_out = cfg["llm_out"]
    images_dir = cfg["images"]
    rank_in = cfg["rank_in"]
    num_images = cfg["num_images_per_prompt"]
    poll_interval = cfg["dit"].get("poll_interval_sec", 10)

    llm_pending = os.path.join(llm_out, "pending")
    llm_claimed = os.path.join(llm_out, "claimed")
    llm_done = os.path.join(llm_out, "done")
    rank_in_pending = os.path.join(rank_in, "pending")

    # Ensure directories exist.
    for d in (llm_pending, llm_claimed, llm_done, images_dir, rank_in_pending):
        os.makedirs(d, exist_ok=True)
    for sub in ("claimed", "done"):
        os.makedirs(os.path.join(rank_in, sub), exist_ok=True)

    worker_id = f"dit-{rank}" if rank is not None else make_worker_id()
    dit_args = make_dit_args(cfg)

    # Write metadata to local /tmp first (HDFS FUSE doesn't support append),
    # then copy to shared FS on completion.
    _local_metadata_dir = os.path.join("/tmp", "dit_metadata")
    os.makedirs(_local_metadata_dir, exist_ok=True)
    metadata_path = os.path.join(_local_metadata_dir, f"metadata_{worker_id}.jsonl")
    _hdfs_metadata_path = os.path.join(images_dir, f"metadata_{worker_id}.jsonl")

    # Resume: scan existing images.
    existing_images = scan_existing_images(images_dir)
    if existing_images:
        log.info("Resume: found %d existing images on disk", len(existing_images))

    # Resume: recover shards claimed by this worker in a prior run.
    recovered = _recover_claimed(llm_claimed, worker_id)
    if recovered:
        log.info("Resume: recovering %d shard(s) claimed by %s", len(recovered), worker_id)

    log.info(
        "DiT consumer started (worker=%s, poll=%ds, images=%d/prompt)",
        worker_id, poll_interval, num_images,
    )

    # Shared state for cross-shard tracking.
    from collections import defaultdict
    index_done_count = defaultdict(int)
    index_query = {}

    # Process recovered shards first, then poll for new ones.
    shard_queue = list(recovered)
    use_burn_gpu = cfg.get("dit", {}).get("burn_gpu", True) and not single_pass
    burn_procs = []

    def _start_burn():
        nonlocal burn_procs
        if not use_burn_gpu or burn_procs:
            return
        try:
            from src.ops.burn_gpu import start_burn_gpu
            burn_procs = start_burn_gpu()
        except Exception as e:
            log.warning("Failed to start burn_gpu: %s", e)

    def _stop_burn():
        nonlocal burn_procs
        if not burn_procs:
            return
        from src.ops.burn_gpu import stop_burn_gpu
        stop_burn_gpu(burn_procs)
        burn_procs = []

    # Start burning if no recovered shards to process immediately.
    if not shard_queue:
        _start_burn()

    while True:
        # Claim as many shards as available (up to a cap) so we can
        # feed all workers at once instead of one shard at a time.
        claimed_shards = []
        if shard_queue:
            claimed_shards = list(shard_queue)
            shard_queue.clear()
        else:
            batch = claim_batch(llm_pending, llm_claimed, worker_id, max_count=0)
            claimed_shards = batch

        if not claimed_shards:
            if single_pass:
                log.info("Single-pass mode: no more shards, exiting")
                _stop_burn()
                return
            # No work — keep burning while we wait.
            _start_burn()
            time.sleep(poll_interval)
            continue

        # Got work — stop burning, give GPU fully to DiT serve.
        _stop_burn()

        log.info("Claimed %d shard(s), processing together...", len(claimed_shards))

        # Collect ALL records from all claimed shards, then render in one
        # big thread pool so workers are fully saturated.
        all_to_render = []
        shard_records_map = {}  # shard_path -> list of records
        for shard_path in claimed_shards:
            shard_name = os.path.basename(shard_path)
            records = parse_shard(shard_path)
            shard_records_map[shard_path] = records

            for record in records:
                index = record["index"]
                image_idx = record["image_idx"]
                index_query[index] = record["prompt"]

                if (index, image_idx) in existing_images:
                    index_done_count[index] += 1
                    check_and_emit_ranking_task(
                        index, record["prompt"], images_dir, shared_root,
                        rank_in_pending, num_images,
                        index_done_count=index_done_count,
                    )
                    continue

                if not record.get("success", True):
                    meta = build_metadata_record(
                        {**record, "image": None}, image_path=None,
                    )
                    append_metadata(meta, metadata_path)
                    index_done_count[index] += 1
                    check_and_emit_ranking_task(
                        index, record["prompt"], images_dir, shared_root,
                        rank_in_pending, num_images,
                        index_done_count=index_done_count,
                    )
                    continue

                all_to_render.append(record)

        if all_to_render:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            num_dit_workers = getattr(dit_args, "num_dit_workers", 32)

            def _render_one(record):
                return run_dit_fn(record, dit_api_bases, dit_args)

            log.info("Rendering %d images from %d shards (%d workers)...",
                     len(all_to_render), len(claimed_shards), num_dit_workers)
            with ThreadPoolExecutor(max_workers=num_dit_workers) as pool:
                futures = {pool.submit(_render_one, r): r for r in all_to_render}
                for future in as_completed(futures):
                    record = futures[future]
                    index = record["index"]
                    image_idx = record["image_idx"]

                    try:
                        dit_result = future.result()
                    except Exception as e:
                        log.error("[DiT %d/%d] Unexpected error: %s", index, image_idx, e)
                        dit_result = {**record, "image": None, "success": False,
                                      "error": f"DiT error: {e}"}

                    image_path = None
                    if dit_result.get("image") is not None:
                        image_path = save_image(
                            dit_result["image"], images_dir, index, image_idx,
                        )
                        existing_images.add((index, image_idx))
                        log.info(
                            "[DiT %d/%d] OK %dx%d (%s)",
                            index, image_idx,
                            dit_result.get("width", 0), dit_result.get("height", 0),
                            dit_result.get("aspect_ratio", "?"),
                        )
                    else:
                        log.warning(
                            "[DiT %d/%d] FAIL: %s",
                            index, image_idx, dit_result.get("error", "unknown"),
                        )

                    meta = build_metadata_record(dit_result, image_path)
                    append_metadata(meta, metadata_path)

                    index_done_count[index] += 1
                    check_and_emit_ranking_task(
                        index, record["prompt"], images_dir, shared_root,
                        rank_in_pending, num_images,
                        index_done_count=index_done_count,
                    )

        # Release all claimed shards to done
        for shard_path in claimed_shards:
            shard_name = os.path.basename(shard_path)
            release_to_done(shard_path, llm_done)
            log.info("Shard %s released to done", shard_name)

        # Copy local metadata to HDFS after each batch.
        try:
            import shutil
            shutil.copy2(metadata_path, _hdfs_metadata_path)
        except Exception as e:
            log.warning("Failed to copy metadata to HDFS: %s", e)

        log.info("Batch of %d shards complete (%d images rendered)",
                 len(claimed_shards), len(all_to_render))

        if single_pass:
            continue


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: DiT image generation daemon")
    parser.add_argument("--config", required=True, help="Path to iter config YAML")
    parser.add_argument("--single-pass", action="store_true",
                        help="Process available shards once and exit (for testing)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Fixed worker rank (enables resume of claimed shards)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg)
    run_dit_consumer(cfg, single_pass=args.single_pass, rank=args.rank)


if __name__ == "__main__":
    main()
