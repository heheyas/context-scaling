# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/dit/gen_images.py.

All tests use a mocked run_dit_fn so we don't need a real DiT backend.
We use PIL.Image to create tiny test images.
"""

import json
import os
import threading

import pytest
from PIL import Image

from src.dit.gen_images import (
    append_metadata,
    build_metadata_record,
    build_ranking_task,
    check_and_emit_ranking_task,
    count_images_for_index,
    make_worker_id,
    parse_shard,
    process_shard,
    run_dit_consumer,
    save_image,
    scan_existing_images,
    make_dit_args,
)
from src.ops.file_queue import emit_pending


# ──────────────────────────────────────────────
# Mock DiT backend
# ──────────────────────────────────────────────

def _make_tiny_image():
    """Create a 4x4 red PNG image."""
    return Image.new("RGB", (4, 4), color="red")


def mock_run_dit(llm_result, dit_api_bases, args):
    """Simulate a successful DiT call, returning a tiny PIL image."""
    if not llm_result.get("success", True):
        return {**llm_result, "image": None}
    return {
        **llm_result,
        "image": _make_tiny_image(),
        "success": True,
    }


def mock_run_dit_failing(llm_result, dit_api_bases, args):
    """Simulate DiT failure for image_idx == 2."""
    if not llm_result.get("success", True):
        return {**llm_result, "image": None}
    if llm_result["image_idx"] == 2:
        return {
            **llm_result,
            "image": None,
            "success": False,
            "error": "DiT error: simulated failure",
        }
    return mock_run_dit(llm_result, dit_api_bases, args)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def make_shard_record(index, image_idx, query="test query"):
    """Build a shard record as produced by Stage 1."""
    return {
        "index": index,
        "image_idx": image_idx,
        "prompt": query,
        "query": query,
        "seed": 42 + image_idx,
        "aspect_ratio": "1:1",
        "llm_raw_response": "<think>reasoning</think>output",
        "structured_prompt": f"a beautiful image of {query}",
        "width": 1024,
        "height": 1024,
        "success": True,
    }


def make_failed_shard_record(index, image_idx, query="test query"):
    """Build a shard record where LLM failed."""
    return {
        "index": index,
        "image_idx": image_idx,
        "prompt": query,
        "query": query,
        "seed": 42 + image_idx,
        "aspect_ratio": "1:1",
        "llm_raw_response": None,
        "structured_prompt": None,
        "width": 1024,
        "height": 1024,
        "success": False,
        "error": "LLM error: simulated",
    }


def write_shard(shard_dir, filename, records):
    """Write a shard JSONL file."""
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, filename)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def shared_root(tmp_path):
    root = tmp_path / "rft_iter0"
    root.mkdir()
    return root


def make_cfg(shared_root, num_images=3):
    return {
        "shared_root": str(shared_root),
        "queries_path": str(shared_root / "queries.jsonl"),
        "llm_out": str(shared_root / "llm_out"),
        "images": str(shared_root / "images"),
        "rank_in": str(shared_root / "rank_in"),
        "rankings": str(shared_root / "rankings"),
        "shard_size": 256,
        "num_images_per_prompt": num_images,
        "seed": 42,
        "dit": {
            "backend": "seedream5p0",
            "psm": "test-psm",
            "num_workers": 2,
            "poll_interval_sec": 1,
        },
        "llm": {"backend": "seed", "psm": "test", "bucket_size": 2048},
    }


# ──────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────

def test_make_worker_id():
    wid = make_worker_id()
    assert "-" in wid
    assert "_" not in wid


def test_parse_shard(tmp_path):
    records = [make_shard_record(0, i) for i in range(3)]
    path = write_shard(str(tmp_path), "shard.jsonl", records)
    parsed = parse_shard(path)
    assert len(parsed) == 3
    assert parsed[0]["index"] == 0


def test_save_image(tmp_path):
    img = _make_tiny_image()
    images_dir = str(tmp_path / "images")
    path = save_image(img, images_dir, 42, 3)
    assert os.path.exists(path)
    assert path.endswith("3.png")
    assert "42" in path
    # Verify it's a valid image.
    loaded = Image.open(path)
    assert loaded.size == (4, 4)


def test_build_metadata_record():
    dit_result = {
        "index": 5, "image_idx": 2, "prompt": "cat",
        "seed": 99, "aspect_ratio": "1:1", "width": 1024, "height": 1024,
        "llm_raw_response": "raw", "structured_prompt": "sp",
        "image": _make_tiny_image(), "success": True,
    }
    rec = build_metadata_record(dit_result, "/images/5/2.png")
    assert rec["index"] == 5
    assert rec["original_prompt"] == "cat"
    assert rec["image_path"] == "/images/5/2.png"
    assert rec["success"] is True
    assert "image" not in rec  # PIL image should not leak into metadata


def test_scan_existing_images(tmp_path):
    images_dir = tmp_path / "images"
    # Create some fake PNGs.
    (images_dir / "10").mkdir(parents=True)
    (images_dir / "10" / "0.png").write_bytes(b"fake")
    (images_dir / "10" / "1.png").write_bytes(b"fake")
    (images_dir / "20").mkdir()
    (images_dir / "20" / "0.png").write_bytes(b"fake")
    # Non-image file should be ignored.
    (images_dir / "20" / "notes.txt").write_text("ignore me")

    existing = scan_existing_images(str(images_dir))
    assert existing == {(10, 0), (10, 1), (20, 0)}


def test_scan_existing_images_empty(tmp_path):
    assert scan_existing_images(str(tmp_path / "nonexistent")) == set()


def test_count_images_for_index(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "5").mkdir(parents=True)
    for i in range(3):
        _make_tiny_image().save(str(images_dir / "5" / f"{i}.png"))
    assert count_images_for_index(str(images_dir), 5) == 3
    assert count_images_for_index(str(images_dir), 999) == 0


def test_append_metadata(tmp_path):
    path = str(tmp_path / "meta" / "metadata_host-1234.jsonl")
    rec = {"index": 0, "success": True}
    append_metadata(rec, path)
    append_metadata({"index": 1, "success": False}, path)

    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert lines[0]["index"] == 0


# ──────────────────────────────────────────────
# Ranking task emission
# ──────────────────────────────────────────────

def test_build_ranking_task(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "42").mkdir(parents=True)
    for i in range(3):
        _make_tiny_image().save(str(images_dir / "42" / f"{i}.png"))

    task = build_ranking_task(42, "a cat", str(images_dir), "/mnt/shared/rft")
    assert task["index"] == 42
    assert task["query"] == "a cat"
    assert task["root"] == "/mnt/shared/rft"
    assert len(task["images"]) == 3
    assert task["images"][0]["path"] == "images/42/0.png"
    assert task["images"][0]["image_idx"] == 0


def test_check_and_emit_ranking_task_below_threshold(tmp_path):
    images_dir = str(tmp_path / "images")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    # Only 2 images, need 3.
    idx_dir = os.path.join(images_dir, "0")
    os.makedirs(idx_dir)
    for i in range(2):
        _make_tiny_image().save(os.path.join(idx_dir, f"{i}.png"))

    result = check_and_emit_ranking_task(
        0, "query", images_dir, "/root", rank_in_pending, 3,
    )
    assert result is False
    assert os.listdir(rank_in_pending) == []


def test_check_and_emit_ranking_task_at_threshold(tmp_path):
    images_dir = str(tmp_path / "images")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    idx_dir = os.path.join(images_dir, "0")
    os.makedirs(idx_dir)
    for i in range(3):
        _make_tiny_image().save(os.path.join(idx_dir, f"{i}.png"))

    result = check_and_emit_ranking_task(
        0, "query", images_dir, "/root", rank_in_pending, 3,
    )
    assert result is True
    assert "idx_0.json" in os.listdir(rank_in_pending)

    # Verify task contents.
    with open(os.path.join(rank_in_pending, "idx_0.json")) as f:
        task = json.loads(f.read())
    assert task["index"] == 0
    assert len(task["images"]) == 3
    assert task["root"] == "/root"


def test_ranking_task_race_two_workers(tmp_path):
    """Two threads try to emit the same ranking task — exactly one wins,
    the other catches FileExistsError gracefully (no crash)."""
    images_dir = str(tmp_path / "images")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    idx_dir = os.path.join(images_dir, "0")
    os.makedirs(idx_dir)
    for i in range(3):
        _make_tiny_image().save(os.path.join(idx_dir, f"{i}.png"))

    results = []
    errors = []

    def worker():
        try:
            r = check_and_emit_ranking_task(
                0, "query", images_dir, "/root", rank_in_pending, 3,
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No crashes.
    assert errors == []
    # All workers should report True (task emitted or already existed).
    assert all(r is True for r in results)
    # Exactly one file in pending/.
    files = os.listdir(rank_in_pending)
    assert files == ["idx_0.json"]


# ──────────────────────────────────────────────
# process_shard integration tests
# ──────────────────────────────────────────────

def test_process_shard_all_succeed(tmp_path):
    """Process a shard where all records succeed."""
    images_dir = str(tmp_path / "images")
    metadata_path = str(tmp_path / "metadata_test-1.jsonl")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    # 1 index, 3 images → should trigger ranking task.
    records = [make_shard_record(0, i) for i in range(3)]
    shard_path = write_shard(str(tmp_path), "shard.jsonl", records)

    from types import SimpleNamespace
    dit_args = SimpleNamespace(dit_backend="test")

    stats = process_shard(
        shard_path=shard_path,
        images_dir=images_dir,
        metadata_path=metadata_path,
        rank_in_pending=rank_in_pending,
        shared_root=str(tmp_path),
        num_images_per_prompt=3,
        dit_args=dit_args,
        dit_api_bases=None,
        existing_images=set(),
        run_dit_fn=mock_run_dit,
    )

    assert stats["processed"] == 3
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert stats["ranking_tasks_emitted"] >= 1

    # Verify PNGs exist.
    for i in range(3):
        assert os.path.exists(os.path.join(images_dir, "0", f"{i}.png"))

    # Verify metadata.
    with open(metadata_path) as f:
        meta_lines = [json.loads(l) for l in f if l.strip()]
    assert len(meta_lines) == 3

    # Verify ranking task.
    assert "idx_0.json" in os.listdir(rank_in_pending)


def test_process_shard_with_resume(tmp_path):
    """Pre-place some PNGs; only missing ones should be generated."""
    images_dir = str(tmp_path / "images")
    metadata_path = str(tmp_path / "metadata_test-1.jsonl")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    # Pre-place images for image_idx 0 and 1.
    idx_dir = os.path.join(images_dir, "0")
    os.makedirs(idx_dir)
    for i in range(2):
        _make_tiny_image().save(os.path.join(idx_dir, f"{i}.png"))

    existing = scan_existing_images(images_dir)
    assert len(existing) == 2

    records = [make_shard_record(0, i) for i in range(3)]
    shard_path = write_shard(str(tmp_path), "shard.jsonl", records)

    called_indices = []

    def tracking_dit(llm_result, api_bases, args):
        called_indices.append((llm_result["index"], llm_result["image_idx"]))
        return mock_run_dit(llm_result, api_bases, args)

    from types import SimpleNamespace
    stats = process_shard(
        shard_path=shard_path,
        images_dir=images_dir,
        metadata_path=metadata_path,
        rank_in_pending=rank_in_pending,
        shared_root=str(tmp_path),
        num_images_per_prompt=3,
        dit_args=SimpleNamespace(dit_backend="test"),
        dit_api_bases=None,
        existing_images=existing,
        run_dit_fn=tracking_dit,
    )

    # Only image_idx=2 should have been called.
    assert called_indices == [(0, 2)]
    assert stats["skipped"] == 2
    assert stats["processed"] == 1
    # Ranking task should be emitted (3 images now exist).
    assert stats["ranking_tasks_emitted"] >= 1


def test_process_shard_partial_failure(tmp_path):
    """Some LLM records have success=false; those skip DiT but still count."""
    images_dir = str(tmp_path / "images")
    metadata_path = str(tmp_path / "metadata_test-1.jsonl")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    records = [
        make_shard_record(0, 0),
        make_shard_record(0, 1),
        make_failed_shard_record(0, 2),  # LLM failed
    ]
    shard_path = write_shard(str(tmp_path), "shard.jsonl", records)

    from types import SimpleNamespace
    stats = process_shard(
        shard_path=shard_path,
        images_dir=images_dir,
        metadata_path=metadata_path,
        rank_in_pending=rank_in_pending,
        shared_root=str(tmp_path),
        num_images_per_prompt=3,
        dit_args=SimpleNamespace(dit_backend="test"),
        dit_api_bases=None,
        existing_images=set(),
        run_dit_fn=mock_run_dit,
    )

    assert stats["processed"] == 2
    assert stats["failed"] == 1
    # All 3 tasks done (2 success + 1 LLM fail). Ranking task IS emitted
    # with the 2 valid images — partial failures still get ranked.
    assert "idx_0.json" in os.listdir(rank_in_pending)

    # Metadata should have 3 records (2 success + 1 failed).
    with open(metadata_path) as f:
        meta = [json.loads(l) for l in f if l.strip()]
    assert len(meta) == 3
    assert sum(1 for m in meta if m["success"]) == 2


def test_process_shard_dit_failure(tmp_path):
    """DiT backend fails for some records."""
    images_dir = str(tmp_path / "images")
    metadata_path = str(tmp_path / "metadata_test-1.jsonl")
    rank_in_pending = str(tmp_path / "rank_in" / "pending")
    os.makedirs(rank_in_pending)

    records = [make_shard_record(0, i) for i in range(5)]
    shard_path = write_shard(str(tmp_path), "shard.jsonl", records)

    from types import SimpleNamespace
    stats = process_shard(
        shard_path=shard_path,
        images_dir=images_dir,
        metadata_path=metadata_path,
        rank_in_pending=rank_in_pending,
        shared_root=str(tmp_path),
        num_images_per_prompt=5,
        dit_args=SimpleNamespace(dit_backend="test"),
        dit_api_bases=None,
        existing_images=set(),
        run_dit_fn=mock_run_dit_failing,  # fails for image_idx==2
    )

    assert stats["processed"] == 4
    assert stats["failed"] == 1
    # All 5 tasks done (4 success + 1 DiT fail). Ranking task IS emitted
    # with 4 valid images — partial failures still get ranked.
    assert "idx_0.json" in os.listdir(rank_in_pending)


def test_metadata_filename_includes_hostname_pid(shared_root):
    """Metadata file should be named metadata_{hostname}-{pid}.jsonl."""
    cfg = make_cfg(shared_root)
    images_dir = cfg["images"]
    os.makedirs(images_dir, exist_ok=True)

    # Write a shard to pending.
    llm_pending = os.path.join(cfg["llm_out"], "pending")
    records = [make_shard_record(0, 0)]
    write_shard(llm_pending, "shard_0000.jsonl", records)

    run_dit_consumer(
        cfg,
        run_dit_fn=mock_run_dit,
        dit_api_bases=None,
        single_pass=True,
    )

    # Check that metadata file has hostname-pid pattern.
    meta_files = [f for f in os.listdir(images_dir)
                  if f.startswith("metadata_") and f.endswith(".jsonl")]
    assert len(meta_files) == 1
    name = meta_files[0]
    assert name.startswith("metadata_")
    assert "-" in name  # hostname-pid
    assert "_" not in name.replace("metadata_", "", 1).replace(".jsonl", "")  # no underscore in worker_id part... actually the worker_id is between metadata_ and .jsonl


def test_run_dit_consumer_full(shared_root):
    """Full integration: write shards → run consumer → verify outputs."""
    cfg = make_cfg(shared_root, num_images=2)

    # Write two shards with different indices.
    llm_pending = os.path.join(cfg["llm_out"], "pending")
    records_s0 = [make_shard_record(0, i) for i in range(2)]
    records_s1 = [make_shard_record(1, i) for i in range(2)]
    write_shard(llm_pending, "shard_0000.jsonl", records_s0)
    write_shard(llm_pending, "shard_0001.jsonl", records_s1)

    run_dit_consumer(
        cfg,
        run_dit_fn=mock_run_dit,
        dit_api_bases=None,
        single_pass=True,
    )

    # Both shards should be in done/.
    llm_done = os.path.join(cfg["llm_out"], "done")
    done_files = sorted(os.listdir(llm_done))
    assert done_files == ["shard_0000.jsonl", "shard_0001.jsonl"]

    # 4 PNGs total.
    images_dir = cfg["images"]
    for idx in (0, 1):
        for img_idx in (0, 1):
            assert os.path.exists(os.path.join(images_dir, str(idx), f"{img_idx}.png"))

    # 2 ranking tasks (one per index).
    rank_in_pending = os.path.join(cfg["rank_in"], "pending")
    rank_files = sorted(os.listdir(rank_in_pending))
    assert rank_files == ["idx_0.json", "idx_1.json"]

    # Verify ranking task contents.
    with open(os.path.join(rank_in_pending, "idx_0.json")) as f:
        task = json.loads(f.read())
    assert task["index"] == 0
    assert task["root"] == str(shared_root)
    assert len(task["images"]) == 2
    assert task["images"][0]["path"] == "images/0/0.png"
