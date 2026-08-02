# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/llm/rollout.py.

All tests use a mocked run_llm_fn and sample_ar_fn so we don't need a
real LLM backend or its external dependencies (json_repair, transformers).
"""

import json
import os
import threading

import pytest

from src.llm.rollout import (
    _stable_hash,
    assemble_shard_payload,
    append_wal,
    compute_seed,
    find_completed_shards,
    load_queries,
    load_wal,
    plan_shards,
    process_shard,
    run_rollout,
    shard_filename,
    wal_path_for_shard,
    make_llm_args,
)
from src.ops.file_queue import emit_pending


# ──────────────────────────────────────────────
# Mock LLM backend
# ──────────────────────────────────────────────

def mock_sample_ar(bucket_size, prompt, seed):
    """Always returns 1024x1024, 1:1."""
    return 1024, 1024, "1:1"


def mock_run_llm(index, image_idx, prompt, seed, args, system_prompt,
                 width, height, ratio_str):
    """Simulate a successful LLM call."""
    return {
        "index": index,
        "image_idx": image_idx,
        "prompt": prompt,
        "seed": seed,
        "aspect_ratio": ratio_str,
        "llm_raw_response": f"<think>reasoning for {index}/{image_idx}</think>structured output",
        "structured_prompt": f"a beautiful image of {prompt}",
        "width": width,
        "height": height,
        "success": True,
    }


def mock_run_llm_with_failures(index, image_idx, prompt, seed, args,
                                system_prompt, width, height, ratio_str):
    """Simulate LLM call that fails for image_idx == 3."""
    if image_idx == 3:
        return {
            "index": index, "image_idx": image_idx, "prompt": prompt,
            "seed": seed, "aspect_ratio": ratio_str,
            "llm_raw_response": None, "structured_prompt": None,
            "width": width, "height": height,
            "success": False, "error": "LLM error: simulated",
        }
    return mock_run_llm(index, image_idx, prompt, seed, args, system_prompt,
                        width, height, ratio_str)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def shared_root(tmp_path):
    """Create a minimal shared root with queries.jsonl."""
    root = tmp_path / "rft_iter0"
    root.mkdir()
    return root


def write_queries(root, n=10):
    """Write n queries to queries.jsonl and return the path."""
    queries_path = root / "queries.jsonl"
    with open(queries_path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"index": i, "query": f"query_{i}"}) + "\n")
    return str(queries_path)


def make_cfg(shared_root, shard_size=4, num_images=3, num_workers=2):
    """Build a config dict for testing."""
    return {
        "shared_root": str(shared_root),
        "queries_path": str(shared_root / "queries.jsonl"),
        "llm_out": str(shared_root / "llm_out"),
        "images": str(shared_root / "images"),
        "rank_in": str(shared_root / "rank_in"),
        "rankings": str(shared_root / "rankings"),
        "shard_size": shard_size,
        "num_images_per_prompt": num_images,
        "seed": 42,
        "llm": {
            "backend": "seed",
            "psm": "test-psm",
            "temperature": 0.9,
            "top_p": 0.95,
            "thinking": True,
            "system_prompt_path": None,
            "system_prompt_role": "system",
            "input_template": "<prompt>",
            "num_workers": num_workers,
            "bucket_size": 2048,
        },
    }


# ──────────────────────────────────────────────
# Unit tests: deterministic functions
# ──────────────────────────────────────────────

def test_stable_hash_deterministic():
    assert _stable_hash("hello") == _stable_hash("hello")
    assert _stable_hash("a") != _stable_hash("b")


def test_compute_seed_deterministic():
    s1 = compute_seed(42, 0)
    s2 = compute_seed(42, 0)
    assert s1 == s2
    assert 0 <= s1 < 2**31


def test_compute_seed_differs_across_pairs():
    seeds = {compute_seed(i, j) for i in range(5) for j in range(5)}
    # All 25 pairs should produce distinct seeds (collision is astronomically unlikely).
    assert len(seeds) == 25


def test_shard_filename():
    assert shard_filename(0) == "shard_0000.jsonl"
    assert shard_filename(42) == "shard_0042.jsonl"


def test_load_queries(shared_root):
    write_queries(shared_root, 5)
    queries = load_queries(str(shared_root / "queries.jsonl"))
    assert len(queries) == 5
    assert queries[0] == {"index": 0, "query": "query_0"}


def test_load_queries_rejects_bad_format(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"index": 0}\n')
    with pytest.raises(ValueError, match="missing"):
        load_queries(str(p))


# ──────────────────────────────────────────────
# Shard planning
# ──────────────────────────────────────────────

def test_plan_shards():
    queries = [{"index": i, "query": f"q{i}"} for i in range(10)]
    shards = plan_shards(queries, num_images_per_prompt=2, shard_size=4)

    # indices 0-3 → shard 0, 4-7 → shard 1, 8-9 → shard 2
    assert set(shards.keys()) == {0, 1, 2}
    assert len(shards[0]) == 4 * 2  # 4 queries × 2 images
    assert len(shards[1]) == 4 * 2
    assert len(shards[2]) == 2 * 2  # 2 queries × 2 images

    # Verify all tasks have required keys.
    for tasks in shards.values():
        for t in tasks:
            assert set(t.keys()) == {"index", "image_idx", "query", "seed"}


def test_plan_shards_deterministic():
    queries = [{"index": i, "query": f"q{i}"} for i in range(5)]
    s1 = plan_shards(queries, 3, 4)
    s2 = plan_shards(queries, 3, 4)
    # Seeds should be identical across runs.
    for sid in s1:
        for t1, t2 in zip(s1[sid], s2[sid]):
            assert t1["seed"] == t2["seed"]


# ──────────────────────────────────────────────
# WAL operations
# ──────────────────────────────────────────────

def test_wal_roundtrip(tmp_path):
    wal = str(tmp_path / "wal" / "shard_0000.partial.jsonl")
    record = {"index": 0, "image_idx": 1, "success": True, "prompt": "test"}
    append_wal(wal, record)
    append_wal(wal, {"index": 0, "image_idx": 2, "success": True, "prompt": "test2"})

    loaded = load_wal(wal)
    assert len(loaded) == 2
    assert loaded[(0, 1)]["prompt"] == "test"
    assert loaded[(0, 2)]["prompt"] == "test2"


def test_load_wal_nonexistent(tmp_path):
    assert load_wal(str(tmp_path / "no_such_file.jsonl")) == {}


def test_load_wal_skips_corrupt_lines(tmp_path):
    wal = tmp_path / "wal.jsonl"
    wal.write_text(
        '{"index": 0, "image_idx": 0, "success": true}\n'
        'not json\n'
        '{"missing_key": true}\n'
        '{"index": 1, "image_idx": 0, "success": true}\n'
    )
    loaded = load_wal(str(wal))
    assert len(loaded) == 2


# ──────────────────────────────────────────────
# Shard assembly
# ──────────────────────────────────────────────

def test_assemble_shard_payload():
    results = {
        (1, 0): {"index": 1, "image_idx": 0, "data": "a"},
        (0, 1): {"index": 0, "image_idx": 1, "data": "b"},
        (0, 0): {"index": 0, "image_idx": 0, "data": "c"},
    }
    payload = assemble_shard_payload(results)
    lines = payload.decode().strip().split("\n")
    assert len(lines) == 3
    # Should be sorted by (index, image_idx).
    assert json.loads(lines[0])["index"] == 0
    assert json.loads(lines[0])["image_idx"] == 0
    assert json.loads(lines[1])["index"] == 0
    assert json.loads(lines[1])["image_idx"] == 1
    assert json.loads(lines[2])["index"] == 1


# ──────────────────────────────────────────────
# find_completed_shards
# ──────────────────────────────────────────────

def test_find_completed_shards(tmp_path):
    llm_out = tmp_path / "llm_out"
    (llm_out / "pending").mkdir(parents=True)
    (llm_out / "done").mkdir()
    (llm_out / "claimed").mkdir()

    (llm_out / "pending" / "shard_0000.jsonl").write_text("x")
    (llm_out / "done" / "shard_0001.jsonl").write_text("x")
    # Claimed file has worker prefix.
    (llm_out / "claimed" / "host1-1234_shard_0002.jsonl").write_text("x")

    completed = find_completed_shards(str(llm_out))
    assert "shard_0000.jsonl" in completed
    assert "shard_0001.jsonl" in completed
    assert "shard_0002.jsonl" in completed


def test_find_completed_shards_empty(tmp_path):
    assert find_completed_shards(str(tmp_path / "nonexistent")) == set()


# ──────────────────────────────────────────────
# process_shard (integration with mock LLM)
# ──────────────────────────────────────────────

def test_process_shard_fresh(tmp_path):
    """Process a shard from scratch — no WAL, no prior results."""
    llm_out = str(tmp_path / "llm_out")
    llm_args = make_llm_args(make_cfg(tmp_path))

    tasks = [
        {"index": 0, "image_idx": i, "query": "cat", "seed": compute_seed(0, i)}
        for i in range(3)
    ]

    process_shard(
        shard_id=0, tasks=tasks, llm_out_dir=llm_out,
        llm_args=llm_args, system_prompts=[(None, 1.0)], bucket_size=2048,
        num_workers=2, run_llm_fn=mock_run_llm, sample_ar_fn=mock_sample_ar,
    )

    # Shard should be in pending/.
    pending = os.path.join(llm_out, "pending")
    assert "shard_0000.jsonl" in os.listdir(pending)

    # WAL should be cleaned up.
    wal = wal_path_for_shard(llm_out, 0)
    assert not os.path.exists(wal)

    # Verify shard contents.
    with open(os.path.join(pending, "shard_0000.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3
    assert all(r["success"] for r in records)
    assert records[0]["index"] == 0
    assert records[0]["image_idx"] == 0


def test_process_shard_with_failures(tmp_path):
    """LLM failures should be recorded in the shard (success=false)."""
    llm_out = str(tmp_path / "llm_out")
    llm_args = make_llm_args(make_cfg(tmp_path))

    tasks = [
        {"index": 0, "image_idx": i, "query": "cat", "seed": compute_seed(0, i)}
        for i in range(5)
    ]

    process_shard(
        shard_id=0, tasks=tasks, llm_out_dir=llm_out,
        llm_args=llm_args, system_prompts=[(None, 1.0)], bucket_size=2048,
        num_workers=2, run_llm_fn=mock_run_llm_with_failures,
        sample_ar_fn=mock_sample_ar,
    )

    with open(os.path.join(llm_out, "pending", "shard_0000.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 5
    # image_idx=3 should have failed.
    failed = [r for r in records if not r["success"]]
    assert len(failed) == 1
    assert failed[0]["image_idx"] == 3


# ──────────────────────────────────────────────
# Resume from partial WAL
# ──────────────────────────────────────────────

def test_resume_from_wal(tmp_path):
    """If a WAL has partial results, resume should skip those tasks."""
    llm_out = str(tmp_path / "llm_out")
    llm_args = make_llm_args(make_cfg(tmp_path))

    tasks = [
        {"index": 0, "image_idx": i, "query": "cat", "seed": compute_seed(0, i)}
        for i in range(4)
    ]

    # Pre-populate WAL with 2 of 4 tasks.
    wal = wal_path_for_shard(llm_out, 0)
    for i in range(2):
        record = mock_run_llm(0, i, "cat", tasks[i]["seed"], llm_args, None, 1024, 1024, "1:1")
        append_wal(wal, record)

    # Track which tasks actually hit the LLM.
    called = []
    def tracking_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                         w, h, ratio_str):
        called.append((index, image_idx))
        return mock_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                           w, h, ratio_str)

    process_shard(
        shard_id=0, tasks=tasks, llm_out_dir=llm_out,
        llm_args=llm_args, system_prompts=[(None, 1.0)], bucket_size=2048,
        num_workers=2, run_llm_fn=tracking_run_llm,
        sample_ar_fn=mock_sample_ar,
    )

    # Only tasks 2 and 3 should have been called (0 and 1 were in WAL).
    assert sorted(called) == [(0, 2), (0, 3)]

    # Final shard should have all 4.
    with open(os.path.join(llm_out, "pending", "shard_0000.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 4


# ──────────────────────────────────────────────
# Resume skipping completed shards
# ──────────────────────────────────────────────

def test_run_rollout_skips_completed(shared_root):
    """run_rollout should skip shards already in pending/ or done/."""
    write_queries(shared_root, 8)
    cfg = make_cfg(shared_root, shard_size=4, num_images=2, num_workers=2)

    # Pre-create shard_0000.jsonl in done/ to simulate a prior run.
    llm_out = shared_root / "llm_out"
    (llm_out / "done").mkdir(parents=True)
    (llm_out / "done" / "shard_0000.jsonl").write_text('{"dummy": true}\n')

    called_indices = set()
    def tracking_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                         w, h, ratio_str):
        called_indices.add(index)
        return mock_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                           w, h, ratio_str)

    run_rollout(cfg, run_llm_fn=tracking_run_llm, sample_ar_fn=mock_sample_ar)

    # Shard 0 covers indices 0-3. Those should NOT have been called.
    assert called_indices.isdisjoint({0, 1, 2, 3})
    # Shard 1 covers indices 4-7. Those should have been called.
    assert {4, 5, 6, 7}.issubset(called_indices)


def test_run_rollout_full(shared_root):
    """Full rollout from scratch with 6 queries."""
    write_queries(shared_root, 6)
    cfg = make_cfg(shared_root, shard_size=4, num_images=2, num_workers=4)

    run_rollout(cfg, run_llm_fn=mock_run_llm, sample_ar_fn=mock_sample_ar)

    pending = str(shared_root / "llm_out" / "pending")
    files = sorted(os.listdir(pending))
    assert files == ["shard_0000.jsonl", "shard_0001.jsonl"]

    # Shard 0: indices 0-3, 2 images each = 8 records.
    with open(os.path.join(pending, "shard_0000.jsonl")) as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) == 8

    # Shard 1: indices 4-5, 2 images each = 4 records.
    with open(os.path.join(pending, "shard_0001.jsonl")) as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) == 4


def test_run_rollout_idempotent(shared_root):
    """Running rollout twice should not re-emit existing shards."""
    write_queries(shared_root, 4)
    cfg = make_cfg(shared_root, shard_size=4, num_images=2, num_workers=2)

    call_count = [0]
    def counting_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                         w, h, ratio_str):
        call_count[0] += 1
        return mock_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                           w, h, ratio_str)

    run_rollout(cfg, run_llm_fn=counting_run_llm, sample_ar_fn=mock_sample_ar)
    first_count = call_count[0]
    assert first_count == 8  # 4 queries × 2 images

    # Second run should make zero LLM calls — shard is already in pending/.
    run_rollout(cfg, run_llm_fn=counting_run_llm, sample_ar_fn=mock_sample_ar)
    assert call_count[0] == first_count  # No new calls.


# ──────────────────────────────────────────────
# make_llm_args
# ──────────────────────────────────────────────

def test_make_llm_args(tmp_path):
    cfg = make_cfg(tmp_path)
    args = make_llm_args(cfg)
    assert args.llm_backend == "seed"
    assert args.llm_temperature == 0.9
    assert args.thinking is True
    assert args.input_template == "<prompt>"
