# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/judge/rank_bon.py.

All tests mock compare_pair and bradley_terry_mle — we never hit the
real Gemini API. We also mock KeyPool to avoid importing openai/httpx.
"""

import json
import os

import numpy as np
import pytest

from src.judge.rank_bon import (
    append_result,
    filter_valid_samples,
    make_worker_id,
    parse_task,
    process_task,
    quarantine_task,
    rank_index,
    resolve_image_paths,
    run_ranker,
)
from src.ops.file_queue import emit_pending


# ──────────────────────────────────────────────
# Mock Gemini backend
# ──────────────────────────────────────────────

class MockKeyPool:
    """Fake KeyPool that doesn't need openai/httpx."""

    def acquire(self):
        return ("fake-key", None)

    def report_429(self, key):
        pass

    def stats(self):
        return "mock"


def mock_compare_pair(key_pool, prompt, path_a, path_b,
                      model=None, system_prompt=None, input_template=None):
    """Return a deterministic comparison result.

    Image A wins if its path sorts before B (alphabetically).
    """
    if path_a < path_b:
        return {"winner": "A", "score_a": 8.0, "score_b": 5.0, "y": 0.95}
    elif path_a > path_b:
        return {"winner": "B", "score_a": 5.0, "score_b": 8.0, "y": 0.05}
    else:
        return {"winner": "TIE", "score_a": 6.0, "score_b": 6.0, "y": 0.5}


def mock_bradley_terry(n, comparisons, reg=1e-2):
    """Simple mock BT: just count wins per candidate."""
    ratings = np.zeros(n)
    for i, j, y in comparisons:
        if y > 0.5:
            ratings[i] += 1
        elif y < 0.5:
            ratings[j] += 1
    ratings -= ratings.mean()
    return ratings


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def make_task(tmp_path, index=42, n_images=5, query="a cat on a mat"):
    """Create a task JSON and corresponding fake PNG files."""
    shared_root = str(tmp_path / "shared")
    images_dir = os.path.join(shared_root, "images", str(index))
    os.makedirs(images_dir, exist_ok=True)

    images = []
    for i in range(n_images):
        rel_path = f"images/{index}/{i}.png"
        abs_path = os.path.join(shared_root, rel_path)
        with open(abs_path, "wb") as f:
            f.write(b"fake png data")
        images.append({"image_idx": i, "path": rel_path, "success": True})

    task = {
        "index": index,
        "query": query,
        "root": shared_root,
        "images": images,
    }
    return task, shared_root


def write_task_to_pending(tmp_path, task, filename="idx_42.json"):
    """Write a task to rank_in/pending/ and return the pending dir."""
    pending = str(tmp_path / "rank_in" / "pending")
    payload = json.dumps(task, ensure_ascii=False).encode()
    emit_pending(pending, filename, payload)
    return pending


@pytest.fixture
def shared_root(tmp_path):
    root = tmp_path / "rft_iter0"
    root.mkdir()
    return root


def make_cfg(shared_root, bt_reg=0.01):
    return {
        "shared_root": str(shared_root),
        "queries_path": str(shared_root / "queries.jsonl"),
        "llm_out": str(shared_root / "llm_out"),
        "images": str(shared_root / "images"),
        "rank_in": str(shared_root / "rank_in"),
        "rankings": str(shared_root / "rankings"),
        "shard_size": 256,
        "num_images_per_prompt": 5,
        "seed": 42,
        "gemini": {
            "model": "test-model",
            "system_prompt_path": None,
            "input_template": None,
            "api_keys_path": None,
            "num_workers": 4,
            "bt_reg": bt_reg,
            "poll_interval_sec": 1,
        },
    }


# ──────────────────────────────────────────────
# Unit tests: task parsing + path resolution
# ──────────────────────────────────────────────

def test_parse_task(tmp_path):
    task_data = {
        "index": 10, "query": "test", "root": "/shared",
        "images": [{"image_idx": 0, "path": "images/10/0.png", "success": True}],
    }
    path = str(tmp_path / "task.json")
    with open(path, "w") as f:
        json.dump(task_data, f)

    parsed = parse_task(path)
    assert parsed["index"] == 10
    assert len(parsed["images"]) == 1


def test_parse_task_missing_field(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        json.dump({"index": 1}, f)
    with pytest.raises(ValueError, match="missing"):
        parse_task(path)


def test_parse_task_malformed_json(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        f.write("not json {{{")
    with pytest.raises(json.JSONDecodeError):
        parse_task(path)


def test_resolve_image_paths():
    task = {
        "index": 5, "query": "q", "root": "/mnt/shared/rft",
        "images": [
            {"image_idx": 0, "path": "images/5/0.png", "success": True},
            {"image_idx": 1, "path": "images/5/1.png", "success": False},
        ],
    }
    samples = resolve_image_paths(task)
    assert len(samples) == 2
    assert samples[0]["image_path"] == "/mnt/shared/rft/images/5/0.png"
    assert samples[0]["image_idx"] == 0
    assert samples[1]["success"] is False


def test_filter_valid_samples(tmp_path):
    # Create one real file.
    real = str(tmp_path / "real.png")
    with open(real, "wb") as f:
        f.write(b"data")

    samples = [
        {"image_idx": 0, "image_path": real, "success": True},
        {"image_idx": 1, "image_path": str(tmp_path / "missing.png"), "success": True},
        {"image_idx": 2, "image_path": real, "success": False},
    ]
    valid = filter_valid_samples(samples)
    assert len(valid) == 1
    assert valid[0]["image_idx"] == 0


# ──────────────────────────────────────────────
# rank_index
# ──────────────────────────────────────────────

def test_rank_index_normal(tmp_path):
    task, shared_root = make_task(tmp_path, n_images=3)
    samples = resolve_image_paths(task)

    result = rank_index(
        idx=42, samples=samples, prompt="a cat",
        key_pool=MockKeyPool(), gemini_model="test",
        gemini_system_prompt=None, gemini_input_template=None,
        gemini_workers=2, bt_reg=1e-2,
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
    )

    assert result["index"] == 42
    assert result["n_samples"] == 3
    assert result["n_valid"] == 3
    assert len(result["rankings"]) == 3
    assert len(result["pairwise_results"]) == 3  # C(3,2)
    assert result["best_image_idx"] in [0, 1, 2]
    assert "ratings" in result


def test_rank_index_trivial_one_valid(tmp_path):
    """With only 1 valid sample, no comparisons needed."""
    task, _ = make_task(tmp_path, n_images=2)
    samples = resolve_image_paths(task)
    # Remove one image file to make it invalid.
    os.unlink(samples[1]["image_path"])

    result = rank_index(
        idx=42, samples=samples, prompt="q",
        key_pool=MockKeyPool(), gemini_model="test",
        gemini_system_prompt=None, gemini_input_template=None,
        gemini_workers=2, bt_reg=1e-2,
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
    )

    assert result["n_valid"] == 1
    assert result["pairwise_results"] == []
    assert result["best_image_idx"] == 0


def test_rank_index_bt_reg_passed(tmp_path):
    """Verify that bt_reg is actually forwarded to the BT function."""
    task, _ = make_task(tmp_path, n_images=3)
    samples = resolve_image_paths(task)

    captured_reg = []

    def tracking_bt(n, comparisons, reg=1e-3):
        captured_reg.append(reg)
        return mock_bradley_terry(n, comparisons, reg=reg)

    rank_index(
        idx=42, samples=samples, prompt="q",
        key_pool=MockKeyPool(), gemini_model="test",
        gemini_system_prompt=None, gemini_input_template=None,
        gemini_workers=2, bt_reg=0.05,
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=tracking_bt,
    )

    assert captured_reg == [0.05]


# ──────────────────────────────────────────────
# process_task
# ──────────────────────────────────────────────

def test_process_task_fresh(tmp_path):
    task, shared_root = make_task(tmp_path, n_images=3)
    task_path = str(tmp_path / "task.json")
    with open(task_path, "w") as f:
        json.dump(task, f)

    results_path = str(tmp_path / "rankings" / "results_test.jsonl")

    result = process_task(
        task_path=task_path,
        rankings_dir=str(tmp_path / "rankings"),
        results_path=results_path,
        key_pool=MockKeyPool(),
        gemini_model="test",
        gemini_system_prompt=None,
        gemini_input_template=None,
        gemini_workers=2,
        bt_reg=1e-2,
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
    )

    assert result["index"] == 42
    # Result should be appended to file.
    with open(results_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1
    assert lines[0]["index"] == 42


def test_process_task_missing_images(tmp_path):
    """Some images are missing — task still runs with reduced n_valid."""
    task, shared_root = make_task(tmp_path, n_images=5)
    # Delete 2 images.
    for i in (2, 4):
        os.unlink(os.path.join(shared_root, f"images/42/{i}.png"))

    task_path = str(tmp_path / "task.json")
    with open(task_path, "w") as f:
        json.dump(task, f)

    results_path = str(tmp_path / "results.jsonl")
    result = process_task(
        task_path=task_path,
        rankings_dir=str(tmp_path / "rankings"),
        results_path=results_path,
        key_pool=MockKeyPool(),
        gemini_model="test",
        gemini_system_prompt=None,
        gemini_input_template=None,
        gemini_workers=2,
        bt_reg=1e-2,
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
    )

    assert result["n_valid"] == 3
    assert result["n_samples"] == 5


# ──────────────────────────────────────────────
# quarantine
# ──────────────────────────────────────────────

def test_quarantine_task(tmp_path):
    """Quarantine receives a claimed path (with worker prefix) and strips it."""
    # Simulate a claimed file: worker_id prefix + original name.
    task_path = str(tmp_path / "host-1234_idx_bad.json")
    with open(task_path, "w") as f:
        f.write("not valid json")

    quarantine_dir = str(tmp_path / "quarantine")
    quarantine_task(task_path, quarantine_dir, "parse error")

    assert not os.path.exists(task_path)
    # Worker prefix should be stripped, restoring original filename.
    assert os.path.exists(os.path.join(quarantine_dir, "idx_bad.json"))
    assert os.path.exists(os.path.join(quarantine_dir, "idx_bad.json.error"))


# ──────────────────────────────────────────────
# Output filename
# ──────────────────────────────────────────────

def test_results_filename_includes_hostname_pid(tmp_path):
    wid = make_worker_id()
    results_path = os.path.join(str(tmp_path), f"results_{wid}.jsonl")
    append_result({"index": 0}, results_path)

    assert os.path.exists(results_path)
    assert f"results_{wid}" in results_path
    assert "-" in wid
    assert "_" not in wid


# ──────────────────────────────────────────────
# run_ranker integration
# ──────────────────────────────────────────────

def test_run_ranker_multiple_tasks(shared_root):
    cfg = make_cfg(shared_root)

    # Create fake images for 3 indices.
    images_base = str(shared_root / "images")
    tasks = []
    for idx in range(3):
        images = []
        idx_dir = os.path.join(images_base, str(idx))
        os.makedirs(idx_dir, exist_ok=True)
        for i in range(3):
            rel_path = f"images/{idx}/{i}.png"
            abs_path = os.path.join(str(shared_root), rel_path)
            with open(abs_path, "wb") as f:
                f.write(b"fake")
            images.append({"image_idx": i, "path": rel_path, "success": True})
        tasks.append({
            "index": idx, "query": f"query_{idx}",
            "root": str(shared_root), "images": images,
        })

    # Write tasks to pending.
    rank_pending = os.path.join(cfg["rank_in"], "pending")
    for t in tasks:
        emit_pending(rank_pending, f"idx_{t['index']}.json",
                     json.dumps(t).encode())

    run_ranker(
        cfg,
        key_pool=MockKeyPool(),
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
        single_pass=True,
    )

    # All tasks should be in done.
    rank_done = os.path.join(cfg["rank_in"], "done")
    done_files = sorted(os.listdir(rank_done))
    assert done_files == ["idx_0.json", "idx_1.json", "idx_2.json"]

    # Pending should be empty.
    assert os.listdir(rank_pending) == []

    # Results file should have 3 lines.
    results_files = [f for f in os.listdir(cfg["rankings"])
                     if f.startswith("results_") and f.endswith(".jsonl")]
    assert len(results_files) == 1
    with open(os.path.join(cfg["rankings"], results_files[0])) as f:
        results = [json.loads(l) for l in f if l.strip()]
    assert len(results) == 3
    indices = {r["index"] for r in results}
    assert indices == {0, 1, 2}


def test_run_ranker_malformed_task_quarantined(shared_root):
    """Malformed task JSON is quarantined, not infinite-looped."""
    cfg = make_cfg(shared_root)

    # Write a malformed task.
    rank_pending = os.path.join(cfg["rank_in"], "pending")
    emit_pending(rank_pending, "idx_bad.json", b"not json {{{")

    run_ranker(
        cfg,
        key_pool=MockKeyPool(),
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
        single_pass=True,
    )

    # Should be in quarantine, not pending or claimed.
    quarantine_dir = os.path.join(cfg["rank_in"], "quarantine")
    assert os.path.exists(os.path.join(quarantine_dir, "idx_bad.json"))
    assert os.path.exists(os.path.join(quarantine_dir, "idx_bad.json.error"))
    assert os.listdir(rank_pending) == []


def test_run_ranker_resume_skips_done(shared_root):
    """Tasks already in done/ should not be re-processed."""
    cfg = make_cfg(shared_root)

    # Pre-create a task in done/.
    rank_done = os.path.join(cfg["rank_in"], "done")
    os.makedirs(rank_done, exist_ok=True)
    with open(os.path.join(rank_done, "idx_0.json"), "w") as f:
        f.write("{}")

    # Put a new task in pending.
    images_base = str(shared_root / "images")
    idx_dir = os.path.join(images_base, "1")
    os.makedirs(idx_dir)
    for i in range(3):
        with open(os.path.join(idx_dir, f"{i}.png"), "wb") as f:
            f.write(b"fake")

    task = {
        "index": 1, "query": "q1", "root": str(shared_root),
        "images": [{"image_idx": i, "path": f"images/1/{i}.png", "success": True}
                   for i in range(3)],
    }
    rank_pending = os.path.join(cfg["rank_in"], "pending")
    emit_pending(rank_pending, "idx_1.json", json.dumps(task).encode())

    run_ranker(
        cfg,
        key_pool=MockKeyPool(),
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=mock_bradley_terry,
        single_pass=True,
    )

    # Task for index 1 should now be in done (alongside pre-existing idx_0).
    done_files = sorted(os.listdir(rank_done))
    assert "idx_0.json" in done_files
    assert "idx_1.json" in done_files

    # Results file should have 1 line (only index 1 was processed).
    results_files = [f for f in os.listdir(cfg["rankings"])
                     if f.startswith("results_") and f.endswith(".jsonl")]
    assert len(results_files) == 1
    with open(os.path.join(cfg["rankings"], results_files[0])) as f:
        results = [json.loads(l) for l in f if l.strip()]
    assert len(results) == 1
    assert results[0]["index"] == 1
