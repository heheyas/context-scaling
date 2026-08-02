# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test: Stage 1 → Stage 2 → Stage 3.

Wires all three stages together with mocks to verify interface
compatibility. No real LLM, DiT, or Gemini API calls are made.

Uses real bradley_terry_mle from scipy (available in test env) to
verify the full BT fitting path works end-to-end.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.optimize import minimize

from src.llm.rollout import run_rollout
from src.dit.gen_images import run_dit_consumer
from src.judge.rank_bon import run_ranker


# ──────────────────────────────────────────────
# Real Bradley-Terry MLE (inlined for test independence).
# ──────────────────────────────────────────────

def real_bradley_terry_mle(n, comparisons, reg=1e-2):
    """Real BT MLE (same as legacy gemini_bon.bradley_terry_mle)."""
    if not comparisons:
        return np.zeros(n)

    def neg_log_likelihood(r):
        loss = 0.0
        for i, j, y in comparisons:
            diff = r[i] - r[j]
            sigma = 1.0 / (1.0 + np.exp(-diff))
            sigma = np.clip(sigma, 1e-10, 1 - 1e-10)
            loss -= y * np.log(sigma) + (1 - y) * np.log(1 - sigma)
        loss += reg * np.sum(r ** 2)
        return loss

    def gradient(r):
        grad = np.zeros(n)
        for i, j, y in comparisons:
            diff = r[i] - r[j]
            sigma = 1.0 / (1.0 + np.exp(-diff))
            err = sigma - y
            grad[i] += err
            grad[j] -= err
        grad += 2 * reg * r
        return grad

    r0 = np.zeros(n)
    result = minimize(neg_log_likelihood, r0, jac=gradient, method="L-BFGS-B")
    ratings = result.x
    ratings -= ratings.mean()
    return ratings


# ──────────────────────────────────────────────
# Mock backends
# ──────────────────────────────────────────────

def _make_tiny_image():
    return Image.new("RGB", (4, 4), color="red")


def mock_sample_ar(bucket_size, prompt, seed):
    return 1024, 1024, "1:1"


def make_mock_run_llm(fail_index=None):
    """Factory for mock run_llm. Fails for (fail_index, any image_idx)."""
    def mock_run_llm(index, image_idx, prompt, seed, args, system_prompt,
                     width, height, ratio_str):
        if index == fail_index:
            return {
                "index": index, "image_idx": image_idx, "prompt": prompt,
                "seed": seed, "aspect_ratio": ratio_str,
                "llm_raw_response": None, "structured_prompt": None,
                "width": width, "height": height,
                "success": False, "error": "LLM error: simulated",
            }
        return {
            "index": index, "image_idx": image_idx, "prompt": prompt,
            "seed": seed, "aspect_ratio": ratio_str,
            "llm_raw_response": f"<think>think_{index}_{image_idx}</think>output",
            "structured_prompt": f"beautiful {prompt}",
            "width": width, "height": height, "success": True,
        }
    return mock_run_llm


def make_mock_run_dit(fail_index=None, fail_image_idx=None):
    """Factory for mock run_dit. Fails for a specific (index, image_idx)."""
    def mock_run_dit(llm_result, dit_api_bases, args):
        if not llm_result.get("success", True):
            return {**llm_result, "image": None}
        if (llm_result["index"] == fail_index
                and llm_result["image_idx"] == fail_image_idx):
            return {
                **llm_result, "image": None,
                "success": False, "error": "DiT error: simulated",
            }
        return {**llm_result, "image": _make_tiny_image(), "success": True}
    return mock_run_dit


def mock_compare_pair(key_pool, prompt, path_a, path_b,
                      model=None, system_prompt=None, input_template=None):
    """Deterministic comparison: lower path wins (alphabetical)."""
    if path_a < path_b:
        return {"winner": "A", "score_a": 8.0, "score_b": 5.0, "y": 0.95}
    elif path_a > path_b:
        return {"winner": "B", "score_a": 5.0, "score_b": 8.0, "y": 0.05}
    return {"winner": "TIE", "score_a": 6.0, "score_b": 6.0, "y": 0.5}


class MockKeyPool:
    def acquire(self):
        return ("fake-key", None)
    def report_429(self, key):
        pass
    def stats(self):
        return "mock"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

EXPECTED_RESULT_KEYS = {
    "index", "original_prompt", "n_samples", "n_valid",
    "rankings", "ratings", "pairwise_results",
    "best_image_idx", "best_image_path",
}


def write_queries(root, queries):
    path = os.path.join(str(root), "queries.jsonl")
    with open(path, "w") as f:
        for i, q in enumerate(queries):
            f.write(json.dumps({"index": i, "query": q}) + "\n")
    return path


def make_cfg(root, num_queries, num_images=3, shard_size=256):
    return {
        "shared_root": str(root),
        "queries_path": os.path.join(str(root), "queries.jsonl"),
        "llm_out": os.path.join(str(root), "llm_out"),
        "images": os.path.join(str(root), "images"),
        "rank_in": os.path.join(str(root), "rank_in"),
        "rankings": os.path.join(str(root), "rankings"),
        "shard_size": shard_size,
        "num_images_per_prompt": num_images,
        "seed": 42,
        "llm": {
            "backend": "seed", "psm": "test",
            "temperature": 0.9, "top_p": 0.95, "thinking": True,
            "system_prompt_path": None, "system_prompt_role": "system",
            "input_template": "<prompt>",
            "num_workers": 4, "bucket_size": 2048,
        },
        "dit": {
            "backend": "seedream5p0", "psm": "test",
            "num_workers": 2, "poll_interval_sec": 1,
        },
        "gemini": {
            "model": "test-model", "system_prompt_path": None,
            "input_template": None, "api_keys_path": None,
            "num_workers": 4, "bt_reg": 0.01, "poll_interval_sec": 1,
        },
    }


def count_files(directory, suffix=None):
    """Count files in a directory, optionally filtered by suffix."""
    if not os.path.isdir(directory):
        return 0
    files = [f for f in os.listdir(directory) if not f.startswith(".")]
    if suffix:
        files = [f for f in files if f.endswith(suffix)]
    return len(files)


def count_pngs_recursive(images_dir):
    """Count all .png files under images/."""
    total = 0
    if not os.path.isdir(images_dir):
        return 0
    for idx_dir in os.listdir(images_dir):
        idx_path = os.path.join(images_dir, idx_dir)
        if os.path.isdir(idx_path):
            total += sum(1 for f in os.listdir(idx_path) if f.endswith(".png"))
    return total


def load_all_results(rankings_dir):
    """Load all result lines from rankings/results_*.jsonl files."""
    results = []
    if not os.path.isdir(rankings_dir):
        return results
    for fname in os.listdir(rankings_dir):
        if fname.startswith("results_") and fname.endswith(".jsonl"):
            with open(os.path.join(rankings_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
    return results


# ──────────────────────────────────────────────
# Happy path: all 3 queries succeed end-to-end
# ──────────────────────────────────────────────

def test_e2e_happy_path(tmp_path):
    """3 queries, 3 images each, all succeed through all stages."""
    root = tmp_path / "rft"
    root.mkdir()
    queries = ["a cat on a mat", "sunset over mountains", "abstract art"]
    num_images = 3

    write_queries(root, queries)
    cfg = make_cfg(root, len(queries), num_images=num_images)

    # ── Stage 1: LLM rollout ──
    run_rollout(
        cfg,
        run_llm_fn=make_mock_run_llm(),
        sample_ar_fn=mock_sample_ar,
    )

    # Verify: shards in llm_out/pending/.
    llm_pending = os.path.join(cfg["llm_out"], "pending")
    shard_count = count_files(llm_pending, ".jsonl")
    assert shard_count == 1  # 3 queries / 256 shard_size = 1 shard

    # ── Stage 2: DiT consumer ──
    run_dit_consumer(
        cfg,
        run_dit_fn=make_mock_run_dit(),
        dit_api_bases=None,
        single_pass=True,
    )

    # Verify: shard moved to llm_out/done/.
    llm_done = os.path.join(cfg["llm_out"], "done")
    assert count_files(llm_done, ".jsonl") == 1
    assert count_files(llm_pending, ".jsonl") == 0

    # Verify: PNG files under images/.
    total_pngs = count_pngs_recursive(cfg["images"])
    assert total_pngs == len(queries) * num_images  # 9

    # Verify: each index has the right number of PNGs.
    for idx in range(len(queries)):
        idx_dir = os.path.join(cfg["images"], str(idx))
        assert count_files(idx_dir, ".png") == num_images

    # Verify: ranking tasks emitted.
    rank_pending = os.path.join(cfg["rank_in"], "pending")
    assert count_files(rank_pending, ".json") == len(queries)

    # ── Stage 3: Gemini ranker ──
    run_ranker(
        cfg,
        key_pool=MockKeyPool(),
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=real_bradley_terry_mle,
        single_pass=True,
    )

    # Verify: ranking tasks moved to done.
    rank_done = os.path.join(cfg["rank_in"], "done")
    assert count_files(rank_done, ".json") == len(queries)
    assert count_files(rank_pending, ".json") == 0

    # Verify: results.
    results = load_all_results(cfg["rankings"])
    assert len(results) == len(queries)

    # Verify: every result has the full expected schema.
    for r in results:
        missing = EXPECTED_RESULT_KEYS - set(r.keys())
        assert missing == set(), f"Result for index {r.get('index')} missing keys: {missing}"

    # Verify: index correspondence — each query has exactly one result.
    result_indices = sorted(r["index"] for r in results)
    assert result_indices == list(range(len(queries)))

    # Verify: results have sensible values.
    for r in results:
        assert r["n_samples"] == num_images
        assert r["n_valid"] == num_images
        assert len(r["rankings"]) == num_images
        assert len(r["pairwise_results"]) == 3  # C(3,2)
        assert r["best_image_idx"] in range(num_images)
        assert r["best_image_path"] != ""
        assert r["original_prompt"] in queries

    # Verify: ratings are real numbers, not all zero.
    for r in results:
        ratings = [float(v) for v in r["ratings"].values()]
        assert any(v != 0.0 for v in ratings), f"All ratings zero for index {r['index']}"


# ──────────────────────────────────────────────
# Partial failure: LLM fails for one query, DiT fails for one image
# ──────────────────────────────────────────────

def test_e2e_partial_failures(tmp_path):
    """3 queries, 5 images each. Query 0 has LLM failure. Query 1 has
    DiT failure for image_idx=2. Query 2 succeeds fully."""
    root = tmp_path / "rft"
    root.mkdir()
    queries = ["query_fail_llm", "query_fail_dit", "query_success"]
    num_images = 5

    write_queries(root, queries)
    cfg = make_cfg(root, len(queries), num_images=num_images)

    # ── Stage 1: LLM rollout ──
    # Index 0 fails at LLM level.
    run_rollout(
        cfg,
        run_llm_fn=make_mock_run_llm(fail_index=0),
        sample_ar_fn=mock_sample_ar,
    )

    # Verify: shard exists (even with failures — they're recorded as success=false).
    llm_pending = os.path.join(cfg["llm_out"], "pending")
    assert count_files(llm_pending, ".jsonl") == 1

    # Verify: shard contains success=false records for index 0.
    shard_files = [f for f in os.listdir(llm_pending) if f.endswith(".jsonl")]
    with open(os.path.join(llm_pending, shard_files[0])) as f:
        records = [json.loads(l) for l in f if l.strip()]
    idx0_records = [r for r in records if r["index"] == 0]
    assert all(not r["success"] for r in idx0_records)

    # ── Stage 2: DiT consumer ──
    # Index 1, image_idx=2 fails at DiT level.
    run_dit_consumer(
        cfg,
        run_dit_fn=make_mock_run_dit(fail_index=1, fail_image_idx=2),
        dit_api_bases=None,
        single_pass=True,
    )

    # Verify: shard moved to done.
    llm_done = os.path.join(cfg["llm_out"], "done")
    assert count_files(llm_done, ".jsonl") == 1

    # Index 0: 0 PNGs (all LLM-failed → no DiT calls).
    assert count_files(os.path.join(cfg["images"], "0"), ".png") == 0

    # Index 1: 4 PNGs (1 DiT failure out of 5).
    assert count_files(os.path.join(cfg["images"], "1"), ".png") == 4

    # Index 2: 5 PNGs (all succeed).
    assert count_files(os.path.join(cfg["images"], "2"), ".png") == 5

    # Verify ranking tasks:
    # Index 0: 0 valid images → NO ranking task (need >=2).
    # Index 1: 4 valid images, all 5 tasks done → YES ranking task.
    # Index 2: 5 valid images → YES ranking task.
    rank_pending = os.path.join(cfg["rank_in"], "pending")
    assert count_files(rank_pending, ".json") == 2

    task_files = sorted(os.listdir(rank_pending))
    assert task_files == ["idx_1.json", "idx_2.json"]

    # ── Stage 3: Gemini ranker ──
    run_ranker(
        cfg,
        key_pool=MockKeyPool(),
        compare_pair_fn=mock_compare_pair,
        bradley_terry_fn=real_bradley_terry_mle,
        single_pass=True,
    )

    # Verify: 2 results (index 1 with partial images + index 2 fully).
    results = load_all_results(cfg["rankings"])
    assert len(results) == 2
    result_by_idx = {r["index"]: r for r in results}

    # Index 1: 4 valid out of 5 total.
    assert result_by_idx[1]["n_valid"] == 4
    assert result_by_idx[1]["n_samples"] == 4  # ranking task only has valid images

    # Index 2: all 5 valid.
    assert result_by_idx[2]["n_valid"] == num_images

    # Verify schema for all results.
    for r in results:
        missing = EXPECTED_RESULT_KEYS - set(r.keys())
        assert missing == set(), f"Index {r['index']} missing keys: {missing}"

    # Verify metadata includes failure records.
    meta_files = [f for f in os.listdir(cfg["images"])
                  if f.startswith("metadata_") and f.endswith(".jsonl")]
    assert len(meta_files) >= 1
    all_meta = []
    for mf in meta_files:
        with open(os.path.join(cfg["images"], mf)) as f:
            all_meta.extend(json.loads(l) for l in f if l.strip())

    # Index 0: 5 records, all failed (LLM failure passed through).
    idx0_meta = [m for m in all_meta if m["index"] == 0]
    assert len(idx0_meta) == num_images
    assert all(not m["success"] for m in idx0_meta)

    # Index 1: 5 records (4 success + 1 DiT failure).
    idx1_meta = [m for m in all_meta if m["index"] == 1]
    assert len(idx1_meta) == num_images
    assert sum(1 for m in idx1_meta if m["success"]) == 4
    assert sum(1 for m in idx1_meta if not m["success"]) == 1
