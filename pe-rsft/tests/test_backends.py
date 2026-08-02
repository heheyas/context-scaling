# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for serving backends and multi-node support.

All tests mock HTTP calls — no real vLLM or DiT serving is started.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from src.llm.backends.vllm import (
    run_llm_vllm,
    resolve_target_size,
    bucket_align_size,
    _parse_ratio_str,
)
from src.dit.backends.native import run_dit_native, check_health
from src.llm.rollout import run_rollout
from src.ops.monitor import (
    count_pngs,
    count_result_lines,
    count_lines,
    format_status,
)


# ──────────────────────────────────────────────
# vLLM backend tests
# ──────────────────────────────────────────────

def _make_vllm_args(**overrides):
    defaults = dict(
        input_template="<prompt>",
        llm_temperature=0.9,
        llm_top_p=0.95,
        thinking=True,
        vllm_url="http://localhost:8000",
        vllm_model="test-model",
        bucket_size=2048,
        max_tokens=65536,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_vllm_response(content, reasoning=None):
    """Build a mock requests.Response for /v1/chat/completions."""
    msg = {"content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    data = {
        "choices": [{"message": msg}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


@patch("src.llm.backends.vllm.requests.post")
def test_vllm_backend_success(mock_post):
    """vLLM backend returns correct schema on success."""
    content = '```json\n{"output": "a beautiful cat"}\n```'
    mock_post.return_value = _mock_vllm_response(content, reasoning="I should expand this")

    args = _make_vllm_args()
    result = run_llm_vllm(0, 1, "a cat", 42, args, "You are helpful.", 1024, 1024, "1:1")

    assert result["success"] is True
    assert result["index"] == 0
    assert result["image_idx"] == 1
    assert result["prompt"] == "a cat"
    assert result["seed"] == 42
    # structured_prompt is the raw json_repair-parsed content (without ratio key → kept as-is).
    assert "a beautiful cat" in result["structured_prompt"]
    assert result["llm_raw_response"].startswith("<think>")
    assert result["width"] == 1024
    assert result["height"] == 1024


@patch("src.llm.backends.vllm.requests.post")
def test_vllm_backend_with_ratio_override(mock_post):
    """vLLM backend handles ratio override in LLM response."""
    content = '```json\n{"output": "wide landscape", "ratio": "16:9"}\n```'
    mock_post.return_value = _mock_vllm_response(content)

    args = _make_vllm_args(thinking=False)
    result = run_llm_vllm(0, 0, "landscape", 42, args, None, 1024, 1024, "1:1")

    assert result["success"] is True
    assert "wide landscape" in result["structured_prompt"]
    # Width should be wider than height due to 16:9 ratio.
    assert result["width"] > result["height"]


@patch("src.llm.backends.vllm.requests.post")
def test_vllm_backend_failure(mock_post):
    """vLLM backend returns success=false on HTTP error."""
    mock_post.side_effect = Exception("Connection refused")

    args = _make_vllm_args()
    result = run_llm_vllm(0, 0, "test", 42, args, None, 1024, 1024, "1:1")

    assert result["success"] is False
    assert "error" in result
    assert result["llm_raw_response"] is None


@patch("src.llm.backends.vllm.requests.post")
def test_vllm_backend_template_substitution(mock_post):
    """vLLM backend performs input template substitution."""
    content = '```json\n{"output": "result"}\n```'
    mock_post.return_value = _mock_vllm_response(content)

    args = _make_vllm_args(
        input_template="<prompt> [width: <width>, height: <height>]",
        thinking=False,
    )
    run_llm_vllm(0, 0, "cat", 42, args, "sys", 512, 768, "2:3")

    # Verify the template was substituted in the API call.
    call_body = mock_post.call_args[1]["json"]
    user_msg = call_body["messages"][-1]["content"]
    assert "cat" in user_msg
    assert "512" in user_msg
    assert "768" in user_msg


def test_resolve_target_size_basic():
    w, h = resolve_target_size(ratio_spec="1:1", bucket_size=1024, factor=32)
    assert w == h
    assert w > 0

    w, h = resolve_target_size(ratio_spec="16:9", bucket_size=1024, factor=32)
    assert w > h

    w, h = resolve_target_size(ratio_spec=None, bucket_size=1024, factor=32)
    assert w == 1024 and h == 1024


def test_parse_ratio_str():
    assert _parse_ratio_str("16:9") == (16.0, 9.0)
    assert _parse_ratio_str("1:1") == (1.0, 1.0)
    assert _parse_ratio_str("") is None
    assert _parse_ratio_str("bad") is None


# ──────────────────────────────────────────────
# Native DiT backend tests
# ──────────────────────────────────────────────

def _make_dit_args(**overrides):
    defaults = dict(
        dit_backend="native",
        num_steps=25,
        cfg_scale=4.0,
        negative_prompt="",
        timeout=10,
        max_retries=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_png_response():
    """Build a mock response containing a tiny PNG."""
    import io
    img = Image.new("RGB", (4, 4), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = MagicMock()
    resp.status_code = 200
    resp.content = buf.getvalue()
    resp.raise_for_status = MagicMock()
    return resp


@patch("src.dit.backends.native.requests.post")
def test_native_dit_success(mock_post):
    """Native DiT returns PIL image on success."""
    mock_post.return_value = _mock_png_response()

    llm_result = {
        "index": 0, "image_idx": 1, "prompt": "cat",
        "structured_prompt": "beautiful cat", "seed": 42,
        "width": 1024, "height": 1024, "aspect_ratio": "1:1",
        "llm_raw_response": "raw", "success": True,
    }
    args = _make_dit_args()
    result = run_dit_native(llm_result, "http://localhost:8091", args)

    assert result["success"] is True
    assert result["image"] is not None
    assert isinstance(result["image"], Image.Image)


@patch("src.dit.backends.native.requests.post")
def test_native_dit_failure(mock_post):
    """Native DiT returns success=false on HTTP error."""
    mock_post.side_effect = Exception("Connection refused")

    llm_result = {
        "index": 0, "image_idx": 0, "prompt": "cat",
        "structured_prompt": "sp", "seed": 42,
        "width": 1024, "height": 1024, "aspect_ratio": "1:1",
        "llm_raw_response": "raw", "success": True,
    }
    args = _make_dit_args()
    result = run_dit_native(llm_result, "http://localhost:8091", args)

    assert result["success"] is False
    assert "error" in result


def test_native_dit_skips_failed_llm():
    """Native DiT skips records where LLM failed."""
    llm_result = {"index": 0, "image_idx": 0, "success": False, "prompt": "x"}
    result = run_dit_native(llm_result, "http://localhost:8091", _make_dit_args())
    assert result["image"] is None


@patch("src.dit.backends.native.requests.get")
def test_check_health(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    assert check_health("http://localhost:8091") is True

    mock_get.side_effect = Exception("refused")
    assert check_health("http://localhost:8091") is False


# ──────────────────────────────────────────────
# Multi-node shard partitioning
# ──────────────────────────────────────────────

def _mock_run_llm(index, image_idx, prompt, seed, args, sys_prompt,
                  w, h, ratio_str):
    return {
        "index": index, "image_idx": image_idx, "prompt": prompt,
        "seed": seed, "aspect_ratio": ratio_str,
        "llm_raw_response": "raw", "structured_prompt": "sp",
        "width": w, "height": h, "success": True,
    }


def _mock_sample_ar(bucket_size, prompt, seed):
    return 1024, 1024, "1:1"


def test_rank_world_size_partitioning(tmp_path):
    """With world_size=3, each rank processes disjoint shards."""
    root = tmp_path / "rft"
    root.mkdir()

    # 12 queries → shard_size=4 → 3 shards (0, 1, 2).
    queries_path = root / "queries.jsonl"
    with open(queries_path, "w") as f:
        for i in range(12):
            f.write(json.dumps({"index": i, "query": f"q{i}"}) + "\n")

    base_cfg = {
        "shared_root": str(root),
        "queries_path": str(queries_path),
        "llm_out": str(root / "llm_out"),
        "images": str(root / "images"),
        "rank_in": str(root / "rank_in"),
        "rankings": str(root / "rankings"),
        "shard_size": 4,
        "num_images_per_prompt": 1,
        "seed": 42,
        "llm": {
            "backend": "seed", "psm": "test",
            "temperature": 0.9, "top_p": 0.95, "thinking": False,
            "system_prompt_path": None, "system_prompt_role": "system",
            "input_template": "<prompt>", "num_workers": 2, "bucket_size": 2048,
        },
    }

    # Run each rank.
    for rank in range(3):
        run_rollout(base_cfg, run_llm_fn=_mock_run_llm,
                    sample_ar_fn=_mock_sample_ar,
                    rank=rank, world_size=3)

    # All 3 shards should be in pending.
    pending = str(root / "llm_out" / "pending")
    files = sorted(os.listdir(pending))
    assert files == ["shard_0000.jsonl", "shard_0001.jsonl", "shard_0002.jsonl"]


def test_rank_world_size_no_overlap(tmp_path):
    """Two ranks should produce disjoint shards."""
    root = tmp_path / "rft"
    root.mkdir()

    queries_path = root / "queries.jsonl"
    with open(queries_path, "w") as f:
        for i in range(8):
            f.write(json.dumps({"index": i, "query": f"q{i}"}) + "\n")

    cfg = {
        "shared_root": str(root),
        "queries_path": str(queries_path),
        "llm_out": str(root / "llm_out"),
        "images": str(root / "images"),
        "rank_in": str(root / "rank_in"),
        "rankings": str(root / "rankings"),
        "shard_size": 4,
        "num_images_per_prompt": 1,
        "seed": 42,
        "llm": {
            "backend": "seed", "psm": "t",
            "temperature": 0.9, "top_p": 0.95, "thinking": False,
            "system_prompt_path": None, "system_prompt_role": "system",
            "input_template": "<prompt>", "num_workers": 2, "bucket_size": 2048,
        },
    }

    # Rank 0 gets shard 0 (indices 0-3). Rank 1 gets shard 1 (indices 4-7).
    run_rollout(cfg, run_llm_fn=_mock_run_llm, sample_ar_fn=_mock_sample_ar,
                rank=0, world_size=2)

    pending = str(root / "llm_out" / "pending")
    assert os.listdir(pending) == ["shard_0000.jsonl"]

    run_rollout(cfg, run_llm_fn=_mock_run_llm, sample_ar_fn=_mock_sample_ar,
                rank=1, world_size=2)

    files = sorted(os.listdir(pending))
    assert files == ["shard_0000.jsonl", "shard_0001.jsonl"]


# ──────────────────────────────────────────────
# Monitor tests
# ──────────────────────────────────────────────

def test_count_pngs(tmp_path):
    images_dir = tmp_path / "images"
    (images_dir / "0").mkdir(parents=True)
    (images_dir / "1").mkdir()
    for i in range(3):
        (images_dir / "0" / f"{i}.png").write_bytes(b"x")
    (images_dir / "1" / "0.png").write_bytes(b"x")

    assert count_pngs(str(images_dir)) == 4
    assert count_pngs(str(tmp_path / "nonexistent")) == 0


def test_count_result_lines(tmp_path):
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    (rankings / "results_a.jsonl").write_text('{"a":1}\n{"b":2}\n')
    (rankings / "results_b.jsonl").write_text('{"c":3}\n')
    (rankings / "other.txt").write_text("ignore\n")

    assert count_result_lines(str(rankings)) == 3


def test_count_lines(tmp_path):
    f = tmp_path / "queries.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n\n{"c":3}\n')
    assert count_lines(str(f)) == 3
    assert count_lines(str(tmp_path / "nonexistent")) == 0


def test_format_status(tmp_path):
    root = tmp_path / "rft"
    root.mkdir()

    # Create minimal structure.
    for sub in ("llm_out/pending", "llm_out/claimed", "llm_out/done",
                "rank_in/pending", "rank_in/claimed", "rank_in/done",
                "images", "rankings"):
        (root / sub).mkdir(parents=True)

    (root / "queries.jsonl").write_text('{"index":0,"query":"q"}\n')

    cfg = {
        "llm_out": str(root / "llm_out"),
        "rank_in": str(root / "rank_in"),
        "images": str(root / "images"),
        "rankings": str(root / "rankings"),
        "queries_path": str(root / "queries.jsonl"),
        "num_images_per_prompt": 5,
    }

    status = format_status(cfg)
    assert "LLM:" in status
    assert "Images:" in status
    assert "Ranked:" in status
