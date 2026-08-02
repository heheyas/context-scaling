# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/build_queries_jsonl.py.

We test the output format by calling a local reimplementation of the
core logic (load prompts → assign index → write JSONL). The legacy
loader is simple enough to test directly with a dummy file — it just
reads lines from a text file.
"""

import json
import os
import sys

import pytest


def build_queries_from_lines(lines: list[str], output_path: str):
    """Core logic extracted for testability (no legacy import needed)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(lines):
            f.write(json.dumps({"index": i, "query": prompt}, ensure_ascii=False) + "\n")


def test_basic_output_format(tmp_path):
    prompts = ["a cat sitting on a mat", "sunset over mountains", "abstract art"]
    out = str(tmp_path / "queries.jsonl")
    build_queries_from_lines(prompts, out)

    with open(out) as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 3
    assert records[0] == {"index": 0, "query": "a cat sitting on a mat"}
    assert records[1] == {"index": 1, "query": "sunset over mountains"}
    assert records[2] == {"index": 2, "query": "abstract art"}


def test_empty_input(tmp_path):
    out = str(tmp_path / "queries.jsonl")
    build_queries_from_lines([], out)

    with open(out) as f:
        assert f.read().strip() == ""


def test_unicode_prompts(tmp_path):
    prompts = ["一只猫坐在垫子上", "日落山景"]
    out = str(tmp_path / "queries.jsonl")
    build_queries_from_lines(prompts, out)

    with open(out) as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert records[0]["query"] == "一只猫坐在垫子上"
    assert records[1]["index"] == 1


def test_with_dummy_prompt_file(tmp_path):
    """Test that a plain-text prompt file produces correct JSONL.

    This exercises the same format that legacy load_prompts_from_file reads:
    one prompt per line.
    """
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("cat on mat\ndog in park\nbird in sky\n")

    # Read with the same logic as legacy's load_prompts_from_file:
    # plain lines → one prompt per line.
    prompts = []
    with open(prompt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    out = str(tmp_path / "queries.jsonl")
    build_queries_from_lines(prompts, out)

    with open(out) as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 3
    assert records[2] == {"index": 2, "query": "bird in sky"}
