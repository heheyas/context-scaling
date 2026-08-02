# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/ops/janitor.py."""

import os
import time

import pytest

from src.ops.file_queue import claim, emit_pending
from src.ops.janitor import run_janitor_pass


@pytest.fixture
def queue_root(tmp_path):
    for sub in ("pending", "claimed", "done"):
        (tmp_path / sub).mkdir()
    return tmp_path


def test_janitor_pass_no_stale(queue_root):
    """Janitor should report depths but re-park nothing when nothing is stale."""
    emit_pending(str(queue_root / "pending"), "f.json", b"data")
    summary = run_janitor_pass([str(queue_root)], t_stale_sec=600)
    info = summary[str(queue_root)]
    assert info["depths"]["pending"] == 1
    assert info["reparked"] == 0


def test_janitor_pass_reparks_stale(queue_root):
    """Janitor should re-park stale claimed files."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    emit_pending(pending, "s0.jsonl", b"d0")
    emit_pending(pending, "s1.jsonl", b"d1")

    # Claim both.
    p0 = claim(pending, claimed, "w0")
    p1 = claim(pending, claimed, "w1")

    # Backdate only one to make it stale.
    old = time.time() - 700
    os.utime(p0, (old, old))

    summary = run_janitor_pass([str(queue_root)], t_stale_sec=600)
    info = summary[str(queue_root)]
    assert info["reparked"] == 1

    # Stale file should be back in pending.
    pending_files = os.listdir(pending)
    assert "s0.jsonl" in pending_files

    # Non-stale file should still be claimed.
    claimed_files = os.listdir(claimed)
    assert len(claimed_files) == 1


def test_janitor_multiple_roots(tmp_path):
    """Janitor should handle multiple queue roots."""
    roots = []
    for name in ("llm_out", "rank_in"):
        root = tmp_path / name
        for sub in ("pending", "claimed", "done"):
            (root / sub).mkdir(parents=True)
        emit_pending(str(root / "pending"), "f.json", b"x")
        roots.append(str(root))

    summary = run_janitor_pass(roots, t_stale_sec=600)
    assert len(summary) == 2
    for root in roots:
        assert summary[root]["depths"]["pending"] == 1
