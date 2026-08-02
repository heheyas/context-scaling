# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for src/ops/file_queue.py."""

import os
import json
import time
import threading
import tempfile

import pytest

from src.ops.file_queue import (
    claim,
    emit_pending,
    list_stale_claims,
    queue_depth,
    release_to_done,
    release_to_pending,
)


@pytest.fixture
def queue_root(tmp_path):
    """Create a queue root with pending/claimed/done subdirs."""
    for sub in ("pending", "claimed", "done"):
        (tmp_path / sub).mkdir()
    return tmp_path


# ──────────────────────────────────────────────
# Basic claim → done flow
# ──────────────────────────────────────────────


def test_basic_claim_and_done(queue_root):
    """Emit a file, claim it, release to done. Verify contents survive."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")
    done = str(queue_root / "done")

    payload = json.dumps({"index": 42, "query": "a cat"}).encode()
    emit_pending(pending, "shard_0000.jsonl", payload)

    # File should be in pending.
    assert os.listdir(pending) == ["shard_0000.jsonl"]

    # Claim it.
    path = claim(pending, claimed, "worker0")
    assert path is not None
    assert os.path.exists(path)
    assert os.listdir(pending) == []

    # Contents intact.
    with open(path, "rb") as f:
        assert f.read() == payload

    # Release to done.
    done_path = release_to_done(path, done)
    assert os.path.exists(done_path)
    assert os.path.basename(done_path) == "shard_0000.jsonl"
    assert os.listdir(claimed) == []

    with open(done_path, "rb") as f:
        assert f.read() == payload


def test_claim_returns_none_when_empty(queue_root):
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")
    assert claim(pending, claimed, "w") is None


def test_claim_returns_none_when_dir_missing(tmp_path):
    assert claim(str(tmp_path / "nonexistent"), str(tmp_path / "c"), "w") is None


def test_claim_skips_hidden_files(queue_root):
    """Hidden files (from in-progress emits) should be ignored."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    # Simulate an in-progress temp file.
    with open(os.path.join(pending, ".tmp_emit_xyz"), "w") as f:
        f.write("partial")

    assert claim(pending, claimed, "w") is None


# ──────────────────────────────────────────────
# Concurrent claim race
# ──────────────────────────────────────────────


def test_concurrent_claim_no_double_claim(queue_root):
    """8 threads racing for 4 files: exactly 4 claims, no double-claims."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    num_files = 4
    num_workers = 8

    for i in range(num_files):
        emit_pending(pending, f"shard_{i:04d}.jsonl", f"data_{i}".encode())

    results = []
    lock = threading.Lock()

    def worker(wid):
        claimed_paths = []
        while True:
            path = claim(pending, claimed, f"w{wid}")
            if path is None:
                break
            claimed_paths.append(path)
        with lock:
            results.extend(claimed_paths)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 4 files should have been claimed total.
    assert len(results) == num_files

    # All claimed files should actually exist.
    for p in results:
        assert os.path.exists(p)

    # Original names should all be distinct (no double-claim).
    basenames = [os.path.basename(p).split("_", 1)[1] for p in results]
    assert len(set(basenames)) == num_files

    # pending should be empty.
    assert os.listdir(pending) == []


def test_concurrent_claim_many_files(queue_root):
    """Stress test: 8 threads, 32 files. All claimed exactly once."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    num_files = 32
    num_workers = 8

    for i in range(num_files):
        emit_pending(pending, f"f_{i:04d}.json", f"payload_{i}".encode())

    results = []
    lock = threading.Lock()

    def worker(wid):
        mine = []
        while True:
            path = claim(pending, claimed, f"w{wid}")
            if path is None:
                break
            mine.append(path)
        with lock:
            results.extend(mine)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == num_files
    assert os.listdir(pending) == []


# ──────────────────────────────────────────────
# Stale claim detection and re-parking
# ──────────────────────────────────────────────


def test_list_stale_claims(queue_root):
    claimed = str(queue_root / "claimed")

    # Create two files, backdate one.
    fresh = os.path.join(claimed, "w0_shard_0000.jsonl")
    stale = os.path.join(claimed, "w1_shard_0001.jsonl")
    for p in (fresh, stale):
        with open(p, "w") as f:
            f.write("x")

    # Backdate stale file by 20 minutes.
    old_time = time.time() - 1200
    os.utime(stale, (old_time, old_time))

    stale_list = list_stale_claims(claimed, t_stale_sec=600)
    assert len(stale_list) == 1
    assert stale_list[0] == stale


def test_list_stale_claims_empty(tmp_path):
    assert list_stale_claims(str(tmp_path / "nonexistent"), 60) == []


def test_release_to_pending(queue_root):
    """Janitor re-parks a stale claimed file back to pending."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    # Simulate a claimed file.
    claimed_path = os.path.join(claimed, "w0_shard_0042.jsonl")
    with open(claimed_path, "w") as f:
        f.write("data")

    result = release_to_pending(claimed_path, pending)
    assert os.path.basename(result) == "shard_0042.jsonl"
    assert os.path.exists(result)
    assert not os.path.exists(claimed_path)


def test_janitor_full_flow(queue_root):
    """End-to-end: emit, claim, let it go stale, janitor re-parks, re-claim."""
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    emit_pending(pending, "task.json", b'{"idx":1}')
    path = claim(pending, claimed, "worker_crash")
    assert path is not None

    # Simulate staleness by backdating mtime.
    old_time = time.time() - 700
    os.utime(path, (old_time, old_time))

    stale = list_stale_claims(claimed, t_stale_sec=600)
    assert len(stale) == 1

    reparked = release_to_pending(stale[0], pending)
    assert os.path.exists(reparked)
    assert os.listdir(claimed) == []

    # Now another worker can claim it.
    path2 = claim(pending, claimed, "worker_ok")
    assert path2 is not None
    with open(path2, "rb") as f:
        assert json.loads(f.read()) == {"idx": 1}


# ──────────────────────────────────────────────
# Atomic emit: partial writes never visible
# ──────────────────────────────────────────────


def test_emit_is_atomic(queue_root):
    """While emit is writing, the file should not appear in pending/."""
    pending = str(queue_root / "pending")
    seen_partial = []

    # We can't truly freeze mid-write, but we can verify that:
    # 1) Before emit returns, only temp files (dot-prefixed) or the final
    #    file exist.
    # 2) After emit returns, the final file has the full payload.

    large_payload = b"x" * (1024 * 1024)  # 1 MB

    # Repeatedly list pending/ from another thread during the write.
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                for name in os.listdir(pending):
                    if not name.startswith(".") and name != "big.bin":
                        seen_partial.append(name)
            except OSError:
                pass

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    emit_pending(pending, "big.bin", large_payload)
    stop.set()
    t.join(timeout=2)

    # No non-temp, non-final files should have been visible.
    assert seen_partial == []

    # Final file should have full payload.
    with open(os.path.join(pending, "big.bin"), "rb") as f:
        assert f.read() == large_payload


def test_emit_creates_dir_if_needed(tmp_path):
    """emit_pending should create the pending dir if it doesn't exist."""
    pending = str(tmp_path / "new_queue" / "pending")
    path = emit_pending(pending, "f.json", b"hello")
    assert os.path.exists(path)


def test_emit_strict_rejects_duplicate(queue_root):
    """Default (overwrite=False) raises FileExistsError on collision."""
    pending = str(queue_root / "pending")
    emit_pending(pending, "f.json", b"v1")
    with pytest.raises(FileExistsError):
        emit_pending(pending, "f.json", b"v2")
    # Original file should be untouched.
    with open(os.path.join(pending, "f.json"), "rb") as f:
        assert f.read() == b"v1"


def test_emit_overwrite_mode(queue_root):
    """overwrite=True silently replaces existing file."""
    pending = str(queue_root / "pending")
    emit_pending(pending, "f.json", b"v1")
    emit_pending(pending, "f.json", b"v2", overwrite=True)
    with open(os.path.join(pending, "f.json"), "rb") as f:
        assert f.read() == b"v2"


# ──────────────────────────────────────────────
# queue_depth
# ──────────────────────────────────────────────


def test_queue_depth(queue_root):
    pending = str(queue_root / "pending")
    claimed = str(queue_root / "claimed")

    for i in range(3):
        emit_pending(pending, f"s{i}.jsonl", b"x")
    claim(pending, claimed, "w0")

    depths = queue_depth(str(queue_root))
    assert depths["pending"] == 2
    assert depths["claimed"] == 1
    assert depths["done"] == 0


def test_queue_depth_missing_dirs(tmp_path):
    depths = queue_depth(str(tmp_path / "nonexistent"))
    assert depths == {"pending": 0, "claimed": 0, "done": 0}
