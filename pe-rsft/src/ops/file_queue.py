# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Atomic file-queue primitives for cross-machine stage coordination.

All coordination between pipeline stages (LLM, DiT, Gemini) is done via
files on a shared filesystem. This module provides the atomic claim/move/emit
helpers that each stage daemon uses.

Protocol:
  - Producers write to a temp file, then os.rename() into pending/.
  - Consumers poll pending/, rename one file into claimed/{worker_id}_{file},
    process it, then rename into done/.
  - On crash, a janitor re-parks stale files from claimed/ back to pending/.

os.rename() is atomic within a single POSIX filesystem and NFSv4. If the
shared FS doesn't guarantee this, fall back to write-then-.ok-marker
(not implemented in this module).
"""

import os
import time
import uuid
import tempfile
import logging

log = logging.getLogger(__name__)


def emit_pending(
    pending_dir: str,
    filename: str,
    payload: bytes,
    overwrite: bool = False,
) -> str:
    """Atomically write payload into pending_dir/filename.

    Writes to a temp file in the same directory first, then renames.
    This ensures consumers never see a partial file in pending/.

    Args:
        pending_dir: Target directory (created if needed).
        filename: Final filename within pending_dir.
        payload: Raw bytes to write.
        overwrite: If False (default), raise FileExistsError when the
            target file already exists in pending_dir. If True, silently
            overwrite. Use strict mode (False) for ranking task emission
            where a collision indicates a bug; use overwrite=True for
            idempotent shard re-emission.

    Returns:
        Final path of the emitted file.

    Raises:
        FileExistsError: If overwrite=False and the file already exists.
    """
    os.makedirs(pending_dir, exist_ok=True)
    final_path = os.path.join(pending_dir, filename)

    if not overwrite and os.path.exists(final_path):
        raise FileExistsError(
            f"File already exists in pending dir: {final_path}"
        )

    # Write to a temp file in the same directory so rename is same-device.
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_emit_", suffix="_" + filename, dir=pending_dir,
    )
    try:
        os.write(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.rename(tmp_path, final_path)
    except BaseException:
        if fd is not None:
            os.close(fd)
        # Clean up the temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return final_path


def claim(pending_dir: str, claimed_dir: str, worker_id: str) -> str | None:
    """Try to claim one file from pending_dir by moving it to claimed_dir.

    Races are safe: if two workers try to claim the same file, only one
    rename will succeed; the loser gets OSError and tries the next file.

    Args:
        pending_dir: Directory to scan for claimable files.
        claimed_dir: Directory to move the claimed file into.
        worker_id: Identifier for this worker (used as filename prefix).

    Returns:
        Path of the claimed file in claimed_dir, or None if nothing available.
    """
    os.makedirs(claimed_dir, exist_ok=True)

    try:
        entries = sorted(os.listdir(pending_dir))
    except FileNotFoundError:
        return None

    for name in entries:
        # Skip hidden/temp files from in-progress emit_pending calls.
        if name.startswith("."):
            continue

        src = os.path.join(pending_dir, name)
        dst = os.path.join(claimed_dir, f"{worker_id}_{name}")

        try:
            os.rename(src, dst)
            return dst
        except OSError:
            # Another worker won the race, or file vanished. Try next.
            continue

    return None


def claim_batch(pending_dir: str, claimed_dir: str, worker_id: str,
                 max_count: int = 0) -> list[str]:
    """Claim up to max_count files in one listdir call.

    Much faster than calling claim() in a loop on HDFS FUSE, because
    we only do one os.listdir() instead of one per file.

    Args:
        pending_dir: Directory to scan for claimable files.
        claimed_dir: Directory to move the claimed file into.
        worker_id: Identifier for this worker (used as filename prefix).
        max_count: Maximum number of files to claim. 0 = no limit.

    Returns:
        List of paths of claimed files in claimed_dir (may be empty).
    """
    os.makedirs(claimed_dir, exist_ok=True)

    try:
        entries = sorted(os.listdir(pending_dir))
    except FileNotFoundError:
        return []

    claimed = []
    for name in entries:
        if name.startswith("."):
            continue

        src = os.path.join(pending_dir, name)
        dst = os.path.join(claimed_dir, f"{worker_id}_{name}")

        try:
            os.rename(src, dst)
            claimed.append(dst)
        except OSError:
            continue

        if max_count > 0 and len(claimed) >= max_count:
            break

    return claimed


def release_to_done(claimed_path: str, done_dir: str) -> str:
    """Move a claimed file to done_dir after successful processing.

    The filename in done_dir strips the worker_id prefix that was added
    during claim, restoring the original filename.

    Args:
        claimed_path: Path of the file in claimed_dir.
        done_dir: Target directory.

    Returns:
        Final path in done_dir.
    """
    os.makedirs(done_dir, exist_ok=True)
    original_name = _strip_worker_prefix(os.path.basename(claimed_path))
    dst = os.path.join(done_dir, original_name)
    os.rename(claimed_path, dst)
    return dst


def release_to_pending(claimed_path: str, pending_dir: str) -> str:
    """Re-park a claimed file back to pending_dir (used by janitor).

    Strips the worker_id prefix to restore the original filename.

    Args:
        claimed_path: Path of the stale file in claimed_dir.
        pending_dir: Target directory to re-park into.

    Returns:
        Final path in pending_dir.
    """
    os.makedirs(pending_dir, exist_ok=True)
    original_name = _strip_worker_prefix(os.path.basename(claimed_path))
    dst = os.path.join(pending_dir, original_name)
    try:
        os.rename(claimed_path, dst)
    except OSError:
        # File may have already been re-parked by another janitor instance,
        # or the original file reappeared in pending. Not an error.
        log.warning("Could not re-park %s -> %s (already moved?)", claimed_path, dst)
        return dst
    return dst


def list_stale_claims(claimed_dir: str, t_stale_sec: float) -> list[str]:
    """Return paths in claimed_dir whose mtime is older than t_stale_sec.

    Args:
        claimed_dir: Directory to scan.
        t_stale_sec: Age threshold in seconds. Files with
            mtime < (now - t_stale_sec) are considered stale.

    Returns:
        List of absolute paths to stale files.
    """
    now = time.time()
    stale = []

    try:
        entries = os.listdir(claimed_dir)
    except FileNotFoundError:
        return stale

    for name in entries:
        if name.startswith("."):
            continue
        path = os.path.join(claimed_dir, name)
        try:
            mtime = os.path.getmtime(path)
            if (now - mtime) >= t_stale_sec:
                stale.append(path)
        except OSError:
            # File vanished between listdir and stat. Ignore.
            continue

    return stale


def queue_depth(queue_root: str) -> dict[str, int]:
    """Count files in pending/, claimed/, done/ under queue_root.

    Returns:
        Dict with keys "pending", "claimed", "done" mapping to file counts.
    """
    counts = {}
    for subdir in ("pending", "claimed", "done"):
        path = os.path.join(queue_root, subdir)
        try:
            entries = [e for e in os.listdir(path) if not e.startswith(".")]
            counts[subdir] = len(entries)
        except FileNotFoundError:
            counts[subdir] = 0
    return counts


def _strip_worker_prefix(claimed_name: str) -> str:
    """Strip the '{worker_id}_' prefix added during claim.

    The worker_id is everything up to and including the first underscore.
    """
    idx = claimed_name.find("_")
    if idx >= 0:
        return claimed_name[idx + 1:]
    return claimed_name
