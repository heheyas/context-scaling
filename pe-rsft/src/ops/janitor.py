# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Janitor: single-pass stale-claim reaper + queue depth reporter.

Walks each queue's claimed/ directory, re-parks files older than T_stale
back to pending/, and prints queue depth metrics. Designed to be run
periodically (e.g., cron every 5 minutes) on any host with access to
the shared filesystem.

Usage:
    python -m src.ops.janitor /mnt/shared/rft_iter0 --t-stale 600

    # Multiple queue roots:
    python -m src.ops.janitor /mnt/shared/rft_iter0/llm_out /mnt/shared/rft_iter0/rank_in
"""

import argparse
import logging
import sys

from src.ops.file_queue import (
    list_stale_claims,
    queue_depth,
    release_to_pending,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [janitor] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_janitor_pass(queue_roots: list[str], t_stale_sec: float) -> dict:
    """Run a single janitor pass over the given queue roots.

    For each queue root (expected to contain pending/, claimed/, done/):
      1. Reports queue depth (pending/claimed/done counts).
      2. Finds stale files in claimed/ (mtime > t_stale_sec ago).
      3. Re-parks stale files back to pending/.

    Args:
        queue_roots: List of queue root directories to scan.
        t_stale_sec: Age threshold in seconds for stale claims.

    Returns:
        Summary dict: {queue_root: {"depths": {...}, "reparked": int}}
    """
    summary = {}

    for root in queue_roots:
        depths = queue_depth(root)
        log.info(
            "%s  pending=%d  claimed=%d  done=%d",
            root, depths["pending"], depths["claimed"], depths["done"],
        )

        claimed_dir = f"{root}/claimed"
        pending_dir = f"{root}/pending"
        stale = list_stale_claims(claimed_dir, t_stale_sec)
        reparked = 0

        for path in stale:
            log.info("Re-parking stale claim: %s", path)
            release_to_pending(path, pending_dir)
            reparked += 1

        if reparked:
            log.info("%s  re-parked %d stale claim(s)", root, reparked)

        summary[root] = {"depths": depths, "reparked": reparked}

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Janitor: re-park stale claims and report queue depths",
    )
    parser.add_argument(
        "queue_roots",
        nargs="+",
        help="One or more queue root directories (each containing pending/claimed/done)",
    )
    parser.add_argument(
        "--t-stale",
        type=float,
        default=600,
        help="Staleness threshold in seconds (default: 600 = 10 min)",
    )
    args = parser.parse_args()

    summary = run_janitor_pass(args.queue_roots, args.t_stale)

    total_reparked = sum(s["reparked"] for s in summary.values())
    if total_reparked == 0:
        log.info("No stale claims found.")
    else:
        log.info("Total re-parked: %d", total_reparked)


if __name__ == "__main__":
    main()
