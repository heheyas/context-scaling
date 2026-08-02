# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pipeline progress monitor.

Polls the shared filesystem and prints a status dashboard at regular
intervals. Writes to both stdout and a log file.

Usage:
    python -m src.ops.monitor --config configs/iter0.yaml [--interval 30]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

from src.ops.file_queue import queue_depth
from src.ops.config import load_config, resolve_paths

log = logging.getLogger(__name__)


def count_pngs(images_dir: str) -> int:
    """Count all .png files under images/."""
    total = 0
    if not os.path.isdir(images_dir):
        return 0
    for entry in os.listdir(images_dir):
        idx_path = os.path.join(images_dir, entry)
        if os.path.isdir(idx_path):
            total += sum(1 for f in os.listdir(idx_path) if f.endswith(".png"))
    return total


def count_result_lines(rankings_dir: str) -> int:
    """Count total lines across all results_*.jsonl files."""
    total = 0
    if not os.path.isdir(rankings_dir):
        return 0
    for fname in os.listdir(rankings_dir):
        if fname.startswith("results_") and fname.endswith(".jsonl"):
            path = os.path.join(rankings_dir, fname)
            with open(path, "r") as f:
                total += sum(1 for line in f if line.strip())
    return total


def count_lines(path: str) -> int:
    """Count non-empty lines in a file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def count_quarantined(rank_in_dir: str) -> int:
    """Count files in rank_in/quarantine/."""
    q_dir = os.path.join(rank_in_dir, "quarantine")
    if not os.path.isdir(q_dir):
        return 0
    return sum(1 for f in os.listdir(q_dir) if not f.endswith(".error"))


def format_status(cfg: dict) -> str:
    """Build a one-line status string."""
    llm = queue_depth(cfg["llm_out"])
    rank = queue_depth(cfg["rank_in"])
    total_images = count_pngs(cfg["images"])
    total_ranked = count_result_lines(cfg["rankings"])
    total_queries = count_lines(cfg["queries_path"])
    expected_images = total_queries * cfg["num_images_per_prompt"]
    quarantined = count_quarantined(cfg["rank_in"])

    img_pct = (total_images / expected_images * 100) if expected_images > 0 else 0
    rank_pct = (total_ranked / total_queries * 100) if total_queries > 0 else 0
    ts = datetime.now().strftime("%H:%M:%S")

    parts = [
        f"[{ts}]",
        f"LLM: p={llm['pending']} c={llm['claimed']} d={llm['done']}",
        f"Images: {total_images}/{expected_images} ({img_pct:.0f}%)",
        f"Rank: p={rank['pending']} c={rank['claimed']} d={rank['done']}",
        f"Ranked: {total_ranked}/{total_queries} ({rank_pct:.0f}%)",
    ]
    if quarantined > 0:
        parts.append(f"QUARANTINED: {quarantined}")

    return " | ".join(parts)


def monitor_loop(cfg: dict, interval_sec: float = 30,
                 log_file: str = "logs/monitor.log"):
    """Print + log pipeline progress at regular intervals.

    Returns when all queries are ranked, or on KeyboardInterrupt.
    """
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    total_queries = count_lines(cfg["queries_path"])

    log.info("Monitor started (interval=%ds, queries=%d, log=%s)",
             interval_sec, total_queries, log_file)

    while True:
        status = format_status(cfg)
        print(status)
        with open(log_file, "a") as f:
            f.write(status + "\n")

        total_ranked = count_result_lines(cfg["rankings"])
        if total_queries > 0 and total_ranked >= total_queries:
            msg = f"Pipeline complete! {total_ranked}/{total_queries} queries ranked."
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            break

        time.sleep(interval_sec)


def main():
    parser = argparse.ArgumentParser(description="Pipeline progress monitor")
    parser.add_argument("--config", required=True, help="Path to iter config YAML")
    parser.add_argument("--interval", type=float, default=30,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--log-file", type=str, default="logs/monitor.log",
                        help="Log file path (default: logs/monitor.log)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    cfg = resolve_paths(cfg)

    try:
        monitor_loop(cfg, interval_sec=args.interval, log_file=args.log_file)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
