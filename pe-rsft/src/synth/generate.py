#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Synthetic reasoning generation using Gemini as teacher.

Reads (prompt, structured_prompt) pairs, generates multiple reasoning
traces per pair in backward and/or forward mode, and writes results
as sharded JSONL with resume support.

Usage:
    python -m src.synth.generate --config configs/synth_reasoning.yaml

    # Smoke test on first 10 pairs
    python -m src.synth.generate --config configs/synth_reasoning.yaml --max-pairs 10
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import openai
import httpx as _httpx

from src.synth.prompts import (
    build_backward_messages,
    build_forward_messages,
    parse_forward_response,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Gemini client
# ──────────────────────────────────────────────

DEFAULT_API_KEY = ""  # set via env var GEMINI_API_KEY
DEFAULT_BASE_URL = "<INTERNAL_URL>"


def make_client(api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL) -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=base_url,
        api_version="2024-03-01-preview",
        http_client=_httpx.Client(),
    )


class KeyPool:
    """Thread-safe API key pool with rotation on rate limits."""

    def __init__(self, key_configs: list[tuple[str, str]]):
        seen = {}
        for api_key, base_url in key_configs:
            seen[api_key] = base_url
        self._keys = list(seen.keys())
        self._base_urls = dict(seen)
        self._lock = threading.Lock()
        self._429_count = {k: 0 for k in self._keys}
        self._429_ts = {k: 0.0 for k in self._keys}
        self._usage = {k: 0 for k in self._keys}
        self._clients = {}

    def acquire(self) -> tuple[str, openai.AzureOpenAI]:
        with self._lock:
            now = time.time()
            # Reset 429 count for keys that have cooled down (>60s since last 429)
            for k in self._keys:
                if self._429_count[k] > 0 and (now - self._429_ts[k]) > 60:
                    self._429_count[k] = 0
            best = min(self._keys, key=lambda k: (
                1 if (now - self._429_ts[k]) < 60 else 0,
                self._429_count[k],
                self._usage[k],
            ))
            self._usage[best] += 1
            if best not in self._clients:
                self._clients[best] = make_client(best, self._base_urls[best])
            return best, self._clients[best]

    def report_429(self, key: str):
        with self._lock:
            self._429_count[key] += 1
            self._429_ts[key] = time.time()

    def stats(self) -> str:
        with self._lock:
            parts = []
            for k in self._keys:
                tag = f"...{k[-8:]}"
                parts.append(f"{tag}: {self._usage[k]} calls, {self._429_count[k]} 429s")
            return " | ".join(parts)


def load_api_keys(arg: Optional[str]) -> list[tuple[str, str]]:
    """Parse API keys from file or CLI string."""
    if arg is None:
        return [(DEFAULT_API_KEY, DEFAULT_BASE_URL)]
    if os.path.exists(arg):
        configs = []
        with open(arg, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                api_key = parts[0]
                base_url = parts[1].strip() if len(parts) > 1 else DEFAULT_BASE_URL
                configs.append((api_key, base_url))
        if configs:
            return configs
    return [(k.strip(), DEFAULT_BASE_URL) for k in arg.split(",") if k.strip()]


# ──────────────────────────────────────────────
# Gemini call with retry
# ──────────────────────────────────────────────

def call_gemini(
    key_pool: KeyPool,
    messages: list[dict],
    model: str = "gemini-3-pro-preview-new",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    max_retries: int = 8,
) -> Optional[str]:
    """Call Gemini and return the response text, or None on failure."""
    for attempt in range(max_retries):
        key, client = key_pool.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content
            if text:
                return text.strip()
            log.warning("Empty response (attempt %d)", attempt + 1)
        except (openai.RateLimitError, openai.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if isinstance(e, openai.RateLimitError) or status == 429:
                key_pool.report_429(key)
                wait = min(1.01 ** attempt, 60)
                log.warning("Rate limited (attempt %d), switching key...", attempt + 1)
                time.sleep(wait)
            elif status and status >= 500:
                log.warning("Server error %d (attempt %d)", status, attempt + 1)
                time.sleep(1.01 ** attempt)
            else:
                log.warning("API error (attempt %d): %s", attempt + 1, e)
                time.sleep(1.01 ** attempt)
        except Exception as e:
            log.warning("Call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.01 ** attempt)

    return None


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_qa_pairs(path: str) -> list[dict]:
    """Load Q-A pairs from JSONL.

    Supports two formats:
    1. SFT messages format: {"messages": [system, user, assistant]}
       - user content: "{prompt} [width: W, height: H]"
       - assistant content: JSON structured prompt
    2. Flat format: {"index", "prompt"/"query", "structured_prompt", "width", "height"}
    """
    import re

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # SFT messages format
            if "messages" in rec:
                msgs = rec["messages"]
                if len(msgs) < 3:
                    continue
                user_content = msgs[1]["content"]
                assistant_content = msgs[2]["content"]

                # Parse user message: "{prompt} [width: W, height: H]"
                m = re.match(
                    r"(.*?)\s*\[width:\s*(\d+),\s*height:\s*(\d+)\]",
                    user_content,
                    re.DOTALL,
                )
                if not m:
                    continue
                prompt = m.group(1).strip()
                width = int(m.group(2))
                height = int(m.group(3))

                pairs.append({
                    "index": line_num,
                    "prompt": prompt,
                    "structured_prompt": assistant_content,
                    "width": width,
                    "height": height,
                })
            else:
                # Flat format
                if "query" in rec and "prompt" not in rec:
                    rec["prompt"] = rec["query"]
                if "prompt" not in rec or "structured_prompt" not in rec:
                    continue
                if "index" not in rec:
                    rec["index"] = line_num
                pairs.append(rec)

    log.info("Loaded %d Q-A pairs from %s", len(pairs), path)
    return pairs


# ──────────────────────────────────────────────
# WAL (write-ahead log) for resume
# ──────────────────────────────────────────────

def _wal_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_wal")


def _wal_file(output_dir: str, mode: str) -> str:
    return os.path.join(_wal_path(output_dir), f"{mode}.partial.jsonl")


def load_completed_keys(output_dir: str, mode: str) -> set[str]:
    """Load keys of already-completed tasks from WAL + done files.

    Key format: "{index}_{mode}_{sample_idx}"
    """
    completed = set()

    # Check WAL
    wal = _wal_file(output_dir, mode)
    if os.path.exists(wal):
        with open(wal, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed.add(rec["_key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Check done files
    done_dir = os.path.join(output_dir, "done")
    if os.path.isdir(done_dir):
        for fname in os.listdir(done_dir):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(done_dir, fname)
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        completed.add(rec["_key"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    return completed


def append_wal(output_dir: str, mode: str, record: dict):
    """Append a completed record to the WAL."""
    wal_dir = _wal_path(output_dir)
    os.makedirs(wal_dir, exist_ok=True)
    wal = _wal_file(output_dir, mode)
    with open(wal, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# Task generation
# ──────────────────────────────────────────────

def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def generate_tasks(
    pairs: list[dict],
    mode: str,
    num_samples: int,
    completed: set[str],
) -> list[dict]:
    """Generate task dicts for all (pair, sample_idx) not yet completed."""
    tasks = []
    for pair in pairs:
        idx = pair["index"]
        for sample_idx in range(num_samples):
            key = f"{idx}_{mode}_{sample_idx}"
            if key in completed:
                continue
            tasks.append({
                "_key": key,
                "index": idx,
                "mode": mode,
                "sample_idx": sample_idx,
                "prompt": pair["prompt"],
                "structured_prompt": pair["structured_prompt"],
                "width": pair.get("width", 1024),
                "height": pair.get("height", 1024),
            })
    return tasks


# ──────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────

def process_one(
    task: dict,
    key_pool: KeyPool,
    model: str,
    max_tokens: int,
    temperature: float,
    backward_system_prompt: str,
    forward_system_prompt: str,
) -> dict:
    """Process a single task: call Gemini and return result record."""
    mode = task["mode"]
    prompt = task["prompt"]
    sp = task["structured_prompt"]
    w, h = task["width"], task["height"]

    if mode == "backward":
        messages = build_backward_messages(backward_system_prompt, prompt, sp, w, h)
    else:
        messages = build_forward_messages(forward_system_prompt, prompt, w, h)

    response = call_gemini(
        key_pool, messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = {
        "_key": task["_key"],
        "index": task["index"],
        "mode": mode,
        "sample_idx": task["sample_idx"],
        "prompt": prompt,
        "width": w,
        "height": h,
        "success": response is not None,
    }

    if response is None:
        result["thinking"] = None
        result["generated_sp"] = None
    elif mode == "backward":
        result["thinking"] = response
        result["generated_sp"] = None  # backward mode doesn't generate SP
        result["reference_sp"] = sp
    else:
        thinking, gen_sp = parse_forward_response(response)
        result["thinking"] = thinking
        result["generated_sp"] = gen_sp
        result["reference_sp"] = sp
        result["raw_response"] = response

    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_generation(cfg: dict, max_pairs: int = 0):
    """Run synthetic reasoning generation from config."""
    import yaml

    synth_cfg = cfg["synth"]
    gemini_cfg = synth_cfg["gemini"]
    output_dir = synth_cfg["output_dir"]

    # Setup output dirs
    for sub in ("done", "_wal"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Load API keys and build pool
    key_configs = load_api_keys(gemini_cfg.get("api_keys_path"))
    key_pool = KeyPool(key_configs)
    log.info("Loaded %d API key(s)", len(key_configs))

    model = gemini_cfg.get("model", "gemini-3-pro-preview-new")
    max_tokens = gemini_cfg.get("max_tokens", 8192)
    temperature = gemini_cfg.get("temperature", 1.0)
    num_workers = gemini_cfg.get("num_workers", 64)

    # Load system prompts
    backward_sp_path = synth_cfg.get("backward", {}).get("system_prompt_path")
    forward_sp_path = synth_cfg.get("forward", {}).get("system_prompt_path")

    backward_system_prompt = ""
    if backward_sp_path and os.path.exists(backward_sp_path):
        with open(backward_sp_path, "r") as f:
            backward_system_prompt = f.read()

    forward_system_prompt = ""
    if forward_sp_path and os.path.exists(forward_sp_path):
        with open(forward_sp_path, "r") as f:
            forward_system_prompt = f.read()

    # Load Q-A pairs
    pairs = load_qa_pairs(synth_cfg["input_jsonl"])
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
        log.info("Limited to %d pairs (smoke test)", max_pairs)

    modes = synth_cfg.get("modes", ["backward", "forward"])

    # Build all tasks
    all_tasks = []
    for mode in modes:
        mode_cfg = synth_cfg.get(mode, {})
        num_samples = mode_cfg.get("num_samples", 4)
        completed = load_completed_keys(output_dir, mode)
        tasks = generate_tasks(pairs, mode, num_samples, completed)
        log.info("Mode %s: %d tasks to run (%d already done)",
                 mode, len(tasks), len(completed))
        all_tasks.extend(tasks)

    if not all_tasks:
        log.info("All tasks already completed. Nothing to do.")
        return

    log.info("Total tasks: %d, workers: %d", len(all_tasks), num_workers)

    # Process with thread pool
    done_count = 0
    fail_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_one, task, key_pool, model, max_tokens, temperature,
                backward_system_prompt, forward_system_prompt,
            ): task
            for task in all_tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    append_wal(output_dir, result["mode"], result)
                    done_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                log.error("Task %s failed: %s", task["_key"], e)
                fail_count += 1

            total = done_count + fail_count
            if total % 100 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d done, %d failed, %.1f tasks/sec | %s",
                    done_count, fail_count, rate, key_pool.stats(),
                )

    elapsed = time.time() - t0
    log.info(
        "Generation complete: %d done, %d failed in %.0fs (%.1f tasks/sec)",
        done_count, fail_count, elapsed,
        (done_count + fail_count) / elapsed if elapsed > 0 else 0,
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Synthetic reasoning generation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit to first N pairs (0=all, for smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_generation(cfg, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
