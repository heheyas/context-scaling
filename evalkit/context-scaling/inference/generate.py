#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Multi-threaded inference pipeline: read rewritten JSONL, generate images via
serving endpoint, save in benchmark-native evaluation format.

Usage:
    # GenEval2 (1 image/prompt, flat PNG + JSON mapping)
    python -m inference.generate \
        --rewritten_jsonl rewritten/geneval2_gemini.jsonl \
        --url http://localhost:8899 \
        --output_dir outputs/geneval2_gemini \
        --height 1024 --width 1024 \
        --workers 8

    # OneIG-Bench (4 images/prompt, 2x2 WEBP grid)
    python -m inference.generate \
        --rewritten_jsonl rewritten/oneig_gemini.jsonl \
        --url http://localhost:8899 \
        --output_dir outputs/oneig_gemini \
        --model_name my_model \
        --height 1024 --width 1024 \
        --workers 8

    # Use original prompt instead of rewritten
    python -m inference.generate \
        --rewritten_jsonl rewritten/geneval2_gemini.jsonl \
        --url http://localhost:8899 \
        --output_dir outputs/geneval2_baseline \
        --disable_rewritten \
        --workers 8
"""

import json
import os
import sys
import logging
import argparse
import threading
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

from inference.output_formatters import get_formatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compact_single_quote_json(data):
    """Compact a JSON-parseable object/string into single-quote format for structured prompts.

    If `data` is a string, try to parse it as JSON first. If it's already a dict/list,
    use it directly. Non-JSON strings are returned as-is.
    """
    PLACEHOLDER_SINGLE = "@@SP_SINGLE_QUOTE@@"
    PLACEHOLDER_DOUBLE = "@@SP_DOUBLE_QUOTE@@"

    # Parse string to object if needed
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return data  # Not JSON, return as-is

    def protect_single_quotes(obj):
        if isinstance(obj, dict):
            return {
                (protect_single_quotes(k) if isinstance(k, str) else k): protect_single_quotes(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [protect_single_quotes(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace("'", PLACEHOLDER_SINGLE).replace('"', PLACEHOLDER_DOUBLE)
        else:
            return obj

    protected_data = protect_single_quotes(data)
    json_str = json.dumps(protected_data, separators=(',', ':'), ensure_ascii=False)
    s_step1 = json_str.replace('"', "'")
    return s_step1.replace(PLACEHOLDER_SINGLE, "\\'").replace(PLACEHOLDER_DOUBLE, '"')


def load_rewritten_jsonl(path: str):
    """Load rewritten JSONL, return list of items.

    Supports two formats:
    - Standard rewritten: {id, benchmark, original_prompt, rewritten_prompt, ...}
    - Random/custom: {filename, json_output, ...} (auto-detected and converted)
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if not items:
        return items

    # Auto-detect random/custom format: has json_output but no benchmark
    if "json_output" in items[0] and "benchmark" not in items[0]:
        log.info("Detected random/custom JSONL format, converting...")
        converted = []
        for i, item in enumerate(items):
            name = item.get("filename", str(i))
            if "." in name:
                name = name.rsplit(".", 1)[0]
            entry = {
                "id": name,
                "benchmark": "random",
                "category": None,
                "original_prompt": item.get("image_key", name),
                "rewritten_prompt": item["json_output"],
            }
            # Preserve per-item resolution if present
            if "width" in item:
                entry["width"] = item["width"]
            if "height" in item:
                entry["height"] = item["height"]
            converted.append(entry)
        return converted

    return items


_GEN_SESSION = None
_GEN_SESSION_LOCK = threading.Lock()


def _patch_getaddrinfo_for_literals():
    """Monkey-patch socket.getaddrinfo to bypass C resolver for IP literals.

    Why: under high concurrency, glibc's getaddrinfo with default AI_ADDRCONFIG
    flag occasionally returns empty list for IPv6 literals when the system
    momentarily sees no configured IPv6 route (resolver cache thrashing).

    For literal IPs (v4 and v6), DNS resolution is unnecessary — we construct
    the addrinfo tuple directly. Falls back to original getaddrinfo for hostnames.
    """
    import socket as _sock
    if getattr(_sock, "_literal_patched", False):
        return
    _orig_getaddrinfo = _sock.getaddrinfo

    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        # Normalize bytes → str for inet_pton (asyncio sometimes passes bytes hosts).
        # If host isn't a string-like literal, fall through to original resolver.
        host_str = host
        if isinstance(host, bytes):
            try:
                host_str = host.decode("ascii")
            except (UnicodeDecodeError, AttributeError):
                host_str = None
        if host_str:
            # IPv4 literal
            try:
                _sock.inet_pton(_sock.AF_INET, host_str)
                return [(_sock.AF_INET, type or _sock.SOCK_STREAM, proto,
                         "", (host_str, port or 0))]
            except (OSError, ValueError, TypeError):
                pass
            # IPv6 literal
            try:
                _sock.inet_pton(_sock.AF_INET6, host_str)
                return [(_sock.AF_INET6, type or _sock.SOCK_STREAM, proto,
                         "", (host_str, port or 0, 0, 0))]
            except (OSError, ValueError, TypeError):
                pass
        # Hostname (or bytes/None we couldn't handle) — fall back to real resolver
        return _orig_getaddrinfo(host, port, family, type, proto, flags)

    _sock.getaddrinfo = _patched_getaddrinfo
    _sock._literal_patched = True


def _get_gen_session():
    """Get a shared requests Session with trust_env=False (ignores HTTP_PROXY etc).

    Critical: when scripts have HTTP_PROXY env var set (e.g., for Claude Code or
    external tools), requests would try to route DiT generation through it. That
    breaks IPv6 PSM backends. Using trust_env=False guarantees direct connection.
    """
    global _GEN_SESSION
    if _GEN_SESSION is None:
        with _GEN_SESSION_LOCK:
            if _GEN_SESSION is None:
                _patch_getaddrinfo_for_literals()
                s = requests.Session()
                s.trust_env = False  # Ignore HTTP_PROXY, HTTPS_PROXY, NO_PROXY env vars
                # Larger connection pool for many parallel workers
                from requests.adapters import HTTPAdapter
                adapter = HTTPAdapter(pool_connections=64, pool_maxsize=256,
                                      max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _GEN_SESSION = s
    return _GEN_SESSION


def send_generate(url, prompt, height, width, num_steps, seed, cfg_scale, negative_prompt,
                  timeout=600, timestep_shift=None, max_retries=10):
    """Send one generation request to serving endpoint. Returns PIL Image or None.

    Retries on 502/503 (backend not ready) with exponential backoff.
    """
    body = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "seed": seed,
        "cfg_scale": cfg_scale,
        # Unified servers that accept true_cfg_scale (e.g. qwenimage-edit) ignore
        # cfg_scale; the older ones ignore true_cfg_scale. Sending both is safe.
        "true_cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
    }
    if timestep_shift is not None:
        body["timestep_shift"] = timestep_shift

    import time as _time
    session = _get_gen_session()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = session.post(
                f"{url}/generate",
                json=body,
                timeout=timeout,
            )
            if resp.status_code in (502, 503):
                wait = min(2 ** attempt, 30)
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            # Common under high concurrency: getaddrinfo failures (FD/resolver
            # exhaustion), connection drops, transient network blips. Backoff and retry.
            last_err = e
            wait = min(2 ** attempt, 30)
            _time.sleep(wait)
            continue

    if last_err is not None:
        raise last_err
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# PSM support: resolve PSM -> URL, verify/switch model before each request
# ---------------------------------------------------------------------------

def _resolve_all_psm_urls(psm: str) -> list:
    """Resolve a PSM to all backend URLs. Requires ray.serve."""
    # ray is installed under <RAY_SERVE_HOME>, which may not be the current HOME
    tiger_site = "<RAY_SERVE_SITE_PACKAGES>"
    if tiger_site not in sys.path:
        sys.path.insert(0, tiger_site)

    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = "<RAY_SERVE_HOME>"
        from ray.serve import get_serve_http_client
        client = get_serve_http_client(psm=psm)
        endpoints = client.endpoint_getter.get_urls()
        urls = []
        for ep in endpoints:
            host, port = ep["Host"], ep["Port"]
            urls.append(f"http://[{host}]:{port}")
        return urls
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            os.environ.pop("HOME", None)


def _check_and_switch(url: str, expect_trial: str = None, expect_step: str = None,
                      expect_ema: bool = False, timeout: int = 30) -> bool:
    """Check /health and send /switch if model params don't match.
    Returns True if ready (either already matching or switch completed)."""
    import time as _time

    # 1. Health check
    try:
        resp = requests.get(f"{url}/health", timeout=timeout, proxies={"http": None, "https": None})
        resp.raise_for_status()
        info = resp.json()
    except Exception as e:
        log.warning("[PSM] Health check failed on %s: %s", url, e)
        return False

    # If no expected params, just check it's healthy
    if not expect_trial:
        return info.get("status") in ("ok", "switching")

    # Wait if currently switching (with timeout)
    max_switch_wait = 120  # 10 min
    wait_count = 0
    while info.get("switching"):
        wait_count += 1
        if wait_count > max_switch_wait:
            log.error("[PSM] Server stuck in switching state for too long")
            return False
        log.info("[PSM] Server is switching, waiting... (%d/%d)", wait_count, max_switch_wait)
        _time.sleep(5)
        try:
            resp = requests.get(f"{url}/health", timeout=timeout, proxies={"http": None, "https": None})
            info = resp.json()
        except Exception as e:
            log.warning("[PSM] Health check failed during switch wait: %s", e)
            _time.sleep(5)
            continue

    # 2. Compare model params
    current_trial = info.get("trial")
    current_step = info.get("step")
    current_ema = info.get("ema", False)

    trial_match = (current_trial == expect_trial)
    step_match = (expect_step is None) or (current_step == expect_step)
    ema_match = (current_ema == expect_ema)

    if trial_match and step_match and ema_match:
        log.info("[PSM] Model already matches: %s/%s (ema=%s)", current_trial, current_step, current_ema)
        return True

    # 3. Send /switch
    log.info("[PSM] Model mismatch: have %s/%s (ema=%s), want %s/%s (ema=%s). Switching...",
             current_trial, current_step, current_ema,
             expect_trial, expect_step, expect_ema)

    switch_body = {"trial": expect_trial, "ema": expect_ema}
    if expect_step:
        switch_body["step"] = expect_step

    try:
        resp = requests.post(f"{url}/switch", json=switch_body, timeout=600, proxies={"http": None, "https": None})
        resp.raise_for_status()
        result = resp.json()
        log.info("[PSM] Switch result: %s", result.get("message", result))
        return result.get("status") == "ok"
    except Exception as e:
        log.error("[PSM] Switch failed: %s", e)
        return False


class PSMResolver:
    """Thread-safe PSM resolver with per-request load balancing and model verification.

    At init, resolves PSM once to get ALL backend URLs. Each get_url() picks a
    random backend (load balancing) and verifies model params. If a switch is
    needed, ALL backends are switched and ALL generation requests are blocked
    until complete.

    Flow:
    1. Init: resolve PSM -> list of all backend URLs
    2. Per request: pick random URL -> fast path if verified -> else verify/switch
    3. Switch: block all requests, switch ALL unverified backends, resume
    """

    def __init__(self, psm: str, expect_trial: str = None,
                 expect_step: str = None, expect_ema: bool = False):
        self.psm = psm
        self.expect_trial = expect_trial
        self.expect_step = expect_step
        self.expect_ema = expect_ema

        # Resolve all backend URLs upfront
        self._all_urls = _resolve_all_psm_urls(psm)
        if not self._all_urls:
            raise RuntimeError(f"PSM {psm} resolved to 0 backends")
        log.info("[PSM] Resolved %s -> %d backends: %s",
                 psm, len(self._all_urls), self._all_urls)

        # Backends that have been verified (model matches)
        self._verified_urls = set()
        self._verified_lock = threading.Lock()
        # Blocks all requests during a switch
        self._ready = threading.Event()
        self._ready.set()
        # Ensures only one switch at a time
        self._switch_lock = threading.Lock()

    def get_url(self) -> str:
        """Pick a random backend, verify model, switch if needed. Called per-request."""
        import random

        # Block while a switch is in progress
        self._ready.wait()

        # Pick a random backend (load balancing)
        url = random.choice(self._all_urls)

        # Fast path: this backend already verified
        with self._verified_lock:
            if url in self._verified_urls:
                return url

        # Slow path: need to verify this backend
        if not self.expect_trial:
            with self._verified_lock:
                self._verified_urls.add(url)
            return url

        with self._switch_lock:
            # Double-check after acquiring lock
            with self._verified_lock:
                if url in self._verified_urls:
                    return url

            # Check /health
            needs_switch = not self._check_model_match(url)

            if needs_switch:
                # Block ALL generation requests, switch ALL unverified backends
                self._ready.clear()
                try:
                    self._switch_all_unverified()
                finally:
                    self._ready.set()
            else:
                with self._verified_lock:
                    self._verified_urls.add(url)

        return url

    def invalidate_url(self, url: str):
        """Remove a URL from verified set (e.g., after a request failure)."""
        with self._verified_lock:
            self._verified_urls.discard(url)

    def refresh_urls(self):
        """Re-resolve PSM to discover new/changed backends."""
        new_urls = _resolve_all_psm_urls(self.psm)
        if new_urls:
            self._all_urls = new_urls
            log.info("[PSM] Refreshed backends: %d URLs", len(new_urls))

    def _check_model_match(self, url: str) -> bool:
        """Check if the backend's model matches expected params."""
        import time as _time
        try:
            resp = requests.get(f"{url}/health", timeout=30,
                                proxies={"http": None, "https": None})
            resp.raise_for_status()
            info = resp.json()
        except Exception as e:
            log.warning("[PSM] Health check failed on %s: %s", url, e)
            return False

        # Wait if currently switching
        max_wait = 120
        wait_count = 0
        while info.get("switching"):
            wait_count += 1
            if wait_count > max_wait:
                log.error("[PSM] Server stuck in switching state")
                return False
            _time.sleep(5)
            try:
                resp = requests.get(f"{url}/health", timeout=30,
                                    proxies={"http": None, "https": None})
                info = resp.json()
            except Exception:
                _time.sleep(5)
                continue

        current_trial = info.get("trial")
        current_step = info.get("step")
        current_ema = info.get("ema", False)

        trial_match = (current_trial == self.expect_trial)
        step_match = (self.expect_step is None) or (current_step == self.expect_step)
        ema_match = (current_ema == self.expect_ema)

        if trial_match and step_match and ema_match:
            log.info("[PSM] Backend %s matches: %s/%s (ema=%s)",
                     url, current_trial, current_step, current_ema)
            return True

        log.info("[PSM] Backend %s mismatch: have %s/%s (ema=%s), want %s/%s (ema=%s)",
                 url, current_trial, current_step, current_ema,
                 self.expect_trial, self.expect_step, self.expect_ema)
        return False

    def _do_switch(self, url: str):
        """Send /switch to a backend. Blocks until complete."""
        switch_body = {"trial": self.expect_trial, "ema": self.expect_ema}
        if self.expect_step:
            switch_body["step"] = self.expect_step

        resp = requests.post(f"{url}/switch", json=switch_body, timeout=600,
                             proxies={"http": None, "https": None})
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "ok":
            raise RuntimeError(f"Switch failed on {url}: {result}")

        log.info("[PSM] Switch done on %s: %s", url, result.get("message", ""))

    def _switch_all_unverified(self):
        """Switch ALL backends that haven't been verified yet."""
        with self._verified_lock:
            to_switch = [u for u in self._all_urls if u not in self._verified_urls]

        log.info("[PSM] Switching %d/%d backends, blocking all requests...",
                 len(to_switch), len(self._all_urls))

        for url in to_switch:
            try:
                if self._check_model_match(url):
                    log.info("[PSM] Backend %s already matches, skip switch", url)
                else:
                    self._do_switch(url)
            except Exception as e:
                log.error("[PSM] Failed to switch %s: %s", url, e)
                raise

            with self._verified_lock:
                self._verified_urls.add(url)

        log.info("[PSM] All %d backends verified/switched", len(self._all_urls))


def send_generate_psm(psm_resolver: 'PSMResolver', prompt, height, width, num_steps,
                      seed, cfg_scale, negative_prompt, timeout=600,
                      timestep_shift=None, max_retries=10):
    """Resolve PSM to a random backend, verify model, then generate."""
    url = psm_resolver.get_url()
    try:
        return send_generate(
            url=url, prompt=prompt, height=height, width=width,
            num_steps=num_steps, seed=seed, cfg_scale=cfg_scale,
            negative_prompt=negative_prompt, timeout=timeout,
            timestep_shift=timestep_shift, max_retries=max_retries,
        )
    except Exception:
        # Backend might be dead — remove from verified so it gets re-checked
        psm_resolver.invalidate_url(url)
        raise


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for benchmark evaluation")

    # Input/output
    parser.add_argument("--rewritten_jsonl", type=str, required=True,
                        help="Rewritten JSONL from rewrite pipeline")
    parser.add_argument("--url", type=str, default="http://localhost:8899",
                        help="Serving endpoint URL")

    # PSM support (overrides --url when set)
    parser.add_argument("--psm", type=str, default=None,
                        help="Ray PSM for serving endpoint (resolves to URL, overrides --url)")
    parser.add_argument("--psm_trial", type=str, default=None,
                        help="Expected trial name on PSM endpoint (triggers /switch if mismatch)")
    parser.add_argument("--psm_step", type=str, default=None,
                        help="Expected step on PSM endpoint")
    parser.add_argument("--psm_ema", action="store_true",
                        help="Expected EMA weights on PSM endpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # Generation params
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--timestep_shift", type=float, default=None,
                        help="Timestep shift for Bagel models (default: None = server default)")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Override num images per prompt (default: auto per benchmark)")

    # Prompt selection
    parser.add_argument("--disable_rewritten", action="store_true",
                        help="Use original_prompt instead of rewritten_prompt")
    parser.add_argument("--fix_lens_key", action="store_true",
                        help="Workaround: rename JSON key 'lens_and_effect' to 'lens-and-effect' "
                             "before sending. Some Bagel ckpts (e.g. unified-json-ct3-ocr) produce "
                             "all-black images on prompts containing the literal key 'lens_and_effect'.")

    # Benchmark-specific
    parser.add_argument("--model_name", type=str, default="model",
                        help="Model name (used in OneIG-Bench directory structure)")
    parser.add_argument("--benchmark_data", type=str, default=None,
                        help="Path to benchmark data file (GenEval needs evaluation_metadata.jsonl)")

    # Concurrency
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Max prompts to process (0=all)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="HTTP request timeout in seconds")

    # Distributed sharding
    parser.add_argument("--rank", type=int, default=0,
                        help="Shard rank (0-indexed). Each rank processes items[rank::world_size].")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of shards. 1 = no sharding (default).")

    args = parser.parse_args()

    # Load items
    all_items = load_rewritten_jsonl(args.rewritten_jsonl)
    if args.max_prompts > 0:
        all_items = all_items[:args.max_prompts]

    if not all_items:
        log.error("No items to process")
        return

    # Distributed sharding: each rank processes a disjoint subset
    if args.world_size > 1:
        if args.rank < 0 or args.rank >= args.world_size:
            log.error("Invalid rank %d for world_size %d (must be 0..%d)",
                      args.rank, args.world_size, args.world_size - 1)
            return
        items = all_items[args.rank::args.world_size]
        log.info("Shard %d/%d: %d items (of %d total)",
                 args.rank, args.world_size, len(items), len(all_items))
    else:
        items = all_items

    if not items:
        log.error("No items for rank %d", args.rank)
        return

    # Detect benchmark from first item
    benchmark = all_items[0]["benchmark"]
    log.info("Detected benchmark: %s", benchmark)

    # Create formatter
    formatter = get_formatter(
        benchmark=benchmark,
        output_dir=args.output_dir,
        model_name=args.model_name,
        benchmark_data=args.benchmark_data,
    )

    # Determine prompt field
    prompt_field = "original_prompt" if args.disable_rewritten else "rewritten_prompt"

    # Build task list: (item, image_idx, save_path, seed)
    # Also recover items whose tmp files are complete but grid wasn't assembled (crash recovery)
    tasks = []
    skip_count = 0
    recovered_count = 0
    for item in items:
        num_images = args.num_images if args.num_images is not None else formatter.get_num_images(item)
        final_path = formatter.get_final_path(item)
        if os.path.exists(final_path):
            skip_count += 1
            continue

        # Check if all tmp/intermediate files already exist (crash recovery)
        existing_paths = []
        all_exist = True
        for i in range(num_images):
            save_path = formatter.get_save_path(item, i)
            if os.path.exists(save_path):
                existing_paths.append(save_path)
            else:
                all_exist = False

        if all_exist and len(existing_paths) == num_images:
            # All intermediate files exist, just need to assemble
            formatter.on_item_complete(item, existing_paths)
            recovered_count += 1
            continue

        for i in range(num_images):
            save_path = formatter.get_save_path(item, i)
            seed = args.seed + i
            raw_prompt = item.get(prompt_field, item["original_prompt"])
            if args.fix_lens_key and isinstance(raw_prompt, str):
                # Rename only the literal JSON key, not arbitrary substrings.
                raw_prompt = raw_prompt.replace('"lens_and_effect"', '"lens-and-effect"')
            task = {
                "item": item,
                "image_idx": i,
                "save_path": save_path,
                "seed": seed,
                "prompt": compact_single_quote_json(raw_prompt),
            }
            # Per-item resolution override (from rewritten JSONL)
            if "height" in item:
                task["height"] = item["height"]
            if "width" in item:
                task["width"] = item["width"]
            tasks.append(task)

    # Group tasks by item id for on_item_complete tracking
    item_tasks = defaultdict(list)
    for t in tasks:
        item_tasks[t["item"]["id"]].append(t)

    items_to_process = {t["item"]["id"]: t["item"] for t in tasks}

    log.info("=" * 60)
    log.info("Inference Pipeline")
    log.info("=" * 60)
    log.info("Benchmark:     %s", benchmark)
    log.info("Prompt field:  %s", prompt_field)
    if recovered_count > 0:
        log.info("Recovered:     %d items from incomplete tmp files", recovered_count)
    log.info("Items:         %d total, %d done, %d recovered, %d to process",
             len(items), skip_count, recovered_count, len(items_to_process))
    log.info("Tasks:         %d image generations", len(tasks))
    log.info("Resolution:    %dx%d", args.width, args.height)
    log.info("Workers:       %d", args.workers)
    log.info("Output:        %s", args.output_dir)

    if not tasks:
        log.info("All items already generated. Running finalize...")
        formatter.finalize(all_items)
        return

    # Set up PSM resolver or use direct URL
    psm_resolver = None
    if args.psm:
        psm_resolver = PSMResolver(
            psm=args.psm,
            expect_trial=args.psm_trial,
            expect_step=args.psm_step,
            expect_ema=args.psm_ema,
        )
        log.info("PSM mode: %s (verify per-request, load-balanced across backends)", args.psm)
    else:
        # Health check (non-fatal — some servers may not implement /health)
        try:
            resp = requests.get(f"{args.url}/health", timeout=5, proxies={"http": None, "https": None})
            resp.raise_for_status()
            log.info("Service health: %s", resp.json().get("status", "unknown"))
        except Exception as e:
            log.warning("Service health check failed: %s (continuing anyway)", e)

    # Track per-item completion
    item_lock = threading.Lock()
    item_completed_paths = defaultdict(list)
    item_expected_count = {}
    for item_id, ts in item_tasks.items():
        item_expected_count[item_id] = len(ts)

    success_count = 0
    fail_count = 0
    pbar = tqdm(total=len(tasks), desc="Generating", unit="img", dynamic_ncols=True)

    def run_task(task):
        """Generate one image and save it."""
        save_path = task["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Per-item resolution or global default
        task_height = task.get("height", args.height)
        task_width = task.get("width", args.width)

        if psm_resolver:
            img = send_generate_psm(
                psm_resolver=psm_resolver,
                prompt=task["prompt"],
                height=task_height,
                width=task_width,
                num_steps=args.num_steps,
                seed=task["seed"],
                cfg_scale=args.cfg_scale,
                negative_prompt=args.negative_prompt,
                timeout=args.timeout,
                timestep_shift=args.timestep_shift,
            )
        else:
            img = send_generate(
                url=args.url,
                prompt=task["prompt"],
                height=task_height,
                width=task_width,
                num_steps=args.num_steps,
                seed=task["seed"],
                cfg_scale=args.cfg_scale,
                negative_prompt=args.negative_prompt,
                timeout=args.timeout,
                timestep_shift=args.timestep_shift,
            )
        img.save(save_path, format="PNG")
        return task

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            item_id = task["item"]["id"]
            try:
                future.result()
                with item_lock:
                    item_completed_paths[item_id].append(task["save_path"])
                    completed = len(item_completed_paths[item_id]) == item_expected_count[item_id]
                if completed:
                    try:
                        formatter.on_item_complete(
                            task["item"],
                            item_completed_paths[item_id],
                        )
                    except Exception as e:
                        log.error("Grid assembly failed id=%s: %s (tmp files kept)", item_id, e)
                success_count += 1
            except Exception as e:
                log.error("Failed id=%s img=%d: %s", item_id, task["image_idx"], e)
                fail_count += 1
            pbar.update(1)

    pbar.close()

    log.info("Generation done. %d success, %d failed.", success_count, fail_count)

    # Finalize (e.g., write image_filepath_data.json for GenEval2)
    # Use all_items so finalize can discover files generated by other ranks
    formatter.finalize(all_items)

    log.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
