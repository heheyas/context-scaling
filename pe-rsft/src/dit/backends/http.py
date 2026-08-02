# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PSM-based DiT backend for image generation.

Resolves a Ray PSM to backend URLs and generates images via HTTP /generate
endpoint with load balancing across backends.

Ported from EvalKit/context-scaling/inference/generate.py (PSMResolver,
send_generate, send_generate_psm).
"""

import os
import sys
import json
import time
import random
import logging
import threading

import requests
from io import BytesIO
from PIL import Image

from src.llm.rollout import compute_dit_seed

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PSM resolution
# ──────────────────────────────────────────────

def _resolve_all_psm_urls(psm: str) -> list[str]:
    """Resolve a PSM to all backend URLs. Requires ray.serve."""
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


# ──────────────────────────────────────────────
# HTTP session (no proxy, large pool)
# ──────────────────────────────────────────────

_GEN_SESSION = None
_GEN_SESSION_LOCK = threading.Lock()


def _patch_getaddrinfo_for_literals():
    """Bypass C resolver for IP literals (avoid IPv6 resolution failures)."""
    import socket as _sock
    if getattr(_sock, "_literal_patched", False):
        return
    _orig_getaddrinfo = _sock.getaddrinfo

    def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        host_str = host
        if isinstance(host, bytes):
            try:
                host_str = host.decode("ascii")
            except (UnicodeDecodeError, AttributeError):
                host_str = None
        if host_str:
            try:
                _sock.inet_pton(_sock.AF_INET, host_str)
                return [(_sock.AF_INET, type or _sock.SOCK_STREAM, proto, "", (host_str, port or 0))]
            except (OSError, ValueError, TypeError):
                pass
            try:
                _sock.inet_pton(_sock.AF_INET6, host_str)
                return [(_sock.AF_INET6, type or _sock.SOCK_STREAM, proto, "", (host_str, port or 0, 0, 0))]
            except (OSError, ValueError, TypeError):
                pass
        return _orig_getaddrinfo(host, port, family, type, proto, flags)

    _sock.getaddrinfo = _patched_getaddrinfo
    _sock._literal_patched = True


def _get_gen_session():
    """Get a shared requests Session with trust_env=False (no proxy)."""
    global _GEN_SESSION
    if _GEN_SESSION is None:
        with _GEN_SESSION_LOCK:
            if _GEN_SESSION is None:
                _patch_getaddrinfo_for_literals()
                s = requests.Session()
                s.trust_env = False
                from requests.adapters import HTTPAdapter
                adapter = HTTPAdapter(pool_connections=128, pool_maxsize=1024, max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _GEN_SESSION = s
    return _GEN_SESSION


# ──────────────────────────────────────────────
# Image generation
# ──────────────────────────────────────────────

def send_generate(url, prompt, height, width, num_steps, seed,
                  cfg_scale, negative_prompt, timeout=600, max_retries=10):
    """Send one generation request. Returns PIL Image."""
    body = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "true_cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
    }

    session = _get_gen_session()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = session.post(f"{url}/generate", json=body, timeout=timeout)
            if resp.status_code in (502, 503):
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
            continue

    if last_err is not None:
        raise last_err
    resp.raise_for_status()


# ──────────────────────────────────────────────
# PSM load balancer
# ──────────────────────────────────────────────

class PSMPool:
    """Thread-safe pool of PSM backend URLs with bad-node cooldown."""

    COOLDOWN_SEC = 120  # exclude a backend for 2 min after failure

    def __init__(self, psm: str):
        self.psm = psm
        self._urls = _resolve_all_psm_urls(psm)
        if not self._urls:
            raise RuntimeError(f"PSM {psm} resolved to 0 backends")
        log.info("[PSM] Resolved %s → %d backends: %s", psm, len(self._urls), self._urls)
        self._lock = threading.Lock()
        self._bad_until: dict[str, float] = {}  # url -> timestamp when cooldown expires

    def get_url(self) -> str:
        now = time.time()
        with self._lock:
            healthy = [u for u in self._urls if self._bad_until.get(u, 0) <= now]
        if not healthy:
            # All backends in cooldown — use any (better than nothing)
            healthy = list(self._urls)
        return random.choice(healthy)

    def report_failure(self, url: str):
        """Mark a backend as bad for COOLDOWN_SEC seconds."""
        with self._lock:
            self._bad_until[url] = time.time() + self.COOLDOWN_SEC
            n_bad = sum(1 for u in self._urls if self._bad_until.get(u, 0) > time.time())
            log.warning("[PSM] Backend %s marked bad for %ds (%d/%d bad)",
                        url, self.COOLDOWN_SEC, n_bad, len(self._urls))

    def refresh(self):
        new = _resolve_all_psm_urls(self.psm)
        if new:
            with self._lock:
                self._urls = new
                self._bad_until.clear()
            log.info("[PSM] Refreshed: %d backends", len(new))


# ──────────────────────────────────────────────
# DiT backend: run_dit_psm
# ──────────────────────────────────────────────

_PSM_POOL = None
_PSM_POOL_LOCK = threading.Lock()


def _get_psm_pool(psm: str) -> PSMPool:
    """Get or create the cached PSM pool."""
    global _PSM_POOL
    if _PSM_POOL is not None:
        return _PSM_POOL
    with _PSM_POOL_LOCK:
        if _PSM_POOL is not None:
            return _PSM_POOL
        _PSM_POOL = PSMPool(psm)
    return _PSM_POOL


def compact_single_quote_json(data):
    """Compact JSON into single-quote format for structured prompts.

    Ported from EvalKit/context-scaling/inference/generate.py.
    """
    PLACEHOLDER_SINGLE = "@@SP_SINGLE_QUOTE@@"
    PLACEHOLDER_DOUBLE = "@@SP_DOUBLE_QUOTE@@"

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return data

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


def run_dit_psm(record: dict, psm_or_pool, dit_args) -> dict:
    """Generate one image via PSM backend.

    Drop-in replacement for legacy run_dit. Takes a record from LLM rollout,
    calls the DiT serve, returns the record with an 'image' field added.

    Args:
        record: Dict with at least {index, image_idx, prompt, seed, width, height,
                structured_prompt, success}.
        psm_or_pool: Either a PSM string or a PSMPool instance.
        dit_args: SimpleNamespace with {num_steps, cfg_scale, negative_prompt,
                  timeout, max_retries}.

    Returns:
        record dict with 'image' (PIL.Image or None) and 'success' updated.
    """
    result = dict(record)

    if not record.get("success", True) or not record.get("structured_prompt"):
        result["image"] = None
        result["success"] = False
        result["error"] = result.get("error", "LLM failed, skipping DiT")
        return result

    # Get PSM pool
    if isinstance(psm_or_pool, str):
        pool = _get_psm_pool(psm_or_pool)
    else:
        pool = psm_or_pool

    url = pool.get_url()

    # Convert SP to single-quote compact format (what the DiT server expects)
    prompt_text = compact_single_quote_json(record["structured_prompt"])

    # DiT seed is shared across all image_idx of a given index, so the N
    # rollouts for one prompt only vary in SP — DiT noise is held fixed.
    dit_seed = compute_dit_seed(record["index"])
    result["dit_seed"] = dit_seed

    try:
        img = send_generate(
            url=url,
            prompt=prompt_text,
            height=record.get("height", 1024),
            width=record.get("width", 1024),
            num_steps=dit_args.num_steps,
            seed=dit_seed,
            cfg_scale=dit_args.cfg_scale,
            negative_prompt=dit_args.negative_prompt,
            timeout=dit_args.timeout,
            max_retries=dit_args.max_retries,
        )
        result["image"] = img
        result["success"] = True
    except Exception as e:
        log.error("[DiT PSM %d/%d] Failed on %s: %s", record["index"], record["image_idx"], url, e)
        # Report bad backend so pool avoids it temporarily
        pool.report_failure(url)
        result["image"] = None
        result["success"] = False
        result["error"] = f"DiT error: {e}"

    return result
