# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Gemini 3 Pro judge for caption-detailness measurement.

Wraps the OpenAI-compatible Azure endpoint used in EvalKit/gemini_bon.py with:
  - a thread-safe KeyPool that rotates on 429s,
  - three task-specific wrappers: decompose / verify / oracle,
  - JSON-output parsing with one bounded repair retry.

All three wrappers accept a `prompt_version` string that the caller is
expected to pass in (hash of the prompt file body); it is only used by the
cache layer, not by the API calls themselves.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import httpx as _httpx
import openai


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Defaults (override via env / CLI)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "<OPENAI_COMPATIBLE_BASE_URL>"
DEFAULT_MODEL = "gemini-3-pro-preview-new"
DEFAULT_KEY_FILE = os.environ.get("GEMINI_API_KEY_FILE", "")


# ──────────────────────────────────────────────────────────────────────
# Key pool
# ──────────────────────────────────────────────────────────────────────

def _make_client(api_key: str, base_url: str) -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=base_url,
        api_version="2024-03-01-preview",
        http_client=_httpx.Client(timeout=120.0),
    )


class KeyPool:
    """Thread-safe pool. Prefers keys not recently rate-limited and evenly used."""

    def __init__(self, key_configs: List[Tuple[str, str]]):
        seen: Dict[str, str] = {}
        for api_key, base_url in key_configs:
            seen[api_key] = base_url
        self._keys = list(seen.keys())
        self._base_urls = dict(seen)
        self._lock = threading.Lock()
        self._429_count: Dict[str, int] = {k: 0 for k in self._keys}
        self._429_ts: Dict[str, float] = {k: 0.0 for k in self._keys}
        self._usage: Dict[str, int] = {k: 0 for k in self._keys}
        self._clients: Dict[str, openai.AzureOpenAI] = {}

    def acquire(self) -> Tuple[str, openai.AzureOpenAI]:
        with self._lock:
            now = time.time()
            best = min(
                self._keys,
                key=lambda k: (
                    1 if (now - self._429_ts[k]) < 60 else 0,
                    self._429_count[k],
                    self._usage[k],
                ),
            )
            self._usage[best] += 1
            if best not in self._clients:
                self._clients[best] = _make_client(best, self._base_urls[best])
            return best, self._clients[best]

    def report_429(self, key: str):
        with self._lock:
            self._429_count[key] += 1
            self._429_ts[key] = time.time()

    def stats(self) -> str:
        with self._lock:
            parts = []
            for k in self._keys:
                host = self._base_urls[k].split("//")[-1].split("/")[0]
                parts.append(
                    f"...{k[-8:]}@{host}: {self._usage[k]} calls, "
                    f"{self._429_count[k]} 429s"
                )
            return " | ".join(parts)


def load_api_keys(path: Optional[str] = None) -> List[Tuple[str, str]]:
    path = path or DEFAULT_KEY_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Key file not found: {path}")
    configs: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            api_key = parts[0]
            base_url = parts[1].strip() if len(parts) > 1 else DEFAULT_BASE_URL
            configs.append((api_key, base_url))
    if not configs:
        raise ValueError(f"No keys parsed from {path}")
    return configs


def load_api_keys_json(path: str) -> List[Tuple[str, str]]:
    """Load keys from JSON config (T2I-CoReBench style):
    [{"api_key": "...", "base_url": "...", "qpm": 300}, ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"API config JSON not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, list):
        raise ValueError(f"Expected JSON list, got {type(cfg)}")
    configs = [(item["api_key"], item["base_url"]) for item in cfg
               if "api_key" in item and "base_url" in item]
    if not configs:
        raise ValueError(f"No valid keys in {path}")
    return configs


# ──────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────

def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def sniff_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "image/webp"
    return "image/png"


# ──────────────────────────────────────────────────────────────────────
# Low-level call with retry / key rotation
# ──────────────────────────────────────────────────────────────────────

_MAX_RETRIES = 6


def _chat(
    pool: KeyPool,
    messages: list,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """Returns the assistant response text, or None after exhausted retries."""
    for attempt in range(_MAX_RETRIES):
        key, client = pool.acquire()
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={"X-TT-LOGID": "bagel_detailness_judge"},
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except (openai.RateLimitError, openai.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if isinstance(e, openai.RateLimitError) or status == 429:
                pool.report_429(key)
                # Gentle 1.01^n backoff for high-concurrency workloads (1024+ workers)
                wait = min(1.01 ** attempt, 30)
                log.warning(
                    "429 on key ...%s (attempt %d), backoff %.2fs",
                    key[-8:], attempt + 1, wait,
                )
                time.sleep(wait)
            elif status and status >= 500:
                wait = min(1.5 ** attempt, 30)
                log.warning("server %s (attempt %d), backoff %.1fs", status, attempt + 1, wait)
                time.sleep(wait)
            else:
                log.warning("API error (attempt %d): %s", attempt + 1, e)
                time.sleep(1.5 ** attempt)
        except Exception as e:
            log.warning("call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.5 ** attempt)
    return None


# ──────────────────────────────────────────────────────────────────────
# JSON output parsing
# ──────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _parse_json_array(text: str) -> Optional[list]:
    if text is None:
        return None
    clean = _strip_fences(text)
    try:
        val = json.loads(clean)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    m = _ARRAY_RE.search(clean)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return val
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Stage A — decompose caption into atomic claims (text only)
# ──────────────────────────────────────────────────────────────────────

def decompose(
    pool: KeyPool,
    prompt_template: str,
    caption: str,
    model: str = DEFAULT_MODEL,
) -> Optional[List[str]]:
    """Return list of atomic claim strings, or None on hard failure.

    `prompt_template` must contain a `SYSTEM` / `USER` split with `{caption}`
    placeholder in the user section — use the loaded decompose.txt body.
    """
    system_text, user_template = _split_template(prompt_template)
    user_text = user_template.replace("{caption}", caption)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    claims = _parse_json_array(raw)
    if claims is None and raw is not None:
        # one repair attempt — ask for strict JSON only
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a JSON "
                "array of claim strings. No other text."
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=8192)
        claims = _parse_json_array(raw)
    if claims is None:
        return None
    return [str(c).strip() for c in claims if str(c).strip()]


# ──────────────────────────────────────────────────────────────────────
# Stage B — verify a batch of claims against an image
# ──────────────────────────────────────────────────────────────────────

_VALID_VERDICTS = ("SUPPORTED", "UNSUPPORTED", "IRRELEVANT")
_VALID_ENTAIL = ("YES", "NO")


def verify(
    pool: KeyPool,
    prompt_template: str,
    image_bytes: bytes,
    claims: List[str],
    model: str = DEFAULT_MODEL,
) -> Optional[List[str]]:
    """Return list of verdict strings, one per claim. None on hard failure.

    The returned length is guaranteed to equal `len(claims)` on success; any
    unparseable / out-of-vocab verdict is normalised to UNSUPPORTED.
    """
    if not claims:
        return []
    system_text, user_template = _split_template(prompt_template)
    claims_json = json.dumps(claims, ensure_ascii=False)
    user_text = user_template.replace("{claims_json}", claims_json)
    data_url = image_bytes_to_data_url(image_bytes, mime=sniff_mime(image_bytes))
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    verdicts = _parse_json_array(raw)
    if verdicts is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a JSON "
                f"array of exactly {len(claims)} verdict strings, each one of "
                f"{list(_VALID_VERDICTS)}, in the same order."
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=8192)
        verdicts = _parse_json_array(raw)
    if verdicts is None:
        return None
    # Normalise length + vocabulary.
    out: List[str] = []
    for i in range(len(claims)):
        if i < len(verdicts):
            v = str(verdicts[i]).strip().upper()
            if v not in _VALID_VERDICTS:
                v = "UNSUPPORTED"
        else:
            v = "UNSUPPORTED"
        out.append(v)
    return out


# ──────────────────────────────────────────────────────────────────────
# v2b — Source tuple deduplication (text-only)
# Converts the raw OAR tuple dict (often with composite A strings)
# into a canonical, atomic tuple dict. Keeps O/R/G intact.
# ──────────────────────────────────────────────────────────────────────

def dedupe_source(
    pool: KeyPool,
    prompt_template: str,
    source_tuples: dict,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Canonicalise `source_tuples` into atomic, non-redundant form.

    Returns {O, A, R, G} dict or None on hard failure. On success, O/R/G
    preserved; A replaced with an atomic, deduplicated set.
    """
    system_text, user_template = _split_template(prompt_template)
    src_js = json.dumps(source_tuples, ensure_ascii=False)
    user_text = user_template.replace("{source_tuples_json}", src_js)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=16384,
                response_format={"type": "json_object"})
    obj = _parse_json_object(raw)
    if obj is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a "
                'JSON object with keys "O", "A", "R", "G".'
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=16384,
                    response_format={"type": "json_object"})
        obj = _parse_json_object(raw)
    if obj is None:
        return None
    out = {}
    for k in _OAR_KEYS:
        v = obj.get(k, [])
        if not isinstance(v, list):
            v = []
        out[k] = [x for x in v if isinstance(x, dict)]
    return out


# ──────────────────────────────────────────────────────────────────────
# Rewrite — JSON source → NL paragraph at a given level (text-only)
# ──────────────────────────────────────────────────────────────────────

def flatten_to_nl(
    pool: KeyPool,
    prompt_template: str,
    json_text: str,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Faithfully flatten a structured JSON caption into a single NL prose
    paragraph. Preserves all visual facts, adds nothing."""
    system_text, user_template = _split_template(prompt_template)
    user_text = user_template.replace("{json_text}", json_text)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    if raw is None:
        return None
    return raw.strip()


def rewrite_dense(
    pool: KeyPool,
    prompt_template: str,
    source_json: str,
    level: str,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Rewrite a raw scene JSON into NL prose at a given detail level."""
    system_text, user_template = _split_template(prompt_template)
    user_text = (user_template
                 .replace("{level}", level)
                 .replace("{source_json}", source_json))
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    if raw is None:
        return None
    return raw.strip()


# ──────────────────────────────────────────────────────────────────────
# Stage D — does caption T entail each oracle fact? (text-only)
# ──────────────────────────────────────────────────────────────────────

def entail(
    pool: KeyPool,
    prompt_template: str,
    caption: str,
    oracle_claims: List[str],
    model: str = DEFAULT_MODEL,
) -> Optional[List[str]]:
    """Return list of YES/NO strings, one per oracle claim. None on hard failure.

    Length is guaranteed to equal len(oracle_claims) on success; unparseable
    tokens are normalised to "NO" (conservative — does not inflate recall).
    """
    if not oracle_claims:
        return []
    system_text, user_template = _split_template(prompt_template)
    claims_json = json.dumps(oracle_claims, ensure_ascii=False)
    user_text = (user_template
                 .replace("{caption}", caption)
                 .replace("{claims_json}", claims_json))
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    verdicts = _parse_json_array(raw)
    if verdicts is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a JSON "
                f"array of exactly {len(oracle_claims)} entries, each one of "
                f"{list(_VALID_ENTAIL)}, same order."
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=8192)
        verdicts = _parse_json_array(raw)
    if verdicts is None:
        return None
    out: List[str] = []
    for i in range(len(oracle_claims)):
        if i < len(verdicts):
            v = str(verdicts[i]).strip().upper()
            if v not in _VALID_ENTAIL:
                v = "NO"
        else:
            v = "NO"
        out.append(v)
    return out


# ──────────────────────────────────────────────────────────────────────
# Stage C — oracle claim list for an image
# ──────────────────────────────────────────────────────────────────────

def oracle(
    pool: KeyPool,
    prompt_template: str,
    image_bytes: bytes,
    model: str = DEFAULT_MODEL,
) -> Optional[List[str]]:
    system_text, user_template = _split_template(prompt_template)
    data_url = image_bytes_to_data_url(image_bytes, mime=sniff_mime(image_bytes))
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_template},
            ],
        },
    ]
    raw = _chat(pool, messages, model=model, max_tokens=8192)
    claims = _parse_json_array(raw)
    if claims is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a JSON "
                "array of claim strings. No other text."
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=8192)
        claims = _parse_json_array(raw)
    if claims is None:
        return None
    return [str(c).strip() for c in claims if str(c).strip()]


# ──────────────────────────────────────────────────────────────────────
# v2 — OAR tuple extraction from a caption (text-only, JSON out)
# ──────────────────────────────────────────────────────────────────────

_OAR_KEYS = ("O", "A", "R", "G")


def _parse_json_object(text: str) -> Optional[dict]:
    if text is None:
        return None
    clean = _strip_fences(text).strip()
    try:
        val = json.loads(clean)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    # Fallback: find first {...} block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    return None


def extract_tuples(
    pool: KeyPool,
    prompt_template: str,
    caption: str,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """Return {O, A, R, G} each a list of dicts, or None on failure."""
    system_text, user_template = _split_template(prompt_template)
    user_text = user_template.replace("{caption}", caption)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    raw = _chat(pool, messages, model=model, max_tokens=16384,
                response_format={"type": "json_object"})
    obj = _parse_json_object(raw)
    if obj is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY a JSON "
                'object with keys "O", "A", "R", "G". No other text.'
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=16384,
                    response_format={"type": "json_object"})
        obj = _parse_json_object(raw)
    if obj is None:
        return None
    # Ensure all keys present and are lists of dicts.
    out = {}
    for k in _OAR_KEYS:
        v = obj.get(k, [])
        if not isinstance(v, list):
            v = []
        out[k] = [x for x in v if isinstance(x, dict)]
    return out


# ──────────────────────────────────────────────────────────────────────
# v2 — Symmetric matching of two tuple lists (text-only, JSON out)
# ──────────────────────────────────────────────────────────────────────

def match_tuples(
    pool: KeyPool,
    prompt_template: str,
    source_tuples: dict,
    caption_tuples: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Optional[dict]:
    """Return {"source_recall": {O:[...], ...}, "caption_precision": {...}}.

    All mask lists are normalised to the length of the corresponding
    tuple list; unparseable entries → NO.
    """
    system_text, user_template = _split_template(prompt_template)
    src_js = json.dumps(source_tuples, ensure_ascii=False)
    cap_js = json.dumps(caption_tuples, ensure_ascii=False)
    user_text = (user_template
                 .replace("{source_tuples_json}", src_js)
                 .replace("{caption_tuples_json}", cap_js))
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    # Big max_tokens: total mask length can exceed 150 YES/NO values for
    # rich scenes; pad generously to avoid truncation cutoffs.
    raw = _chat(pool, messages, model=model, max_tokens=16384,
                temperature=temperature,
                response_format={"type": "json_object"})
    obj = _parse_json_object(raw)
    if obj is None and raw is not None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                "Your previous reply was not valid JSON. Output ONLY the "
                'required JSON object with keys "source_recall" and '
                '"caption_precision".'
            ),
        })
        raw = _chat(pool, messages, model=model, max_tokens=16384,
                    temperature=temperature,
                    response_format={"type": "json_object"})
        obj = _parse_json_object(raw)
    if obj is None:
        return None

    def _normalise_mask(d, expected_lengths):
        out = {}
        for k in _OAR_KEYS:
            L = expected_lengths.get(k, 0)
            mask = d.get(k, []) if isinstance(d, dict) else []
            if not isinstance(mask, list):
                mask = []
            vs = []
            for i in range(L):
                v = str(mask[i]).strip().upper() if i < len(mask) else "NO"
                if v not in ("YES", "NO"):
                    v = "NO"
                vs.append(v)
            out[k] = vs
        return out

    src_lens = {k: len(source_tuples.get(k, [])) for k in _OAR_KEYS}
    cap_lens = {k: len(caption_tuples.get(k, [])) for k in _OAR_KEYS}
    return {
        "source_recall": _normalise_mask(obj.get("source_recall", {}), src_lens),
        "caption_precision": _normalise_mask(obj.get("caption_precision", {}), cap_lens),
    }


# ──────────────────────────────────────────────────────────────────────
# Prompt template helpers
# ──────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^SYSTEM\s*\n(.*?)\n\s*USER\s*\n(.*)$", re.DOTALL)


def _split_template(body: str) -> Tuple[str, str]:
    m = _SECTION_RE.match(body)
    if not m:
        raise ValueError("Prompt file must begin with SYSTEM/USER sections.")
    return m.group(1).strip(), m.group(2).rstrip()


def load_prompt(path: str) -> Tuple[str, str]:
    """Returns (body, version). Version is sha1 of the raw body."""
    import hashlib
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    version = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    return body, version
