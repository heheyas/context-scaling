# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Base class for rewrite backends and shared utilities (KeyPool)."""

import json
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import openai
import httpx

log = logging.getLogger(__name__)


def strip_markdown_codeblock(text: str) -> str:
    """Strip markdown code block wrapper (```json ... ``` or ``` ... ```) if present."""
    s = text.strip()
    if s.startswith("```"):
        # Remove opening line (```json, ```markdown, ``` etc.)
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        # Remove closing ```
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s


def apply_input_template(input_template: str, prompt: str,
                         width: int = None, height: int = None) -> str:
    """Apply input_template by replacing placeholders with actual values.

    Supported placeholders: <prompt>, <width>, <height>
    Example: "<prompt> [width: <width>, height: <height>]"
           → "a cat [width: 1024, height: 1024]"
    """
    assert "<prompt>" in input_template, \
        f"input_template must contain '<prompt>' placeholder, got: {input_template!r}"
    result = input_template.replace("<prompt>", prompt)
    if "<width>" in result and width is not None:
        result = result.replace("<width>", str(width))
    if "<height>" in result and height is not None:
        result = result.replace("<height>", str(height))
    return result


class RewriteClient(ABC):
    """Abstract base class for all rewrite backends."""

    @abstractmethod
    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None) -> dict:
        """Rewrite a single prompt.

        Args:
            prompt: Original prompt text.
            system_prompt: System prompt instructing how to rewrite.
            input_template: Template for the user message. Supported placeholders:
                            <prompt> (required), <width>, <height>.
                            Default "<prompt>" sends raw prompt as-is.
            width: Image width (for <width> placeholder).
            height: Image height (for <height> placeholder).

        Returns:
            {"rewritten_prompt": str, "reasoning": str|None, "usage": dict}
        """
        raise NotImplementedError


class KeyPool:
    """Thread-safe API key pool with automatic rotation on rate limits.

    Supports two config formats:
      1. JSON file (api_config.json): [{"api_key": "...", "base_url": "...", "qpm": 300}, ...]
      2. Text file (keys.txt): one line per key, format: "api_key [base_url]"

    Selection strategy (per acquire call):
      1. Avoid keys rate-limited in the last 60s
      2. Prefer keys with fewer total 429s
      3. Round-robin via usage count for even distribution
    """

    def __init__(self, key_configs: List[Dict[str, str]], api_version: str = "2024-03-01-preview"):
        self._api_version = api_version
        # Deduplicate by api_key, keep last seen base_url
        seen: Dict[str, str] = {}
        for cfg in key_configs:
            seen[cfg["api_key"]] = cfg["base_url"]
        self._keys = list(seen.keys())
        self._base_urls = dict(seen)
        self._lock = threading.Lock()
        self._429_count: Dict[str, int] = {k: 0 for k in self._keys}
        self._429_ts: Dict[str, float] = {k: 0.0 for k in self._keys}
        self._usage: Dict[str, int] = {k: 0 for k in self._keys}
        self._clients: Dict[str, openai.AzureOpenAI] = {}

    def _make_client(self, api_key: str, base_url: str) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=self._api_version,
            http_client=httpx.Client(trust_env=False),
        )

    def acquire(self) -> Tuple[str, openai.AzureOpenAI]:
        """Return (key, client) for the best available key."""
        with self._lock:
            now = time.time()
            best = min(self._keys, key=lambda k: (
                1 if (now - self._429_ts[k]) < 60 else 0,
                self._429_count[k],
                self._usage[k],
            ))
            self._usage[best] += 1
            if best not in self._clients:
                self._clients[best] = self._make_client(best, self._base_urls[best])
            return best, self._clients[best]

    def report_429(self, key: str):
        """Record a 429 rate-limit hit for the given key."""
        with self._lock:
            self._429_count[key] += 1
            self._429_ts[key] = time.time()

    def stats(self) -> str:
        with self._lock:
            parts = []
            for k in self._keys:
                parts.append(f"...{k[-8:]}: {self._usage[k]} calls, {self._429_count[k]} 429s")
            return " | ".join(parts)

    @property
    def size(self) -> int:
        return len(self._keys)


def load_api_config(path: str) -> List[Dict[str, str]]:
    """Load API key config from either JSON or text file.

    JSON format: [{"api_key": "...", "base_url": "...", "qpm": 300}, ...]
    Text format: one line per key, "api_key [base_url]"
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        # JSON format
        configs = json.loads(content)
        return [{"api_key": c["api_key"], "base_url": c["base_url"]} for c in configs]
    else:
        # Text format (like keys.txt)
        results = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            api_key = parts[0]
            base_url = parts[1].strip() if len(parts) > 1 else None
            results.append({"api_key": api_key, "base_url": base_url})
        return results
