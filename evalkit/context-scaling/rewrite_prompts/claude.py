# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Claude rewrite backend via OpenRouter (OpenAI-compatible API)."""

import time
import logging

import httpx
import openai
from openai import OpenAI

from .base import RewriteClient, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://vip.aipro.love/v1"
DEFAULT_MODEL = "anthropic/claude-opus-4"


class ClaudeRewriteClient(RewriteClient):
    """Rewrite client for Claude models via OpenRouter."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 max_tokens: int = 16384, temperature: float = 0.0,
                 max_retries: int = 8):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
        http_client = httpx.Client(trust_env=True, timeout=timeout, limits=limits)
        self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        log.info("Claude client: %s (model=%s)", base_url, model)

    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None,
                image_base64: str = None) -> dict:
        user_content = apply_input_template(input_template, prompt, width=width, height=height)
        if image_base64 is not None:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": user_content},
                ],
            }
        else:
            user_msg = {"role": "user", "content": user_content}
        messages = [
            {"role": "system", "content": system_prompt},
            user_msg,
        ]

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                text = strip_markdown_codeblock(resp.choices[0].message.content)
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                return {"rewritten_prompt": text, "reasoning": None, "usage": usage}

            except (openai.RateLimitError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                wait = min(2 ** attempt, 60)
                if isinstance(e, openai.RateLimitError) or status == 429:
                    log.warning("Claude rate limited (attempt %d/%d), waiting %.1fs...",
                                attempt + 1, self.max_retries, wait)
                elif status and status >= 500:
                    log.warning("Claude server error %d (attempt %d/%d)",
                                status, attempt + 1, self.max_retries)
                else:
                    log.warning("Claude API error (attempt %d/%d): %s",
                                attempt + 1, self.max_retries, e)
                time.sleep(wait)
            except Exception as e:
                wait = min(2 ** attempt, 60)
                log.warning("Claude call failed (attempt %d/%d): %s",
                            attempt + 1, self.max_retries, e)
                time.sleep(wait)

        raise RuntimeError(f"Claude rewrite failed after {self.max_retries} retries")
