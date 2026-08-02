# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""GPT rewrite backend using AzureOpenAI + KeyPool."""

import time
import logging

import openai

from .base import RewriteClient, KeyPool, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-chat-2025-08-07"
DEFAULT_API_VERSION = "2024-02-01"


class GPTRewriteClient(RewriteClient):
    def __init__(self, key_pool: KeyPool, model: str = DEFAULT_MODEL,
                 max_tokens: int = 16384, temperature: float = 0.0,
                 max_retries: int = 8):
        self.key_pool = key_pool
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None) -> dict:
        user_content = apply_input_template(input_template, prompt, width=width, height=height)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}],
            },
        ]

        for attempt in range(self.max_retries):
            key, client = self.key_pool.acquire()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_headers={"X-TT-LOGID": "rewrite_prompt"},
                )
                text = strip_markdown_codeblock(resp.choices[0].message.content)
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                return {"rewritten_prompt": text, "reasoning": None, "usage": usage}

            except (openai.RateLimitError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if isinstance(e, openai.RateLimitError) or status == 429:
                    self.key_pool.report_429(key)
                    wait = min(1.01 ** attempt, 60)
                    log.warning("GPT rate limited key=...%s (attempt %d), switching...",
                                key[-8:], attempt + 1)
                    time.sleep(wait)
                elif status and status >= 500:
                    log.warning("GPT server error %d (attempt %d)", status, attempt + 1)
                    time.sleep(1.01 ** attempt)
                else:
                    log.warning("GPT API error (attempt %d): %s", attempt + 1, e)
                    time.sleep(1.01 ** attempt)
            except Exception as e:
                log.warning("GPT call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(1.01 ** attempt)

        raise RuntimeError(f"GPT rewrite failed after {self.max_retries} retries")
