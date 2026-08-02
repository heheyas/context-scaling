# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Generic OpenAI-compatible rewrite backend (any base_url + api_key).

Supports reasoning/thinking via extra_body (e.g. OpenRouter's Qwen models):
    extra_body={"reasoning": {"enabled": True}}
"""

import time
import logging

import httpx
import openai
from openai import OpenAI

from .base import RewriteClient, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)


class OpenAIGenericRewriteClient(RewriteClient):
    """Rewrite client for any OpenAI-compatible API (vLLM, local models, third-party, etc.).

    Args:
        reasoning: Enable reasoning/thinking via extra_body.
                   Sends {"reasoning": {"enabled": True}} to the API.
                   The response's reasoning_content is extracted and returned.
    """

    def __init__(self, base_url: str, api_key: str = "test",
                 model: str = "default", max_tokens: int = 16384,
                 temperature: float = 0.0, max_retries: int = 5,
                 reasoning: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning = reasoning

        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
        # trust_env=True for external APIs (need proxy), False for localhost
        is_local = any(x in base_url for x in ("localhost", "127.0.0.1", "0.0.0.0"))
        http_client = httpx.Client(trust_env=not is_local, timeout=timeout, limits=limits)
        self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        log.info("OpenAI generic client: %s (model=%s, reasoning=%s)",
                 base_url, model, reasoning)

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

        extra_kwargs = {}
        if self.reasoning:
            extra_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **extra_kwargs,
                )
                msg = resp.choices[0].message
                content = msg.content or ""

                # Extract reasoning from response
                reasoning_text = None
                if self.reasoning:
                    # OpenRouter / some providers put it in reasoning_content
                    reasoning_text = getattr(msg, "reasoning_content", None)
                    # Some providers use reasoning_details
                    if not reasoning_text:
                        details = getattr(msg, "reasoning_details", None)
                        if details:
                            reasoning_text = details if isinstance(details, str) else str(details)
                    # Fallback: parse <think>...</think> from content
                    if not reasoning_text and "</think>" in content:
                        parts = content.split("</think>", 1)
                        reasoning_text = parts[0].replace("<think>", "").strip()
                        content = parts[1].strip()

                text = strip_markdown_codeblock(content)

                # If reasoning ate all tokens and content is empty, retry without reasoning
                if not text.strip() and self.reasoning:
                    log.warning("Reasoning produced empty content, retrying without reasoning...")
                    try:
                        fb_resp = self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stream=False,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                        )
                        text = strip_markdown_codeblock(fb_resp.choices[0].message.content or "")
                        if fb_resp.usage:
                            return {
                                "rewritten_prompt": text,
                                "reasoning": reasoning_text,
                                "usage": {
                                    "prompt_tokens": getattr(fb_resp.usage, "prompt_tokens", None),
                                    "completion_tokens": getattr(fb_resp.usage, "completion_tokens", None),
                                    "total_tokens": getattr(fb_resp.usage, "total_tokens", None),
                                },
                            }
                    except Exception as e:
                        log.warning("Reasoning fallback failed: %s", e)

                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                return {"rewritten_prompt": text, "reasoning": reasoning_text, "usage": usage}

            except (openai.RateLimitError, openai.APIStatusError) as e:
                log.warning("OpenAI API error (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                time.sleep(min(2 ** attempt, 60))
            except Exception as e:
                log.warning("OpenAI call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                time.sleep(min(2 ** attempt, 60))

        raise RuntimeError(f"OpenAI generic rewrite failed after {self.max_retries} retries")
