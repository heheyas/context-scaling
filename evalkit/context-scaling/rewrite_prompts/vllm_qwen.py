# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 rewrite backend via vLLM OpenAI-compatible API.

Use with scripts/serve_qwen_llm.sh to serve the model, then call this backend.
Much faster than the local HuggingFace backend thanks to vLLM's continuous
batching, PagedAttention, and expert parallelism.
"""

import json
import time
import logging

import requests as _requests

from .base import RewriteClient, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)


class VllmQwenRewriteClient(RewriteClient):
    """Qwen3.5 rewrite via vLLM serving (OpenAI chat API)."""

    def __init__(self, url: str = "http://localhost:8000",
                 model: str = None,
                 max_tokens: int = 16384,
                 min_tokens: int = 0,
                 temperature: float = 0.0,
                 thinking: bool = False,
                 max_retries: int = 5):
        self.url = url.rstrip("/")
        self.model = model  # auto-detected from server if None
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.max_retries = max_retries

    def _detect_model(self):
        """Auto-detect model name from vLLM server."""
        if self.model is None or self.model == "auto":
            resp = _requests.get(f"{self.url}/v1/models", timeout=10)
            resp.raise_for_status()
            self.model = resp.json()["data"][0]["id"]
            log.info("Auto-detected model: %s", self.model)

    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None,
                image_base64: str = None) -> dict:
        self._detect_model()

        user_content = apply_input_template(input_template, prompt, width=width, height=height)

        if image_base64 is not None:
            # Vision mode: text before image so system_prompt + text can hit prefix cache
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        else:
            user_msg = {"role": "user", "content": user_content}

        messages = [
            {"role": "system", "content": system_prompt},
            user_msg,
        ]

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if self.temperature > 0 else 0,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.min_tokens > 0:
            body["min_tokens"] = self.min_tokens

        for attempt in range(self.max_retries):
            try:
                resp = _requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=body,
                    timeout=3600,
                )
                resp.raise_for_status()
                data = resp.json()

                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""

                # vLLM with --reasoning-parser puts thinking in "reasoning"
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")

                # Fallback: if reasoning_content not in response, parse from content
                if not reasoning and self.thinking and "</think>" in content:
                    parts = content.split("</think>", 1)
                    reasoning = parts[0].replace("<think>", "").strip()
                    content = parts[1].strip()

                content = strip_markdown_codeblock(content)

                usage = {}
                if "usage" in data and data["usage"]:
                    usage = {
                        "prompt_tokens": data["usage"].get("prompt_tokens"),
                        "completion_tokens": data["usage"].get("completion_tokens"),
                        "total_tokens": data["usage"].get("total_tokens"),
                    }

                # If thinking mode produced empty content (thinking used all tokens),
                # retry without thinking as fallback
                if not content.strip() and self.thinking:
                    log.warning("Thinking produced empty content, retrying without thinking as fallback...")
                    fallback_body = dict(body)
                    fallback_body["chat_template_kwargs"] = {"enable_thinking": False}
                    fallback_body["max_tokens"] = 16384
                    try:
                        fb_resp = _requests.post(
                            f"{self.url}/v1/chat/completions",
                            json=fallback_body,
                            timeout=3600,
                        )
                        fb_resp.raise_for_status()
                        fb_data = fb_resp.json()
                        fb_msg = fb_data["choices"][0]["message"]
                        content = fb_msg.get("content") or ""
                        content = strip_markdown_codeblock(content)
                        reasoning = "(fallback: thinking exceeded token limit)"
                        if fb_data.get("usage"):
                            usage = {
                                "prompt_tokens": fb_data["usage"].get("prompt_tokens"),
                                "completion_tokens": fb_data["usage"].get("completion_tokens"),
                                "total_tokens": fb_data["usage"].get("total_tokens"),
                            }
                    except Exception as fb_e:
                        log.warning("Fallback also failed: %s", fb_e)

                return {"rewritten_prompt": content, "reasoning": reasoning, "usage": usage}

            except Exception as e:
                log.warning("vLLM call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        # All retries failed - last resort: try without thinking
        if self.thinking:
            log.warning("All retries failed, final fallback without thinking...")
            try:
                fallback_body = dict(body)
                fallback_body["chat_template_kwargs"] = {"enable_thinking": False}
                fallback_body["max_tokens"] = 16384
                fb_resp = _requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=fallback_body,
                    timeout=3600,
                )
                fb_resp.raise_for_status()
                fb_data = fb_resp.json()
                content = fb_data["choices"][0]["message"].get("content") or ""
                content = strip_markdown_codeblock(content)
                return {
                    "rewritten_prompt": content,
                    "reasoning": "(fallback: all thinking attempts failed)",
                    "usage": {},
                }
            except Exception as fb_e:
                log.error("Final fallback also failed: %s", fb_e)

        raise RuntimeError(f"vLLM rewrite failed after {self.max_retries} retries")
