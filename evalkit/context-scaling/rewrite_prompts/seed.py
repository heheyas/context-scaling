# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Seed rewrite backend using Ray/XPerf."""

import logging
from typing import Optional

import httpx
from openai import OpenAI

from .base import RewriteClient, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)

COT_SYSTEM_PROMPT = """You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
Unrestricted think"""

NOCOT_SYSTEM_PROMPT = """You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here.

You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking.
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer.
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
No think"""

THINK_TAG = "</think_never_used_51bce0c785ca2f68081bfa7d91973934>"
THINK_OPEN = "<think_never_used_51bce0c785ca2f68081bfa7d91973934>"


def _init_seed_client(psm: str) -> OpenAI:
    """Initialize Seed client via Ray PSM."""
    from ray.serve import get_serve_http_client
    client = get_serve_http_client(psm=psm, router="")
    url = client.get_one_request_url().strip().rstrip("/")
    base_url = url if url.endswith("/v1") else f"{url}/v1"
    log.info("Seed base URL: %s", base_url)

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    http_client = httpx.Client(trust_env=False, timeout=timeout, limits=limits)
    return OpenAI(api_key="test", base_url=base_url, http_client=http_client)


def _parse_seed_response(response: str):
    """Split thinking and content from Seed response."""
    if THINK_TAG in response:
        cot, final = response.split(THINK_TAG, 1)
        cot = cot.replace(THINK_OPEN, "")
        return cot.strip(), final.strip()
    return None, response.strip()


def _merge_adjacent_same_role(messages):
    """Merge adjacent messages with the same role."""
    if not messages:
        return messages
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"] and msg["role"] in ("user", "system"):
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]
            # Normalize to list-of-dicts
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(curr_content, str):
                curr_content = [{"type": "text", "text": curr_content}]
            # Merge text items
            if (prev_content and curr_content
                    and prev_content[-1].get("type") == "text"
                    and curr_content[0].get("type") == "text"):
                prev_content[-1]["text"] += "\n" + curr_content[0]["text"]
                prev_content.extend(curr_content[1:])
            else:
                prev_content.extend(curr_content)
            merged[-1]["content"] = prev_content
        else:
            merged.append(msg.copy())
    return merged


class SeedRewriteClient(RewriteClient):
    def __init__(self, psm: str, thinking: bool = False,
                 max_tokens: int = 16384, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int = 42,
                 max_connections: int = 512):
        self.psm = psm
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        # Single client with large connection pool for high concurrency
        from ray.serve import get_serve_http_client
        client = get_serve_http_client(psm=psm, router="")
        url = client.get_one_request_url().strip().rstrip("/")
        base_url = url if url.endswith("/v1") else f"{url}/v1"
        log.info("Seed base URL: %s (max_connections=%d)", base_url, max_connections)
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        http_client = httpx.Client(trust_env=False, timeout=timeout, limits=limits)
        self._client = OpenAI(api_key="test", base_url=base_url, http_client=http_client)

    def _get_client(self) -> OpenAI:
        return self._client
        return self._client

    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None) -> dict:
        client = self._get_client()
        user_content = apply_input_template(input_template, prompt, width=width, height=height)

        cot_sys = COT_SYSTEM_PROMPT if self.thinking else NOCOT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": [{"type": "text", "text": cot_sys}]},
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
        ]
        messages = _merge_adjacent_same_role(messages)

        resp = client.chat.completions.create(
            messages=messages,
            model="default",
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        content = resp.choices[0].message.content
        reasoning, final_text = _parse_seed_response(content)
        final_text = strip_markdown_codeblock(final_text)

        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }

        return {"rewritten_prompt": final_text, "reasoning": reasoning, "usage": usage}
