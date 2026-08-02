# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""vLLM backend for LLM rollout.

Drop-in replacement for legacy run_llm. Calls a local vLLM server's
/v1/chat/completions endpoint instead of the internal PSM-based
call_seed_for_structured_prompt.

Everything else (input template substitution, response parsing,
structured prompt extraction) is identical to legacy run_llm.
"""

import json
import math
import re
import logging
import traceback
from math import gcd

import requests
import json_repair

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# resolve_target_size — ratio string → (w, h) with bucket alignment
# ──────────────────────────────────────────────

def _round_to_factor(x: float, factor: int) -> int:
    return max(factor, round(x / factor) * factor)


def _parse_ratio_str(s: str):
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
    if not m:
        return None
    w, h = float(m.group(1)), float(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


def bucket_align_size(*, height, width, bucket_size, factor=32,
                      min_shift_range=0.95, max_shift_range=1.05,
                      min_side=256, max_side=16384):
    if width <= 0 or height <= 0:
        return bucket_size, bucket_size
    target_pixels = float(bucket_size) ** 2
    min_pixels = target_pixels * min_shift_range
    max_pixels = target_pixels * max_shift_range
    r = float(height) / float(width)
    w0 = math.sqrt(target_pixels / r)
    h0 = w0 * r

    def quantize(hf, wf):
        wq = _round_to_factor(wf, factor)
        hq = _round_to_factor(hf, factor)
        return int(min(max(hq, min_side), max_side)), int(min(max(wq, min_side), max_side))

    hq, wq = quantize(h0, w0)
    for _ in range(8):
        px = float(hq) * float(wq)
        if min_pixels <= px <= max_pixels:
            break
        scale = math.sqrt((min_pixels if px < min_pixels else max_pixels) / max(px, 1.0))
        hq, wq = quantize(float(hq) * scale, float(wq) * scale)
    return int(hq), int(wq)


def resolve_target_size(*, ratio_spec, bucket_size, factor=32):
    """Convert ratio string to (width, height) with bucket alignment."""
    default = (int(bucket_size), int(bucket_size))
    if ratio_spec is None:
        return default
    s = str(ratio_spec).strip()
    wh = _parse_ratio_str(s)
    if wh is not None:
        w_r, h_r = wh
        new_h, new_w = bucket_align_size(height=h_r, width=w_r,
                                         bucket_size=bucket_size, factor=factor)
        return int(new_w), int(new_h)
    return default


# ──────────────────────────────────────────────
# vLLM run_llm
# ──────────────────────────────────────────────

_cached_model_name = {}  # {url: model_name} — auto-detect cache


def _get_model_name(url, model_hint):
    """Return model name, auto-detecting from server once and caching."""
    if model_hint:
        return model_hint
    if url in _cached_model_name:
        return _cached_model_name[url]
    try:
        resp = requests.get(f"{url}/v1/models", timeout=10)
        resp.raise_for_status()
        name = resp.json()["data"][0]["id"]
        _cached_model_name[url] = name
        log.info("Auto-detected vLLM model: %s", name)
        return name
    except Exception:
        return "default"


def run_llm_vllm(index, image_idx, prompt, seed, args, system_prompt,
                 width, height, ratio_str):
    """Drop-in replacement for legacy run_llm using vLLM /v1/chat/completions.

    The `args` namespace must have:
        vllm_url, input_template, llm_temperature, llm_top_p, thinking,
        bucket_size (for ratio override), and optionally vllm_model.
    """
    try:
        g = gcd(width, height)
        aspect_ratio = f"{width // g}:{height // g}"

        input_template = args.input_template
        assert "<prompt>" in input_template
        llm_input = input_template.replace("<prompt>", prompt)
        if "<aspect_ratio>" in input_template:
            llm_input = llm_input.replace("<aspect_ratio>", aspect_ratio)
        if "<width>" in input_template:
            llm_input = llm_input.replace("<width>", str(width))
        if "<height>" in input_template:
            llm_input = llm_input.replace("<height>", str(height))

        # Build messages.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": llm_input})

        # Call vLLM.
        url = getattr(args, "vllm_url", "http://localhost:8000")
        model = _get_model_name(url, getattr(args, "vllm_model", None))

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": getattr(args, "max_tokens", 65536),
            "temperature": args.llm_temperature if args.llm_temperature > 0 else 0,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": getattr(args, "thinking", False)},
        }

        resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=3600)
        resp.raise_for_status()
        data = resp.json()

        msg = data["choices"][0]["message"]
        final_content = msg.get("content") or ""

        # Extract reasoning from vLLM response.
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if not reasoning and getattr(args, "thinking", False) and "</think>" in final_content:
            parts = final_content.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            final_content = parts[1].strip()

        # Build llm_raw_response with standard Qwen3.5 <think> tags.
        # (Legacy PSM backend used a custom tag; vLLM uses the real one.)
        if reasoning:
            llm_raw_response = (
                "<think>" + reasoning + "</think>" + final_content
            )
        else:
            llm_raw_response = final_content

        # Parse structured prompt from final_content.
        # Strip <scene_imagine>...</scene_imagine> if present (some system
        # prompts ask for a scene description before the JSON output).
        content_for_parsing = final_content
        if "</scene_imagine>" in content_for_parsing:
            content_for_parsing = content_for_parsing.split("</scene_imagine>", 1)[1]

        # Extract JSON: try ```json fenced block first, then raw content.
        sp = content_for_parsing.split('```json')[-1].split('```')[0].strip()
        sp_json = json_repair.loads(sp)
        if isinstance(sp_json, dict) and "ratio" in sp_json:
            ratio = sp_json["ratio"]
            if isinstance(sp_json.get("output"), dict):
                sp_body = sp_json["output"]
            elif isinstance(sp_json.get("structured_prompt"), dict):
                sp_body = sp_json["structured_prompt"]
            else:
                # Flat schema — strip ratio key from body
                sp_body = {k: v for k, v in sp_json.items() if k != "ratio"}
            sp = json.dumps(sp_body, ensure_ascii=False)
            bucket_size = getattr(args, "bucket_size", max(height, width))
            width, height = resolve_target_size(
                ratio_spec=ratio, bucket_size=bucket_size, factor=32,
            )

        # Build full trajectory for training.
        trajectory = list(messages)  # copy (system + user already there)
        trajectory.append({
            "role": "assistant",
            "content": llm_raw_response,
        })

        return {
            "index": index, "image_idx": image_idx, "prompt": prompt,
            "seed": seed, "aspect_ratio": ratio_str,
            "llm_raw_response": llm_raw_response, "structured_prompt": sp,
            "width": width, "height": height, "success": True,
            "messages": trajectory,
        }
    except Exception as e:
        log.error("[LLM vllm %d/%d] Error: %s", index, image_idx, e)
        traceback.print_exc()
        return {
            "index": index, "image_idx": image_idx, "prompt": prompt,
            "seed": seed, "aspect_ratio": ratio_str,
            "llm_raw_response": None, "structured_prompt": None,
            "width": width, "height": height,
            "success": False, "error": f"LLM error: {e}",
            "messages": None,
        }
