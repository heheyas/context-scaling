# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native HTTP backend for DiT image generation.

Drop-in replacement for legacy run_dit. Calls a local FastAPI server's
/generate endpoint instead of the internal PSM-based generate_image_xperf.

The native serve endpoint (serve_qwenimage_multigpu.py) accepts:
  POST /generate {prompt, height, width, seed, num_steps, cfg_scale, ...}
  → PNG bytes

Ported from EvalKit/inference/generate.py.
"""

import io
import json
import time
import logging
import traceback

import requests
from PIL import Image

from src.llm.rollout import compute_dit_seed

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompt encoding (matches training format)
# ──────────────────────────────────────────────

def compact_single_quote_json(data):
    """Compact a JSON-parseable object/string into single-quote format.

    The DiT model is trained with structured prompts encoded in this format
    (compact, single-quoted JSON). We must apply the same encoding at
    inference time.

    Ported from EvalKit/inference/generate.py.
    """
    PLACEHOLDER_SINGLE = "@@SP_SINGLE_QUOTE@@"
    PLACEHOLDER_DOUBLE = "@@SP_DOUBLE_QUOTE@@"

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


# ──────────────────────────────────────────────
# Native DiT backend
# ──────────────────────────────────────────────

def run_dit_native(llm_result, dit_url, args):
    """Drop-in replacement for legacy run_dit using native HTTP /generate.

    Args:
        llm_result: Dict from LLM rollout (index, image_idx, prompt,
            structured_prompt, width, height, seed, success, ...).
        dit_url: Base URL of the native DiT serve (e.g. "http://localhost:8091").
        args: Namespace with dit_backend, num_steps, cfg_scale, and
            optionally negative_prompt, timeout.

    Returns:
        Dict matching legacy run_dit schema ({...llm_result, image, success}).
    """
    index = llm_result["index"]
    image_idx = llm_result["image_idx"]

    if not llm_result.get("success", True):
        return {**llm_result, "image": None}

    try:
        # Convert structured prompt to compact single-quote JSON format
        # to match the training data encoding.
        prompt = compact_single_quote_json(llm_result["structured_prompt"])

        # DiT seed is shared across all image_idx of a given index, so the
        # N rollouts for one prompt only vary in SP — DiT noise is fixed.
        dit_seed = compute_dit_seed(llm_result["index"])

        body = {
            "prompt": prompt,
            "height": llm_result["height"],
            "width": llm_result["width"],
            "seed": dit_seed,
            "num_steps": getattr(args, "num_steps", 25),
            "cfg_scale": getattr(args, "cfg_scale", 4.0),
        }

        body["negative_prompt"] = getattr(args, "negative_prompt", "") or ""

        timeout = getattr(args, "timeout", 600)
        max_retries = getattr(args, "max_retries", 5)

        image_pil = _generate_with_retry(
            url=f"{dit_url}/generate",
            body=body,
            timeout=timeout,
            max_retries=max_retries,
            tag=f"DiT {index}/{image_idx}",
        )

        return {
            **llm_result,
            "image": image_pil,
            "success": image_pil is not None,
            "dit_seed": dit_seed,
        }
    except Exception as e:
        log.error("[DiT native %d/%d] Error: %s", index, image_idx, e)
        traceback.print_exc()
        return {
            **llm_result,
            "image": None,
            "success": False,
            "error": f"DiT error: {e}",
            "dit_seed": dit_seed,
        }


def _generate_with_retry(url, body, timeout, max_retries, tag):
    """POST to /generate with retry on 502/503."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, timeout=timeout)

            if resp.status_code in (502, 503):
                wait = min(2 ** attempt, 30)
                log.warning("[%s] %d (attempt %d/%d), retrying in %ds...",
                            tag, resp.status_code, attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()

            image_pil = Image.open(io.BytesIO(resp.content))
            return image_pil

        except requests.exceptions.Timeout:
            log.warning("[%s] Timeout (attempt %d/%d)", tag, attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                log.warning("[%s] Error (attempt %d/%d): %s, retrying in %ds...",
                            tag, attempt + 1, max_retries, e, wait)
                time.sleep(wait)
            else:
                raise

    log.error("[%s] All %d retries exhausted", tag, max_retries)
    return None


def check_health(url, timeout=10):
    """Check if the DiT serve is healthy.

    Returns True if /health responds with 200.
    """
    try:
        resp = requests.get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False
