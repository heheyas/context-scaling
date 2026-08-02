# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prompt-Expansion + Captioner backend.

Wraps a Qwen3.5-35B-A3B VLM checkpoint (the same one released alongside
this repo) in two entry points:

* :meth:`PEBackend.expand` — natural-language prompt → Structured
  Prompt JSON. Uses the text-only chat template with the RFT/SP-synthesis
  system prompt (``rft_iter0_with_ratio_v6_synthesis_student.txt``).
* :meth:`PEBackend.caption_image` — PIL image → Structured Prompt JSON.
  Uses the multimodal chat template with the ``image2json.txt`` system
  prompt; needs the same VLM checkpoint and is served from the same
  weights that ``expand()`` uses (no second model load).

Loaded once at server startup and reused for every /api/generate_sp
and /api/caption_image request. Both entry points are thread-hostile —
serialise calls through an asyncio lock or a single worker.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


class PEBackend:
    """Text LLM used to synthesise Structured Prompt JSON from an NL prompt."""

    def __init__(
        self,
        ckpt: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        system_prompt_path: Optional[str] = None,
        caption_system_prompt_path: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.ckpt = ckpt
        # `device` may be either a single-device string (``cuda:0``) or one of
        # the accelerate device-map keywords (``auto``, ``balanced``). We store
        # the requested value verbatim; the generation code uses
        # ``self.model.device`` when moving input tensors, so a sharded
        # (device_map='auto') load is fine.
        self.device_map = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        log.info("PE: loading tokenizer from %s", ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

        # A VLM ckpt (Qwen3.5-35B-A3B) also has an image processor. Load it
        # via AutoProcessor so caption_image() can build multimodal chat
        # inputs. Fall back gracefully if the ckpt is text-only.
        self.processor = None
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
            log.info("PE: loaded AutoProcessor (multimodal caption endpoint enabled)")
        except Exception as e:  # noqa: BLE001
            log.info("PE: no AutoProcessor for %s (%s); "
                     "caption_image() will be unavailable", ckpt, e)

        # Optional per-device VRAM cap for the accelerate device-map planner.
        # Read from PE_MAX_MEMORY as a JSON dict, e.g.:
        #     '{"1": "45GiB", "cpu": "128GiB"}'
        # This is how we make room for a large DiT living on GPU 0: cap PE's
        # GPU 1 share and let the rest of the MoE experts land on CPU (they
        # will be pulled onto GPU per token by accelerate's dispatch hooks —
        # slower, but the two models fit simultaneously).
        max_memory_env = os.environ.get("PE_MAX_MEMORY")
        max_memory: Optional[Dict[Any, str]] = None
        if max_memory_env:
            try:
                raw = json.loads(max_memory_env)
                # accelerate wants int keys for GPUs; keep 'cpu'/'disk' as str.
                max_memory = {(int(k) if k.isdigit() else k): v for k, v in raw.items()}
                log.info("PE: max_memory constraint applied: %s", max_memory)
            except Exception as e:  # noqa: BLE001
                log.warning("PE: could not parse PE_MAX_MEMORY=%r (%s); ignoring", max_memory_env, e)

        log.info("PE: loading model from %s onto %s (dtype=%s)", ckpt, device, dtype)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        from_pretrained_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        if max_memory is not None:
            from_pretrained_kwargs["max_memory"] = max_memory
            # If any layers end up offloaded to CPU/disk, accelerate needs a
            # spot on disk to spill safetensors shards. Use /tmp by default.
            from_pretrained_kwargs["offload_folder"] = os.environ.get(
                "PE_OFFLOAD_FOLDER", "/tmp/pe_offload"
            )
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, **from_pretrained_kwargs)
        self.model.eval()
        # Resolve where input tensors should land. When device_map='auto',
        # accelerate places the embedding layer on cuda:0 by default; using
        # ``self.model.device`` picks that up for us.
        self.input_device = self.model.device if str(device) in {"auto", "balanced", "balanced_low_0", "sequential"} else device

        self.system_prompt = ""
        if system_prompt_path and Path(system_prompt_path).is_file():
            self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
            log.info("PE: loaded expand system prompt (%d chars) from %s",
                     len(self.system_prompt), system_prompt_path)
        else:
            log.warning("PE: expand system prompt path is missing or invalid: %r", system_prompt_path)

        # Separate system prompt for image → SP (loaded lazily; ok to be absent).
        self.caption_system_prompt = ""
        if caption_system_prompt_path and Path(caption_system_prompt_path).is_file():
            self.caption_system_prompt = Path(caption_system_prompt_path).read_text(encoding="utf-8")
            log.info("PE: loaded caption system prompt (%d chars) from %s",
                     len(self.caption_system_prompt), caption_system_prompt_path)
        else:
            log.info("PE: caption system prompt not configured — /api/caption_image will 501")

    # ---------- Prompt formatting ----------

    def _build_user_message(self, user_prompt: str, width: int, height: int) -> str:
        """Append the target aspect ratio to the user prompt, matching the
        `rft_iter0_with_ratio_v6_synthesis_student` prompt's convention."""
        return f"{user_prompt.strip()} [width: {width}, height: {height}]"

    # ---------- JSON post-processing ----------

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    @classmethod
    def _extract_json(cls, text: str) -> Dict[str, Any]:
        """Best-effort parse: strip ```json fences and anything before the
        first '{' / after the last '}'. Raises on total failure."""
        # 1. Fenced block
        m = cls._JSON_FENCE_RE.search(text)
        if m:
            candidate = m.group(1).strip()
        else:
            # 2. Fallback: first '{' to last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("PE output contains no JSON object")
            candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Retry after removing trailing commas — a common LLM quirk.
            fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
            return json.loads(fixed)

    # ---------- Main entry point ----------

    @torch.inference_mode()
    def expand(self, user_prompt: str, width: int, height: int) -> Dict[str, Any]:
        """Generate a Structured Prompt JSON for the given natural-language
        prompt and target image aspect ratio."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._build_user_message(user_prompt, width, height)})

        # transformers >= 5 returns a BatchEncoding from apply_chat_template
        # when return_tensors is set. Older versions return the tensor
        # directly. Support both.
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(enc, "input_ids"):
            input_ids = enc["input_ids"].to(self.input_device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.input_device)
        else:
            input_ids = enc.to(self.input_device)
            attention_mask = None

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        output_ids = self.model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        try:
            return self._extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("PE: JSON parse failed (%s); returning raw text under 'raw'", e)
            return {"raw": text}

    # ---------- Image → SP ----------

    @torch.inference_mode()
    def caption_image(self, image: Image.Image) -> Dict[str, Any]:
        """Given a PIL image, return a Structured Prompt JSON describing it.

        Uses the same model weights as :meth:`expand`, but through the
        multimodal chat template and the ``image2json.txt`` system prompt
        (bounding boxes normalized to a 1000×1000 grid, per the prompt).
        """
        if self.processor is None:
            raise RuntimeError(
                "caption_image() requires an AutoProcessor for the VLM ckpt; "
                "none was loaded at startup.")
        if not self.caption_system_prompt:
            raise RuntimeError(
                "caption_image() requires PE_CAPTION_SYSTEM_PROMPT to point "
                "at a system prompt file (e.g. demo/system_prompts/image2json.txt).")

        img = image.convert("RGB")
        messages = [
            {"role": "system", "content": self.caption_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image as a Structured Prompt JSON."},
                ],
            },
        ]

        # AutoProcessor.apply_chat_template returns a BatchFeature with
        # input_ids + pixel_values (and image_grid_thw for Qwen-VL-family).
        # Some processor versions want tokenize=False + a separate call to
        # __call__ with `text=`; try the modern one-shot form first.
        try:
            enc = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            # Older transformers: build text + call processor.__call__ manually.
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            enc = self.processor(
                text=[text], images=[img], return_tensors="pt", padding=True,
            )

        # Move all tensor fields onto the model's input device.
        enc = {k: (v.to(self.input_device) if hasattr(v, "to") else v)
               for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        output_ids = self.model.generate(**enc, **gen_kwargs)
        new_tokens = output_ids[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        try:
            return self._extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("PE(caption): JSON parse failed (%s); returning raw text under 'raw'", e)
            return {"raw": text}
