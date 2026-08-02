# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 local rewrite backend using HuggingFace transformers.

For small models (single-GPU), supports multi-GPU data parallelism:
load one model copy per GPU, dispatch prompts across them for higher throughput.
"""

import logging
import os
import threading
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import RewriteClient, apply_input_template, strip_markdown_codeblock

log = logging.getLogger(__name__)

# Map short names to full model IDs / local paths
QWEN_MODELS = {
    "qwen3.5-0.8b": "Qwen3.5-0.8B",
    "qwen3.5-4b": "Qwen3.5-4B-Base",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.5-35b": "Qwen3.5-35B-A3B",
    "qwen3.5-122b": "Qwen3.5-122B-A10B",
    "qwen3.5-397b": "Qwen3.5-397B-A17B",
    "qwen3.5-397b-fp8": "Qwen3.5-397B-A17B-FP8",
}

DEFAULT_WEIGHTS_DIR = "<HDFS_ROOT>/weights"

# Models that fit on a single GPU (BF16), used to decide DP strategy
# Rough sizes: 0.8B~1.7G, 4B~8.8G, 9B~19G, 35B(MoE)~67G
SINGLE_GPU_MODELS = {"qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-35b"}


def _estimate_model_size(model_name: str) -> str:
    """Estimate if model fits on single GPU."""
    lower = model_name.lower()
    for name in SINGLE_GPU_MODELS:
        if name in lower:
            return "single"
    return "multi"


class QwenRewriteClient(RewriteClient):
    """Local Qwen3.5 model for prompt rewriting.

    For small models: loads one copy per GPU for data-parallel throughput.
    For large models: uses device_map="auto" to shard across GPUs.
    """

    def __init__(self, model_path: str, max_tokens: int = 16384,
                 temperature: float = 0.0, thinking: bool = False,
                 device: str = "auto", dtype: str = "auto",
                 num_gpus: int = 0, batch_size: int = 1):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._num_gpus = num_gpus or torch.cuda.device_count()
        self._workers = []  # list of (model, tokenizer, device) per GPU
        self._lock = threading.Lock()
        self._robin = 0
        self._loaded = False
        self._model_size = _estimate_model_size(os.path.basename(model_path))
        # Batch queue: accumulate prompts, process in batch
        self._batch_queue = []  # list of (prompt_args, future)
        self._batch_lock = threading.Lock()

    def _get_dtype(self):
        if self._dtype == "auto" or self._dtype == "bf16":
            return torch.bfloat16
        elif self._dtype == "fp16":
            return torch.float16
        return torch.bfloat16

    def _load(self):
        with self._lock:
            if self._loaded:
                return

            dtype = self._get_dtype()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            if self._model_size == "single" and self._num_gpus > 1:
                # Data parallel: load one model per GPU
                log.info("Loading %s in data-parallel mode (%d GPUs)...",
                         self.model_path, self._num_gpus)
                for gpu_id in range(self._num_gpus):
                    device = f"cuda:{gpu_id}"
                    log.info("  Loading on %s...", device)
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        dtype=dtype,
                        device_map={"": gpu_id},
                        trust_remote_code=True,
                    ).eval()
                    self._workers.append((model, tokenizer, device))
                log.info("Data-parallel loading complete: %d replicas", len(self._workers))
            else:
                # Large model: shard across GPUs with device_map
                log.info("Loading %s with device_map=auto...", self.model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=dtype,
                    device_map=self._device,
                    trust_remote_code=True,
                ).eval()
                self._workers.append((model, tokenizer, model.device))
                log.info("Model loaded (sharded)")

            self._loaded = True

    def _acquire_worker(self):
        """Round-robin select a worker (thread-safe)."""
        with self._lock:
            idx = self._robin % len(self._workers)
            self._robin += 1
            return self._workers[idx]

    def _build_chat_text(self, prompt, system_prompt, input_template, width, height, tokenizer):
        """Build the chat-formatted text for one prompt."""
        user_content = apply_input_template(input_template, prompt, width=width, height=height)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def _parse_response(self, response):
        """Parse a raw model response, strip special tokens, separate thinking."""
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            response = response.replace(tok, "")
        response = response.strip()

        reasoning = None
        if self.thinking and "</think>" in response:
            parts = response.split("</think>", 1)
            reasoning = parts[0].replace("<think>", "").strip()
            response = parts[1].strip()
        elif self.thinking:
            response = response.replace("<think>", "").strip()

        response = strip_markdown_codeblock(response)
        return response, reasoning

    def _generate_batch(self, texts, model, tokenizer, device):
        """Run batched generation on a list of chat-formatted texts."""
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        # Decode each sequence, skipping input tokens
        results = []
        for i in range(len(texts)):
            input_len = inputs.attention_mask[i].sum().item()
            output_ids = generated_ids[i][input_len:]
            response = tokenizer.decode(output_ids, skip_special_tokens=False)
            text, reasoning = self._parse_response(response)
            results.append({"rewritten_prompt": text, "reasoning": reasoning, "usage": {}})
        return results

    def rewrite(self, prompt: str, system_prompt: str, input_template: str = "<prompt>",
                width: int = None, height: int = None) -> dict:
        self._load()
        model, tokenizer, device = self._acquire_worker()

        text = self._build_chat_text(prompt, system_prompt, input_template, width, height, tokenizer)
        results = self._generate_batch([text], model, tokenizer, device)
        return results[0]

    def rewrite_batch(self, prompts, system_prompt, input_template="<prompt>",
                      width=None, height=None):
        """Rewrite multiple prompts in one batched forward pass.

        Args:
            prompts: list of prompt strings
        Returns:
            list of result dicts
        """
        self._load()
        model, tokenizer, device = self._acquire_worker()

        texts = [self._build_chat_text(p, system_prompt, input_template, width, height, tokenizer)
                 for p in prompts]

        # Split into sub-batches of self.batch_size
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_results.extend(self._generate_batch(batch, model, tokenizer, device))
        return all_results


def resolve_model_path(model_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> str:
    """Resolve a model short name or path to a full local path.

    Accepts:
        - Short name: "qwen3.5-9b" -> <HDFS_ROOT>/.../Qwen3.5-9B
        - Directory name: "Qwen3.5-9B" -> <HDFS_ROOT>/.../Qwen3.5-9B
        - Full path: "/path/to/model" -> as-is
    """
    # Full path
    if os.path.isdir(model_name):
        return model_name

    # Short name lookup
    lower = model_name.lower()
    if lower in QWEN_MODELS:
        return os.path.join(weights_dir, QWEN_MODELS[lower])

    # Try as directory name under weights_dir
    candidate = os.path.join(weights_dir, model_name)
    if os.path.isdir(candidate):
        return candidate

    raise ValueError(
        f"Cannot resolve model: {model_name}. "
        f"Known short names: {list(QWEN_MODELS.keys())}. "
        f"Or provide a full path."
    )
