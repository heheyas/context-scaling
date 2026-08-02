# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""In-process backends for the FastAPI demo.

Two model wrappers are exposed:

    PEBackend   — natural-language prompt → Structured Prompt JSON
                  (Qwen3.5-35B-A3B via transformers)
    DiTBackend  — Structured Prompt JSON → PIL.Image
                  (QwenImage DiT + Qwen2.5-VL text encoder + causal VAE)

Both are loaded once at server startup and remain resident. GPU placement
is set via demo.backend.config.
"""

from .config import Config
from .pe import PEBackend
from .dit import DiTBackend

__all__ = ["Config", "PEBackend", "DiTBackend"]
