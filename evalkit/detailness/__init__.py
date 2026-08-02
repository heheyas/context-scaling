# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Detailness metric package.

Two metrics, one evaluator:
    ed/    Effective Detailness (caption-only, model-free LLM-API metric)
    gpg/   Grounded Perplexity Gain (Qwen3-VL forward-pass PMI metric)
    evaluator.Evaluator
           Unified driver — runs ED, GPG, or both on arbitrary
           (image, caption) inputs and returns a single dict per sample.

Convenience re-exports:
    >>> from detailness import Evaluator
    >>> ev = Evaluator(metrics=("ed",))
    >>> ev.score("/path/to/img.png", "A short caption.")
    {'ed': 0.27, 'ed_F05_A': 0.27, 'ed_P_A': 1.0, 'ed_R_A': 0.07, ...}

See `evalkit/detailness/README.md` for the metric definitions, fit
numbers, and paper framing.
"""

from .evaluator import Evaluator

__all__ = ["Evaluator"]
