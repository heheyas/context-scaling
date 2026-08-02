# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""ED (Effective Detailness) scoring & analysis.

A fully model-free caption-only Spearman-validated metric of caption
detailness for T2I training. ED is the per-cell trim20 mean of
F_{β=0.5}(P_A, R_A), where the source side is image-grounded OARG
tuples (Gemini-vision exhaustive) and the caption side is OARG tuples
decomposed from the caption text. P_A/R_A come from a lenient text-only
matcher (GPT-4o by default; Gemini-3-pro available for cross-matcher
robustness).

Modules:
    extract_image      — image → OARG tuples via Gemini-vision (per uid)
    extract_caption    — caption text → OARG tuples via GPT-4o text
    match              — paired (source, caption) tuple match (GPT-4o default,
                         --matcher gemini for cross-matcher robustness run)
    aggregate          — per-cell trim20 of F_{β=0.5}(P_A, R_A), 16-point
                         table, power-law fit, main figure
    bootstrap          — B=10,000 resample CI on per-cell ED, σ vs N scaling

Prompts:
    prompts/image_source.txt    — exhaustive OARG enumeration from image
    prompts/extract_tuples.txt  — caption-side OARG (closed 16-attr schema,
                                  paper-final version)
    prompts/match_tuples.txt    — paraphrase-tolerant lenient matcher

See `README.md` for the metric definition and fit numbers. The 16-cell
ground-truth MSE is the same as GPG (same Bagel runs, same unified token
budget) — re-uses `detailness.gpg.io` for MSE-side loaders.
"""
