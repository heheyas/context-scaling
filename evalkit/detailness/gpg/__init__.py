# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""GPG (Grounded Perplexity Gain) scoring & analysis.

Modules:
    score              — main per-sample PMI scorer (sharded driver)
    content_mask       — JSON value/scaffold token mask for --content-only
    analyze            — aggregate per-(kind,level) GPG, join with Bagel MSE
                         CSVs, produce 16-point table + main figure
    metric_candidates  — explore alternative GPG → MSE shapes (capacity-
                         bounded GPG_eff, power, exponential, etc.)

Prompts:
    prompts/structured_schema.txt  — schema system prompt (tested in v5/v7/v8,
                                     dropped from v6_dropmore final because it
                                     collapsed the spatial gap)
"""
