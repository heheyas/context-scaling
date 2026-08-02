# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""YAML config loader for the RFT pipeline.

Each iteration has a config file (e.g., configs/iter0.yaml) that
specifies paths, sizes, and per-stage parameters. This module loads
and validates it into a plain dict.
"""

import os
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping, got {type(cfg)}")
    return cfg


def resolve_paths(cfg: dict) -> dict:
    """Resolve relative paths in the config against shared_root.

    Adds commonly-used derived paths as top-level keys:
      - queries_path: full path to queries.jsonl
      - llm_out: shared_root/llm_out
      - images: shared_root/images
      - rank_in: shared_root/rank_in
      - rankings: shared_root/rankings

    Does not modify the original dict; returns a new one.
    """
    cfg = dict(cfg)
    root = cfg["shared_root"]
    cfg["queries_path"] = os.path.join(root, cfg.get("queries_file", "queries.jsonl"))
    cfg["llm_out"] = os.path.join(root, "llm_out")
    cfg["images"] = os.path.join(root, "images")
    cfg["rank_in"] = os.path.join(root, "rank_in")
    cfg["rankings"] = os.path.join(root, "rankings")
    return cfg
