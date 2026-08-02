# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for the demo backend.

Values may be overridden via environment variables so the same code runs
locally against on-disk checkpoints and on a Hugging Face Space against
HF Hub repositories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Model checkpoints ─────────────────────────────────────────────
    # Prompt-expansion (PE) LLM. Local dir OR HF Hub repo id.
    # Default: our released Structured-Prompt-fine-tuned Qwen3.5-35B-A3B.
    pe_ckpt: str = os.environ.get(
        "PE_CKPT",
        "heheyas/SP-PE-Qwen3.5-35B-A3B",
    )
    # QwenImage DiT overlay ckpt. Local diffusers-layout dir OR HF Hub
    # repo id containing the sharded `dit_model-*.safetensors` overlay
    # (in HF mode the base pipeline is pulled from `dit_base_repo` below
    # and the overlay is loaded on top of it).
    dit_ckpt: str = os.environ.get(
        "DIT_CKPT",
        "heheyas/Qwen-Image-SP",
    )
    # HF Hub repo that supplies the base pipeline (transformer config +
    # VAE + text encoder + tokenizer) when dit_ckpt is an HF Hub id.
    # Ignored when dit_ckpt points at a local dir.
    dit_base_repo: str = os.environ.get(
        "DIT_BASE_REPO",
        "Qwen/Qwen-Image",
    )
    # Glob for the DiT overlay shards inside dit_ckpt (HF mode only).
    dit_overlay_glob: str = os.environ.get(
        "DIT_OVERLAY_GLOB",
        "dit_model-*.safetensors",
    )
    # HF token for pulling private repos (or `huggingface-cli login`
    # once and leave this unset). Never checked into source.
    hf_token: str | None = os.environ.get("HF_TOKEN")

    # ── Device placement ──────────────────────────────────────────────
    # Cap visible GPUs before loading torch. Two devices are enough:
    # GPU 0 for the DiT + text encoder + VAE, GPU 1 for the PE model.
    cuda_visible_devices: str = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    # PE placement — either a single-device string ("cuda:1") or an
    # accelerate device-map keyword ("auto", "balanced", "sequential").
    # Default "auto" lets accelerate spread a large PE across all
    # visible GPUs alongside the DiT.
    pe_device: str = os.environ.get("PE_DEVICE", "auto")
    dit_device: str = os.environ.get("DIT_DEVICE", "cuda:0")
    dtype: str = os.environ.get("DTYPE", "bfloat16")

    # ── PE system prompts ─────────────────────────────────────────────
    # System prompt fed to the PE model for /api/generate_sp
    # (NL → Structured Prompt). Default is the RFT student synthesis
    # prompt shipped in demo/system_prompts/.
    pe_system_prompt_path: str = os.environ.get(
        "PE_SYSTEM_PROMPT",
        str(Path(__file__).resolve().parents[1] / "system_prompts"
            / "rft_iter0_with_ratio_v6_synthesis_student.txt"),
    )
    # System prompt fed to the PE model for /api/caption_image
    # (image → Structured Prompt). Uses the same VLM ckpt as expand()
    # but goes through the multimodal chat template.
    pe_caption_system_prompt_path: str = os.environ.get(
        "PE_CAPTION_SYSTEM_PROMPT",
        str(Path(__file__).resolve().parents[1] / "system_prompts"
            / "image2json.txt"),
    )

    # ── Generation defaults ───────────────────────────────────────────
    default_height: int = int(os.environ.get("DEFAULT_HEIGHT", 1024))
    default_width: int = int(os.environ.get("DEFAULT_WIDTH", 1024))
    default_num_steps: int = int(os.environ.get("DEFAULT_NUM_STEPS", 25))
    default_cfg_scale: float = float(os.environ.get("DEFAULT_CFG_SCALE", 4.0))
    default_seed: int = int(os.environ.get("DEFAULT_SEED", 42))

    # PE generation
    pe_max_new_tokens: int = int(os.environ.get("PE_MAX_NEW_TOKENS", 2048))
    pe_temperature: float = float(os.environ.get("PE_TEMPERATURE", 0.7))
    pe_top_p: float = float(os.environ.get("PE_TOP_P", 0.9))

    # ── Server ────────────────────────────────────────────────────────
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", 7860))


def load_config() -> Config:
    cfg = Config()
    # Apply CUDA_VISIBLE_DEVICES early — before any torch.cuda init call
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg
