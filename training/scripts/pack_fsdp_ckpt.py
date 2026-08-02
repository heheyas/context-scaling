#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Merge QwenImage FSDP sharded checkpoints (DCP format) into a SINGLE .safetensors
file using CPU only — no GPU required, no OOM.

Usage (single process, no torchrun needed):
  python pack_fsdp_ckpt_qwenimage.py \
    --ckpt_dir /path/to/fsdp/checkpoint_dir/model \
    --dit_path /path/to/qwenimage_dit \
    --output   /path/to/model_bf16.safetensors \
    --dtype bf16

Notes:
- 完全在 CPU 上运行，不需要 GPU，不会 OOM。
- 直接读取 DCP (torch.distributed.checkpoint) 格式的分片文件，重组为完整 state dict。
- --strip_text_encoder 可在导出时去掉 frozen text encoder 权重（只保留 DiT）。
"""

import os
import gc
import json
import argparse
from types import SimpleNamespace
from typing import Dict

import torch
from safetensors.torch import save_file

from modeling.qwenimage.model import CausalFusionQwenImage, CausalFusionQwenImageConfig
from modeling.qwenimage.transformer import QwenImageTransformer2DModel
from modeling.qwenimage.text_encoder_navit import PackedQwen2TextEncoder


# ================== Model construction (CPU) ==================

DEFAULT_TEXT_ENC_CONFIG = dict(
    hidden_size=3584,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
    intermediate_size=18944,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    vocab_size=152064,
)


def build_model_cpu(dit_path: str, text_enc_config_overrides: dict = None):
    """Build CausalFusionQwenImage on CPU with real (zero-initialized) weights.

    We need the correct tensor shapes/dtypes so that DCP load_state_dict
    can fill them in from the sharded checkpoint files.
    """
    with open(os.path.join(dit_path, "config.json"), "r") as f:
        dit_config = json.load(f)

    text_cfg = {**DEFAULT_TEXT_ENC_CONFIG, **(text_enc_config_overrides or {})}
    text_enc_config = SimpleNamespace(**text_cfg)

    # Build on CPU with real (empty) tensors — only allocates memory for shapes,
    # actual values will be overwritten by DCP load.
    dit_model = QwenImageTransformer2DModel(
        patch_size=dit_config.get("patch_size", 2),
        in_channels=dit_config.get("in_channels", 64),
        out_channels=dit_config.get("out_channels", 16),
        num_layers=dit_config.get("num_layers", 60),
        attention_head_dim=dit_config.get("attention_head_dim", 128),
        num_attention_heads=dit_config.get("num_attention_heads", 24),
        joint_attention_dim=dit_config.get("joint_attention_dim", 3584),
        axes_dims_rope=tuple(dit_config.get("axes_dims_rope", [16, 56, 56])),
        guidance_embeds=dit_config.get("guidance_embeds", False),
    )
    text_encoder = PackedQwen2TextEncoder(text_enc_config)
    text_encoder.eval()
    model = CausalFusionQwenImage(
        dit_model,
        CausalFusionQwenImageConfig(),
        text_encoder=text_encoder,
    )
    return model


# ================== Utilities ==================

def _drop_buffers_inplace(d: Dict[str, torch.Tensor]):
    for k in list(d.keys()):
        if any(s in k for s in ["running_mean", "running_var", "_float_tensor"]):
            d.pop(k, None)


def _clean_fsdp_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        kk = k.replace("._fsdp_wrapped_module.", ".")
        if kk.startswith("module."):
            kk = kk[len("module."):]
        out[kk] = v
    return out


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove keys starting with prefix."""
    return {k: v for k, v in sd.items() if not k.startswith(prefix)}


# ================== DCP Loading ==================

def _load_dcp_to_state_dict(ckpt_dir: str, state_dict: Dict[str, torch.Tensor]):
    """Load a DCP sharded checkpoint into a regular state dict on CPU.

    Tries multiple approaches in order of preference:
    1. dcp_to_torch_save (PyTorch 2.3+, cleanest)
    2. load_state_dict with no_dist=True (PyTorch 2.1+)
    3. load_state_dict with gloo single-process (fallback)
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader

    # --- Method 1: dcp_to_torch_save (no dist required) ---
    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        import tempfile
        print("[*] Method 1: using dcp_to_torch_save (CPU only, no distributed)")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        dcp_to_torch_save(ckpt_dir, tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        os.remove(tmp_path)
        state_dict.update(loaded)
        del loaded
        gc.collect()
        return
    except ImportError:
        print("[!] dcp_to_torch_save not available, trying next method...")
    except Exception as e:
        print(f"[!] dcp_to_torch_save failed: {e}, trying next method...")

    # --- Method 2: load_state_dict with no_dist=True ---
    from torch.distributed.checkpoint import load_state_dict as dcp_load
    try:
        reader = FileSystemReader(ckpt_dir)
        print("[*] Method 2: using load_state_dict(no_dist=True)")
        dcp_load(state_dict, reader, no_dist=True)
        return
    except TypeError:
        # no_dist parameter not available in this PyTorch version
        print("[!] no_dist not supported, trying next method...")
    except Exception as e:
        print(f"[!] load_state_dict(no_dist=True) failed: {e}, trying next method...")

    # --- Method 3: single-process gloo backend ---
    print("[*] Method 3: using gloo backend with single process")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    reader = FileSystemReader(ckpt_dir)
    dcp_load(state_dict, reader)
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== Main logic ==================

def pack_checkpoint(
    ckpt_dir: str,
    dit_path: str,
    output_path: str,
    dtype_str: str = "bf16",
    keep_buffers: bool = False,
    strip_text_encoder: bool = False,
):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    target_dtype = dtype_map[dtype_str]

    # 1) Build model on CPU (need correct shapes for DCP loading)
    print(f"[*] Building model on CPU from {dit_path} ...")
    model = build_model_cpu(dit_path)
    sd = model.state_dict()
    del model
    gc.collect()
    print(f"    {len(sd)} parameters/buffers in state dict")

    # 2) Load DCP sharded checkpoint into state dict (all on CPU)
    print(f"[*] Loading DCP checkpoint from {ckpt_dir} ...")
    _load_dcp_to_state_dict(ckpt_dir, sd)
    print(f"    Loaded successfully")

    # 3) Clean FSDP key prefixes (if any)
    sd = _clean_fsdp_keys(sd)

    if not keep_buffers:
        _drop_buffers_inplace(sd)

    if strip_text_encoder:
        before = len(sd)
        sd = _strip_prefix(sd, "text_encoder.")
        print(f"[*] Stripped text_encoder keys: {before} → {len(sd)} tensors")

    # 4) Cast dtype
    casted = {}
    for k, t in sd.items():
        if torch.is_floating_point(t) and t.dtype != target_dtype:
            t = t.to(dtype=target_dtype)
        casted[k] = t.detach().contiguous()
    del sd
    gc.collect()

    # 5) Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(casted, output_path)
    print(f"[✓] Wrote {output_path}  ({len(casted)} tensors, dtype={dtype_str})")


def main():
    ap = argparse.ArgumentParser(
        description="Pack QwenImage FSDP sharded checkpoints into a single .safetensors file (CPU only)"
    )
    ap.add_argument("--ckpt_dir", required=True,
                    help="FSDP sharded checkpoint directory (e.g. .../0003000/model or .../0003000/ema)")
    ap.add_argument("--dit_path", required=True,
                    help="Path to QwenImage DiT checkpoint (contains config.json for architecture)")
    ap.add_argument("--output", default="model.safetensors",
                    help="Output single-file safetensors path")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="Export dtype (recommended bf16)")
    ap.add_argument("--no-drop-buffers", action="store_true",
                    help="Keep non-persistent buffers (default: drop them)")
    ap.add_argument("--strip-text-encoder", action="store_true",
                    help="Remove frozen text encoder weights from output (keep only DiT)")
    args = ap.parse_args()

    pack_checkpoint(
        ckpt_dir=args.ckpt_dir,
        dit_path=args.dit_path,
        output_path=args.output,
        dtype_str=args.dtype,
        keep_buffers=args.no_drop_buffers,
        strip_text_encoder=args.strip_text_encoder,
    )


if __name__ == "__main__":
    main()


# python pack_fsdp_ckpt_qwenimage.py \
#   --ckpt_dir /path/to/results/0003000/model \
#   --dit_path /path/to/qwenimage_dit \
#   --output  /path/to/results/0003000/model.safetensors \
#   --dtype bf16
