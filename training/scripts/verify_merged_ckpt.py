#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Verify a merged QwenImage safetensors checkpoint:
  1. Load and check keys/shapes match the model architecture
  2. Check weight statistics (no all-zeros, reasonable values)
  3. (Optional) Compare with original sharded checkpoint
  4. (Optional, --forward) Run a dummy forward pass on GPU

Usage:
  # Basic check (CPU only, no GPU needed):
  python scripts/merge/verify_merged_ckpt.py \
    --ckpt /path/to/model.safetensors \
    --dit_path /path/to/qwenimage_dit

  # Compare with original checkpoint:
  python scripts/merge/verify_merged_ckpt.py \
    --ckpt /path/to/model.safetensors \
    --dit_path /path/to/qwenimage_dit \
    --compare_original /path/to/original/transformer

  # With forward pass test (needs GPU):
  python scripts/merge/verify_merged_ckpt.py \
    --ckpt /path/to/model.safetensors \
    --dit_path /path/to/qwenimage_dit \
    --forward
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modeling.qwenimage.model import CausalFusionQwenImage, CausalFusionQwenImageConfig
from modeling.qwenimage.transformer import QwenImageTransformer2DModel
from modeling.qwenimage.text_encoder_navit import PackedQwen2TextEncoder


DEFAULT_TEXT_ENC_CONFIG = dict(
    hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
    num_key_value_heads=4, intermediate_size=18944, rms_norm_eps=1e-6,
    rope_theta=1000000.0, vocab_size=152064,
)


def build_model(dit_path):
    with open(os.path.join(dit_path, "config.json")) as f:
        cfg = json.load(f)
    dit = QwenImageTransformer2DModel(
        patch_size=cfg.get("patch_size", 2),
        in_channels=cfg.get("in_channels", 64),
        out_channels=cfg.get("out_channels", 16),
        num_layers=cfg.get("num_layers", 60),
        attention_head_dim=cfg.get("attention_head_dim", 128),
        num_attention_heads=cfg.get("num_attention_heads", 24),
        joint_attention_dim=cfg.get("joint_attention_dim", 3584),
        axes_dims_rope=tuple(cfg.get("axes_dims_rope", [16, 56, 56])),
        guidance_embeds=cfg.get("guidance_embeds", False),
    )
    te = PackedQwen2TextEncoder(SimpleNamespace(**DEFAULT_TEXT_ENC_CONFIG))
    te.eval()
    return CausalFusionQwenImage(dit, CausalFusionQwenImageConfig(), text_encoder=te)


def check_weight_stats(sd, prefix=""):
    """Print statistics and flag suspicious tensors."""
    n_zero = 0
    n_nan = 0
    n_total = 0
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        if not torch.is_floating_point(v):
            continue
        n_total += 1
        if torch.isnan(v).any():
            print(f"  [NaN!] {k}: shape={list(v.shape)}")
            n_nan += 1
        elif v.abs().max().item() == 0.0:
            # Allow norm biases and some special params to be zero
            if not any(s in k for s in ["bias", "norm", "embed"]):
                print(f"  [ZERO] {k}: shape={list(v.shape)}")
                n_zero += 1

    print(f"\n  Total float tensors: {n_total}")
    if n_zero > 0:
        print(f"  WARNING: {n_zero} non-trivial tensors are all zeros!")
    else:
        print(f"  All-zero non-bias tensors: 0  (OK)")
    if n_nan > 0:
        print(f"  ERROR: {n_nan} tensors contain NaN!")
    else:
        print(f"  NaN tensors: 0  (OK)")
    return n_zero == 0 and n_nan == 0


def compare_with_original(merged_sd, original_dir):
    """Compare merged checkpoint values with original sharded checkpoint."""
    print(f"\n{'='*60}")
    print(f"Comparing with original: {original_dir}")
    print(f"{'='*60}")

    # Load original shards
    shards = sorted(glob.glob(os.path.join(original_dir, "*.safetensors")))
    if not shards:
        print(f"  No .safetensors files found in {original_dir}")
        return False

    orig_sd = {}
    for s in shards:
        orig_sd.update(load_file(s, device="cpu"))
    print(f"  Original: {len(orig_sd)} tensors")

    # The merged checkpoint has `dit_model.` prefix; original doesn't
    n_matched = 0
    n_mismatched = 0
    max_diff = 0.0

    for orig_key, orig_val in orig_sd.items():
        merged_key = f"dit_model.{orig_key}"
        if merged_key not in merged_sd:
            continue
        merged_val = merged_sd[merged_key]
        if orig_val.shape != merged_val.shape:
            print(f"  Shape mismatch: {orig_key}: orig={list(orig_val.shape)} merged={list(merged_val.shape)}")
            n_mismatched += 1
            continue

        diff = (orig_val.float() - merged_val.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-3:
            print(f"  VALUE DIFF: {orig_key}: max_abs_diff={diff:.6f}")
            n_mismatched += 1
        else:
            n_matched += 1

    print(f"\n  Matched: {n_matched}, Mismatched: {n_mismatched}")
    print(f"  Max absolute difference: {max_diff:.8f}")
    if n_mismatched == 0 and n_matched > 0:
        print(f"  PASS: All {n_matched} compared tensors match!")
        return True
    else:
        print(f"  FAIL: {n_mismatched} tensors differ!")
        return False


def test_forward(dit_path, merged_sd):
    """Run a small dummy forward pass to verify the model works."""
    print(f"\n{'='*60}")
    print("Running dummy forward pass on GPU...")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("  SKIP: No GPU available")
        return True

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load only the DiT (skip text encoder for a lightweight test)
    with open(os.path.join(dit_path, "config.json")) as f:
        cfg = json.load(f)
    dit = QwenImageTransformer2DModel(
        patch_size=cfg.get("patch_size", 2),
        in_channels=cfg.get("in_channels", 64),
        out_channels=cfg.get("out_channels", 16),
        num_layers=cfg.get("num_layers", 60),
        attention_head_dim=cfg.get("attention_head_dim", 128),
        num_attention_heads=cfg.get("num_attention_heads", 24),
        joint_attention_dim=cfg.get("joint_attention_dim", 3584),
        axes_dims_rope=tuple(cfg.get("axes_dims_rope", [16, 56, 56])),
        guidance_embeds=cfg.get("guidance_embeds", False),
    )

    # Load dit_model weights from merged checkpoint
    dit_sd = {}
    prefix = "dit_model."
    for k, v in merged_sd.items():
        if k.startswith(prefix):
            dit_sd[k[len(prefix):]] = v
    missing, unexpected = dit.load_state_dict(dit_sd, strict=False)
    print(f"  DiT load: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    del dit_sd

    dit = dit.to(device, dtype=dtype).eval()

    # Dummy inputs: 1 image with 4x4 patches (small)
    patch_h, patch_w = 4, 4
    img_seq_len = patch_h * patch_w  # 16 tokens
    txt_seq_len = 8
    z_dim = 16
    hidden_dim = cfg.get("num_attention_heads", 24) * cfg.get("attention_head_dim", 128)

    with torch.no_grad():
        packed_img = torch.randn(img_seq_len, z_dim * 4, device=device, dtype=dtype)
        packed_txt = torch.randn(txt_seq_len, cfg.get("joint_attention_dim", 3584), device=device, dtype=dtype)
        timestep = torch.tensor([0.5], device=device, dtype=dtype)

        try:
            out = dit(
                hidden_states=packed_img.unsqueeze(0),
                encoder_hidden_states=packed_txt.unsqueeze(0),
                timestep=timestep,
                img_shapes=[[(1, patch_h, patch_w)]],
            )
            if isinstance(out, tuple):
                out = out[0]
            print(f"  Output shape: {list(out.shape)}")
            print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
            print(f"  Output mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")

            if torch.isnan(out).any():
                print("  FAIL: Output contains NaN!")
                return False
            if out.abs().max().item() == 0.0:
                print("  FAIL: Output is all zeros!")
                return False

            print("  PASS: Forward pass succeeded with non-trivial output")
            return True
        except Exception as e:
            print(f"  FAIL: Forward pass error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    ap = argparse.ArgumentParser(description="Verify merged QwenImage checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to merged .safetensors file")
    ap.add_argument("--dit_path", required=True, help="Path to DiT config dir (for model architecture)")
    ap.add_argument("--compare_original", default=None,
                    help="Path to original transformer/ dir to compare values")
    ap.add_argument("--forward", action="store_true",
                    help="Run a dummy forward pass (needs GPU)")
    args = ap.parse_args()

    all_pass = True

    # ── Step 1: Load merged checkpoint ──
    print(f"{'='*60}")
    print(f"Loading merged checkpoint: {args.ckpt}")
    print(f"{'='*60}")
    merged_sd = load_file(args.ckpt, device="cpu")
    print(f"  Loaded {len(merged_sd)} tensors")

    # Print summary by prefix
    prefixes = {}
    for k in merged_sd:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print(f"  Key prefixes: {dict(sorted(prefixes.items()))}")

    # Print dtypes
    dtypes = {}
    total_bytes = 0
    for k, v in merged_sd.items():
        dt = str(v.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
        total_bytes += v.nelement() * v.element_size()
    print(f"  Dtypes: {dtypes}")
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")

    # ── Step 2: Check keys match model ──
    print(f"\n{'='*60}")
    print("Checking key compatibility with model architecture...")
    print(f"{'='*60}")
    model = build_model(args.dit_path)
    expected_sd = model.state_dict()
    print(f"  Expected keys: {len(expected_sd)}")
    print(f"  Merged keys:   {len(merged_sd)}")

    expected_keys = set(expected_sd.keys())
    merged_keys = set(merged_sd.keys())

    missing = expected_keys - merged_keys
    extra = merged_keys - expected_keys

    if missing:
        print(f"\n  Missing keys ({len(missing)}):")
        for k in sorted(missing)[:20]:
            print(f"    - {k}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    if extra:
        print(f"\n  Extra keys ({len(extra)}):")
        for k in sorted(extra)[:20]:
            print(f"    + {k}")
        if len(extra) > 20:
            print(f"    ... and {len(extra) - 20} more")

    # Check shapes match
    n_shape_match = 0
    n_shape_mismatch = 0
    for k in expected_keys & merged_keys:
        if expected_sd[k].shape != merged_sd[k].shape:
            print(f"  Shape mismatch: {k}: expected={list(expected_sd[k].shape)}, got={list(merged_sd[k].shape)}")
            n_shape_mismatch += 1
        else:
            n_shape_match += 1

    print(f"\n  Shape matches: {n_shape_match}")
    if n_shape_mismatch:
        print(f"  Shape mismatches: {n_shape_mismatch}")
        all_pass = False
    else:
        print(f"  Shape mismatches: 0  (OK)")

    del model, expected_sd

    # ── Step 3: Weight statistics ──
    print(f"\n{'='*60}")
    print("Checking weight statistics...")
    print(f"{'='*60}")

    # Sample a few key tensors for detailed stats
    sample_keys = [k for k in sorted(merged_sd.keys())
                   if "weight" in k and "norm" not in k][:5]
    for k in sample_keys:
        v = merged_sd[k]
        print(f"  {k}:")
        print(f"    shape={list(v.shape)}, dtype={v.dtype}")
        print(f"    mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}")
        print(f"    min={v.min().item():.6f}, max={v.max().item():.6f}")

    stats_ok = check_weight_stats(merged_sd)
    if not stats_ok:
        all_pass = False

    # ── Step 4: Compare with original (optional) ──
    if args.compare_original:
        cmp_ok = compare_with_original(merged_sd, args.compare_original)
        if not cmp_ok:
            all_pass = False

    # ── Step 5: Forward pass (optional) ──
    if args.forward:
        fwd_ok = test_forward(args.dit_path, merged_sd)
        if not fwd_ok:
            all_pass = False

    # ── Summary ──
    print(f"\n{'='*60}")
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED (see above)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
