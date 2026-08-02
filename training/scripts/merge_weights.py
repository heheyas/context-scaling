# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Merge multiple QwenImage checkpoints by weighted combination.

Usage:
  # Simple average of two checkpoints:
  python scripts/merge/merge_weights.py \
    --ckpts /path/to/a.safetensors /path/to/b.safetensors \
    --weights 0.5 0.5 \
    --output /path/to/merged.safetensors

  # Weighted merge (70% model A + 30% model B):
  python scripts/merge/merge_weights.py \
    --ckpts /path/to/a.safetensors /path/to/b.safetensors \
    --weights 0.7 0.3 \
    --output /path/to/merged.safetensors

  # Three-way merge:
  python scripts/merge/merge_weights.py \
    --ckpts a.safetensors b.safetensors c.safetensors \
    --weights 0.5 0.3 0.2 \
    --output merged.safetensors

  # Merge with original QwenImage as base (SLERP-like interpolation):
  python scripts/merge/merge_weights.py \
    --ckpts /path/to/original.safetensors /path/to/finetuned.safetensors \
    --weights 0.3 0.7 \
    --output /path/to/merged.safetensors

  # Use trial names (auto-resolve to safetensors path):
  python scripts/merge/merge_weights.py \
    --trials trial_a:0016400:ema trial_b:0018000:model \
    --weights 0.5 0.5 \
    --output /path/to/merged.safetensors

  # Delta mode: base + weighted sum of (finetuned - base) deltas
  # merged = base + w1*(ft1 - base) + w2*(ft2 - base)
  python scripts/merge/merge_weights.py \
    --mode delta \
    --base /path/to/original_qwenimage.safetensors \
    --ckpts /path/to/ft1.safetensors /path/to/ft2.safetensors \
    --weights 0.7 0.3 \
    --output /path/to/merged.safetensors

  # Delta mode with single model (scale the delta):
  # merged = base + 1.5 * (finetuned - base)
  python scripts/merge/merge_weights.py \
    --mode delta \
    --base /path/to/original.safetensors \
    --ckpts /path/to/finetuned.safetensors \
    --weights 1.5 \
    --output /path/to/amplified.safetensors \
    --no-normalize
"""

import argparse
import gc
import os
import sys

import torch
from safetensors.torch import load_file, save_file


TRIALS_ROOT = "<HDFS_ROOT>/<TRIALS_ROOT>"
ORIGINAL_QWENIMAGE = "<WEIGHTS_ROOT>/qwenimage/origin/raw_data/transformer"


def load_safetensors(path):
    """Load safetensors, supporting both single file and sharded diffusers directory.
    Returns state dict with keys normalized (no dit_model. prefix)."""
    import glob

    if os.path.isdir(path):
        # Sharded diffusers format (e.g. original QwenImage transformer/ directory)
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not shards:
            raise ValueError(f"No .safetensors files in directory: {path}")
        sd = {}
        for s in shards:
            sd.update(load_file(s, device="cpu"))
        print(f"  Loaded {len(sd)} tensors from {len(shards)} shards in {path}")
        return sd, "diffusers"

    # Single file
    sd = load_file(path, device="cpu")

    # Detect format: if keys start with dit_model., it's our CausalFusion format
    has_prefix = any(k.startswith("dit_model.") for k in sd)
    if has_prefix:
        return sd, "causalfusion"
    else:
        return sd, "diffusers"


def normalize_keys(sd, fmt, target_fmt):
    """Convert between key formats.
    diffusers: 'img_in.weight'
    causalfusion: 'dit_model.img_in.weight', 'text_encoder.xxx'
    """
    if fmt == target_fmt:
        return sd

    prefix = "dit_model."
    if fmt == "causalfusion" and target_fmt == "diffusers":
        # Strip dit_model. prefix, drop text_encoder keys
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            # skip text_encoder.* keys
        return out
    elif fmt == "diffusers" and target_fmt == "causalfusion":
        # Add dit_model. prefix
        return {f"{prefix}{k}": v for k, v in sd.items()}

    return sd


def resolve_trial_path(trial_spec):
    """Resolve 'trial_name:step:weight_type' to safetensors path."""
    parts = trial_spec.split(":")
    trial_name = parts[0]
    step = parts[1] if len(parts) > 1 else None
    weight_type = parts[2] if len(parts) > 2 else "model"

    trial_dir = os.path.join(TRIALS_ROOT, trial_name)
    if not os.path.isdir(trial_dir):
        raise ValueError(f"Trial not found: {trial_dir}")

    if step is None:
        # Auto-detect latest step
        steps = sorted([d for d in os.listdir(trial_dir) if d.isdigit()])
        if not steps:
            raise ValueError(f"No step dirs in {trial_dir}")
        step = steps[-1]

    path = os.path.join(trial_dir, step, f"{weight_type}.safetensors")
    if not os.path.isfile(path):
        raise ValueError(f"Not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Weighted merge of QwenImage checkpoints")
    parser.add_argument("--mode", default="linear", choices=["linear", "delta"],
                        help="linear: weighted sum; delta: base + weighted deltas")
    parser.add_argument("--base", default=None,
                        help="Base checkpoint for delta mode (path or trial:step:type)")
    parser.add_argument("--ckpts", nargs="+", default=None,
                        help="Paths to .safetensors files")
    parser.add_argument("--trials", nargs="+", default=None,
                        help="Trial specs: trial_name:step:type (e.g. my_trial:0016400:ema)")
    parser.add_argument("--weights", nargs="+", type=float, required=True,
                        help="Weights for each checkpoint")
    parser.add_argument("--output", required=True,
                        help="Output .safetensors path")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no-normalize", action="store_true",
                        help="Don't normalize weights to sum to 1.0")
    args = parser.parse_args()

    # Resolve checkpoint paths
    if args.ckpts:
        paths = args.ckpts
    elif args.trials:
        paths = [resolve_trial_path(t) for t in args.trials]
    else:
        parser.error("Specify --ckpts or --trials")

    weights = args.weights
    assert len(paths) == len(weights), f"Got {len(paths)} checkpoints but {len(weights)} weights"

    # Resolve base for delta mode
    base_path = None
    if args.mode == "delta":
        if args.base is None:
            parser.error("--base is required for delta mode")
        base_path = args.base
        if not os.path.isfile(base_path) and not os.path.isdir(base_path):
            # Not a file or directory — try as trial spec
            base_path = resolve_trial_path(base_path)

    # Normalize weights (only for linear mode by default)
    if not args.no_normalize and args.mode == "linear":
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            print(f"Normalizing weights: {weights} (sum={total:.4f}) -> ", end="")
            weights = [w / total for w in weights]
            print(f"{weights}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    target_dtype = dtype_map[args.dtype]

    print(f"Mode: {args.mode}")
    if args.mode == "delta":
        print(f"Base: {base_path}")
        print(f"Deltas ({len(paths)}):")
        for p, w in zip(paths, weights):
            print(f"  {w:+.4f} x ({os.path.basename(p)} - base)")
        print(f"Formula: merged = base + sum(w_i * (ckpt_i - base))")
    else:
        print(f"Merging {len(paths)} checkpoints:")
        for p, w in zip(paths, weights):
            print(f"  {w:.4f} x {p}")
    print(f"Output: {args.output} ({args.dtype})")
    print()

    # Detect output format from first checkpoint
    print(f"Detecting key formats ...")
    first_sd, first_fmt = load_safetensors(paths[0])
    target_fmt = first_fmt
    print(f"  Target format: {target_fmt}")
    del first_sd; gc.collect()

    if args.mode == "delta":
        # Delta mode: merged = base + sum(w_i * (ckpt_i - base))
        print(f"\nLoading base ...")
        base_sd, base_fmt = load_safetensors(base_path)
        base_sd = normalize_keys(base_sd, base_fmt, target_fmt)
        print(f"  {len(base_sd)} tensors (format: {base_fmt} -> {target_fmt})")

        # Start from base
        merged = {}
        for k, v in base_sd.items():
            merged[k] = v.float() if torch.is_floating_point(v) else v

        # Add weighted deltas
        for i, (path, weight) in enumerate(zip(paths, weights)):
            print(f"[{i+1}/{len(paths)}] Loading {os.path.basename(path)} (delta weight={weight:+.4f}) ...")
            sd, fmt = load_safetensors(path)
            sd = normalize_keys(sd, fmt, target_fmt)
            print(f"  {len(sd)} tensors (format: {fmt} -> {target_fmt})")
            for k, v in sd.items():
                if k in merged and k in base_sd and torch.is_floating_point(v):
                    delta = v.float() - base_sd[k].float()
                    merged[k] += delta * weight
            del sd
            gc.collect()

        del base_sd
        gc.collect()

    else:
        # Linear mode: merged = sum(w_i * ckpt_i)
        merged = None
        for i, (path, weight) in enumerate(zip(paths, weights)):
            print(f"[{i+1}/{len(paths)}] Loading {os.path.basename(path)} (weight={weight:.4f}) ...")
            sd, fmt = load_safetensors(path)
            sd = normalize_keys(sd, fmt, target_fmt)
            print(f"  {len(sd)} tensors (format: {fmt} -> {target_fmt})")

            if merged is None:
                merged = {}
                for k, v in sd.items():
                    if torch.is_floating_point(v):
                        merged[k] = v.float() * weight
                    else:
                        merged[k] = v
            else:
                for k, v in sd.items():
                    if k in merged and torch.is_floating_point(v):
                        merged[k] += v.float() * weight

            del sd
            gc.collect()

    # Cast to target dtype and save
    print(f"\nCasting to {args.dtype} and saving ...")
    casted = {}
    for k, v in merged.items():
        if torch.is_floating_point(v) and v.dtype != target_dtype:
            v = v.to(dtype=target_dtype)
        casted[k] = v.contiguous()
    del merged
    gc.collect()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(casted, args.output)
    print(f"Saved {len(casted)} tensors to {args.output}")


if __name__ == "__main__":
    main()
