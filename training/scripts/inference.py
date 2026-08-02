# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
QwenImage inference using the codebase's modeling.

Usage:
    python scripts/inference_qwenimage.py \
        --prompt "A cat sitting on a wooden table" \
        --output output.png
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CKPT_ROOT = "<WEIGHTS_ROOT>/qwenimage/origin/raw_data"


# ---------------------------------------------------------------------------
# Latent pack / unpack (2x2 patches)
# ---------------------------------------------------------------------------

def pack_latents(latents, batch_size, num_channels, height, width):
    """[B, C, H, W] → [B, (H/2)*(W/2), C*4]"""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    """[B, S, C*4] → [B, C, 1, H, W]"""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, height, width)


# ---------------------------------------------------------------------------
# Timestep schedule (flow matching Euler with dynamic shifting)
# ---------------------------------------------------------------------------

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192,
                    base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def get_schedule(num_steps, image_seq_len, device):
    """Compute shifted sigmas for Euler sampling."""
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = calculate_shift(image_seq_len)
    # Exponential shift (matching scheduler config time_shift_type=exponential)
    shifted = []
    for s in sigmas:
        shifted.append(math.exp(mu) * s / (1 + (math.exp(mu) - 1) * s))
    shifted.append(0.0)
    return torch.tensor(shifted, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt(text_encoder_model, tokenizer, prompt, device, dtype):
    """Encode prompt using QwenImage's template + Qwen2.5-VL."""
    TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects "
        "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    DROP_IDX = 34

    text = TEMPLATE.format(prompt)
    tokens = tokenizer(
        [text], padding=True, return_tensors="pt",
    ).to(device)

    lm = text_encoder_model.model.language_model
    out = lm(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state

    # Extract valid tokens, drop template prefix
    hidden = hidden[0]  # [T, D]
    mask = tokens.attention_mask[0].bool()
    valid = hidden[mask]
    valid = valid[DROP_IDX:]  # drop template prefix

    prompt_embeds = valid.unsqueeze(0).to(dtype)  # [1, T, D]
    prompt_mask = torch.ones(1, valid.shape[0], dtype=torch.bool, device=device)
    return prompt_embeds, prompt_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cat sitting on a wooden table, realistic photo")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--ckpt_root", type=str, default=CKPT_ROOT)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    ckpt = args.ckpt_root

    # ── Load transformer ──────────────────────────────────────────────
    print("Loading transformer...")
    from modeling.qwenimage.transformer import QwenImageTransformer2DModel
    with open(os.path.join(ckpt, "transformer/config.json")) as f:
        tf_config = json.load(f)
    model = QwenImageTransformer2DModel(
        patch_size=tf_config.get("patch_size", 2),
        in_channels=tf_config.get("in_channels", 64),
        out_channels=tf_config.get("out_channels", 16),
        num_layers=tf_config.get("num_layers", 60),
        attention_head_dim=tf_config.get("attention_head_dim", 128),
        num_attention_heads=tf_config.get("num_attention_heads", 24),
        joint_attention_dim=tf_config.get("joint_attention_dim", 3584),
        axes_dims_rope=tuple(tf_config.get("axes_dims_rope", [16, 56, 56])),
    )
    # Load all safetensor shards
    import glob
    shards = sorted(glob.glob(os.path.join(ckpt, "transformer/*.safetensors")))
    state = {}
    for s in shards:
        state.update(load_file(s, device="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded transformer: {len(state)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    del state
    model = model.to(device, dtype=dtype).eval()

    # ── Load VAE ──────────────────────────────────────────────────────
    print("Loading VAE...")
    from modeling.qwenimage.vae import AutoencoderKLQwenImage
    with open(os.path.join(ckpt, "vae/config.json")) as f:
        vae_config = json.load(f)
    vae = AutoencoderKLQwenImage(
        base_dim=vae_config.get("base_dim", 96),
        z_dim=vae_config.get("z_dim", 16),
        dim_mult=vae_config.get("dim_mult", [1, 2, 4, 4]),
        num_res_blocks=vae_config.get("num_res_blocks", 2),
        temperal_downsample=vae_config.get("temperal_downsample", [False, True, True]),
        latents_mean=vae_config.get("latents_mean"),
        latents_std=vae_config.get("latents_std"),
    )
    vae_state = load_file(os.path.join(ckpt, "vae/diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(vae_state, strict=False)
    del vae_state
    vae = vae.to(device, dtype=dtype).eval()

    # ── Load text encoder ─────────────────────────────────────────────
    print("Loading text encoder...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
    te_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(ckpt, "text_encoder"), torch_dtype=dtype,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ckpt, "tokenizer"))

    # ── Encode prompt ─────────────────────────────────────────────────
    print(f"Encoding prompt: \"{args.prompt}\"")
    prompt_embeds, prompt_mask = encode_prompt(te_model, tokenizer, args.prompt, device, dtype)
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    neg_embeds, neg_mask = None, None
    do_cfg = args.cfg_scale > 1.0 and args.negative_prompt != ""
    if do_cfg:
        neg_embeds, neg_mask = encode_prompt(te_model, tokenizer, args.negative_prompt, device, dtype)

    # Free text encoder memory
    del te_model
    torch.cuda.empty_cache()

    # ── Prepare latents ───────────────────────────────────────────────
    vae_scale = 8  # spatial compression
    latent_h = args.height // vae_scale
    latent_w = args.width // vae_scale
    patch_h, patch_w = latent_h // 2, latent_w // 2
    image_seq_len = patch_h * patch_w
    z_dim = vae_config.get("z_dim", 16)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(1, z_dim, latent_h, latent_w, generator=generator, device=device, dtype=dtype)
    latents = pack_latents(noise, 1, z_dim, latent_h, latent_w)  # [1, S, 64]

    img_shapes = [[(1, patch_h, patch_w)]]

    # ── Denoising loop ────────────────────────────────────────────────
    sigmas = get_schedule(args.num_steps, image_seq_len, device)
    print(f"Denoising: {args.num_steps} steps, cfg={args.cfg_scale}, "
          f"latent={latent_h}x{latent_w}, patches={patch_h}x{patch_w}")

    with torch.no_grad():
        for i in tqdm(range(args.num_steps)):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            t = sigma.expand(1).to(dtype)

            # Conditional prediction
            pred = model(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask if not prompt_mask.all() else None,
                timestep=t,
                img_shapes=img_shapes,
            )[0]

            if do_cfg:
                neg_pred = model(
                    hidden_states=latents,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask if not neg_mask.all() else None,
                    timestep=t,
                    img_shapes=img_shapes,
                )[0]
                combined = neg_pred + args.cfg_scale * (pred - neg_pred)
                # Norm rescaling
                cond_norm = torch.norm(pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=1e-8)
                pred = combined * (cond_norm / comb_norm)

            # Euler step
            latents = latents + (sigma_next - sigma) * pred

    # ── Decode ────────────────────────────────────────────────────────
    print("Decoding...")
    latents_5d = unpack_latents(latents, args.height, args.width, vae_scale)
    latents_5d = latents_5d.to(dtype)

    # Denormalize
    latents_mean = torch.tensor(vae.config["latents_mean"]).view(1, -1, 1, 1, 1).to(latents_5d)
    latents_std_inv = 1.0 / torch.tensor(vae.config["latents_std"]).view(1, -1, 1, 1, 1).to(latents_5d)
    latents_5d = latents_5d / latents_std_inv + latents_mean

    with torch.no_grad():
        decoded = vae.decode(latents_5d)
        if isinstance(decoded, dict):
            decoded = decoded["sample"]
    image = decoded[:, :, 0]  # remove temporal dim → [1, 3, H, W]
    image = image.clamp(-1, 1)
    image = (image + 1) / 2  # → [0, 1]

    # Save
    img_np = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pil_img.save(args.output)
    print(f"\nSaved: {args.output} ({pil_img.size[0]}x{pil_img.size[1]})")

    # Sanity checks
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Mean pixel: {img_np.mean():.1f}")
    is_black = img_np.mean() < 10
    is_white = img_np.mean() > 245
    if is_black:
        print("  WARNING: Image appears mostly black!")
    elif is_white:
        print("  WARNING: Image appears mostly white!")
    else:
        print("  Image looks non-trivial (not black/white).")


if __name__ == "__main__":
    main()
