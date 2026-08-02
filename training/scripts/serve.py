# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
QwenImage serving: keeps DiT + VAE + text encoder in GPU memory,
accepts prompts via HTTP and returns generated images.

Usage:
  python scripts/merge/serve_qwenimage.py \
    --merged_ckpt /path/to/model.safetensors \
    --ckpt_root /path/to/<QWENIMAGE_MODEL>/origin/raw_data \
    --port 8899

  # Then from anywhere:
  curl -X POST http://localhost:8899/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "A cat sitting on a wooden table"}' \
    --output cat.png

  # With parameters:
  curl -X POST http://localhost:8899/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "A sunset over mountains", "height": 768, "width": 1024, "num_steps": 30, "seed": 123}' \
    --output sunset.png

  # Health check:
  curl http://localhost:8899/health

Dependencies: pip install fastapi uvicorn
"""

import argparse
import glob
import io
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CKPT_ROOT = "<WEIGHTS_ROOT>/qwenimage/origin/raw_data"


# ---------------------------------------------------------------------------
# Latent pack / unpack
# ---------------------------------------------------------------------------

def pack_latents(latents, batch_size, num_channels, height, width):
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, height, width)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192,
                    base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def get_schedule(num_steps, image_seq_len, device):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = calculate_shift(image_seq_len)
    shifted = [math.exp(mu) * s / (1 + (math.exp(mu) - 1) * s) for s in sigmas]
    shifted.append(0.0)
    return torch.tensor(shifted, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pipeline: holds all models, does encode → denoise → decode
# ---------------------------------------------------------------------------

class QwenImagePipeline:
    def __init__(self, merged_ckpt: str, ckpt_root: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self._lock = threading.Lock()

        print("=" * 60)
        print("Loading QwenImage pipeline...")
        print("=" * 60)
        t0 = time.time()

        # ── DiT ──
        print("[1/3] Loading DiT...")
        from modeling.qwenimage.transformer import QwenImageTransformer2DModel
        config_path = os.path.join(ckpt_root, "transformer/config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.dit = QwenImageTransformer2DModel(
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
        full_sd = load_file(merged_ckpt, device="cpu")
        dit_sd = {k[len("dit_model."):]: v for k, v in full_sd.items() if k.startswith("dit_model.")}
        if not dit_sd:
            # Fallback: maybe the ckpt is already a plain DiT (no prefix)
            dit_sd = full_sd
        missing, unexpected = self.dit.load_state_dict(dit_sd, strict=False)
        print(f"       {len(dit_sd)} params loaded, missing={len(missing)}, unexpected={len(unexpected)}")
        del full_sd, dit_sd
        self.dit = self.dit.to(self.device, dtype=self.dtype).eval()

        # ── VAE ──
        print("[2/3] Loading VAE...")
        from modeling.qwenimage.vae import AutoencoderKLQwenImage
        with open(os.path.join(ckpt_root, "vae/config.json")) as f:
            self.vae_config = json.load(f)
        self.vae = AutoencoderKLQwenImage(
            base_dim=self.vae_config.get("base_dim", 96),
            z_dim=self.vae_config.get("z_dim", 16),
            dim_mult=self.vae_config.get("dim_mult", [1, 2, 4, 4]),
            num_res_blocks=self.vae_config.get("num_res_blocks", 2),
            temperal_downsample=self.vae_config.get("temperal_downsample", [False, True, True]),
            latents_mean=self.vae_config.get("latents_mean"),
            latents_std=self.vae_config.get("latents_std"),
        )
        vae_state = load_file(os.path.join(ckpt_root, "vae/diffusion_pytorch_model.safetensors"))
        self.vae.load_state_dict(vae_state, strict=False)
        del vae_state
        self.vae = self.vae.to(self.device, dtype=self.dtype).eval()

        # ── Text encoder ──
        print("[3/3] Loading text encoder...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
        self.te_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(ckpt_root, "text_encoder"), torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(ckpt_root, "tokenizer"))

        self.z_dim = self.vae_config.get("z_dim", 16)

        elapsed = time.time() - t0
        print(f"Pipeline ready in {elapsed:.1f}s")
        print("=" * 60)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        TEMPLATE = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, "
            "size, texture, quantity, text, spatial relationships of the objects "
            "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        DROP_IDX = 34

        text = TEMPLATE.format(prompt)
        tokens = self.tokenizer(
            [text], padding=True, return_tensors="pt",
        ).to(self.device)

        lm = self.te_model.model.language_model
        out = lm(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state

        hidden = hidden[0]
        mask = tokens.attention_mask[0].bool()
        valid = hidden[mask][DROP_IDX:]

        prompt_embeds = valid.unsqueeze(0).to(self.dtype)
        prompt_mask = torch.ones(1, valid.shape[0], dtype=torch.bool, device=self.device)
        return prompt_embeds, prompt_mask

    @torch.no_grad()
    def _denoise(self, latents, prompt_embeds, prompt_mask, img_shapes,
                 num_steps, image_seq_len,
                 neg_embeds=None, neg_mask=None, cfg_scale=1.0):
        sigmas = get_schedule(num_steps, image_seq_len, self.device)
        do_cfg = cfg_scale > 1.0 and neg_embeds is not None
        for i in range(num_steps):
            t = sigmas[i].expand(1).to(self.dtype)
            pred = self.dit(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask if not prompt_mask.all() else None,
                timestep=t,
                img_shapes=img_shapes,
            )[0]
            if do_cfg:
                neg_pred = self.dit(
                    hidden_states=latents,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask if not neg_mask.all() else None,
                    timestep=t,
                    img_shapes=img_shapes,
                )[0]
                combined = neg_pred + cfg_scale * (pred - neg_pred)
                cond_norm = torch.norm(pred, dim=-1, keepdim=True)
                comb_norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=1e-8)
                pred = combined * (cond_norm / comb_norm)
            latents = latents + (sigmas[i + 1] - sigmas[i]) * pred
        return latents

    @torch.no_grad()
    def _decode(self, latents, height, width):
        vae_scale = 8
        latents_5d = unpack_latents(latents, height, width, vae_scale).to(self.dtype)

        latents_mean = torch.tensor(self.vae.config["latents_mean"]).view(1, -1, 1, 1, 1).to(latents_5d)
        latents_std_inv = 1.0 / torch.tensor(self.vae.config["latents_std"]).view(1, -1, 1, 1, 1).to(latents_5d)
        latents_5d = latents_5d / latents_std_inv + latents_mean

        decoded = self.vae.decode(latents_5d)
        if isinstance(decoded, dict):
            decoded = decoded["sample"]
        image = decoded[:, :, 0].clamp(-1, 1)
        image = (image + 1) / 2
        return image

    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 50,
        seed: int = 42,
        cfg_scale: float = 1.0,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate an image from a text prompt. Thread-safe (one at a time)."""
        with self._lock:
            t0 = time.time()

            # Encode
            prompt_embeds, prompt_mask = self._encode_prompt(prompt)

            neg_embeds, neg_mask = None, None
            if cfg_scale > 1.0 and negative_prompt is not None:
                if negative_prompt:
                    neg_embeds, neg_mask = self._encode_prompt(negative_prompt)
                else:
                    # Empty prompt for unconditional: encode a single empty string
                    neg_embeds, neg_mask = self._encode_prompt("")

            # Prepare latents
            vae_scale = 8
            latent_h = height // vae_scale
            latent_w = width // vae_scale
            patch_h, patch_w = latent_h // 2, latent_w // 2
            image_seq_len = patch_h * patch_w
            img_shapes = [[(1, patch_h, patch_w)]]

            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(1, self.z_dim, latent_h, latent_w,
                                generator=generator, device=self.device, dtype=self.dtype)
            latents = pack_latents(noise, 1, self.z_dim, latent_h, latent_w)

            # Denoise
            latents = self._denoise(latents, prompt_embeds, prompt_mask,
                                    img_shapes, num_steps, image_seq_len,
                                    neg_embeds=neg_embeds, neg_mask=neg_mask,
                                    cfg_scale=cfg_scale)

            # Decode
            image = self._decode(latents, height, width)

            img_np = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            elapsed = time.time() - t0
            print(f"  Generated {width}x{height} in {elapsed:.1f}s "
                  f"(steps={num_steps}, seed={seed}, cfg={cfg_scale}, mean_pixel={img_np.mean():.0f})")
            return pil_img


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(pipeline: QwenImagePipeline):
    from fastapi import FastAPI
    from fastapi.responses import Response, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="QwenImage Serving")

    class GenerateRequest(BaseModel):
        prompt: str
        height: int = 1024
        width: int = 1024
        num_steps: int = 50
        seed: int = 42
        cfg_scale: float = 1.0
        negative_prompt: str = ""

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "QwenImage"}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        try:
            pil_img = pipeline.generate(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_steps=req.num_steps,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                negative_prompt=req.negative_prompt,
            )
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/generate_base64")
    def generate_base64(req: GenerateRequest):
        """Same as /generate but returns base64-encoded PNG in JSON."""
        import base64
        try:
            pil_img = pipeline.generate(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_steps=req.num_steps,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                negative_prompt=req.negative_prompt,
            )
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {
                "image_base64": b64,
                "width": pil_img.size[0],
                "height": pil_img.size[1],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QwenImage HTTP serving")
    parser.add_argument("--merged_ckpt", required=True,
                        help="Path to merged .safetensors")
    parser.add_argument("--ckpt_root", default=CKPT_ROOT,
                        help="Original QwenImage checkpoint root (VAE, text encoder)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    pipeline = QwenImagePipeline(args.merged_ckpt, args.ckpt_root)

    import uvicorn
    app = create_app(pipeline)
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  POST /generate        → returns PNG image")
    print(f"  POST /generate_base64 → returns JSON with base64 image")
    print(f"  GET  /health          → health check\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
