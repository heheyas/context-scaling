# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
QwenImage multi-GPU serving via Ray Serve with PSM support.

Wraps QwenImagePipeline in a Ray Serve deployment so it gets automatic
PSM registration (via BYTED_RAY_SERVE_PROXY_PSM env var), health checks,
and Ray's built-in load balancing.

Usage:
  # Basic (all GPUs on this node):
  python scripts/merge/serve_qwenimage_ray.py \
    --merged_ckpt /path/to/model.safetensors \
    --ckpt_root /path/to/<QWENIMAGE_MODEL>/origin/raw_data

  # Specify GPU count and replicas:
  python scripts/merge/serve_qwenimage_ray.py \
    --merged_ckpt /path/to/model.safetensors \
    --ckpt_root /path/to/<QWENIMAGE_MODEL>/origin/raw_data \
    --gpu_per_replica 1 \
    --num_replicas 8

  # With PSM (set env before running):
  export BYTED_RAY_SERVE_PROXY_PSM=your_psm_suffix
  python scripts/merge/serve_qwenimage_ray.py ...

PSM will be registered as: inf.ray.serve_{BYTED_RAY_SERVE_PROXY_PSM}

API endpoints (same as serve_qwenimage_multigpu.py):
  POST /generate          -> PNG image
  POST /generate_base64   -> JSON {image_base64, width, height}
  GET  /health            -> {status: ok, ...}
"""

import argparse
import asyncio
import copy
import gc
import io
import json
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Optional

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
# Pipeline (one per replica, owns one GPU)
# ---------------------------------------------------------------------------

class QwenImagePipeline:
    def __init__(self, merged_ckpt: str, ckpt_root: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        from modeling.qwenimage.transformer import QwenImageTransformer2DModel
        from modeling.qwenimage.vae import AutoencoderKLQwenImage
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

        # DiT
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
            dit_sd = full_sd
        self.dit.load_state_dict(dit_sd, strict=False)
        del full_sd, dit_sd; gc.collect()
        self.dit = self.dit.to(self.device, dtype=self.dtype).eval()

        # VAE
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
        del vae_state; gc.collect()
        self.vae = self.vae.to(self.device, dtype=self.dtype).eval()

        # Text encoder
        self.te_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(ckpt_root, "text_encoder"), torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(ckpt_root, "tokenizer"))
        self.z_dim = self.vae_config.get("z_dim", 16)

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
        valid = hidden[0][tokens.attention_mask[0].bool()][DROP_IDX:]
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
        return (image + 1) / 2

    def generate(self, prompt, height=1024, width=1024, num_steps=50,
                 seed=42, cfg_scale=1.0, negative_prompt=""):
        t0 = time.time()
        prompt_embeds, prompt_mask = self._encode_prompt(prompt)
        neg_embeds, neg_mask = None, None
        if cfg_scale > 1.0:
            neg_embeds, neg_mask = self._encode_prompt(negative_prompt if negative_prompt else "")

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
        latents = self._denoise(latents, prompt_embeds, prompt_mask,
                                img_shapes, num_steps, image_seq_len,
                                neg_embeds=neg_embeds, neg_mask=neg_mask,
                                cfg_scale=cfg_scale)
        image = self._decode(latents, height, width)
        img_np = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        elapsed = time.time() - t0
        print(f"  Generated {width}x{height} in {elapsed:.1f}s "
              f"(steps={num_steps}, seed={seed}, cfg={cfg_scale})")
        return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------

import ray
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from ray import serve

app = FastAPI(title="QwenImage Ray Serve")


class GenerateRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_steps: int = 50
    seed: int = 42
    cfg_scale: float = 1.0
    negative_prompt: str = ""


@serve.ingress(app)
class QwenImageServing:
    def __init__(self, merged_ckpt: str, ckpt_root: str):
        print(f"[QwenImageServing] Initializing pipeline ...")
        print(f"  merged_ckpt: {merged_ckpt}")
        print(f"  ckpt_root:   {ckpt_root}")
        t0 = time.time()
        self.pipeline = QwenImagePipeline(merged_ckpt, ckpt_root, device="cuda")
        self._lock = threading.Lock()
        elapsed = time.time() - t0
        print(f"[QwenImageServing] Ready in {elapsed:.1f}s")

    @app.get("/health")
    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "model": "QwenImage"}

    @app.post("/generate")
    async def generate(self, req: GenerateRequest):
        try:
            loop = asyncio.get_event_loop()
            pil_img = await loop.run_in_executor(None, self._generate_sync, req)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/generate_base64")
    async def generate_base64(self, req: GenerateRequest):
        import base64
        try:
            loop = asyncio.get_event_loop()
            pil_img = await loop.run_in_executor(None, self._generate_sync, req)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"image_base64": b64, "width": pil_img.size[0], "height": pil_img.size[1]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    def _generate_sync(self, req: GenerateRequest) -> Image.Image:
        with self._lock:
            return self.pipeline.generate(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_steps=req.num_steps,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                negative_prompt=req.negative_prompt,
            )

    def check_health(self):
        """Ray Serve health check — verifies GPU is accessible."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QwenImage Ray Serve with PSM support")
    parser.add_argument("--merged_ckpt", required=True)
    parser.add_argument("--ckpt_root", default=CKPT_ROOT)
    parser.add_argument("--gpu_per_replica", type=int, default=1,
                        help="GPUs per replica (default: 1)")
    parser.add_argument("--num_replicas", type=int, default=0,
                        help="Number of replicas (default: auto = total_gpus / gpu_per_replica)")
    parser.add_argument("--max_ongoing_requests", type=int, default=8,
                        help="Max concurrent requests per replica")
    parser.add_argument("--port", type=int, default=8899,
                        help="HTTP port (also used by Ray Serve proxy)")
    parser.add_argument("--health_check_period_s", type=float, default=30)
    parser.add_argument("--health_check_timeout_s", type=float, default=60)
    args = parser.parse_args()

    # PSM info
    ray_psm_suffix = os.getenv("BYTED_RAY_SERVE_PROXY_PSM", None)
    if ray_psm_suffix is not None:
        ray_psm = f"inf.ray.serve_{ray_psm_suffix}" if "." not in ray_psm_suffix else ray_psm_suffix
        print(f"PSM: {ray_psm}")
    else:
        print("No PSM configured (set BYTED_RAY_SERVE_PROXY_PSM to enable)")

    # Auto-detect replicas
    ray.init()
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if args.num_replicas <= 0:
        args.num_replicas = max(1, total_gpus // args.gpu_per_replica)

    print("=" * 60)
    print("QwenImage Ray Serve")
    print(f"  Total GPUs:       {total_gpus}")
    print(f"  GPU per replica:  {args.gpu_per_replica}")
    print(f"  Num replicas:     {args.num_replicas}")
    print(f"  Max ongoing reqs: {args.max_ongoing_requests}")
    print(f"  Port:             {args.port}")
    print("=" * 60)

    # Deploy
    placement_group_bundles = [{"GPU": args.gpu_per_replica, "CPU": 1}]

    try:
        deployment_cls = serve.deployment(
            ray_actor_options={"num_gpus": args.gpu_per_replica},
            num_replicas=args.num_replicas,
            max_ongoing_requests=args.max_ongoing_requests,
            health_check_period_s=args.health_check_period_s,
            health_check_timeout_s=args.health_check_timeout_s,
            placement_group_bundles=placement_group_bundles,
            placement_group_strategy="PACK",
        )(QwenImageServing)
    except TypeError:
        # Fallback for older Ray versions without placement_group args
        deployment_cls = serve.deployment(
            ray_actor_options={"num_gpus": args.gpu_per_replica},
            num_replicas=args.num_replicas,
            max_ongoing_requests=args.max_ongoing_requests,
            health_check_period_s=args.health_check_period_s,
            health_check_timeout_s=args.health_check_timeout_s,
        )(QwenImageServing)

    ray_services = deployment_cls.bind(
        args.merged_ckpt,
        args.ckpt_root,
    )

    serve.run(
        ray_services,
        name="qwenimage_server",
        host="0.0.0.0",
        port=args.port,
    )

    print(f"\nServing on http://0.0.0.0:{args.port}")
    print(f"  POST /generate        -> PNG image")
    print(f"  POST /generate_base64 -> JSON with base64 image")
    print(f"  GET  /health          -> health check")
    if ray_psm_suffix:
        print(f"  PSM: inf.ray.serve_{ray_psm_suffix}")
    print()

    # Keep process alive
    signal.pause()


if __name__ == "__main__":
    main()
