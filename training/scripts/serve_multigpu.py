# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Multi-GPU QwenImage serving with request queue and GPU dispatch.

Each GPU loads its own pipeline. Incoming requests are queued and dispatched
to the first available GPU. If all GPUs are busy (up to --max_per_gpu pending
each), the request blocks until a slot opens.

Usage:
  python scripts/merge/serve_qwenimage_multigpu.py \
    --merged_ckpt /path/to/model.safetensors \
    --ckpt_root /path/to/<QWENIMAGE_MODEL>/origin/raw_data \
    --gpus 0,1,2,3 \
    --max_per_gpu 4 \
    --port 8899

  # Use all visible GPUs:
  python scripts/merge/serve_qwenimage_multigpu.py \
    --merged_ckpt /path/to/model.safetensors \
    --ckpt_root /path/to/<QWENIMAGE_MODEL>/origin/raw_data \
    --port 8899

API (same as single-GPU version):
  POST /generate          → PNG image
  POST /generate_base64   → JSON {image_base64, width, height}
  GET  /health            → {status, gpus, queue_size, queue_capacity}

Dependencies: pip install fastapi uvicorn
"""

import argparse
import asyncio
import io
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
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
# Single-GPU pipeline (loaded once per GPU)
# ---------------------------------------------------------------------------

def _build_dit_cpu(dit_config):
    from modeling.qwenimage.transformer import QwenImageTransformer2DModel
    return QwenImageTransformer2DModel(
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


def _build_vae_cpu(vae_config):
    from modeling.qwenimage.vae import AutoencoderKLQwenImage
    return AutoencoderKLQwenImage(
        base_dim=vae_config.get("base_dim", 96),
        z_dim=vae_config.get("z_dim", 16),
        dim_mult=vae_config.get("dim_mult", [1, 2, 4, 4]),
        num_res_blocks=vae_config.get("num_res_blocks", 2),
        temperal_downsample=vae_config.get("temperal_downsample", [False, True, True]),
        latents_mean=vae_config.get("latents_mean"),
        latents_std=vae_config.get("latents_std"),
    )


def prepare_cpu_models(merged_ckpt: str, ckpt_root: str, num_gpus: int):
    """Load weights from disk ONCE, build N complete CPU models (one per GPU).
    Workers only need to call .to(device) — no load_state_dict, no deepcopy."""
    import copy, gc
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

    with open(os.path.join(ckpt_root, "transformer/config.json")) as f:
        dit_config = json.load(f)
    with open(os.path.join(ckpt_root, "vae/config.json")) as f:
        vae_config = json.load(f)

    # ── Read files from disk (once) ──
    print("[Preload] Reading DiT checkpoint ...")
    t0 = time.time()
    full_sd = load_file(merged_ckpt, device="cpu")
    dit_sd = {k[len("dit_model."):]: v for k, v in full_sd.items() if k.startswith("dit_model.")}
    if not dit_sd:
        dit_sd = full_sd
    del full_sd; gc.collect()
    print(f"[Preload] DiT: {len(dit_sd)} tensors ({time.time()-t0:.1f}s)")

    print("[Preload] Reading VAE checkpoint ...")
    t0 = time.time()
    vae_sd = load_file(os.path.join(ckpt_root, "vae/diffusion_pytorch_model.safetensors"), device="cpu")
    print(f"[Preload] VAE: {len(vae_sd)} tensors ({time.time()-t0:.1f}s)")

    print("[Preload] Reading text encoder ...")
    t0 = time.time()
    te_model_0 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(ckpt_root, "text_encoder"), torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ckpt_root, "tokenizer"))
    print(f"[Preload] Text encoder ({time.time()-t0:.1f}s)")

    # ── Build first complete model on CPU ──
    print("[Preload] Building model #0 on CPU ...")
    t0 = time.time()
    dit_0 = _build_dit_cpu(dit_config)
    dit_0.load_state_dict(dit_sd, strict=False)
    dit_0.eval()
    vae_0 = _build_vae_cpu(vae_config)
    vae_0.load_state_dict(vae_sd, strict=False)
    vae_0.eval()
    del dit_sd, vae_sd; gc.collect()
    print(f"[Preload] Model #0 ready ({time.time()-t0:.1f}s)")

    # ── Deepcopy for remaining GPUs (sequential in main thread, no GIL fight) ──
    cpu_models = [{"dit": dit_0, "vae": vae_0, "te": te_model_0,
                   "vae_config": vae_config, "tokenizer": tokenizer}]
    for i in range(1, num_gpus):
        print(f"[Preload] Copying model #{i} ...")
        t0 = time.time()
        cpu_models.append({
            "dit": copy.deepcopy(dit_0),
            "vae": copy.deepcopy(vae_0),
            "te": copy.deepcopy(te_model_0),
            "vae_config": vae_config,
            "tokenizer": tokenizer,
        })
        print(f"[Preload] Model #{i} copied ({time.time()-t0:.1f}s)")

    print(f"[Preload] All {num_gpus} CPU models ready\n")
    return cpu_models


class QwenImagePipeline:
    def __init__(self, cpu_model: dict, device: str = "cuda:0"):
        """Move pre-built CPU models to GPU. No disk I/O, no load_state_dict,
        no deepcopy — just .to(device)."""
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        self.dit = cpu_model["dit"].to(self.device, dtype=self.dtype).eval()
        self.vae = cpu_model["vae"].to(self.device, dtype=self.dtype).eval()
        self.te_model = cpu_model["te"].to(self.device).eval()
        self.tokenizer = cpu_model["tokenizer"]
        self.vae_config = cpu_model["vae_config"]
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

    @torch.no_grad()
    def generate(self, prompt, height=1024, width=1024, num_steps=50,
                 seed=42, cfg_scale=1.0, negative_prompt=""):
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
        return Image.fromarray(img_np)


# ---------------------------------------------------------------------------
# GPU Worker: one thread per GPU, pulls tasks from its own queue
# ---------------------------------------------------------------------------

class GPUWorker(threading.Thread):
    """Worker thread that owns a pipeline on a single GPU and processes tasks
    from a bounded queue. The queue size = max_per_gpu, so at most that many
    requests can be pending on this GPU before the dispatcher must wait."""

    def __init__(self, gpu_id: int, cpu_model: dict, max_per_gpu: int):
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.cpu_model = cpu_model
        self.task_queue: queue.Queue = queue.Queue(maxsize=max_per_gpu)
        self.pipeline: Optional[QwenImagePipeline] = None
        self.ready = threading.Event()

    def run(self):
        device = f"cuda:{self.gpu_id}"
        print(f"[GPU {self.gpu_id}] Moving weights to {device} ...")
        t0 = time.time()
        self.pipeline = QwenImagePipeline(self.cpu_model, device=device)
        self.cpu_model = None  # release CPU copy
        elapsed = time.time() - t0
        print(f"[GPU {self.gpu_id}] Ready in {elapsed:.1f}s")
        self.ready.set()

        while True:
            future, kwargs = self.task_queue.get()
            try:
                t0 = time.time()
                result = self.pipeline.generate(**kwargs)
                elapsed = time.time() - t0
                print(f"[GPU {self.gpu_id}] Generated {kwargs.get('width','?')}x{kwargs.get('height','?')} "
                      f"in {elapsed:.1f}s (steps={kwargs.get('num_steps',50)}, "
                      f"seed={kwargs.get('seed',42)}, cfg={kwargs.get('cfg_scale',1.0)}, "
                      f"queue_remaining={self.task_queue.qsize()})")
                future.set_result(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                future.set_exception(e)
            finally:
                self.task_queue.task_done()

    @property
    def pending(self) -> int:
        return self.task_queue.qsize()

    @property
    def is_full(self) -> bool:
        return self.task_queue.full()


# ---------------------------------------------------------------------------
# Worker Pool: dispatches to least-loaded GPU
# ---------------------------------------------------------------------------

class WorkerPool:
    def __init__(self, gpu_ids, cpu_models, max_per_gpu):
        self.workers = []
        self.max_per_gpu = max_per_gpu

        # Each worker gets its own pre-built CPU model, only does .to(device).
        # Safe to run in parallel — no shared state, no GIL-heavy ops.
        print(f"Starting {len(gpu_ids)} GPU worker(s) in parallel ...")
        for gpu_id, cpu_model in zip(gpu_ids, cpu_models):
            w = GPUWorker(gpu_id, cpu_model, max_per_gpu)
            w.start()
            self.workers.append(w)

        for w in self.workers:
            w.ready.wait()
        print(f"All {len(self.workers)} GPU(s) ready!\n")

        # Condition variable for blocking when all GPUs are full
        self._cv = threading.Condition()

    @property
    def total_capacity(self) -> int:
        return len(self.workers) * self.max_per_gpu

    @property
    def total_pending(self) -> int:
        return sum(w.pending for w in self.workers)

    def _find_worker(self) -> Optional[GPUWorker]:
        """Find the worker with the least pending tasks that isn't full."""
        candidates = [w for w in self.workers if not w.is_full]
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.pending)

    def submit(self, **kwargs) -> Future:
        """Submit a generation task. Blocks if all GPUs are at capacity."""
        future = Future()

        with self._cv:
            while True:
                worker = self._find_worker()
                if worker is not None:
                    break
                # All full — wait until a slot opens
                self._cv.wait(timeout=0.1)

            worker.task_queue.put((future, kwargs))

        # When this task completes, notify waiters that a slot opened
        def _on_done(_):
            with self._cv:
                self._cv.notify_all()
        future.add_done_callback(_on_done)

        return future

    def status(self):
        return {
            "gpus": [
                {"gpu_id": w.gpu_id, "pending": w.pending, "capacity": self.max_per_gpu}
                for w in self.workers
            ],
            "total_pending": self.total_pending,
            "total_capacity": self.total_capacity,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(pool: WorkerPool):
    from fastapi import FastAPI
    from fastapi.responses import Response, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="QwenImage Multi-GPU Serving")

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
        return {"status": "ok", "model": "QwenImage", **pool.status()}

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        try:
            future = pool.submit(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_steps=req.num_steps,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                negative_prompt=req.negative_prompt,
            )
            # Await the future without blocking the event loop
            loop = asyncio.get_event_loop()
            pil_img = await loop.run_in_executor(None, future.result)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/generate_base64")
    async def generate_base64(req: GenerateRequest):
        import base64
        try:
            future = pool.submit(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_steps=req.num_steps,
                seed=req.seed,
                cfg_scale=req.cfg_scale,
                negative_prompt=req.negative_prompt,
            )
            loop = asyncio.get_event_loop()
            pil_img = await loop.run_in_executor(None, future.result)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"image_base64": b64, "width": pil_img.size[0], "height": pil_img.size[1]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QwenImage multi-GPU HTTP serving")
    parser.add_argument("--merged_ckpt", required=True)
    parser.add_argument("--ckpt_root", default=CKPT_ROOT)
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2,3'. Default: all visible GPUs")
    parser.add_argument("--max_per_gpu", type=int, default=4,
                        help="Max pending requests per GPU before blocking")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    # Determine GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        print("ERROR: No GPUs available")
        sys.exit(1)

    print("=" * 60)
    print(f"QwenImage Multi-GPU Serving")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Max per GPU: {args.max_per_gpu}")
    print(f"  Total capacity: {len(gpu_ids) * args.max_per_gpu}")
    print("=" * 60)

    # Step 1: Read weights once, build N complete CPU models (one per GPU)
    cpu_models = prepare_cpu_models(args.merged_ckpt, args.ckpt_root, len(gpu_ids))

    # Step 2: Each GPU worker just does .to(device) — parallel, fast
    pool = WorkerPool(gpu_ids, cpu_models, args.max_per_gpu)

    # Free CPU models (each worker took ownership and moved to GPU)
    del cpu_models
    import gc; gc.collect()

    import uvicorn
    app = create_app(pool)
    print(f"\nServing on http://{args.host}:{args.port}")
    print(f"  POST /generate        → PNG image")
    print(f"  POST /generate_base64 → JSON with base64 image")
    print(f"  GET  /health          → status + per-GPU queue info\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
