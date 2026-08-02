# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""FastAPI entry point for the Structured Prompt Studio demo.

Two POST endpoints back the JS studio in ``static/``:

    POST /api/generate_sp     — natural language → Structured Prompt JSON
    POST /api/generate_image  — Structured Prompt → rendered image (base64 PNG)

Both models are loaded once at startup. The default device layout is
GPU 0 for the DiT stack and GPU 1 for the PE model; override via env vars
(see backend/config.py).

Run locally:

    CUDA_VISIBLE_DEVICES=0,1 python -m demo.app

Or via uvicorn:

    CUDA_VISIBLE_DEVICES=0,1 uvicorn demo.app:app --host 0.0.0.0 --port 7860

Then open http://localhost:7860/ in a browser.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# When PE and DiT share two GPUs, GPU 0 ends up packed to within a few GB of
# capacity by the DiT stack. cuDNN needs room *outside* the PyTorch caching
# allocator to bring up its handle for the VAE's Conv3d — under this layout
# it can't get any, and the whole VAE decode fails with
# CUDNN_STATUS_NOT_INITIALIZED. The QwenImage VAE is small enough that
# PyTorch's native conv kernels handle it comfortably, so we disable cuDNN
# by default in the demo. Set DEMO_ENABLE_CUDNN=1 to override.
if os.environ.get("DEMO_ENABLE_CUDNN", "0") != "1":
    import torch as _torch
    _torch.backends.cudnn.enabled = False

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.backend import Config, DiTBackend, PEBackend
from demo.backend.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SPRequest(BaseModel):
    user_prompt: str = Field(..., description="Natural-language description of the image to generate")
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)


class SPResponse(BaseModel):
    sp: Any


class ImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt or compact-single-quote SP JSON")
    height: int = Field(1024, ge=64, le=4096)
    width: int = Field(1024, ge=64, le=4096)
    num_steps: int = Field(25, ge=1, le=100)
    seed: int = 42
    cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    negative_prompt: str = ""


class ImageResponse(BaseModel):
    image_base64: str


class CaptionRequest(BaseModel):
    image_base64: str = Field(..., description=(
        "Base64-encoded PNG/JPEG bytes, with or without a "
        "'data:image/...;base64,' prefix."))


# ---------------------------------------------------------------------------
# App + lifespan (model loading)
# ---------------------------------------------------------------------------

state: Dict[str, Any] = {"pe": None, "dit": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load PE and DiT once at startup, unload on shutdown."""
    cfg = load_config()
    state["config"] = cfg
    log.info("Config loaded: %s", cfg)

    # Load the DiT first — it's the smaller of the two and pins the DiT
    # device up front. Loading the PE afterwards with device_map='auto'
    # lets accelerate see the already-allocated DiT memory and spread the
    # PE's expert weights across the remaining space on both GPUs.
    log.info("Loading DiT backend (%s on %s) ...", cfg.dit_ckpt, cfg.dit_device)
    state["dit"] = DiTBackend(
        ckpt_root=cfg.dit_ckpt,
        device=cfg.dit_device,
        dtype=cfg.dtype,
        base_repo=cfg.dit_base_repo,
        overlay_shard_glob=cfg.dit_overlay_glob,
        hf_token=cfg.hf_token,
    )
    log.info("Loading PE backend (%s, device_map=%s) ...", cfg.pe_ckpt, cfg.pe_device)
    state["pe"] = PEBackend(
        ckpt=cfg.pe_ckpt,
        device=cfg.pe_device,
        dtype=cfg.dtype,
        system_prompt_path=cfg.pe_system_prompt_path,
        caption_system_prompt_path=cfg.pe_caption_system_prompt_path,
        max_new_tokens=cfg.pe_max_new_tokens,
        temperature=cfg.pe_temperature,
        top_p=cfg.pe_top_p,
    )
    log.info("Both backends resident. Ready to serve.")
    yield
    # Nothing to clean up — process exit reclaims GPU memory.


app = FastAPI(
    title="Context-Scaling Structured Prompt Studio",
    description="PE (prompt-expansion) → DiT (QwenImage) in-process demo.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    cfg: Optional[Config] = state["config"]
    return {
        "status": "ok" if state["pe"] and state["dit"] else "loading",
        "pe_ckpt": cfg.pe_ckpt if cfg else None,
        "dit_ckpt": cfg.dit_ckpt if cfg else None,
    }


@app.post("/api/generate_sp", response_model=SPResponse)
async def generate_sp(req: SPRequest) -> SPResponse:
    pe: Optional[PEBackend] = state["pe"]
    if pe is None:
        raise HTTPException(status_code=503, detail="PE backend not yet loaded")
    try:
        sp = pe.expand(req.user_prompt, width=req.width, height=req.height)
        return SPResponse(sp=sp)
    except Exception as e:  # noqa: BLE001
        log.exception("PE expansion failed")
        raise HTTPException(status_code=500, detail=f"PE failed: {e}") from e


@app.post("/api/caption_image", response_model=SPResponse)
async def caption_image(req: CaptionRequest) -> SPResponse:
    """Image → Structured Prompt JSON, using the same VLM ckpt as PE plus
    the image2json.txt system prompt (1000×1000 normalized bboxes)."""
    pe: Optional[PEBackend] = state["pe"]
    if pe is None:
        raise HTTPException(status_code=503, detail="PE backend not yet loaded")
    try:
        from PIL import Image
        b64 = req.image_base64
        if "," in b64 and b64.lstrip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        sp = pe.caption_image(img)
        return SPResponse(sp=sp)
    except RuntimeError as e:
        # Caption endpoint not configured — surface as 501, not 500.
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("PE caption failed")
        raise HTTPException(status_code=500, detail=f"PE caption failed: {e}") from e


@app.post("/api/generate_image", response_model=ImageResponse)
async def generate_image(req: ImageRequest) -> ImageResponse:
    dit: Optional[DiTBackend] = state["dit"]
    if dit is None:
        raise HTTPException(status_code=503, detail="DiT backend not yet loaded")
    try:
        # Release any PyTorch caching allocator scratch back to the driver
        # before running the DiT — under a tight two-GPU layout the cuDNN
        # handle initialisation for the VAE conv3d requires a few hundred MB
        # of *externally-visible* free memory, not just PyTorch cache.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        pil = dit.render(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_steps=req.num_steps,
            seed=req.seed,
            cfg_scale=req.cfg_scale,
            negative_prompt=req.negative_prompt,
        )
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return ImageResponse(image_base64=b64)
    except Exception as e:  # noqa: BLE001
        log.exception("DiT render failed")
        raise HTTPException(status_code=500, detail=f"DiT failed: {e}") from e


# ---------------------------------------------------------------------------
# Static single-page app
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = _STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=500, detail="static/index.html is missing; run `python demo/build_index.py` first")
    return FileResponse(str(index))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    uvicorn.run(
        "demo.app:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
