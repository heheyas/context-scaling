# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""DiT backend for the demo.

Wraps the QwenImage pipeline (DiT + VAE + Qwen2.5-VL text encoder) in a
tiny render() method. All heavy lifting lives in
``training/scripts/serve.py`` — this file just re-exposes the pipeline
without booting the FastAPI server, so we can call it directly from
demo/app.py inside a single process.

The DiT checkpoint can be:

* **A local directory** (existing behaviour), with any of the QwenImage
  layouts described in :meth:`DiTBackend._resolve_layout`.

* **An HF Hub repo id** such as ``heheyas/Qwen-Image-SP``, in which case
  the class transparently:

  1. ``snapshot_download`` s the base pipeline repo (``Qwen/Qwen-Image``
     by default) — used for the transformer *config*, the VAE, the text
     encoder, and the tokenizer.
  2. ``snapshot_download`` s the overlay repo — expected to contain
     N sharded ``dit_model-*.safetensors`` files with the
     ``dit_model.`` prefix already stripped inside each shard.
  3. Merges those shards into a single ``dit_model.merged.safetensors``
     under the HF cache root, keyed by repo id + revision, so subsequent
     launches skip the merge.

Turnkey: with the two defaults, calling ``DiTBackend(dit_ckpt=None,
base_repo="Qwen/Qwen-Image", overlay_repo="heheyas/Qwen-Image-SP")``
pulls ~55 GB from HF the first time, ~0 s afterwards.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Make `training/` importable so the DiT / VAE / text-encoder classes resolve.
# ---------------------------------------------------------------------------

_TRAINING_ROOT = Path(__file__).resolve().parents[2] / "training"
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))
_TRAINING_SCRIPTS = _TRAINING_ROOT / "scripts"
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))


# ---------------------------------------------------------------------------
# HF Hub resolution — download base + overlay, merge shards if needed.
# ---------------------------------------------------------------------------

def _looks_like_hf_repo(s: str) -> bool:
    """Heuristic: HF repo ids contain '/' and don't look like local paths."""
    if not s:
        return False
    if os.path.exists(s):
        return False
    if s.startswith(("/", ".", "~", "<")):
        return False
    return "/" in s


def _hf_snapshot(repo_id: str, token: str | None = None,
                 allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download
    log.info("DiT: snapshot_download(%s) ...", repo_id)
    t0 = time.time()
    local = snapshot_download(
        repo_id=repo_id,
        token=token,
        allow_patterns=allow_patterns,
    )
    log.info("DiT: snapshot ready at %s (%.1fs)", local, time.time() - t0)
    return Path(local)


def _merge_shards_into_one(shard_paths: list[Path], out_path: Path) -> Path:
    """Load N safetensors shards → union state dict → save as one file.

    Only runs when out_path doesn't already exist (subsequent launches
    hit the cached merged file directly).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out_path.is_file():
        log.info("DiT: reusing cached merged ckpt %s (%.2f GB)",
                 out_path, out_path.stat().st_size / 1e9)
        return out_path

    log.info("DiT: merging %d shards into %s (first launch — takes a minute)",
             len(shard_paths), out_path)
    t0 = time.time()
    merged: dict = {}
    for i, p in enumerate(shard_paths, 1):
        log.info("DiT: reading shard %d/%d (%s)", i, len(shard_paths), p.name)
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                merged[k] = f.get_tensor(k)

    # QwenImagePipeline expects `dit_model.` prefix; if shards were
    # stripped (they are in heheyas/Qwen-Image-SP), add it back.
    sample_key = next(iter(merged))
    if not sample_key.startswith("dit_model."):
        log.info("DiT: shard keys have no `dit_model.` prefix; adding it")
        merged = {f"dit_model.{k}": v for k, v in merged.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out_path))
    log.info("DiT: merged %d tensors → %s (%.2f GB, %.1fs)",
             len(merged), out_path,
             out_path.stat().st_size / 1e9, time.time() - t0)
    del merged
    import gc; gc.collect()
    return out_path


def _resolve_hf(overlay_repo: str,
                base_repo: str,
                overlay_shard_glob: str,
                token: str | None) -> "tuple[Path, Path]":
    """Return (ckpt_root_dir, merged_dit_ckpt) after pulling from HF Hub.

    Merged DiT ckpt is cached under
        <HF_HOME>/context-scaling/<overlay_repo_hash>/dit_model.merged.safetensors
    so first launch does the merge, subsequent launches skip it.
    """
    base_dir = _hf_snapshot(base_repo, token=token)

    # Some QwenImage snapshots ship the diffusers subdirs under
    # ``origin/raw_data/``. Descend if needed.
    if (base_dir / "raw_data" / "transformer" / "config.json").is_file():
        ckpt_root = base_dir / "raw_data"
    elif (base_dir / "origin" / "raw_data" / "transformer" / "config.json").is_file():
        ckpt_root = base_dir / "origin" / "raw_data"
    elif (base_dir / "transformer" / "config.json").is_file():
        ckpt_root = base_dir
    else:
        raise FileNotFoundError(
            f"base_repo {base_repo!r} did not expose transformer/config.json")

    overlay_dir = _hf_snapshot(overlay_repo, token=token,
                               allow_patterns=[overlay_shard_glob])
    from glob import glob
    shards = sorted(glob(str(overlay_dir / overlay_shard_glob)))
    if not shards:
        raise FileNotFoundError(
            f"overlay_repo {overlay_repo!r} has no shards matching "
            f"{overlay_shard_glob!r}")

    cache_dir = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    merged_out = (cache_dir / "context-scaling" /
                  overlay_repo.replace("/", "__") /
                  "dit_model.merged.safetensors")
    merged = _merge_shards_into_one([Path(p) for p in shards], merged_out)
    return ckpt_root, merged


class DiTBackend:
    """Thin adapter around ``training/scripts/serve.py:QwenImagePipeline``.

    All the interesting work — DiT / VAE / text-encoder loading, the
    flow-matching schedule, the denoising loop, VAE decode — is delegated
    to the pipeline class. This class only exists to (a) hold the loaded
    instance for the FastAPI lifespan and (b) expose a small render()
    surface matched to the /api/generate_image request schema.
    """

    def __init__(
        self,
        ckpt_root: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        base_repo: str | None = None,
        overlay_shard_glob: str = "dit_model-*.safetensors",
        hf_token: str | None = None,
    ):
        # Import lazily so importing this module doesn't force torch init
        # on machines without CUDA (e.g. HF Space build step).
        from serve import QwenImagePipeline  # type: ignore  # sys.path-injected

        token = hf_token or os.environ.get("HF_TOKEN")

        if _looks_like_hf_repo(ckpt_root):
            base_repo = base_repo or "Qwen/Qwen-Image"
            log.info("DiT: HF Hub mode — overlay=%s base=%s",
                     ckpt_root, base_repo)
            resolved_root, merged_ckpt = _resolve_hf(
                overlay_repo=ckpt_root,
                base_repo=base_repo,
                overlay_shard_glob=overlay_shard_glob,
                token=token,
            )
        else:
            resolved_root, merged_ckpt = self._resolve_layout(ckpt_root)

        log.info(
            "DiT: loading pipeline; ckpt_root=%s  merged_ckpt=%s  device=%s  dtype=%s",
            resolved_root, merged_ckpt, device, dtype,
        )
        self.pipeline = QwenImagePipeline(
            merged_ckpt=str(merged_ckpt),
            ckpt_root=str(resolved_root),
            device=device,
        )

    # ------------------------------------------------------------------
    # Local-path resolution (unchanged, kept as fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_layout(ckpt_root: str) -> "tuple[Path, Path]":
        """Return (ckpt_root_dir, merged_safetensors_path).

        Accepts either of the two common QwenImage layouts:

        1. A directory that directly contains ``transformer/config.json`` +
           ``vae/`` + ``text_encoder/`` + ``tokenizer/`` and a merged
           ``*.safetensors`` next to those subdirs (or one directory up).

        2. The nested release layout
               ``<root>/{qwenimage_*_merged.safetensors, raw_data/{transformer/,vae/,...}}``
           where the raw diffusers-style subdirs live under ``raw_data/``.

        3. A double-nested variant ``<root>/origin/...`` produced by some
           ByteDance mirroring — descends once more.
        """
        root = Path(ckpt_root)

        # If the user pointed at a parent of `origin/`, descend once.
        if (root / "origin").is_dir() and not (root / "transformer" / "config.json").is_file():
            root = root / "origin"

        # If the diffusers subdirs are one level deeper under raw_data/, use that.
        if (root / "raw_data" / "transformer" / "config.json").is_file():
            resolved_root = root / "raw_data"
            merged_search_dir = root
        elif (root / "transformer" / "config.json").is_file():
            resolved_root = root
            merged_search_dir = root
        else:
            raise FileNotFoundError(
                f"Could not find transformer/config.json under {ckpt_root!r} "
                f"(tried {root} and {root}/raw_data)."
            )

        # Merged single-file checkpoint: prefer a name that looks canonical,
        # then fall back to any *.safetensors at the same level.
        candidates = sorted(merged_search_dir.glob("*.safetensors"))
        if not candidates:
            # As a last resort, take the first transformer/ shard — the
            # pipeline will complain if it isn't a full-model file.
            candidates = sorted((resolved_root / "transformer").glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(
                f"No .safetensors merged checkpoint found near {ckpt_root!r}."
            )
        # Prefer files with 'merged' in the name, otherwise take the first.
        merged = next((p for p in candidates if "merged" in p.name), candidates[0])
        return resolved_root, merged

    def render(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 25,
        seed: int = 42,
        cfg_scale: float = 4.0,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate an image for a text prompt (natural language OR
        compact-single-quote-JSON Structured Prompt)."""
        return self.pipeline.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_steps=num_steps,
            seed=seed,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
        )
