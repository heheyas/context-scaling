# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Grounded Perplexity Gain (GPG) for the detailness pool.

For each (uid, kind, level) sample we compute:

    GPG = mean_nll(caption | text-only prompt) - mean_nll(caption | image + text prompt)

Per-token NLL is computed over the assistant-content span only (not the
chat scaffolding), under Qwen2.5-VL-7B-Instruct.

GPG > 0 means the image lowers the caption's loss → caption is grounded
in the image. This is the M3ID / mutual-information per-token PMI
(Favero et al. CVPR 2024, arXiv 2403.14003), aggregated to caption level.

Usage:
    HOME=<RAY_SERVE_HOME> PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \\
    python -m detailness.gpg.score \\
        --pool /tmp/detailness_real_n100/pool.jsonl \\
        --pool-dir /tmp/detailness_real_n100 \\
        --model <HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct \\
        --out /tmp/detailness_real_n100/gpg_v6_dropmore/shard0.jsonl \\
        --content-only --canonicalize-json \\
        --extra-drop-keys "atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect" \\
        --max-pixels $((1024*28*28))
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
# Qwen-native bbox conversion: <bbox>x y x y</bbox> (0-1000 normalized) →
# [x,y,x,y] in resized-pixel space (post-smart_resize).
# Qwen2.5-VL was trained with absolute pixel coords; using this format
# lets the vision features actually ground the bbox numeric tokens.
# --------------------------------------------------------------------------- #

_BBOX_QUOTED_RE = re.compile(r'"<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>"')
_BBOX_INLINE_RE = re.compile(r'<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>')


# Fields whose values should be DROPPED from the JSON caption before tokenizing.
# These are non-visual metadata (integer IDs, depth integers) that have ~0 PMI
# under any VLM judge but can shift PMI of surrounding tokens via autoregressive
# context. Drop them entirely to get a clean canonical caption.
CANONICAL_DROP_KEYS = {"depth", "id"}


def canonicalize_json_caption(caption: str, drop_keys=None) -> str:
    """If caption is valid JSON, remove specified metadata keys recursively.
    Returns compact JSON string. Non-JSON captions returned as-is.
    """
    if drop_keys is None:
        drop_keys = CANONICAL_DROP_KEYS
    try:
        obj = json.loads(caption)
    except Exception:
        return caption

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items() if k not in drop_keys}
        elif isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    cleaned = _walk(obj)
    return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False)


def convert_bboxes_to_qwen_native(caption_text: str, img_w: int, img_h: int,
                                  min_pixels: int, max_pixels: int) -> str:
    """Replace <bbox>x y x y</bbox> tokens (0-1000 normalized) with numeric
    arrays [x,y,x,y] in the smart-resized pixel space.

    Two patterns handled:
      - "<bbox>...</bbox>"  (a JSON-string value of a position field)
        →  [x,y,x,y]   (becomes a JSON array)
      - <bbox>...</bbox>  embedded in a string value (e.g. relationships)
        →  [x,y,x,y]   (still inside the string, but no scaffold)
    """
    # Defer the smart_resize import so the module loads even if the qwen2_vl
    # subpackage isn't present.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    rh, rw = smart_resize(img_h, img_w, factor=28,
                          min_pixels=min_pixels, max_pixels=max_pixels)

    def _conv(m):
        x1, y1, x2, y2 = map(int, m.groups())
        rx1 = round(x1 / 1000 * rw); ry1 = round(y1 / 1000 * rh)
        rx2 = round(x2 / 1000 * rw); ry2 = round(y2 / 1000 * rh)
        return f"[{rx1},{ry1},{rx2},{ry2}]"

    out = _BBOX_QUOTED_RE.sub(_conv, caption_text)
    out = _BBOX_INLINE_RE.sub(_conv, out)
    return out

# --------------------------------------------------------------------------- #
# Model load
# --------------------------------------------------------------------------- #

def load_model(model_path: str, device: str = "cuda:0",
               max_memory_per_gpu: Optional[str] = None):
    """Auto-detect model_type from config and load the appropriate VL class.

    If `device` is "auto", uses device_map='auto' across all visible GPUs
    (for >24GB models). Optional max_memory_per_gpu like '80GiB'.
    """
    import json as _json
    from transformers import AutoProcessor
    cfg = _json.load(open(os.path.join(model_path, "config.json")))
    model_type = cfg.get("model_type", "")
    print(f"[load] model_type={model_type} from {model_path}", flush=True)
    print(f"[load] processor", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"[load] model bf16 → {device}", flush=True)
    if model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_5_moe",
                      "internvl", "internvl_chat"):
        # Qwen2/2.5-VL, Qwen3.5-MoE-VL, and InternVL3 all hit cuDNN
        # CUDNN_STATUS_NOT_INITIALIZED on conv3d/conv2d in their vision
        # encoder under this PyTorch+cuDNN combo. Disable cuDNN; native
        # CUDA kernels work and the vision-side slowdown is negligible.
        torch.backends.cudnn.enabled = False
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as Cls
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    elif model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif model_type == "qwen3_5_moe":
        from transformers import Qwen3_5MoeForConditionalGeneration as Cls
    elif model_type in ("internvl", "internvl_chat"):
        from transformers import InternVLForConditionalGeneration as Cls
    else:
        # Fallback to the generic auto-class (transformers ≥4.50)
        try:
            from transformers import AutoModelForImageTextToText as Cls
            print(f"[load] using AutoModelForImageTextToText fallback for "
                  f"model_type={model_type}", flush=True)
        except Exception:
            raise ValueError(f"Unsupported model_type: {model_type}")
    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if device == "auto":
        kw["device_map"] = "auto"
        if max_memory_per_gpu:
            n_gpu = torch.cuda.device_count()
            kw["max_memory"] = {i: max_memory_per_gpu for i in range(n_gpu)}
    else:
        kw["device_map"] = device
    model = Cls.from_pretrained(model_path, **kw)
    model.eval()
    print(f"[ok] model loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)",
          flush=True)
    return model, processor


# --------------------------------------------------------------------------- #
# Per-sample NLL
# --------------------------------------------------------------------------- #

USER_TURN = "Describe the image in detail."

def _build_messages(caption: str, image: Optional[Image.Image],
                    system_prompt: Optional[str] = None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system",
                     "content": [{"type": "text", "text": system_prompt}]})
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": USER_TURN})
    msgs.append({"role": "user", "content": user_content})
    msgs.append({"role": "assistant",
                 "content": [{"type": "text", "text": caption}]})
    return msgs


def _locate_caption_span(input_ids: list[int], cap_ids: list[int]) -> tuple[int, int]:
    """Find caption token subseq in input_ids. Returns (start, end) inclusive-exclusive.

    Tries exact match first, then a fallback search using the last K caption
    tokens (template wrapping sometimes shifts the leading whitespace token).
    """
    L = len(cap_ids)
    if L == 0:
        raise ValueError("empty caption tokens")
    # exact
    for i in range(len(input_ids) - L, -1, -1):
        if input_ids[i:i+L] == cap_ids:
            return i, i + L
    # fallback: match last 16 tokens
    K = min(16, L)
    tail = cap_ids[-K:]
    for i in range(len(input_ids) - K, -1, -1):
        if input_ids[i:i+K] == tail:
            end = i + K
            return end - L, end
    raise RuntimeError(
        f"Caption span not found (cap_len={L}, prompt_len={len(input_ids)})"
    )


@torch.no_grad()
def score(model, processor, caption: str, image: Optional[Image.Image],
          system_prompt: Optional[str] = None,
          content_mask: Optional[list] = None):
    """Return (sum_nll, n_tokens, mean_nll) over the caption span.

    image=None ⇒ text-only prior pass. The caption is scored under the
    same system_prompt in both passes, so it cancels in the GPG
    difference and only changes how easy the LM finds the format.

    content_mask: optional bool list of length len(cap_ids). If provided,
    only sum NLL over tokens marked True. The returned n_tokens is the
    number of content tokens (not the full caption length).
    """
    messages = _build_messages(caption, image, system_prompt=system_prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    images_arg = [image] if image is not None else None
    inputs = processor(
        text=[text], images=images_arg,
        return_tensors="pt", padding=True,
    ).to(getattr(model, 'device', 'cuda:0'))

    # Caption span
    cap_ids = getattr(processor, "tokenizer", processor)(caption, add_special_tokens=False).input_ids
    seq = inputs["input_ids"][0].tolist()
    start, end = _locate_caption_span(seq, cap_ids)

    out = model(**inputs)
    logits = out.logits  # (1, T, V)
    shift_logits = logits[0, :-1, :]
    shift_labels = inputs["input_ids"][0, 1:]
    sl, el = max(start - 1, 0), end - 1
    log_probs = F.log_softmax(shift_logits[sl:el].float(), dim=-1)
    target = shift_labels[sl:el]
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if content_mask is not None:
        # Mask matches cap_ids order; nll[i] corresponds to predicting cap_ids[i]
        # given the prefix. So mask[i] applies directly to nll[i].
        if len(content_mask) != nll.numel():
            # Mismatch — fall back to all-content
            pass
        else:
            keep = torch.tensor(content_mask, dtype=torch.bool, device=nll.device)
            nll = nll[keep]
    n = nll.numel()
    s = float(nll.sum().item())
    return s, n, s / max(n, 1)


# --------------------------------------------------------------------------- #
# Pool driver
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool.jsonl with uid/kind/level/caption/image_path")
    ap.add_argument("--pool-dir", required=True, help="directory containing image_path entries (relative)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="per-sample GPG jsonl (append-only, resumable)")
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:N for single-GPU, or 'auto' for multi-GPU (device_map=auto)")
    ap.add_argument("--max-memory-per-gpu", default=None,
                    help="e.g. '80GiB' (used with --device auto)")
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28,
                    help="cap for processor min/max pixels (Qwen2.5-VL default ~12.5M)")
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only score the first N samples (debug)")
    ap.add_argument("--shard", type=int, default=0, help="this worker index in [0, num_shards)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--system-prompt-file", default=None,
                    help="path to schema doc; applied as system message ONLY for the kinds in --system-prompt-kinds")
    ap.add_argument("--system-prompt-kinds", default="structured",
                    help="comma-separated kinds that get the system prompt (default: structured)")
    ap.add_argument("--qwen-native-bbox", action="store_true",
                    help="convert <bbox>x y x y</bbox> (0-1000 norm) into Qwen2.5-VL native "
                         "format [x,y,x,y] in resized-pixel space. Applied ONLY for the kinds "
                         "in --qwen-native-bbox-kinds (default: structured).")
    ap.add_argument("--qwen-native-bbox-kinds", default="structured")
    ap.add_argument("--content-only", action="store_true",
                    help="for JSON captions, only sum PMI over VALUE tokens "
                         "(skip keys/brackets/commas/quotes). Removes cross-field "
                         "context-shift noise from ablation comparisons. Non-JSON "
                         "captions are unaffected (all tokens treated as content).")
    ap.add_argument("--canonicalize-json", action="store_true",
                    help="Remove non-visual metadata fields (depth, id) from JSON "
                         "captions before tokenizing. Ensures abl_no_depth ≡ "
                         "abl_full ≡ struct_l10 (canonical equivalent).")
    ap.add_argument("--extra-drop-keys", default=None,
                    help="comma-separated extra JSON keys to drop during canonicalize (used with --canonicalize-json)")
    ap.add_argument("--kinds-filter", default=None,
                    help="comma-separated kinds to include; others are skipped")
    args = ap.parse_args()

    system_prompt = None
    sp_kinds = set()
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()
        sp_kinds = set(args.system_prompt_kinds.split(","))
        print(f"[cfg] system prompt loaded ({len(system_prompt)} chars), "
              f"applied to kinds: {sorted(sp_kinds)}", flush=True)

    qb_kinds = set()
    if args.qwen_native_bbox:
        qb_kinds = set(args.qwen_native_bbox_kinds.split(","))
        print(f"[cfg] qwen-native-bbox conversion ENABLED for kinds: "
              f"{sorted(qb_kinds)}", flush=True)

    # Load existing results to support resume
    done: set[tuple[str,str,str]] = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["uid"], r["kind"], r["level"]))
                except Exception:
                    pass
        print(f"[resume] {len(done)} samples already scored, skipping", flush=True)

    rows = []
    with open(args.pool) as f:
        for line in f:
            rows.append(json.loads(line))
    # Optional kind filter
    if args.kinds_filter:
        keep_kinds = set(k.strip() for k in args.kinds_filter.split(",") if k.strip())
        rows = [r for r in rows if r["kind"] in keep_kinds]
        print(f"[filter] keeping {len(keep_kinds)} kinds, {len(rows)} rows", flush=True)
    # Deterministic shard split by row index
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    todo = [r for r in rows if (r["uid"], r["kind"], r["level"]) not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[pool] shard={args.shard}/{args.num_shards}  rows={len(rows)}  todo={len(todo)}",
          flush=True)

    model, processor = load_model(args.model, device=args.device,
                                   max_memory_per_gpu=args.max_memory_per_gpu)

    # Set processor pixel caps so 2k×2k JPEGs don't blow up image tokens
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels
        print(f"[cfg] image processor pixels: [{args.min_pixels}, {args.max_pixels}]",
              flush=True)

    out_f = open(args.out, "a", buffering=1)  # line-buffered
    t0 = time.time()
    n_done = 0
    n_fail = 0
    for r in todo:
        uid, kind, level, caption = r["uid"], r["kind"], r["level"], r["caption"]
        img_path = os.path.join(args.pool_dir, r["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {uid} image read fail: {e}", flush=True)
            n_fail += 1
            continue
        sp_arg = system_prompt if (kind in sp_kinds) else None
        cap_for_score = caption
        if args.canonicalize_json:
            drop_keys = set(CANONICAL_DROP_KEYS)
            if args.extra_drop_keys:
                drop_keys |= set(k.strip() for k in args.extra_drop_keys.split(",") if k.strip())
            cap_for_score = canonicalize_json_caption(cap_for_score, drop_keys=drop_keys)
        if kind in qb_kinds:
            try:
                cap_for_score = convert_bboxes_to_qwen_native(
                    cap_for_score, img.size[0], img.size[1],
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            except Exception as e:
                print(f"[bbox-conv-skip] {uid} {kind} {level}: {e}", flush=True)
        cmask = None
        if args.content_only:
            from .content_mask import token_content_mask
            cmask = token_content_mask(cap_for_score, getattr(processor, "tokenizer", processor))
        try:
            sg, ng, mg = score(model, processor, cap_for_score, img,
                               system_prompt=sp_arg, content_mask=cmask)
            sp, np_, mp = score(model, processor, cap_for_score, None,
                                system_prompt=sp_arg, content_mask=cmask)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] {uid} {kind} {level}  (cap_len={len(caption)})", flush=True)
            n_fail += 1
            continue
        except Exception as e:
            print(f"[err] {uid} {kind} {level}: {e}", flush=True)
            n_fail += 1
            continue

        # Sanity: caption tokens should match between the two passes
        rec = {
            "uid": uid, "kind": kind, "level": level,
            "n_tokens": ng,
            "n_tokens_prior": np_,
            "nll_grounded_sum": sg, "nll_grounded_mean": mg,
            "nll_prior_sum": sp,    "nll_prior_mean": mp,
            "gpg": mp - mg,
        }
        out_f.write(json.dumps(rec) + "\n")
        n_done += 1

        if n_done % 10 == 0 or n_done == len(todo):
            dt = time.time() - t0
            rate = n_done / max(dt, 1e-3)
            eta = (len(todo) - n_done) / max(rate, 1e-3)
            print(f"[{n_done}/{len(todo)}] {kind:10s}{level} "
                  f"GPG={rec['gpg']:+.4f} ng={ng}  "
                  f"({rate:.2f}/s, eta {eta/60:.1f} min)", flush=True)

    out_f.close()
    print(f"[done] scored {n_done}, failed {n_fail}, total time {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
