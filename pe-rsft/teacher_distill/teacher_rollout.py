# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Teacher rollout — vision LLM produces `<analysis>...</analysis>{SP JSON}`.

Reads a manifest of (image, short_prompt) pairs and calls an OpenAI-compatible
vision LLM under a Structured-Prompt system prompt. Output is one line per
pair, resume-safe (append-only; already-completed uids are skipped on
restart).

Usage:
    python -m pe-rsft.teacher_distill.teacher_rollout \\
        --dataset-dir     /path/to/dataset \\
        --manifest        pairs_manifest.jsonl \\
        --system-prompt   pe-rsft/teacher_distill/system_prompts/teacher.txt \\
        --out             /path/to/dataset/rollout_v6_structured_full.jsonl \\
        --model           <your-teacher-model> \\
        --workers         500

Requires OPENAI_API_KEY (or --api-key) and either OPENAI_BASE_URL
(or --base-url) if you're not hitting api.openai.com.
"""
import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from openai import OpenAI


def teacher_rollout(client: OpenAI, model: str, sp_text: str,
                    prompt: str, image_b64: str, timeout: float = 1800.0):
    """Single teacher call. Returns (content, n_completion_tokens)."""
    messages = [
        {"role": "system", "content": sp_text},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]},
    ]
    kwargs = dict(model=model, messages=messages, max_tokens=65536,
                  temperature=1.0, top_p=0.9, timeout=timeout)
    # `reasoning_effort` is vendor-specific; ignore if not accepted.
    try:
        r = client.chat.completions.create(**kwargs, reasoning_effort="high")
    except TypeError:
        r = client.chat.completions.create(**kwargs)
    m = r.choices[0].message
    content = m.content or ""
    reasoning = getattr(m, "reasoning_content", None) or ""
    if reasoning and "<analysis>" not in content and "</think>" not in content:
        content = f"<think>\n{reasoning}\n</think>\n\n{content}"
    ntok = r.usage.completion_tokens if r.usage else 0
    return content, ntok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True,
                   help="Dir containing images/ and the manifest jsonl")
    p.add_argument("--manifest", default="pairs_manifest.jsonl",
                   help="Manifest filename (relative to --dataset-dir)")
    p.add_argument("--system-prompt", required=True,
                   help="Path to the teacher SP system prompt (v6 synthesis)")
    p.add_argument("--out", required=True,
                   help="Output JSONL (append-only, resume-safe)")
    p.add_argument("--model", required=True,
                   help="Teacher model id / deployment name")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--workers", type=int, default=500,
                   help="Concurrent in-flight teacher calls "
                        "(warm up smaller; some vendors cap connections)")
    p.add_argument("--uid-prefix", default="row",
                   help="Prefix for stable per-row uids")
    args = p.parse_args()

    if not args.api_key:
        sys.exit("ERROR: pass --api-key or set OPENAI_API_KEY")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    sys_text = open(args.system_prompt).read().rstrip()

    manifest_path = os.path.join(args.dataset_dir, args.manifest)
    rows = [json.loads(l) for l in open(manifest_path)]
    for i, r in enumerate(rows):
        r["uid"] = f"{args.uid_prefix}_{i:05d}"

    # Resume: skip uids whose rollout_text is already non-error
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                row = json.loads(l)
                rt = row.get("rollout_text", "")
                if rt and not rt.startswith("__ERROR__"):
                    done.add(row["uid"])
            except Exception:
                pass
    todo = [r for r in rows if r["uid"] not in done]
    print(f"[teacher] manifest={len(rows)} done={len(done)} todo={len(todo)}",
          flush=True)

    def work(r):
        image_path = os.path.join(args.dataset_dir, r["image"])
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        w, h = r.get("width"), r.get("height")
        if not (w and h):
            with Image.open(image_path) as im:
                w, h = im.size
        try:
            content, ntok = teacher_rollout(
                client, args.model, sys_text, r["prompt"], b64)
        except Exception as e:  # noqa: BLE001
            return r, f"__ERROR__: {type(e).__name__}: {e}", 0, w, h
        return r, content, ntok, w, h

    lock = threading.Lock()
    with open(args.out, "a", encoding="utf-8") as fo, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r, content, ntok, w, h = fut.result()
            with lock:
                fo.write(json.dumps({
                    "uid": r["uid"], "prompt": r["prompt"],
                    "source": r.get("source"), "image": r["image"],
                    "orig_width": w, "orig_height": h,
                    "rollout_text": content,
                    "n_completion_tokens": ntok,
                }, ensure_ascii=False) + "\n")
                fo.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done", flush=True)

    print(f"DONE: wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
