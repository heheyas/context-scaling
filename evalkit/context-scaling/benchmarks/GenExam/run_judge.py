"""GenExam judge: evaluate generated images against ground-truth with VLM.

Wraps the original GenExam eval pipeline with KeyPool-based API key rotation
(same as rewrite pipeline) and multi-threaded execution.

Usage:
  # Using internal GPT proxy (api_config.json with KeyPool):
  python -m benchmarks.GenExam.run_judge \
      --img_dir outputs/genexam_gemini/0005200_1024x1024_cfg4_0_ema \
      --eval_dir eval_results/genexam_gemini \
      --api_config /path/to/api_config.json \
      --model gpt-4o-2024-11-20 --workers 16

  # Mini subset (251 samples):
  python -m benchmarks.GenExam.run_judge \
      --img_dir outputs/genexam_gemini/0005200_1024x1024_cfg4_0_ema \
      --eval_dir eval_results/genexam_gemini \
      --api_config /path/to/api_config.json \
      --model gpt-4o-2024-11-20 --workers 16 --mini

  # Calculate scores only (no API calls):
  python -m benchmarks.GenExam.run_judge \
      --eval_dir eval_results/genexam_gemini --score_only
"""

import argparse
import ast
import base64
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image
from tqdm import tqdm

# Add project root to path for imports
GENEXAM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GENEXAM_DIR)
from eval_prompt import prompt_for_eval
from cal_score import calculate_score

# Reuse KeyPool from rewrite pipeline
PROJECT_ROOT = os.path.abspath(os.path.join(GENEXAM_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from rewrite_prompts.base import KeyPool, load_api_config

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-2024-11-20"
DEFAULT_API_VERSION = "2024-02-01"
MAX_RETRIES = 5


def encode_image(image_path, target_size=1024, fmt="JPEG"):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if target_size and target_size > 0:
        w, h = img.size
        if max(w, h) > target_size:
            if w >= h:
                new_w, new_h = target_size, int(h * target_size / w)
            else:
                new_h, new_w = target_size, int(w * target_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm(gen_img_path, gt_img_path, text_prompt, key_pool, model,
             max_tokens=16384, img_size=768):
    """Call VLM via KeyPool (AzureOpenAI), same pattern as rewrite GPT backend."""
    import openai

    b64_gen = encode_image(gen_img_path, target_size=img_size)
    b64_gt = encode_image(gt_img_path, target_size=img_size)

    content = [
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_gen}", "detail": "high"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_gt}", "detail": "high"}},
    ]

    messages = [{"role": "user", "content": content}]

    for attempt in range(MAX_RETRIES):
        key, client = key_pool.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                extra_headers={"X-TT-LOGID": "genexam_judge"},
            )
            return resp.choices[0].message.content

        except (openai.RateLimitError, openai.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if isinstance(e, openai.RateLimitError) or status == 429:
                key_pool.report_429(key)
                wait = min(1.01 ** attempt, 60)
                log.warning("Rate limited key=...%s (attempt %d), switching...",
                            key[-8:], attempt + 1)
                time.sleep(wait)
            elif status and status >= 500:
                log.warning("Server error %d (attempt %d)", status, attempt + 1)
                time.sleep(1.01 ** attempt)
            else:
                raise
        except Exception as e:
            log.warning("API call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.01 ** attempt)

    raise RuntimeError(f"VLM call failed after {MAX_RETRIES} retries")


def eval_single(data, img_dir, eval_dir, data_dir, key_pool, model, img_size):
    sample_id = data["id"]
    json_path = os.path.join(eval_dir, f"{sample_id}.json")

    if os.path.exists(json_path):
        return "skip", sample_id

    gen_img_path = os.path.join(img_dir, f"{sample_id}.png")
    if not os.path.exists(gen_img_path):
        return "missing", sample_id

    gt_img_path = os.path.join(data_dir, "images", data["image_path"])
    if not os.path.exists(gt_img_path):
        return "no_gt", sample_id

    scoring_questions = [pt["question"] for pt in data["scoring_points"]]
    eval_prompt_text = prompt_for_eval.format(prompt=data["prompt"], scoring_points=scoring_questions)

    for attempt in range(3):
        try:
            response_text = call_vlm(
                gen_img_path, gt_img_path, eval_prompt_text,
                key_pool, model, img_size=img_size,
            )
        except Exception as e:
            log.warning("Eval failed for %s (attempt %d): %s", sample_id, attempt + 1, e)
            continue

        # Parse JSON from response
        try:
            result = json.loads(response_text.split("```json")[-1].split("```")[0])
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(response_text.split("```json")[-1].split("```")[0])
            except Exception:
                log.warning("JSON parse failed for %s (attempt %d)", sample_id, attempt + 1)
                continue

        # Validate
        try:
            assert "global_evaluation" in result and "answers" in result
            assert len(result["answers"]) == len(data["scoring_points"])
            assert all(a["answer"] in [0, 1] for a in result["answers"])
            assert all(k in result["global_evaluation"] for k in
                       ["Spelling", "Clarity and Readability", "Logical Consistency"])
        except Exception:
            log.warning("Validation failed for %s (attempt %d)", sample_id, attempt + 1)
            continue

        # Save
        result.update(data)
        result["gen_img_path"] = gen_img_path
        result["gt_img_path"] = gt_img_path
        if "image_path" in result:
            del result["image_path"]

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return "done", sample_id

    return "fail", sample_id


def main():
    parser = argparse.ArgumentParser(description="GenExam judge")
    parser.add_argument("--img_dir", type=str, help="Directory with generated {id}.png images")
    parser.add_argument("--eval_dir", type=str, required=True, help="Output directory for eval JSONs")
    parser.add_argument("--data_dir", type=str, default=os.path.join(GENEXAM_DIR, "data"),
                        help="GenExam data directory (default: benchmarks/GenExam/data)")
    parser.add_argument("--api_config", type=str, required=False,
                        help="Path to API key config (JSON or text, same format as rewrite pipeline)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_version", type=str, default=DEFAULT_API_VERSION)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mini", action="store_true", help="Evaluate mini subset (251 samples)")
    parser.add_argument("--score_only", action="store_true", help="Only calculate scores, no eval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Score only mode
    if args.score_only:
        mini_path = os.path.join(args.data_dir, "mini_sample_ids.txt") if args.mini else None
        calculate_score(args.eval_dir, sampled_id_path=mini_path)
        return

    if not args.img_dir:
        parser.error("--img_dir is required unless --score_only")
    if not args.api_config:
        parser.error("--api_config is required unless --score_only")

    os.makedirs(args.eval_dir, exist_ok=True)

    # Build KeyPool (same as rewrite pipeline)
    configs = load_api_config(args.api_config)
    key_pool = KeyPool(configs, api_version=args.api_version)
    log.info("KeyPool: %d key(s)", key_pool.size)

    # Load data
    data_path = os.path.join(args.data_dir, "annotations", "All_Subjects.jsonl")
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # Filter mini subset
    if args.mini:
        mini_path = os.path.join(args.data_dir, "mini_sample_ids.txt")
        with open(mini_path) as f:
            mini_ids = {x.strip() for x in f}
        all_data = [d for d in all_data if d["id"] in mini_ids]

    log.info("Evaluating %d samples (workers=%d, model=%s)", len(all_data), args.workers, args.model)

    stats = {"done": 0, "skip": 0, "fail": 0, "missing": 0, "no_gt": 0}
    lock = Lock()

    def _task(data):
        return eval_single(data, args.img_dir, args.eval_dir, args.data_dir,
                           key_pool, args.model, args.img_size)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_task, d): d for d in all_data}
        with tqdm(total=len(futures), desc="Judging") as pbar:
            for future in as_completed(futures):
                status, sid = future.result()
                with lock:
                    stats[status] += 1
                pbar.update(1)
                pbar.set_postfix(**stats)

    log.info("Results: %s", stats)
    log.info("Key pool stats: %s", key_pool.stats())

    # Auto calculate scores
    print("\n" + "=" * 60)
    print("Score Summary")
    print("=" * 60)
    mini_path = os.path.join(args.data_dir, "mini_sample_ids.txt") if args.mini else None
    calculate_score(args.eval_dir, sampled_id_path=mini_path)


if __name__ == "__main__":
    main()
