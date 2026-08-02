from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

from scripts.text.text_utils import preprocess_string, clean_and_remove_hallucinations, levenshtein_distance, calculate_char_match_ratio

import torch
import torch.multiprocessing as mp

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

OCR_BATCH_SIZE = int(os.environ.get("ONEIG_OCR_BATCH_SIZE", "128"))


def worker_fn(gpu_id, tasks, model_path, batch_size, result_dict):
    """Worker: load Qwen on one GPU, batch OCR across all items."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from scripts.utils.inference import Qwen2_5VLBatchInferencer
    inferencer = Qwen2_5VLBatchInferencer(model_path, device="cuda")

    cache_dir = f"tmp_{formatted_time}_gpu{gpu_id}"
    os.makedirs(cache_dir, exist_ok=True)

    # Phase 1: Pre-split grids and collect all OCR queries
    item_data = {}  # task_id -> {text_gt, num_images, max_new_tokens}
    all_queries = []  # (query_idx, task_id, image_path, max_new_tokens)

    for task_id, task_info in tasks:
        text_gt = task_info["text_gt"]
        word_count = len(text_gt.split())
        max_new_tokens = 256 if word_count > 60 else 128

        if len(task_info["img_path"]) != 1:
            item_data[task_id] = None
            continue

        item_cache = os.path.join(cache_dir, f"item_{task_id}")
        os.makedirs(item_cache, exist_ok=True)
        split_imgs = split_mxn_grid(task_info["img_path"][0], task_info["img_grid"], item_cache)
        if len(split_imgs) == 0:
            item_data[task_id] = None
            continue

        item_data[task_id] = {
            "text_gt": text_gt,
            "num_images": len(split_imgs),
            "max_new_tokens": max_new_tokens,
        }

        for img_path in split_imgs:
            all_queries.append((len(all_queries), task_id, img_path, max_new_tokens))

    print(f"[GPU {gpu_id}] {len(tasks)} items, {len(all_queries)} OCR queries", flush=True)

    # Phase 2: Batch OCR inference
    # Group by max_new_tokens to avoid padding waste
    all_answers = [None] * len(all_queries)

    for max_tokens in [128, 256]:
        token_queries = [(i, q) for i, q in enumerate(all_queries) if q[3] == max_tokens]
        if not token_queries:
            continue

        for batch_start in tqdm(range(0, len(token_queries), batch_size),
                                desc=f"GPU {gpu_id} tok={max_tokens}", position=gpu_id):
            batch = token_queries[batch_start:batch_start + batch_size]

            messages = []
            for _, (query_idx, _, img_path, _) in batch:
                messages.append([{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": inferencer.TEXT_PROMPT}
                    ],
                }])

            answers = inferencer.batch_inference(messages, max_new_tokens=max_tokens)

            for i, (_, (query_idx, _, _, _)) in enumerate(batch):
                all_answers[query_idx] = answers[i]

    # Phase 3: Reconstruct per-item results
    results = {}
    # Group answers by task_id
    task_answers = {}
    for query_idx, task_id, _, _ in all_queries:
        if task_id not in task_answers:
            task_answers[task_id] = []
        task_answers[task_id].append(all_answers[query_idx])

    for task_id, data in item_data.items():
        if data is None:
            results[task_id] = None
            continue

        ocr_results = task_answers.get(task_id, [])
        text_ocr_list = clean_and_remove_hallucinations(ocr_results)
        text_gt = data["text_gt"]
        text_gt_preprocessed = preprocess_string(text_gt)

        edit_distances, completion_ratios, match_word_counts, gt_word_counts = [], [], [], []
        ED_score, CR_score, WAC_score = [], [], []

        for text_ocr in text_ocr_list:
            text_ocr_preprocessed = preprocess_string(text_ocr)
            edit_distance = levenshtein_distance(text_ocr_preprocessed, text_gt_preprocessed)
            completion_ratio = 1 if edit_distance == 0 else 0
            match_word_count, text_word_accuracy, gt_word_count = calculate_char_match_ratio(
                text_gt_preprocessed, text_ocr_preprocessed)

            edit_distances.append(edit_distance)
            completion_ratios.append(completion_ratio)
            match_word_counts.append(match_word_count)
            gt_word_counts.append(gt_word_count)
            ED_score.append(edit_distance)
            CR_score.append(completion_ratio)
            WAC_score.append(text_word_accuracy)

        if ED_score:
            ed_val = sum(ED_score) / len(ED_score)
            prompt_score = [
                ed_val.item() if hasattr(ed_val, 'item') else ed_val,
                sum(CR_score) / len(CR_score),
                sum(WAC_score) / len(WAC_score),
            ]
        else:
            prompt_score = None

        results[task_id] = {
            "prompt_score": prompt_score,
            "edit_distances": edit_distances,
            "completion_ratios": completion_ratios,
            "match_word_counts": match_word_counts,
            "gt_word_counts": gt_word_counts,
        }

    result_dict[gpu_id] = results

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

    print(f"[GPU {gpu_id}] Done. {len(tasks)} items.", flush=True)


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    model_path = os.environ.get("ONEIG_QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Using {num_gpus} GPU(s), model: {model_path}, OCR batch size: {OCR_BATCH_SIZE}")

    if args.mode == "EN":
        text_csv_path = "scripts/text/text_content.csv"
        MAX_EDIT_DISTANCE = 100
    else:
        text_csv_path = "scripts/text/text_content_zh.csv"
        MAX_EDIT_DISTANCE = 50
    text_df = pd.read_csv(text_csv_path, dtype=str)

    results_dir = args.results_dir
    text_score_csv = f"{results_dir}/text_score_{args.mode}_{formatted_time}.csv"
    text_prompt_score_csv = f"{results_dir}/text_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(results_dir, exist_ok=True)

    score_csv = pd.DataFrame(index=args.model_names, columns=["ED", "CR", "WAC", "text score"])
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)

    for model_id, model_name in enumerate(args.model_names):
        print(f"It is {model_name} time.")
        img_grid = (int(args.image_grid[model_id].split(',')[0]),
                    int(args.image_grid[model_id].split(',')[-1]))

        all_tasks = []
        for idx, (item_id, text_gt) in enumerate(zip(text_df["id"], text_df["text_content"])):
            img_path = megfile.smart_glob(args.image_dirname + '/' + model_name + '/' + item_id + '*')
            all_tasks.append((item_id, {
                "img_path": img_path,
                "img_grid": img_grid,
                "text_gt": text_gt,
            }))

        print(f"Total tasks: {len(all_tasks)}")

        if num_gpus <= 1:
            result_dict = {}
            worker_fn(0, all_tasks, model_path, OCR_BATCH_SIZE, result_dict)
            all_results = result_dict[0]
        else:
            mp.set_start_method("spawn", force=True)
            manager = mp.Manager()
            result_dict = manager.dict()

            gpu_tasks = [[] for _ in range(num_gpus)]
            for i, task in enumerate(all_tasks):
                gpu_tasks[i % num_gpus].append(task)

            processes = []
            for gpu_id in range(num_gpus):
                p = mp.Process(target=worker_fn,
                               args=(gpu_id, gpu_tasks[gpu_id], model_path, OCR_BATCH_SIZE, result_dict))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            all_results = {}
            for gpu_id, results in result_dict.items():
                all_results.update(results)

        # Aggregate
        edit_distances, completion_ratios, match_word_counts, gt_word_counts = [], [], [], []
        for task_id, result in all_results.items():
            if result is None:
                score_of_prompt_csv.loc[task_id, model_name] = None
            else:
                score_of_prompt_csv.loc[task_id, model_name] = result["prompt_score"]
                edit_distances.extend(result["edit_distances"])
                completion_ratios.extend(result["completion_ratios"])
                match_word_counts.extend(result["match_word_counts"])
                gt_word_counts.extend(result["gt_word_counts"])

        if edit_distances:
            ED = sum(edit_distances) / len(edit_distances)
            CR = sum(completion_ratios) / len(completion_ratios)
            WAC = sum(match_word_counts) / sum(gt_word_counts)
            score_csv.loc[model_name, "ED"] = ED
            score_csv.loc[model_name, "CR"] = CR
            score_csv.loc[model_name, "WAC"] = WAC
            score_csv.loc[model_name, "text score"] = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

    save2csv(score_csv, text_score_csv)
    save2csv(score_of_prompt_csv, text_prompt_score_csv)


if __name__ == "__main__":
    main()
