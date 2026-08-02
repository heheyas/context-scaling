from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

import json
import torch
import torch.multiprocessing as mp
from copy import deepcopy

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# Batch size for VQA inference (number of (image, question) pairs per batch)
VQA_BATCH_SIZE = int(os.environ.get("ONEIG_VQA_BATCH_SIZE", "128"))


def apply_dependency_filter(score, dependencies, num_images):
    """Apply parent-child dependency filtering to raw scores."""
    filter_score = deepcopy(score)
    for img_idx in range(num_images):
        for id, parent_ids in dependencies.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                try:
                    if score[parent_id][img_idx] == 0:
                        any_parent_answered_no = True
                        break
                except:
                    pass
            if any_parent_answered_no:
                filter_score[id][img_idx] = 0
    return filter_score


def compute_alignment_score(filter_score, num_images):
    """Compute final alignment score from filtered per-question scores."""
    sum_scores = [0] * num_images
    for question_id in range(len(filter_score)):
        for img_idx in range(num_images):
            sum_scores[img_idx] += filter_score[question_id + 1][img_idx]
    sum_scores = [s / len(filter_score) for s in sum_scores]
    return sum(sum_scores) / len(sum_scores)


def worker_fn(gpu_id, tasks, model_path, batch_size, result_dict):
    """Worker: load Qwen on one GPU, batch all VQA queries across items."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from scripts.utils.inference import Qwen2_5VLBatchInferencer
    inferencer = Qwen2_5VLBatchInferencer(model_path, device="cuda")

    cache_dir = f"tmp_{formatted_time}_gpu{gpu_id}"
    os.makedirs(cache_dir, exist_ok=True)

    # Phase 1: Pre-split all grids and collect all VQA queries
    item_data = {}  # task_key -> {split_imgs, questions, dependencies, num_images}
    all_queries = []  # (query_idx, task_key, question_id, image_path)

    for task_idx, (task_key, task_info) in enumerate(tasks):
        if len(task_info["img_path"]) != 1:
            item_data[task_key] = None
            continue

        # Use per-item sub-directory to avoid file name collisions
        item_cache = os.path.join(cache_dir, f"item_{task_idx}")
        os.makedirs(item_cache, exist_ok=True)
        split_imgs = split_mxn_grid(task_info["img_path"][0], task_info["img_grid"], item_cache)
        if len(split_imgs) == 0:
            item_data[task_key] = None
            continue

        item_data[task_key] = {
            "split_imgs": split_imgs,
            "questions": task_info["questions"],
            "dependencies": task_info["dependencies"],
            "num_images": len(split_imgs),
        }

        # Create one query per (question, image) pair
        for qid, question in task_info["questions"].items():
            for img_path in split_imgs:
                all_queries.append((len(all_queries), task_key, qid, img_path, question))

    print(f"[GPU {gpu_id}] {len(tasks)} items, {len(all_queries)} VQA queries, batch_size={batch_size}",
          flush=True)

    # Phase 2: Batch inference
    all_answers = [None] * len(all_queries)

    for batch_start in tqdm(range(0, len(all_queries), batch_size),
                            desc=f"GPU {gpu_id}", position=gpu_id):
        batch = all_queries[batch_start:batch_start + batch_size]

        messages = []
        for _, _, _, img_path, question in batch:
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": f"{question}. Please answer 'Yes' or 'No' only."}
                ],
            }])

        answers = inferencer.batch_inference(messages, max_new_tokens=8)

        for i, (query_idx, _, _, _, _) in enumerate(batch):
            all_answers[query_idx] = answers[i]

    # Phase 3: Reconstruct per-item scores
    results = {}
    # Group answers by (task_key, question_id)
    query_results = {}  # (task_key, qid) -> [answers per image]
    for query_idx, task_key, qid, _, _ in all_queries:
        key = (task_key, qid)
        if key not in query_results:
            query_results[key] = []
        ans = all_answers[query_idx].strip() if all_answers[query_idx] else ""
        query_results[key].append(float(ans == "Yes"))

    for task_key, data in item_data.items():
        if data is None:
            results[task_key] = None
            continue

        score = {}
        for qid in data["questions"]:
            score[qid] = query_results.get((task_key, qid), [0.0] * data["num_images"])

        filter_score = apply_dependency_filter(score, data["dependencies"], data["num_images"])
        results[task_key] = compute_alignment_score(filter_score, data["num_images"])

    result_dict[gpu_id] = results

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

    print(f"[GPU {gpu_id}] Done. {len(tasks)} items.", flush=True)


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    model_path = os.environ.get("ONEIG_QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Using {num_gpus} GPU(s), model: {model_path}, VQA batch size: {VQA_BATCH_SIZE}")

    question_dependency_dir = "scripts/alignment"

    results_dir = args.results_dir
    alignment_score_csv = f"{results_dir}/alignment_score_{args.mode}_{formatted_time}.csv"
    alignment_prompt_score_csv = f"{results_dir}/alignment_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(results_dir, exist_ok=True)

    score_csv = pd.DataFrame(index=args.model_names, columns=["alignment"])
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)

    # Build all tasks
    all_tasks = []
    for class_item in args.class_items:
        print(f"Loading {class_item}...")
        if args.mode == "EN":
            qd_path = question_dependency_dir + '/Q_D/' + class_item + '.json'
        else:
            qd_path = question_dependency_dir + '/Q_D/' + class_item + '_zh.json'

        with open(qd_path, "r", encoding="utf-8") as f:
            question_dependency = json.load(f)

        for key, item in question_dependency.items():
            if isinstance(item["question"], str):
                item["question"] = {int(k): v for k, v in json.loads(item["question"]).items()}
            if isinstance(item["dependency"], str):
                item["dependency"] = {int(k): v for k, v in json.loads(item["dependency"]).items()}

            for model_id, model_name in enumerate(args.model_names):
                img_grid = (int(args.image_grid[model_id].split(',')[0]),
                            int(args.image_grid[model_id].split(',')[-1]))
                image_path = megfile.smart_glob(
                    args.image_dirname + '/' + class_item + '/' + model_name + '/' + key + '*')

                task_key = f"{class_item}_{key}_{model_name}"
                all_tasks.append((task_key, {
                    "img_path": image_path,
                    "questions": item["question"],
                    "dependencies": item["dependency"],
                    "img_grid": img_grid,
                    "class_item": class_item,
                    "model_name": model_name,
                }))

    print(f"Total tasks: {len(all_tasks)}")

    if num_gpus <= 1:
        # Single GPU — use same batched approach
        result_dict = {}
        worker_fn(0, all_tasks, model_path, VQA_BATCH_SIZE, result_dict)
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
                           args=(gpu_id, gpu_tasks[gpu_id], model_path, VQA_BATCH_SIZE, result_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_results = {}
        for gpu_id, results in result_dict.items():
            all_results.update(results)

    # Merge into DataFrames
    for tk, td in all_tasks:
        if tk in all_results:
            prompt_key = f"{td['class_item']}_{tk.split('_')[1]}"
            score_of_prompt_csv.loc[prompt_key, td["model_name"]] = all_results[tk]

    mean_values = score_of_prompt_csv.mean()
    score_csv["alignment"] = mean_values.values
    save2csv(score_csv, alignment_score_csv)

    score_of_prompt_csv = score_of_prompt_csv.sort_index()
    save2csv(score_of_prompt_csv, alignment_prompt_score_csv)


if __name__ == "__main__":
    main()
