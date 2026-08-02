"""
Multi-GPU GenEval2 evaluation. Loads one Qwen3-VL-8B per GPU and evaluates
prompts in parallel across all GPUs.

Usage:
    python benchmarks/GenEval2/evaluation_multigpu.py \
        --benchmark_data benchmarks/GenEval2/geneval2_data.jsonl \
        --image_filepath_data outputs/geneval2_xxx/image_filepath_data.json \
        --method soft_tifa_gm \
        --output_file outputs/geneval2_xxx/scores.json \
        --num_gpus 8

    # Use specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/GenEval2/evaluation_multigpu.py ...
"""

import json
import argparse
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from scipy.stats import gmean
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "<HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct"


def return_numeric_string(number):
    mapping = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    }
    return mapping.get(number, 'other')


def construct_message_with_image(prompt, image_filepath):
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image_filepath},
            {"type": "text", "text": prompt},
        ]}
    ]


def send_message_with_image(prompt, image_filepath, answer_list, model, processor, device):
    messages = construct_message_with_image(prompt, image_filepath)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True
        )
    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(scores, dim=-1)

    if answer_list:
        lm_prob = 0
        for answer in answer_list:
            ans_token_id = processor.tokenizer.encode(answer)[0]
            lm_prob += probs[0, ans_token_id].item()
    else:
        lm_prob = None

    pred = processor.batch_decode([torch.argmax(probs)])[0]
    return pred, lm_prob


def soft_tifa(vqa_list, image_filepath, model, processor, device):
    score_list = []
    for vqa in vqa_list:
        question, answer = vqa
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), ' ' + answer,
                           ' ' + answer.capitalize(), return_numeric_string(answer),
                           ' ' + return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        _, ans_prob = send_message_with_image(
            '{} Answer in one word.'.format(question), image_filepath,
            answer_list, model, processor, device
        )
        score_list.append(ans_prob)
    return score_list


def tifa(vqa_list, image_filepath, model, processor, device):
    score_list = []
    for vqa in vqa_list:
        question, answer = vqa
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), ' ' + answer,
                           ' ' + answer.capitalize(), return_numeric_string(answer),
                           ' ' + return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        pred, _ = send_message_with_image(
            '{} Answer in one word.'.format(question), image_filepath,
            answer_list, model, processor, device
        )
        score_list.append(1 if pred.lower() in answer_list else 0)
    return score_list


def vqa_score(prompt, image_filepath, model, processor, device):
    message_prompt = 'Does this image show "{}"? Answer the question with Yes or No.'.format(prompt)
    _, ans_prob = send_message_with_image(
        message_prompt, image_filepath,
        ['Yes', 'yes', ' yes', ' Yes'], model, processor, device
    )
    return [ans_prob]


def worker_fn(gpu_id, tasks, method, result_dict, model_path):
    """Worker: load model on one GPU, evaluate assigned tasks."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model...", flush=True)

    processor = AutoProcessor.from_pretrained(model_path, torch_dtype='auto')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto"
    ).to(device).eval()

    print(f"[GPU {gpu_id}] Model loaded. Processing {len(tasks)} prompts.", flush=True)

    results = {}
    for idx, d, image_filepath in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        if method == 'vqascore':
            score_list = vqa_score(d['prompt'], image_filepath, model, processor, device)
        elif method == 'tifa':
            score_list = tifa(d['vqa_list'], image_filepath, model, processor, device)
        elif method in ('soft_tifa_am', 'soft_tifa_gm'):
            score_list = soft_tifa(d['vqa_list'], image_filepath, model, processor, device)
        else:
            raise NotImplementedError(f"Unknown method: {method}")
        results[idx] = score_list

    result_dict[gpu_id] = results
    print(f"[GPU {gpu_id}] Done.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU GenEval2 Evaluation")
    parser.add_argument("--benchmark_data", type=str, required=True)
    parser.add_argument("--image_filepath_data", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["vqascore", "tifa", "soft_tifa_am", "soft_tifa_gm"])
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use (0 = all available)")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s)")

    benchmark_data = [json.loads(l) for l in open(args.benchmark_data).readlines()]
    image_data = json.load(open(args.image_filepath_data))

    # Build task list: (global_index, data_item, image_path)
    all_tasks = []
    for idx, d in enumerate(benchmark_data):
        if d['prompt'] not in image_data:
            raise Exception(f"Missing filepath for prompt: {d['prompt']}")
        all_tasks.append((idx, d, image_data[d['prompt']]))

    print(f"Total prompts: {len(all_tasks)}")

    # Split tasks across GPUs
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_tasks[i % num_gpus].append(task)

    for g in range(num_gpus):
        print(f"  GPU {g}: {len(gpu_tasks[g])} prompts")

    # Run workers
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, gpu_tasks[gpu_id], args.method, result_dict, args.model_path)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results in original order
    all_score_lists = [None] * len(all_tasks)
    for gpu_id, results in result_dict.items():
        for idx, score_list in results.items():
            all_score_lists[idx] = score_list

    # Check completeness
    missing = [i for i, s in enumerate(all_score_lists) if s is None]
    if missing:
        print(f"WARNING: {len(missing)} prompts missing results: {missing[:10]}...")

    # Save scores
    json.dump(all_score_lists, open(args.output_file, 'w'))

    # ── Summary ──
    valid_scores = [s for s in all_score_lists if s is not None]
    all_skill_lists = [d['skills'] for d in benchmark_data]
    atomicity_list = [d['atom_count'] for d in benchmark_data]

    print(f"\n{'='*60}")
    print(f"GenEval2 Results ({len(valid_scores)} prompts)")
    print(f"{'='*60}")

    # Overall: Soft-TIFA AM (atom-level)
    am_per_prompt = [sum(s) / len(s) for s in valid_scores]
    am_total = 100 * sum(am_per_prompt) / len(am_per_prompt)
    print(f"\nSoft-TIFA AM (atom-level):  {am_total:.2f}")

    # Overall: Soft-TIFA GM (prompt-level)
    gm_per_prompt = [gmean(s) for s in valid_scores]
    gm_total = 100 * sum(gm_per_prompt) / len(gm_per_prompt)
    print(f"Soft-TIFA GM (prompt-level): {gm_total:.2f}")

    # Per-skill breakdown (AM)
    print(f"\n--- Per Skill (Soft-TIFA AM) ---")
    skill_scores = {}
    for score_list, skill_list in zip(valid_scores, all_skill_lists):
        for score, skill in zip(score_list, skill_list):
            if skill not in skill_scores:
                skill_scores[skill] = {"sum": 0, "count": 0}
            skill_scores[skill]["sum"] += score
            skill_scores[skill]["count"] += 1

    for skill in ["object", "attribute", "count", "position", "verb"]:
        if skill in skill_scores:
            acc = 100 * skill_scores[skill]["sum"] / skill_scores[skill]["count"]
            print(f"  {skill:<12} = {acc:.2f}  ({skill_scores[skill]['count']} atoms)")

    # Per-atomicity breakdown (GM)
    print(f"\n--- Per Atomicity (Soft-TIFA GM) ---")
    atom_buckets = {}
    for score_list, atom_count in zip(valid_scores, atomicity_list):
        if atom_count not in atom_buckets:
            atom_buckets[atom_count] = {"sum": 0, "count": 0}
        atom_buckets[atom_count]["sum"] += gmean(score_list)
        atom_buckets[atom_count]["count"] += 1

    for atom_count in sorted(atom_buckets.keys()):
        b = atom_buckets[atom_count]
        acc = 100 * b["sum"] / b["count"]
        print(f"  atomicity={atom_count:<3} = {acc:.2f}  ({b['count']} prompts)")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
