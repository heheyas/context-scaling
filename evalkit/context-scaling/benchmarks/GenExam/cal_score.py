import json
import os
from tqdm import tqdm
from glob import glob
import argparse
from collections import defaultdict

def cal_score_single(eval_result):
    assert "global_evaluation" in eval_result and "answers" in eval_result, f"Invalid eval result: {eval_result}"
    scoring_point_scores = 0
    for idx, answer in enumerate(eval_result["answers"]):
        if answer["answer"] == 1:
            scoring_point_scores += eval_result["scoring_points"][idx]["score"]
    
    assert round(sum(item["score"] for item in eval_result["scoring_points"]), 4) == 1, f"Invalid sum of scoring point scores: {sum(item['score'] for item in eval_result['scoring_points'])}"
    
    scores = {
        "semantic_correctness": round(scoring_point_scores, 4),
        "readability": eval_result["global_evaluation"]["Clarity and Readability"]["score"],
        "logical_consistency": eval_result["global_evaluation"]["Logical Consistency"]["score"],
        "spelling": eval_result["global_evaluation"]["Spelling"]["score"],
    }
    
    scores["scoring_point_all_correct"] = all(item["answer"] == True for item in eval_result["answers"])
    
    
    scores["strict_score"] = int(scores["scoring_point_all_correct"] and scores["spelling"] == 2 and scores["readability"] == 2 and scores["logical_consistency"] == 2)
    scores["relaxed_score"] = round(scores["semantic_correctness"] * 0.7 + scores["spelling"] * 0.1 / 2 + scores["readability"] * 0.1 / 2 + scores["logical_consistency"] * 0.1 / 2, 3)
    
    return scores


def calculate_score(eval_results_dir, sampled_id_path=None):
    all_score_keys = ["semantic_correctness", "spelling", "readability", "logical_consistency"]
    
    all_scores = defaultdict(list)
    
    if sampled_id_path is not None:
        with open(sampled_id_path) as f:
            sampled_ids = [x.strip() for x in f.readlines()]
    
    eval_results = glob(os.path.join(eval_results_dir, "**/*.json"), recursive=True)
    for eval_result_path in eval_results:
        with open(eval_result_path, "r") as f:
            eval_result = json.load(f)
        
        if sampled_id_path is not None and eval_result["id"] not in sampled_ids:
            continue
            
        scores = cal_score_single(eval_result)
        
        data_name = eval_result.get("subject", "all")
        
        all_scores[data_name].append(scores)
    
    data_names = sorted(list(all_scores.keys()))
    total_count = sum(len(scores) for scores in all_scores.values())
    if total_count == 0:
        return {}
    
    print("=" * 80)
    print("Each score dimension:")
    for key in all_score_keys:
        score_avg = sum(score[key] for scores in all_scores.values() for score in scores) / total_count
        print(f"- {key}: {round(score_avg, 2)}")

    print("=" * 80)
    print("Each score dimension (average) for each subject:")
    for data_name, scores in all_scores.items():
        if len(scores) == 0:
            continue
        print(f"- {data_name}:")
        for key in all_score_keys:
            score_sum = sum(score[key] for score in scores)
            score_avg = score_sum / len(scores)
            print(f"  {key}: {round(score_avg, 2)}")
    
    print("-" * 80)
    print("Total number of eval results: ", total_count)
    
    print("-" * 80)
    print("Strict score:")
    for data_name in data_names:
        scores = all_scores[data_name]
        if len(scores) == 0:
            continue
        print(f"- {data_name}({len(scores)} samples): {round(sum(score['strict_score'] for score in scores) / len(scores) * 100, 1)}%", end=" ")
    


    print("-" * 80)
    print("Relaxed score:")
    for data_name in data_names:
        scores = all_scores[data_name]
        if len(scores) == 0:
            continue
        print(f"- {data_name}({len(scores)} samples): {round(sum(score['relaxed_score'] for score in scores) / len(scores) * 100, 1)}%", end=" ")

    avg_strict_score = sum(score['strict_score'] for scores in all_scores.values() for score in scores) / total_count
    avg_relaxed_score = sum(score['relaxed_score'] for scores in all_scores.values() for score in scores) / total_count
    
    print(f"\nAverage strict score: {round(avg_strict_score * 100, 1)}%")
    print(f"Average relaxed score: {round(avg_relaxed_score * 100, 1)}%")

    avg_semantic_correctness = sum(score['semantic_correctness'] for scores in all_scores.values() for score in scores) / total_count
    avg_spelling = sum(score['spelling'] for scores in all_scores.values() for score in scores) / total_count
    avg_readability = sum(score['readability'] for scores in all_scores.values() for score in scores) / total_count
    avg_logical_consistency = sum(score['logical_consistency'] for scores in all_scores.values() for score in scores) / total_count
    
    return {
        "strict_score": avg_strict_score,
        "relaxed_score": avg_relaxed_score,
        "semantic_correctness": avg_semantic_correctness,
        "spelling": avg_spelling,
        "readability": avg_readability,
        "logical_consistency": avg_logical_consistency,
    }
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_results_dir", type=str, default="eval_results")
    parser.add_argument("--sampled_id_path", type=str, default="data/mini_sample_ids.txt")
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()
    
    if args.mini:
        calculate_score(eval_results_dir=args.eval_results_dir, sampled_id_path=args.sampled_id_path)
    else:
        calculate_score(eval_results_dir=args.eval_results_dir, sampled_id_path=None)
    
    
