import numpy as np
import json
import os
from scipy.optimize import linear_sum_assignment

def normalized_edit_distance(s1, s2):
    """Calculate the normalized edit distance (NED) between two strings."""
    len_s1 = len(s1)
    len_s2 = len(s2)
    max_len = max(len_s1, len_s2)
    if max_len == 0:
        return 0.0
    # Calculate the edit distance
    dp = np.zeros((len_s1 + 1, len_s2 + 1))
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,      # Deletion
                               dp[i][j-1] + 1,      # Insertion
                               dp[i-1][j-1] + cost) # Substitution
    # Normalize the edit distance
    ned = dp[len_s1][len_s2] / max_len
    return ned

def calculate_recall(list1, list2, threshold=0.3):
    """
    Calculate the recall of list2 with respect to list1.
    Ensure that each element in list1 and list2 can only be used once.
    """
    true_positives = 0
    total = len(list1)
    used_indices_list1 = set()  # Record the indices of matched elements in list1
    used_indices_list2 = set()  # Record the indices of matched elements in list2

    new_list2 = []
    for _ in list2:
        new_list2.extend(_.split(" ", -1))

    list2 = new_list2

    for i, gt in enumerate(list1):
        if i in used_indices_list1:
            continue  # If the element in list1 has already been matched, skip
        for j, pred in enumerate(list2):
            if j in used_indices_list2:
                continue  # If the element in list2 has already been matched, skip
            ned = normalized_edit_distance(gt, pred)
            if ned <= threshold:
                true_positives += 1
                used_indices_list1.add(i)  # Mark the element in list1 as used
                used_indices_list2.add(j)  # Mark the element in list2 as used
                break  # Break the inner loop after a successful match
    
    recall = true_positives / total
    return recall

def matching_based_nled(gt_list, test_list):
    new_test_list = []
    for _ in test_list:
        new_test_list.extend(_.split(" ", -1))
    test_list = new_test_list

    len_gt, len_test = len(gt_list), len(test_list)
    if len_gt == 0 and len_test == 0:
        return 0.0  # 两者均为空时无误差

    cost_matrix = np.zeros((len_test, len_gt))
    for i, test_item in enumerate(test_list):
        for j, gt_item in enumerate(gt_list):
            cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_cost = cost_matrix[row_ind, col_ind].sum()
    unmatched_penalty = abs(len_test - len_gt)
    total_cost = matched_cost + unmatched_penalty

    max_len = max(len_gt, len_test)
    normalized_gned = total_cost / max_len
    return normalized_gned

def parse_text(simple_caption):
    text = simple_caption.split("\"", -1)[-2].lower()
    return [text]

def collect_data(path):
    try:
        data_list = json.load(open(path, "r", encoding="utf-8"))
        ocr_results_simple_image = [item['short_image_ocr_results'] for item in data_list]
        ocr_results_enhanced_image = [item['long_image_ocr_results'] for item in data_list]
        # Process the OCR results for simple_image
        ocr_results_simple_image = [
            [item[1][0].lower() for item in ocr_list] if ocr_list else [""]
            for ocr_list in ocr_results_simple_image
        ]

        # Process the OCR results for enhanced_image
        ocr_results_enhanced_image = [
            [item[1][0].lower() for item in ocr_list] if ocr_list else [""]
            for ocr_list in ocr_results_enhanced_image
        ]
        gt_text_list = [[_.lower() for _ in item['text']] for item in data_list]

        return ocr_results_simple_image, ocr_results_enhanced_image, gt_text_list
    except:
        return [], [], []

def process_json_file(file_path):
    all_nled_simple_image, all_nled_enhanced_image = [], []
    all_recall_simple_image, all_recall_enhanced_image = [], []

    ocr_results_simple_image, ocr_results_enhanced_image, gt_text_list = collect_data(file_path)
    
    # Calculate NLED
    nled_list_simple_image = [matching_based_nled(A, B) for A, B in zip(gt_text_list, ocr_results_simple_image)]
    nled_list_enhanced_image = [matching_based_nled(A, B) for A, B in zip(gt_text_list, ocr_results_enhanced_image)]

    # Calculate Recall
    recall_list_simple_image = [calculate_recall(A, B) for A, B in zip(gt_text_list, ocr_results_simple_image)]
    recall_list_enhanced_image = [calculate_recall(A, B) for A, B in zip(gt_text_list, ocr_results_enhanced_image)]

    # Aggregate results
    all_nled_simple_image += nled_list_simple_image
    all_nled_enhanced_image += nled_list_enhanced_image
    all_recall_simple_image += recall_list_simple_image
    all_recall_enhanced_image += recall_list_enhanced_image

    # Calculate averages
    avg_nled_simple = sum(all_nled_simple_image)/len(all_nled_simple_image) if all_nled_simple_image else 0
    avg_nled_enhanced = sum(all_nled_enhanced_image)/len(all_nled_enhanced_image) if all_nled_enhanced_image else 0
    avg_recall_simple = sum(all_recall_simple_image)/len(all_recall_simple_image) if all_recall_simple_image else 0
    avg_recall_enhanced = sum(all_recall_enhanced_image)/len(all_recall_enhanced_image) if all_recall_enhanced_image else 0

    return {
        "file_name": os.path.basename(file_path),
        "GNED_score_short": avg_nled_simple,
        "GNED_score_long": avg_nled_enhanced,
        "Recall_score_short": avg_recall_simple,
        "Recall_score_long": avg_recall_enhanced
    }

if __name__ == "__main__":
    folder_path = '/mnt/data/cpfs/cfps/personal/why/text/'
    output_file = '/mnt/data/cpfs/cfps/personal/why/eval_text_methods_results_GNED.json'
    
    # Get all json files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    all_results = []
    
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        print(f"Processing {json_file}...")
        result = process_json_file(file_path)
        all_results.append(result)
    
    # Save all results to a new json file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"All results saved to {output_file}")