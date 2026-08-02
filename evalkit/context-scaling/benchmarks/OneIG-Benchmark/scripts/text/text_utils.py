import re
import numpy as np
from collections import Counter

def preprocess_string(s):
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", '', s)
    if contains_chinese(cleaned):
        pattern = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]")
        s = ''.join(pattern.findall(s))

        return s.strip()

    normalized = re.sub(r'\s+', ' ', cleaned)  

    return normalized.strip()

def clean_and_remove_hallucinations(texts):
    # keywords_list can be added to process ocr results to a cleaner version
    keywords_list = ["addCriterion", "No text recognized."] 
    for keyword in keywords_list:
        texts = [text.replace(keyword, "").replace(f"\n{keyword}", "").replace(f"{keyword}\n", "") for text in texts]
    
    return texts

def levenshtein_distance(s1, s2):
    matrix = np.zeros((len(s1) + 1, len(s2) + 1))

    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # delete
                matrix[i][j - 1] + 1,      # insert
                matrix[i - 1][j - 1] + cost  # replace
            )

    return matrix[len(s1)][len(s2)]

def contains_chinese(text):
    # check whether it contains Chinese
    return bool(re.search('[\u4e00-\u9fff]', text))

def calculate_char_match_ratio(text_gt, ocr_str):
    if contains_chinese(text_gt):
        gt_counter = Counter(text_gt)
        ocr_counter = Counter(ocr_str)
        total_match_count = sum((gt_counter & ocr_counter).values())
        ratio = total_match_count / len(text_gt) if len(text_gt) > 0 else 0.0
    else:
        words_gt = text_gt.split()
        words_ocr = ocr_str.split()
        gt_counter = Counter(words_gt)
        ocr_counter = Counter(words_ocr)
        total_match_count = sum((gt_counter & ocr_counter).values())
        total_gt_count = len(words_gt)
        ratio = total_match_count / total_gt_count if total_gt_count > 0 else 0.0
    return total_match_count, ratio, sum(gt_counter.values())
