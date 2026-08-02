# Copyright: Meta Platforms, Inc. and affiliates

import re
import pdb
import json
import argparse
from tqdm import tqdm
from scipy.stats import gmean


# Per-skill analysis (Soft-TIFA AM)
def per_skill_analysis(all_score_lists, all_skill_lists):
    object_score, object_total = 0, 0
    count_score, count_total = 0, 0
    position_score, position_total = 0, 0
    verb_score, verb_total = 0, 0
    attribute_score, attribute_total = 0, 0

    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        for i in range(len(score_list)):
            if skill_list[i] == 'object':
                object_score += score_list[i]
                object_total += 1
            elif skill_list[i] == 'count':
                count_score += score_list[i]
                count_total += 1
            elif skill_list[i] == 'position':
                position_score += score_list[i]
                position_total += 1
            elif skill_list[i] == 'verb':
                verb_score += score_list[i]
                verb_total += 1
            elif skill_list[i] == 'attribute':
                attribute_score += score_list[i]
                attribute_total += 1
            else:
                raise Exception("Unrecognized skill")
    
    # Note that we say "accuracy", but it's an estimate.
    object_accuracy = 100 * object_score / object_total
    attribute_accuracy = 100 * attribute_score / attribute_total
    count_accuracy = 100 * count_score / count_total
    position_accuracy = 100 * position_score / position_total
    verb_accuracy = 100 * verb_score / verb_total
    return object_accuracy, attribute_accuracy, count_accuracy, \
            position_accuracy, verb_accuracy


# Per-atomicity analysis (Soft-TIFA GM)
def per_atomicity_analysis(all_score_lists, atomicity_list):
    all_atomicity_dict = {k: {} for k in range(3, 11)}
    for k in all_atomicity_dict:
        all_atomicity_dict[k] = {'correct': 0, 'total': 0}
    
    for score_list, atomicity in zip(all_score_lists, atomicity_list):
        all_atomicity_dict[atomicity]['correct'] += gmean(score_list)
        all_atomicity_dict[atomicity]['total'] += 1

    # Here, too, "accuracy" is an estimate.
    for atomicity in all_atomicity_dict:
        all_atomicity_dict[atomicity]['accuracy'] = \
                (all_atomicity_dict[atomicity]['correct'])*100 / \
                all_atomicity_dict[atomicity]['total']
    return all_atomicity_dict


def main():
    parser = argparse.ArgumentParser(description="Analyze T2I Performance")
    parser.add_argument("--benchmark_data", type=str, required=False, \
            default='./geneval2_data.jsonl', help="File with benchmark data")
    parser.add_argument("--score_data", type=str, required=True, \
            help="File with lists of scores per prompt")
    args = parser.parse_args()

    benchmark_data = [json.loads(l) for l in open(args.benchmark_data).readlines()]
    all_score_lists = json.load(open(args.score_data))

    atomicity_list = [b['atom_count'] for b in benchmark_data]
    all_skill_lists = [b['skills'] for b in benchmark_data]

    # Check that the lists all line up:
    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        assert len(score_list) == len(skill_list)

    print("Per Atom Type Analysis (Soft-TIFA AM)")
    object_accuracy, attribute_accuracy, count_accuracy, \
            position_accuracy, verb_accuracy = per_skill_analysis(all_score_lists, \
            all_skill_lists)
    print("Object:")
    print(round(object_accuracy, 2))
    print("Attribute:")
    print(round(attribute_accuracy, 2))
    print("Count:")
    print(round(count_accuracy, 2))
    print("Position:")
    print(round(position_accuracy, 2))
    print("Verb:")
    print(round(verb_accuracy, 2))
    print()
    
    print("Per Atomicity Analysis (Soft-TIFA GM)")
    all_atomicity_dict = per_atomicity_analysis(all_score_lists, \
            atomicity_list)
    for atomicity in all_atomicity_dict:
        print("Results for Atomicity={}".format(atomicity))
        print(round(all_atomicity_dict[atomicity]['accuracy'], 2))


if __name__ == "__main__":
        main()
