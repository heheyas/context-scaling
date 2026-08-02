# Copyright: Meta Platforms, Inc. and affiliates

import re
import pdb
import json
import torch
import base64
import argparse
from PIL import Image
from tqdm import tqdm
from scipy.stats import gmean
from transformers import AutoModelForCausalLM, AutoProcessor, \
        GenerationConfig, AutoTokenizer, Qwen3VLForConditionalGeneration


print("Loading Qwen")
qwen_processor = AutoProcessor.from_pretrained(
                "<HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct",
                torch_dtype='auto',
                device_map='auto'
            )

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "<HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct", 
            dtype="auto", 
            device_map="auto"
            )


def return_numeric_string(number):
    match number:
        case 'one':
            return '1'
        case 'two':
            return '2'
        case 'three':
            return '3'
        case 'four':
            return '4'
        case 'five':
            return '5'
        case 'six':
            return '6'
        case 'seven':
            return '7'
        case 'eight':
            return '8'
        case 'nine':
            return '9'
        case 'ten':
            return '10'
    return 'other'


def construct_message_with_image(prompt, image_filepath):
    """
    Constructs the message structure with image.
    """
    return [
        { "role": "user", "content": [  
            { 
                "type": "image",
                "image": image_filepath,
            },
            { 
                "type": "text", 
                "text": prompt 
            },
        ] } 
    ]


def send_message_with_image(prompt, image_filepath, answer_list=None):
    messages = construct_message_with_image(prompt, image_filepath)
    inputs = qwen_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, 
                return_dict=True, return_tensors="pt"
                )
    inputs = inputs.to(qwen_model.device)
    outputs = qwen_model.generate(**inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
            )
    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(scores, dim=-1)

    if answer_list:
        lm_prob = 0
        for answer in answer_list:
            ans_token_id = qwen_processor.tokenizer.encode(answer)[0]
            lm_prob += probs[0, ans_token_id].item()

    else:
        lm_prob = None

    argmax_token = qwen_processor.batch_decode([torch.argmax(probs)])[0]
    pred = argmax_token
    return pred, lm_prob


def vqa_score(prompt, image_filepath):
    message_prompt = 'Does this image show "{}"? Answer the question with Yes or No.'.format(prompt)
    pred, ans_prob = send_message_with_image(message_prompt.format(prompt), image_filepath, answer_list=['Yes', 'yes', ' yes', ' Yes'])
    return ans_prob


def tifa(vqa_list, image_filepath):
    score = 0
    score_list = []
    for vqa in vqa_list:
        question, answer = vqa
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), ' '+answer, \
                    ' '+answer.capitalize(), return_numeric_string(answer), \
                    ' '+return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        pred, ans_prob = send_message_with_image('{} Answer in one word.'.format(question), image_filepath, \
                answer_list=answer_list)
        if pred.lower() in answer_list:
            score += 1
            score_list.append(1)
        else:
            score_list.append(0)
    score = score / len(vqa_list)
    return score, score_list


def soft_tifa(vqa_list, image_filepath):
    score = 0
    score_list = []
    for vqa in vqa_list:
        question, answer = vqa
        if question.startswith("How many"):
            answer_list = [answer, answer.capitalize(), ' '+answer, \
                    ' '+answer.capitalize(), return_numeric_string(answer), \
                    ' '+return_numeric_string(answer)]
        else:
            answer_list = ['Yes', 'yes', ' yes', ' Yes']
        pred, ans_prob = send_message_with_image('{} Answer in one word.'.format(question), image_filepath, \
                answer_list=answer_list)
        score += ans_prob
        score_list.append(ans_prob)
    score = score / len(vqa_list)
    return score, score_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate T2I Images")
    parser.add_argument("--benchmark_data", type=str, required=True, \
            help="File with benchmark data")
    parser.add_argument("--image_filepath_data", type=str, required=True, \
            help="File with prompts and image filepaths")
    parser.add_argument("--method", type=str, required=True, choices=["vqascore", \
            "tifa", "soft_tifa_am", "soft_tifa_gm"], help="Method name")
    parser.add_argument("--output_file", type=str, required=True, \
            help="Output filepath name")
    args = parser.parse_args()

    benchmark_data = [json.loads(l) for l in open(args.benchmark_data).readlines()]
    # File with prompts and image filepaths as a dictionary:
    # {prompt: image_filepath}
    image_data = json.load(open(args.image_filepath_data))
    all_score_lists = []

    for d in tqdm(benchmark_data):
        if d['prompt'] not in image_data.keys():
            raise Exception("Missing filepath for the prompt: {}".format(d['prompt']))
        image_filepath = image_data[d['prompt']]
        if args.method == 'vqascore':
            score = vqa_score(d['prompt'], image_filepath)
            score_list = [score]
        elif args.method == 'tifa':
            score, score_list = tifa(d['vqa_list'], image_filepath) 
        elif args.method == 'soft_tifa_am' or args.method == 'soft_tifa_gm':
            score, score_list = soft_tifa(d['vqa_list'], image_filepath)
        else:
            raise NotImplementedError
        all_score_lists.append(score_list)

    # Save scores for later analysis
    json.dump(all_score_lists, open(args.output_file, 'w'))
    
    # Calculating total score
    if args.method == 'soft_tifa_gm':
        per_prompt_scores = [gmean(s) for s in all_score_lists]
    else:
        per_prompt_scores = [sum(s)/len(s) for s in all_score_lists]
    total_score = 100 * sum(per_prompt_scores)/len(per_prompt_scores)
    print("Score: {}".format(total_score))


if __name__ == "__main__":
        main()

