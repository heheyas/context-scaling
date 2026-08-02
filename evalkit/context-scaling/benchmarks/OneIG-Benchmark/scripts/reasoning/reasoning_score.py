from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

import json
from scripts.utils.inference import LLM2CLIP

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def main():
    args = parse_args()
    cache_dir = f"tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)
    
    LLM2CLIP_Model = LLM2CLIP()
    
    if args.mode == "EN":
        answer_json_dir = "scripts/reasoning/gt_answer.json"
    else:
        answer_json_dir = "scripts/reasoning/gt_answer_zh.json"
    with open(answer_json_dir, 'r', encoding='utf-8') as f:
        answer_gt = json.load(f)
        
    results_dir = args.results_dir
    reasoning_score_csv = f"{results_dir}/reasoning_score_{args.mode}_{formatted_time}.csv"
    reasoning_prompt_score_csv = f"{results_dir}/reasoning_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(results_dir, exist_ok=True)
    
    score_csv = pd.DataFrame(index=args.model_names, columns=["reasoning"])
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)

    for model_id, model_name in enumerate(args.model_names):
        
        print(f"It is {model_name} time.")
        
        img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1])) 
        
        image_dir = args.image_dirname + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        img_list = sorted(img_list)
        
        print(f"We fetch {len(img_list)} images.")
        
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            
            split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
            
            img_id = img_path.split('/')[-1][:3]
            answer_text = answer_gt[img_id]

            score = LLM2CLIP_Model.text_img_similarity_score(split_img_list, answer_text)

            if len(score) != 0:
                score = [x for x in score if x is not None]
                score_of_prompt_csv.loc[img_id, model_name] = sum(score)/len(score)
            else:
                score_of_prompt_csv.loc[img_id, model_name] = None
    
    mean_values = score_of_prompt_csv.mean()
    score_csv["reasoning"] = mean_values.values
    save2csv(score_csv, reasoning_score_csv)
    
    score_of_prompt_csv = score_of_prompt_csv.sort_index()
    save2csv(score_of_prompt_csv, reasoning_prompt_score_csv)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
            