from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from dreamsim import dreamsim

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

device = "cuda"

# Skip GitHub API validation for torch.hub (no network needed if repo is cached)
torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True

_dreamsim_cache = os.environ.get("ONEIG_DREAMSIM_CACHE", "./models")
model, preprocess = dreamsim(pretrained=True, device=device, cache_dir=_dreamsim_cache)

def img_similar_score(image_1_path, image_2_path):
    image_1 = preprocess(Image.open(image_1_path)).to(device)
    image_2 = preprocess(Image.open(image_2_path)).to(device)
    distance = model(image_1, image_2)     
    return distance.item()

def main():
    args = parse_args()
    
    cache_dir = f"tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)
    
    results_dir = args.results_dir
    diversity_score_csv = f"{results_dir}/diversity_score_{args.mode}_{formatted_time}.csv"
    diversity_prompt_score_csv = f"{results_dir}/diversity_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(results_dir, exist_ok=True)

    column_items = args.class_items.copy().append("total average")
    score_csv = pd.DataFrame(index=args.model_names, columns=column_items)
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)

    for model_id, model_name in enumerate(args.model_names):
        
        print(f"It is {model_name} time.")
        
        img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1])) 

        model_score = []
        
        for class_item in args.class_items:
            
            print(f"We process {class_item} now.")
            
            image_dir = args.image_dirname + '/' + class_item + '/' + model_name
            img_list = megfile.smart_glob(image_dir + '/*')
            img_list = sorted(img_list)
            
            print(f"We fetch {len(img_list)} images.")
            
            diversity_score = []
            
            for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
                
                split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)
                if len(split_img_list) <= 1:
                    continue
                
                score = []
                
                for i in range(len(split_img_list)):
                    for j in range(i+1, len(split_img_list)):
                        prob = img_similar_score(split_img_list[i], split_img_list[j])
                        score.append(prob)
                
                avg_score = sum(score)/len(score)
                
                diversity_score.append(avg_score)
                model_score.append(avg_score)
                
                score_of_prompt_csv.loc[f"{class_item}_{img_path.split('/')[-1][:3]}", model_name] = avg_score

            if len(diversity_score) != 0:
                score_csv.loc[model_name, class_item] = sum(diversity_score)/len(diversity_score)
            else:
                score_csv.loc[model_name, class_item] = None

    mean_values = score_of_prompt_csv.mean()
    score_csv["total average"] = mean_values.values
    save2csv(score_csv, diversity_score_csv)
    
    score_of_prompt_csv = score_of_prompt_csv.sort_index()
    save2csv(score_of_prompt_csv, diversity_prompt_score_csv)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()