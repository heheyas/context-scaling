from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

import torch
torch.cuda.empty_cache()
from scripts.utils.inference import CSDStyleEmbedding, SEStyleEmbedding

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

style_list = ['abstract_expressionism', 'art_nouveau', 'baroque', 'chinese_ink_painting', 'cubism', 'fauvism', 'impressionism', 'line_art', 'minimalism', 'pointillism', 'pop_art', 'rococo',  'ukiyo-e', 'clay', 'crayon',  'graffiti','lego', 'comic', 'pencil_sketch', 'stone_sculpture', 'watercolor', 'celluloid', 'chibi',   'cyberpunk',  'ghibli',  'impasto', 'pixar', 'pixel_art',  '3d_rendering']

def main():
    args = parse_args()
    cache_dir = f"tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)

    style_csv_path = "scripts/style/style.csv"
    df = pd.read_csv(style_csv_path, dtype=str)
    
    CSD_Encoder = CSDStyleEmbedding(model_path=os.environ.get("ONEIG_CSD_CKPT", "scripts/style/models/checkpoint.pth"))
    SE_Encoder = SEStyleEmbedding(pretrained_path=os.environ.get("ONEIG_STYLE_ENCODER_PATH", "xingpng/OneIG-StyleEncoder"))

    CSD_embed_pt = "scripts/style/CSD_embed.pt"
    CSD_ref = torch.load(CSD_embed_pt, weights_only=False)
    SE_embed_pt = "scripts/style/SE_embed.pt"
    SE_ref = torch.load(SE_embed_pt)

    results_dir = args.results_dir
    style_score_csv = f"{results_dir}/style_score_{args.mode}_{formatted_time}.csv"
    style_style_score_csv = f"{results_dir}/style_style_score_{args.mode}_{formatted_time}.csv"
    style_prompt_score_csv = f"{results_dir}/style_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(results_dir, exist_ok=True)

    score_csv = pd.DataFrame(index=args.model_names, columns=["style"])
    score_of_style_csv = pd.DataFrame(index=args.model_names, columns=style_list)
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)  
    
    for model_id, model_name in enumerate(args.model_names):
        
        print(f"It is {model_name} time.")
        
        img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1])) 
        
        image_dir = args.image_dirname + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        img_list = sorted(img_list)
        
        print(f"We fetch {len(img_list)} images.")
        
        style_dict = {style: [] for style in style_list}

        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            
            id = img_path.split('/')[-1][:3]
            
            image_style =  str(df.loc[df["id"] == id, "class"].values[0])
            if (image_style[:3] == "nan"):
                continue
            else:
                image_style = image_style.lower().replace(' ', '_')
            
            split_img_list = split_mxn_grid(img_path, img_grid, cache_dir)

            CSD_ref_embeds = CSD_ref[image_style]
            SE_ref_embeds = SE_ref[image_style]
            
            score = []
            for num, split_img_path in enumerate(split_img_list):
                
                CSD_embed = CSD_Encoder.get_style_embedding(split_img_path)
                SE_embed = SE_Encoder.get_style_embedding(split_img_path)
                
                CSD_max_style_score = max(torch.max(CSD_embed @ CSD_ref_embeds.T).item(), 0)
                SE_max_style_score = max(torch.max(SE_embed @ SE_ref_embeds.T).item(), 0)
                
                max_style_score = (CSD_max_style_score + SE_max_style_score) / 2
                score.append(max_style_score)
            
            if len(score) != 0:
                score_of_prompt_csv.loc[id, model_name] = sum(score)/len(score)
                style_dict[image_style].append(sum(score)/len(score))
            else:
                score_of_prompt_csv.loc[id, model_name] = None        
                    
        for style in style_list:
            if len(style_dict[style]) != 0:
                score_of_style_csv.loc[model_name, style] = sum(style_dict[style]) / len(style_dict[style])

    mean_values = score_of_prompt_csv.mean()
    score_csv["style"] = mean_values.values
    save2csv(score_csv, style_score_csv)
    
    save2csv(score_of_style_csv, style_style_score_csv)

    score_of_prompt_csv = score_of_prompt_csv.sort_index()
    save2csv(score_of_prompt_csv, style_prompt_score_csv)    

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()
