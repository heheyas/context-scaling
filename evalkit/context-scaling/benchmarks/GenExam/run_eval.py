import json
import os
import base64
import requests
from PIL import Image
import io
import ast
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor

from eval_prompt import prompt_for_eval

#! change with your own openai api key
openai_api_key = "sk-xxxx"
openai_base_url = "https://api.openai.com/v1/"


def encode_image(image_path, target_size=1024, fmt="JPEG"):
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    if target_size is not None and target_size > 0:
        w, h = img.size
        if max(w, h) > target_size:
            if w >= h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')



def call_gpt5(image_path, image_path2, text_prompt, max_tokens=4096, img_size=768):
    base64_image = encode_image(image_path, target_size=img_size)
    base64_image2 = encode_image(image_path2, target_size=img_size)
        
    content = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}",
                "detail": "high"
            }
        }
    ]
                
    json_parameters = {
        "model": "gpt-5-2025-08-07",
        "messages": [
            {"role": "user", "content": content}
        ],
        "stream": False,
        "reasoning_effort": "low",
        "verbosity": "medium",
        "max_completion_tokens": max_tokens,
    }
    
    try:
        response = requests.post(
            openai_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json=json_parameters,
            stream=False,
            # proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"} #! Uncomment to use proxy
        )
        response.raise_for_status()
        
        response_json = response.json()
        ret = response_json["choices"][0]["message"]["content"]
        return ret
    
    
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        try:
            response_json = response.json()
            if 'error' in response_json:
                print(response_json["error"])
        finally:
            return None



def call_gpt_img_gen(text_prompt, save_path):
    response = requests.post(
        openai_base_url.rstrip("/") + "/images/generations", 
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
        json={
            "model": "gpt-image-1",
            "prompt": text_prompt,
            "n": 1,  # Number of images to generate
            "moderation": "low",
            "quality": "high",
        },
        # proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"} #! Uncomment to use proxy
    )
    response.raise_for_status()
    image_base64 = response.json()["data"][0]["b64_json"]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"Image saved to {save_path}")
    


def eval_single(gen_img_path, gt_img_path, t2i_prompt, scoring_points):
    assert os.path.exists(gen_img_path), f"Image {gen_img_path} not found"
    assert os.path.exists(gt_img_path), f"Image {gt_img_path} not found"

    count = 0
    while True:
        count += 1
        if count > 3:
            print(f"Failed to get response from GPT-5 after {count} times")
            raise ValueError("Failed to get response from GPT-5")
        
        response = call_gpt5(
            image_path=gen_img_path, 
            image_path2=gt_img_path,
            text_prompt=prompt_for_eval.format(prompt=t2i_prompt, scoring_points=scoring_points),
            max_tokens=16384,
            img_size=768,
        )
        if response is not None:
            try:
                response = json.loads(response.split("```json")[-1].split("```")[0])
            except:
                try:
                    response = ast.literal_eval(response.split("```json")[-1].split("```")[0])
                except:
                    continue
            try:
                assert "global_evaluation" in response and "answers" in response, f"Invalid response: {response}"
                assert len(response["answers"]) == len(scoring_points), f"Invalid number of answers: {len(response['answers'])}"
                assert all(item["answer"] in [0, 1] for item in response["answers"]), f"Invalid answer: {response['answers']}"
                assert "Clarity and Readability" in response["global_evaluation"] and "Logical Consistency" in response["global_evaluation"] and "Spelling" in response["global_evaluation"], f"Invalid global evaluation: {response['global_evaluation']}"
            except:
                continue
            break
        
    
    
    
    return response


def inference_and_eval_single(data, img_save_dir, eval_save_dir, data_dir, inference_function, sampled_ids):
    json_save_path = os.path.join(eval_save_dir, f"{data['id']}.json")
    gen_img_save_path = os.path.join(img_save_dir, f"{data['id']}.png")
    
    if sampled_ids is not None and data["id"] not in sampled_ids:
        return
    
    if os.path.exists(json_save_path):
        print("Skipping already evaluated data ...", data["id"])
        return
    
    if os.path.exists(gen_img_save_path):
        print(f"Image already generated: {gen_img_save_path}")
    
    else:
        os.makedirs(os.path.dirname(gen_img_save_path), exist_ok=True)
        print(f"Generating image ...")
        assert inference_function is not None, f"Image {data['id']} has not been generated and inference function is not set"
        inference_function(text_prompt=data["prompt"], save_path=gen_img_save_path)
    
    gt_img_path = os.path.join(data_dir, "images", data["image_path"])
    
    eval_result = eval_single(
        gen_img_path=gen_img_save_path, 
        gt_img_path=gt_img_path, 
        t2i_prompt=data["prompt"], 
        scoring_points=[item["question"] for item in data["scoring_points"]], 
    )
    
    eval_result.update(data)
    eval_result["gen_img_path"] = gen_img_save_path
    eval_result["gt_img_path"] = gt_img_path
    del eval_result["image_path"]

    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(eval_result, f, indent=4, ensure_ascii=False)
    print(f"Saved eval results to {json_save_path}")


def _inference_and_eval_single(args):
    data, img_save_dir, eval_save_dir, data_dir, inference_function, sampled_ids = args
    return inference_and_eval_single(data, img_save_dir, eval_save_dir, data_dir, inference_function, sampled_ids)



def inference_and_eval(
    img_save_dir, 
    eval_save_dir, 
    data_dir="./data/full", 
    inference_function=call_gpt_img_gen, 
    sampled_id_path=None,
    max_workers=-1,
):   
        
    with open(os.path.join(data_dir, "annotations", "All_Subjects.jsonl"), "r") as f:
        all_data = [json.loads(line) for line in f.readlines()]
    
    if sampled_id_path is not None:
        with open(sampled_id_path) as f:
            sampled_ids = [x.strip() for x in f.readlines()]
    else:
        sampled_ids = None
    
    
    if max_workers > 0:   
        print(f"Evaluating with {max_workers} workers ...")
        args_list = [
            (data, img_save_dir, eval_save_dir, data_dir, inference_function, sampled_ids)
            for data in all_data
        ]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_map(
                _inference_and_eval_single, 
                args_list, 
                max_workers=max_workers,
                desc="Evaluating",
            )
            
    else:
        print("Evaluating without multiprocessing ...")
        for data in tqdm(all_data):
            inference_and_eval_single(data, img_save_dir, eval_save_dir, data_dir, inference_function, sampled_ids)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--img_save_dir", type=str, default="./gen_imgs")
    parser.add_argument("--eval_save_dir", type=str, default="./eval_results")
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--max_workers", type=int, default=-1)

    args = parser.parse_args()
    
    if args.run_inference:
        #! change with your own inference function that takes `text_prompt` and `save_path` as input
        inference_function = call_gpt_img_gen
    else:
        #! first generate images offline to `args.img_save_dir` before running this script
        inference_function = None
    
    inference_and_eval(
        img_save_dir=args.img_save_dir, 
        eval_save_dir=args.eval_save_dir, 
        inference_function=inference_function,
        data_dir=args.data_dir,
        sampled_id_path=os.path.join(args.data_dir, "mini_sample_ids.txt") if args.mini else None,
        max_workers=args.max_workers,
    )

