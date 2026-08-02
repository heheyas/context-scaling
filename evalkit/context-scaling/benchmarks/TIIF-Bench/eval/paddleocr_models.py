import os
import json
from paddleocr import PaddleOCR
from tqdm import tqdm

def extract_text_with_paddleocr(image_path, ocr_engine):
    result = ocr_engine.ocr(image_path, cls=False)
    ocr_results = []
    for idx in range(len(result)):
        res = result[idx]
    return res

def process_images(short_prompt_dir, long_prompt_dir, output_json_path):

    ocr = PaddleOCR(lang='en')

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_names = sorted(
        [f for f in os.listdir(short_prompt_dir) if os.path.splitext(f.lower())[1] in image_extensions],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    results = []

    for img_name in tqdm(image_names, desc="Processing images"):
        short_img_path = os.path.join(short_prompt_dir, img_name)
        long_img_path = os.path.join(long_prompt_dir, img_name)

        simple_results = extract_text_with_paddleocr(short_img_path, ocr)
        enhanced_results = extract_text_with_paddleocr(long_img_path, ocr)

        result_dict = {
            "image_name": img_name,
            "short_image_ocr_results": simple_results,
            "long_image_ocr_results": enhanced_results
        }
        results.append(result_dict)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"OCR results saved: {output_json_path}")

def process_all_folders(base_dir, output_base_dir):
    """Process all folders under base_dir"""

    os.makedirs(output_base_dir, exist_ok=True)

    folders = [f for f in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, f))]

    for folder in folders:
        print(f"\n Processing: {folder}")

        short_prompt_dir = os.path.join(base_dir, folder, "short_description")
        long_prompt_dir = os.path.join(base_dir, folder, "long_description")
        output_json_path = os.path.join(output_base_dir, f"ocr_results_{folder}.json")

        if not os.path.exists(short_prompt_dir) or not os.path.exists(long_prompt_dir):
            print(f"Warning: {folder} is missing either the short_description or long_description directory, skipping.")
            continue

        process_images(short_prompt_dir, long_prompt_dir, output_json_path)

if __name__ == "__main__":
    base_dir = "/mnt/data/cpfs/cfps/personal/why/methods/"
    output_base_dir="/mnt/data/cpfs/cfps/personal/why/text"

    process_all_folders(base_dir,output_base_dir)