import megfile
import pandas as pd
from PIL import Image

def create_image_gallery(images, rows=2, cols=2):
    assert len(images) >= rows * cols, "Not enough images provided!"

    img_height, img_width = images[0].size

    # Create a blank image as the gallery background
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))

    # Paste each image onto the gallery canvas
    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]  # Convert numpy array to PIL image
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))

    return gallery_image

# category to subfolder name
class_item = {
    "Anime_Stylization" : "anime",
    "Portrait" : "human",
    "General_Object" : "object",
    "Text_Rendering" : "text",
    "Knowledge_Reasoning" : "reasoning",
    "Multilingualism" : "multilingualism"
}

image_dir = "images"
model_name = "xxx"

# you can choose the language mode here.
df = pd.read_csv("OneIG-Bench.csv", dtype=str)
# df = pd.read_csv("OneIG-Bench-ZH.csv", dtype=str)

# you can change the grid here. (2, 2), (1, 4), (3, 3) ...
grid = (2, 2)


for idx, row in df.iterrows():
    
    # You can select the desired category for image generation by adding "if row['cateogory'] in { , , }"
    prompt = row['prompt_en']
    # prompt = row['prompt_cn']
    
    images = []
    # inference
    from inference import inference
    for cnt in range(grid[0] * grid[1]):
        image = inference(prompt)
        # image is suggested to save as PIL format.
        images.append(image)
    
    # If the number of generated images is insufficient, fill the remaining slots with black images.
    total_slots = grid[0] * grid[1]
    if len(images) == 0:
        # If there are no images at all, fill with black images of size 1024x1024.
        black_img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        images.extend([black_img] * total_slots)
    elif len(images) < total_slots:
        # If there are some images but not enough, fill with black images using the size of the first image.
        img_w, img_h = images[0].size
        black_img = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
        images.extend([black_img] * (total_slots - len(images)))

    image_gallery = create_image_gallery(images, grid[0], grid[1])
    
    file_path = megfile.smart_path_join(image_dir, class_item[row['category']], model_name, f"{row['id']}.webp")
    with megfile.smart_open(file_path, "wb") as f:
        image_gallery.save(f)  