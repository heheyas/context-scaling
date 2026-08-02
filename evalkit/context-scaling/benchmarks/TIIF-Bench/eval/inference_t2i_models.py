import os
import json
import glob
import argparse
import torch
from tqdm import tqdm



class T2IModel():
    def __init__(self, model_name, ckpt_path, height=None, width=None, guidance_scale=None, num_inference_steps=None, max_sequence_length=None):
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self._build_pipe()
        print("\nPipeline components and their configs:")
        for name, component in self.pipe.components.items():
            print(f"{name}: {getattr(component, 'config', 'No config attribute')}\n")

    def _build_pipe(self):
        self.pipe = None
    def generate_and_save(self, prompt, save_path):
        pipe_kwargs = {
            "prompt": prompt,
            "height": self.height,
            "width": self.width,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "max_sequence_length": self.max_sequence_length
        }

        # 移除值为 None 的参数
        pipe_kwargs = {key: value for key, value in pipe_kwargs.items() if value is not None}

        # 调用 pipe 并生成图像
        image = self.pipe(**pipe_kwargs).images[0]

        # 保存图像
        image.save(save_path)



class SANA15(T2IModel):
    def _build_pipe(self):
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SANA(T2IModel):
    def _build_pipe(self):
        pipe = SanaPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.bfloat16)
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SANASprint(T2IModel):
    def _build_pipe(self):
        from diffusers import SanaSprintPipeline
        pipe = SanaSprintPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SD3(T2IModel):
    def _build_pipe(self):
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(self.ckpt_path)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SDXL(T2IModel):
    def _build_pipe(self):
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SD1_5(T2IModel):
    def _build_pipe(self):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class SD2_1(T2IModel):
    def _build_pipe(self):
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.float16)
        # pipe = pipe.to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        self.pipe = pipe

class FLUX(T2IModel):
    def _build_pipe(self):
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        self.pipe = pipe

def build_t2i_model(model_name):

    if model_name == 'sana_1_5':
        return SANA15(
            model_name=model_name,
            ckpt_path='ckpts/SANA1.5_4.8B_1024px_diffusers',
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
        )
    elif model_name == 'sana':
        return SANA15(
            model_name=model_name,
            ckpt_path='ckpts/Sana_1600M_1024px_diffusers',
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
        )
    elif model_name == 'sana_sprint':
        return SANASprint(
            model_name=model_name,
            ckpt_path='ckpts/Sana_Sprint_1.6B_1024px_diffusers',
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=2,
        )
    elif model_name == 'sd35':
        return SD3(
           model_name=model_name,
            ckpt_path='ckpts/stable-diffusion-3.5-large',
            guidance_scale=3.5,
            num_inference_steps=28, 
        )
    elif model_name == 'sd3':
        return SD3(
           model_name=model_name,
            ckpt_path='ckpts/stable-diffusion-3-medium-diffusers',
            guidance_scale=7,
            num_inference_steps=28, 
        )
    elif model_name == 'sd_xl':
        return SDXL(
            model_name=model_name,
            ckpt_path='/ckpts/stable-diffusion-xl-base-1.0',
        )
    elif model_name == 'sd1_5':
        return SD1_5(
            model_name=model_name,
            ckpt_path='ckpts/stable-diffusion-v1-5',
            guidance_scale=3.5,
            num_inference_steps=28, 
        )
    elif model_name == 'sd2_1':
        return SD2_1(
            model_name=model_name,
            ckpt_path='ckpts/stable-diffusion-2-1',
            guidance_scale=3.5,
            num_inference_steps=28, 
        )
    elif model_name == 'flux1_dev':
        return FLUX(
            model_name=model_name,
            ckpt_path='ckpts/FLUX.1-dev',
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512
        )
    elif model_name == 'flux1_schnell':
        return FLUX(
            model_name=model_name,
            ckpt_path='ckpts/FLUX.1-schnell',
            height=1024,
            width=1024,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256
        )
    else:
        raise NotImplementedError



def process_jsonl_files(input_folder, output_folder, model_name, specific_file):
    # 创建模型实例
    model = build_t2i_model(model_name)

    # 获取所有jsonl文件

    if specific_file is not None:
        jsonl_files = [os.path.join(input_folder, specific_file)]
    else:
        jsonl_files = glob.glob(os.path.join(input_folder, '*.jsonl'))


    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as file:
            for idx, line in tqdm(enumerate(file)):
                data = json.loads(line)
                data_type = data['type']
                short_description = data['short_description']
                long_description = data['long_description']

                # 创建目录结构
                type_dir = os.path.join(output_folder, data_type, model_name)
                short_desc_dir = os.path.join(type_dir, 'short_description')
                long_desc_dir = os.path.join(type_dir, 'long_description')

                os.makedirs(short_desc_dir, exist_ok=True)
                os.makedirs(long_desc_dir, exist_ok=True)

                # 生成并保存图像，文件名使用行号作为索引
                short_desc_image_path = os.path.join(short_desc_dir, f"{idx}.png")
                long_desc_image_path = os.path.join(long_desc_dir, f"{idx}.png")

                # 检查文件是否存在
                if os.path.exists(short_desc_image_path):
                    print(f"File {short_desc_image_path} already exists. Skipping...")
                else:
                    model.generate_and_save(short_description, short_desc_image_path)

                if os.path.exists(long_desc_image_path):
                    print(f"File {long_desc_image_path} already exists. Skipping...")
                else:
                    model.generate_and_save(long_description, long_desc_image_path)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Generate images from JSONL files using a specified T2I model.")
    parser.add_argument('--model', required=True, help="Name of the T2I model to use.")
    parser.add_argument('--specific_file',type=str, default=None)
    parser.add_argument('--input_folder', type=str, default='data/testmini_eval_prompts', help="Path to the folder containing JSONL files.")
    parser.add_argument('--output_folder', type=str, default='./outputs', help="Path to the folder where generated images will be saved.")

    # 解析命令行参数
    args = parser.parse_args()

    # 处理 JSONL 文件
    process_jsonl_files(args.input_folder, args.output_folder, args.model, args.specific_file)

if __name__ == '__main__':
    main()
