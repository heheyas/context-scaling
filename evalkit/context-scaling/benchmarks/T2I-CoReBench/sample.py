import os
import json
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist


def seed_everything(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@torch.no_grad()
def sampling_diffusers(args, prompts, batch_size=4):

    from diffusers import StableDiffusion3Pipeline, FluxPipeline, PixArtAlphaPipeline, DiffusionPipeline

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"

    if args.model == "FLUX.1-schnell":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"guidance_scale": 0.0, "num_inference_steps": 4, "max_sequence_length": 512}
    elif args.model == "FLUX.1-dev":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"height": 1024, "width": 1024, "guidance_scale": 3.5, "num_inference_steps": 50, "max_sequence_length": 512}
    elif args.model == "FLUX.1-Krea-dev":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"height": 1024, "width": 1024, "guidance_scale": 4.5}
    elif args.model == "SD-3-Medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        kwargs = {"num_inference_steps": 28, "guidance_scale": 7.0}
    elif args.model == "SD-3.5-Medium":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"num_inference_steps": 40, "guidance_scale": 4.5}
    elif args.model == "SD-3.5-Large":
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"num_inference_steps": 28, "guidance_scale": 3.5}
    elif args.model == "PixArt-Alpha":
        pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        kwargs = {}
    elif args.model == "PixArt-Sigma":
        pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
        kwargs = {}
    elif args.model == "Qwen-Image":
        batch_size = 1
        pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        kwargs = {"width": 1328, "height": 1328, "num_inference_steps": 50, "true_cfg_scale": 4.0, "negative_prompt": " "}
    pipe.to(device)

    total_metadatas = len(prompts)
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):

        (ID, prompt) = prompts[idx]
        save_path = f"{args.output_path}/{args.model}/{'-'.join(ID.split('-')[:2])}"
        os.makedirs(save_path, exist_ok=True)

        if sum(1 for fname in os.listdir(save_path) if fname.startswith(ID)) == args.num_samples: 
            print(f"GPU {rank} skipping generation for prompt: {ID}"); continue
        else:
            print(f"[{args.model}] GPU {rank} processing prompt {idx - start + 1}/{end - start}: {ID}")

        image_list = []

        for i in range(args.num_samples // batch_size):
            tmp_image_list = pipe([prompt] * batch_size, **kwargs).images
            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample.save(os.path.join(save_path, f"{ID}-{sample_count}.png"))
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="""
        FLUX.1-schnell, FLUX.1-dev, FLUX.1-Krea-dev | SD-3-Medium, SD-3.5-Medium, SD-3.5-Large | PixArt-Alpha, PixArt-Sigma | Qwen-Image
    """)
    parser.add_argument('--gen_eval_file', type=str, help="C-MI, C-MA, C-MR, C-TR | R-LR, R-BR, R-HR, R-PR | R-GR, R-AR | R-CR, R-RR")
    parser.add_argument('--output_path', type=str, default="logs")
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    MODELS = {
        "FLUX.1-schnell"    : "black-forest-labs/FLUX.1-schnell",
        "FLUX.1-dev"        : "black-forest-labs/FLUX.1-dev",
        "FLUX.1-Krea-dev"   : "black-forest-labs/FLUX.1-Krea-dev",
        "SD-3-Medium"       : "stabilityai/stable-diffusion-3-medium-diffusers",
        "SD-3.5-Medium"     : "stabilityai/stable-diffusion-3.5-medium",
        "SD-3.5-Large"      : "stabilityai/stable-diffusion-3.5-large",
        "PixArt-Alpha"      : "PixArt-alpha/PixArt-XL-2-1024-MS",
        "PixArt-Sigma"      : "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "Qwen-Image"        : "Qwen/Qwen-Image",
    }
    args.model_path = MODELS[args.model]

    PROMPTS = []
    for task in args.gen_eval_file.split(","):
        with open(f"data/{task.strip()}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        PROMPTS.extend([[key, entry["Prompt"]] for key, entry in data.items()])

    sampling_diffusers(args, PROMPTS)
