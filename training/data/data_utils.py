# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import or_masks, and_masks

import re

# 定义所有提示模板列表（与之前相同）
prompt_candidates = [
    "请依据所提供的文本描述，创作出一幅与之对应的图像。",
    "麻烦根据给定的文本内容，绘制出一幅能够准确呈现其描述的图像。",
    "请按照文本中的描述信息，生成一幅符合要求的图像。",
    "依据所给的文本描述，进行图像创作，生成一幅与之契合的图像。",
    "请根据文本叙述的内容，绘制出一幅可以体现其内涵的图像。",
    "按照文本里的描述，生成一幅与之匹配的、形象生动的图像。",
    "请根据具体的文本描述，制作一幅相应的图像来展现其内容。",
    "麻烦根据文本的详细描述，生成一幅能够反映其特征的图像。",
    "依据文本给出的描述信息，生成一幅精美且准确的图像。",
    "请根据所提供的文本详细描述，绘制出一幅与之相符的高质量图像。",
    "Generate an image based on the textual description provided.",
    "Create an image that corresponds to the given text description.",
    "Produce an image according to the information described in the text.",
    "Develop an image that accurately represents the textual description.",
    "Render an image based on the details described in the text.",
    "Craft an image in line with the text's narrative and description.",
    "Formulate an image that visually interprets the provided text description.",
    "Generate a visual representation of the content described in the text.",
    "Construct an image that matches the textual description precisely.",
    "Create a picture that captures the essence of the text's description.",
]

instruction_templates_for_diff = [
    "I need you to modify the image in the following way: {}",
    "Please edit the picture according to this instruction: {}",
    "Help me modify the image as described: {}",
    "Adjust the image based on the following description: {}",
    "Please make the following changes to the picture: {}",
    "Can you edit the image with this requirement: {}",
    "I want you to transform the image like this: {}",
    "Please process the image according to: {}",
    "Could you modify the picture by doing: {}",
    "I'd like you to alter the image as follows: {}",
    "Please apply these modifications to the image: {}",
    "Help me transform the picture with this change: {}",
    "I need the image to be edited in this manner: {}",
    "Please perform the following image editing: {}",
    "Could you implement this image modification: {}"
]

instruction_templates_for_diff_cn = [
    "我需要你按照如下的方式修改图像：{}",
    "请按照以下要求对图片进行编辑：{}",
    "帮我按照这个指令修改图片：{}",
    "根据以下描述来调整图像：{}",
    "请对图片做如下修改：{}",
    "按照下面的描述对图像进行处理：{}",
    "我想要你对这张图片做出以下改变：{}",
    "请你帮忙实现这样的图像编辑效果：{}",
    "能否按照这个要求来修改图片：{}",
    "希望你能够这样调整图像：{}",
    "麻烦按照以下指示编辑图片：{}",
    "请实现如下的图像变换：{}",
    "我希望对图片进行这样的修改：{}",
    "能帮我按这个描述改图吗：{}",
    "请按照这个方案修改图像：{}"
]

instruction_templates_for_image1 = [
    "Please generate an image of {}",
    "Create an image showing {}",
    "I want an image of {}",
    "Generate a picture of {}",
    "Make an image depicting {}",
    "I need you to produce an image featuring {}",
    "Could you create a visual representation of {}",
    "Please design an image that shows {}",
    "I'd like to get a picture of {}",
    "Help me generate a visual of {}",
    "Can you make a picture displaying {}",
    "I want you to create an artwork of {}",
    "Please produce a visual showing {}",
    "Could you generate an illustration of {}",
    "I need an image that presents {}",
    "Please craft a picture of {}",
    "Generate a visual representation of {}"
]

instruction_templates_for_image1_cn = [
    "帮我生成一张{}图片",
    "请为我创建一张{}的图像",
    "我想要一张{}的图片",
    "生成一个{}的图像",
    "制作一张{}的图片",
    "请生成一幅{}的画面",
    "我需要一张关于{}的图像",
    "能否为我制作{}的图片",
    "希望得到一张{}的图像",
    "想要获得一个{}的视觉效果",
    "麻烦帮我创作一张{}的图片",
    "请设计一张展示{}的图像",
    "我希望看到一张{}的图片",
    "能生成一个{}的图像吗",
    "请制造一张{}的画面",
    "想要一个呈现{}的图像",
    "帮忙创造一张{}的图片"
]

def remove_prompts_inner(text: str, all_templates) -> str:
    for tpl in all_templates:
        has_placeholder = '{}' in tpl

        # Build regex pattern depending on whether the placeholder is at the end
        if has_placeholder:
            if tpl.endswith('{}'):
                # Variable goes to the first punctuation (Chinese/English)
                prefix = re.escape(tpl[:-2])
                pattern = prefix + r'([^。.!?]+)' + r'([。.!?])?'
            else:
                # Placeholder has suffix; match the minimal content before suffix
                pattern = re.escape(tpl).replace(re.escape('{}'), r'(.+?)') + r'([。.!?])?'
        else:
            # No placeholder; match entire template and keep only trailing punctuation
            pattern = re.escape(tpl) + r'([。.!?])?'

        regex = re.compile(pattern)
        while True:
            m = regex.search(text)
            if not m:
                break

            start, end = m.span()
            if has_placeholder:
                if tpl.endswith('{}'):
                    inner = m.group(1)
                    punct = m.group(2) or ''
                    replacement = inner + punct
                else:
                    inner = m.group(1)
                    punct = m.group(2) or ''
                    replacement = inner + punct
            else:
                # Remove whole template; keep punctuation if captured
                replacement = m.group(1) or ''

            text = text[:start] + replacement + text[end:]
        # early stop if found match
        if m:
            break

    return text.strip()

def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def resize_position_ids_by_pooling(pos_ids, num_patches_h, num_patches_w, pooling_size):
    pos_ids_4d = pos_ids.view(num_patches_h, num_patches_w).unsqueeze(0).unsqueeze(0).float()
    pooled_pos_ids_4d = F.avg_pool2d(
        pos_ids_4d,
        kernel_size=pooling_size,
        stride=pooling_size,
        ceil_mode=False
    )
    pooled_pos_ids = pooled_pos_ids_4d.flatten().round().long()
    return pooled_pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def double_uniform_sampling(mean):
    if not (0 < mean < 1):
        raise ValueError("mean should be in (0, 1)")
    m = mean
    p = 1 - mean
    if np.random.random() < p:
        return np.random.uniform(0, m)
    else:
        return np.random.uniform(m, 1)
