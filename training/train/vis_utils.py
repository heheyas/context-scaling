# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import bisect
import re
from concurrent.futures import ProcessPoolExecutor
from heapq import merge
from typing import Callable, List, Literal, Optional, Tuple
import torch
import wandb
from einops import rearrange
from transformers import PreTrainedTokenizer

words_to_remove = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
pattern = re.compile("|".join(map(re.escape, words_to_remove)))


def get_consecutive_ranges(indices: torch.Tensor):
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    device = indices.device
    diff = torch.diff(indices)
    breaks = (diff != 1).nonzero(as_tuple=True)[0]
    starts = torch.cat([torch.tensor([0], device=device), breaks + 1])
    ends = torch.cat([breaks + 1, torch.tensor([len(indices)], device=device)])
    return starts, ends


def find_consecutive_segments(indices: torch.Tensor):
    indices = indices.long()
    starts, ends = get_consecutive_ranges(indices)
    return [(indices[s].item(), indices[e - 1].item() + 1) for s, e in zip(starts, ends)]


def assign_segments(
    segments: List[Tuple[int, int]], split_len: List[int], modality: Literal["txt", "vae"]
):
    split_offsets = torch.tensor(split_len).cumsum(0).tolist()
    split_starts = [0] + split_offsets[:-1]

    relative_results = [[] for _ in split_len]

    for seg_start, seg_end in segments:
        slot_idx = bisect.bisect_right(split_offsets, seg_end - 1)

        sample_start = split_starts[slot_idx]

        if seg_start >= sample_start:
            relative_results[slot_idx].append(
                ((seg_start - sample_start, seg_end - sample_start), modality)
            )
        else:
            relative_results[slot_idx].append(((0, seg_end - sample_start), modality))

    return relative_results


def split_text_by_indices(txt_indices: torch.Tensor, text_ids):
    starts, ends = get_consecutive_ranges(txt_indices)
    text_pointer = 0
    text_chunks = []
    for s_idx, e_idx in zip(starts, ends):
        length = e_idx.item() - s_idx.item()
        chunk = text_ids[text_pointer : text_pointer + length]
        text_chunks.append(chunk)
        text_pointer += length

    assert text_pointer == len(text_ids), "Mismatch between txt_indices and text_ids length"
    return text_chunks


def merge_and_tag_intervals(txt_edpts, vae_edpts):
    tagged_txt = [(interval, "txt") for interval in txt_edpts]
    tagged_vae = [(interval, "vae") for interval in vae_edpts]
    return list(merge(tagged_txt, tagged_vae, key=lambda x: x[0]))


def encode_video(array, fps=4):
    return wandb.Video(array.transpose(0, 1).numpy(), fps=fps, format="gif")


# Image encoding
def encode_image(array):
    return wandb.Image(array.permute(1, 2, 0).cpu().numpy())


def latent_unflatten(
    indices: torch.Tensor,
    pred: torch.Tensor,
    noised: torch.Tensor,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    shapes: List[Tuple[int, int]],  # (h, w)
    p: int,
    c: int,
):
    def tensor_unflatten_image(tensor: torch.Tensor, shape: Tuple[int, int]):
        h, w = shape
        # QwenImage token layout: [C, p, q] per token (from _pack_latents)
        tensor = tensor.reshape(h, w, c, p, p)  # [h, w, C, p, q]
        tensor = tensor.permute(2, 0, 3, 1, 4).reshape(c, h * p, w * p)  # [C, H, W]
        # Add fake time dimension for 1-frame video: (C, H, W) -> (C, 1, H, W)
        return tensor.unsqueeze(1)

    latents, timesteps_pred = [], []
    for idx, (h, w) in enumerate(shapes):
        if sum(indices == idx) == 0:
            latents.append(None)
            timesteps_pred.append(None)
            continue
        latent_pred = pred[indices == idx]
        latent_noised = noised[indices == idx]
        latent_clean = clean[indices == idx]
        timestep = timesteps[indices == idx]
        assert max(timestep) == min(timestep), "Timestep Mismatch."
        assert len(latent_pred) == h * w, "Latent size Mismatch."
        latents.append(
            torch.stack(
                [
                    tensor_unflatten_image(latent_clean, (h, w)),
                    tensor_unflatten_image(latent_noised, (h, w)),
                    tensor_unflatten_image(latent_pred, (h, w)),
                ],
                dim=0,
            )
        )
        timesteps_pred.append(timestep[0].item())
    return latents, timesteps_pred


def construct_vis_table(
    token_ids: torch.Tensor,
    sample_len: List[int],
    vae_indices: torch.Tensor,
    txt_indices: torch.Tensor,
    vae_decoding: Callable,
    tokenizer: PreTrainedTokenizer,
    latent_clean: torch.Tensor,
    latent_noised: torch.Tensor,
    latent_pred: torch.Tensor,
    latent_indices: torch.Tensor,
    latent_timesteps: torch.Tensor,
    latent_shapes: List[Tuple[int, int, int]],
    patch_size: int,
    latent_channel: int,
    max_decoding: int = 15,
):
    latents, timesteps = latent_unflatten(
        indices=latent_indices,
        pred=latent_pred,
        noised=latent_noised,
        clean=latent_clean,
        timesteps=latent_timesteps,
        shapes=latent_shapes,
        p=patch_size,
        c=latent_channel,
    )

    # 1. distribute indices into samples
    vae_edpts = find_consecutive_segments(vae_indices)
    split_vae_edpts = assign_segments(vae_edpts, sample_len, modality="vae")
    txt_edpts = find_consecutive_segments(txt_indices)
    split_txt_edpts = assign_segments(txt_edpts, sample_len, modality="txt")
    all_token_segs = split_text_by_indices(txt_indices, token_ids)

    # 2. decoding
    output_images = []
    decoded_image_num = 0
    with torch.no_grad():
        for latent in latents:
            if latent is None:
                output_images.append(None)
            else:
                if decoded_image_num < max_decoding:
                    output_images.append(vae_decoding(latent.squeeze(2).float()).cpu())
                    decoded_image_num += 1
                else:
                    break

    # 3. Assemble visualization table
    vae_ptr, txt_ptr = 0, 0
    rows = []
    image_futures = []
    executor = ProcessPoolExecutor()
    cnt_vis_dec = 0
    num_good = 0
    for group_idx, (txt_edpts, vae_edpts) in enumerate(zip(split_txt_edpts, split_vae_edpts)):
        if len(vae_edpts) == 0:
            txt_ptr += len(txt_edpts)
            continue
        merged = list(merge(txt_edpts, vae_edpts, key=lambda x: x[0]))
        prefix = ""
        for interval, source in merged:
            if source == "txt":
                length = interval[1] - interval[0]
                txt = tokenizer.decode(all_token_segs[txt_ptr][-length:])
                prefix += re.sub(r"\s+", " ", re.sub(pattern, "", txt)).strip() + "\n"
                txt_ptr += 1
            else:
                out, t = map(lambda x: x[vae_ptr], (output_images, timesteps))
                if out is None:
                    print('is none')
                    vae_ptr += 1
                    continue
                else:
                    num_good += 1
                    rec, noised, pred = out.unbind(0)
                    cnt_vis_dec += 1
                row_idx = len(rows)
                rows.append(
                    [group_idx, prefix, None, None, None, t if t is not None else -1]
                )
                future_rec = executor.submit(encode_image, rec)
                future_noised = executor.submit(encode_image, noised)
                future_pred = executor.submit(encode_image, pred)
                image_futures.append((row_idx, future_rec, future_noised, future_pred))
                prefix = ""
                vae_ptr += 1
                if cnt_vis_dec >= max_decoding:
                    break
            if cnt_vis_dec >= max_decoding:
                break

    for row_idx, f_rec, f_noised, f_pred in image_futures:
        rows[row_idx][2] = f_rec.result()
        rows[row_idx][3] = f_noised.result()
        rows[row_idx][4] = f_pred.result()
    executor.shutdown()
    columns = ["group_idx", "prefix", "rec", "noised", "pred", "t"]
    table = wandb.Table(data=rows, columns=columns)
    return table
