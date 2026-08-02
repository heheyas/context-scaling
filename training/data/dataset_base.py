# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import random
import json

import numpy as np
import torch

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
    resize_position_ids_by_pooling,
    double_uniform_sampling,
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .transforms import ImageTransform
from .video_utils import FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
        timestep_shift=-1,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.timestep_shift = timestep_shift


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos

        # QwenImage template: if set, text segments use the template instead of BOS/EOS,
        # and vae_image segments skip vision_start/vision_end boundary tokens.
        self.qwenimage_template_prefix_ids = None
        self.qwenimage_template_suffix_ids = None
        self.qwenimage_template_prefix_len = 0

        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def setup_qwenimage_template(self, tokenizer):
        """Configure QwenImage prompt template for text encoding.

        When enabled, pack_sequence will:
          - Wrap caption tokens with template prefix+suffix (instead of BOS/EOS)
          - Skip vision_start/vision_end for vae_image segments
          - Track condition_sample_lens (text encoder input length per sample)
        """
        TEMPLATE = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, "
            "size, texture, quantity, text, spatial relationships of the objects "
            "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        TEMPLATE_PREFIX_LEN = 34

        dummy = TEMPLATE.format("")
        all_ids = tokenizer.encode(dummy)
        self.qwenimage_template_prefix_ids = all_ids[:TEMPLATE_PREFIX_LEN]
        self.qwenimage_template_suffix_ids = all_ids[TEMPLATE_PREFIX_LEN:]
        self.qwenimage_template_prefix_len = TEMPLATE_PREFIX_LEN

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            has_num_used_data = True
            if 'num_used_data' not in dataset_args.keys():
                dataset_args['num_used_data'] = []
                has_num_used_data = False

            for item in dataset_names:
                if not has_num_used_data:
                    dataset_name = list(item.keys())[0]
                    dataset_args['num_used_data'].append(item[dataset_name][0])
                else:
                    dataset_name = item
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{dataset_name}')
                meta_info = DATASET_INFO[grouped_dataset_name][dataset_name]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)
                    
                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            caption_list                = list(),
            condition_sample_lens       = list(),
            actual_txt_indexes          = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        # pass through raw caption strings (for external text encoders like Qwen2.5-VL)
        if len(sequence_status['caption_list']) > 0:
            data['caption_list'] = sequence_status['caption_list']

        # QwenImage: per-sample text encoder input length + actual indexes
        if len(sequence_status['condition_sample_lens']) > 0:
            data['condition_sample_lens'] = sequence_status['condition_sample_lens']
            data['actual_txt_indexes'] = torch.tensor(sequence_status['actual_txt_indexes'])

        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            fail_times = 0
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            # HACK to prevent too long sequence
                            if num_tokens < self.max_num_tokens_per_sample and sequence_status['curr'] + num_tokens< self.max_num_tokens:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                fail_times+=1
                                if fail_times>5:
                                    print(f"skip the current dataset as fail_times > 5")
                                    break
                                continue
            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    ###
                    if any([d<0 for d in data['sample_lens']]):
                        print("shit!!!!!!")
                        print(data['sample_lens'])
                        print(sequence_status['curr'])
                    ###
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                ###
                if any([d<0 for d in data['sample_lens']]):
                    print("shit2!!!!!!")
                    print(data['sample_lens'])
                    print(sequence_status['curr'])
                ###
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        caption_list = sample.get('caption_list', [])
        sequence_status['caption_list'].extend(caption_list)

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                text_dropped = (item['enable_cfg'] == 1 and
                                random.random() < self.data_config.text_cond_dropout_prob)

                if self.qwenimage_template_prefix_ids is not None:
                    # QwenImage mode: [template_prefix + caption + template_suffix]
                    # When dropped (CFG), use empty caption: just template prefix + suffix
                    if text_dropped:
                        text_ids = []  # empty caption
                    prefix = self.qwenimage_template_prefix_ids
                    suffix = self.qwenimage_template_suffix_ids
                    prefix_len = self.qwenimage_template_prefix_len
                    full_ids = prefix + text_ids + suffix
                    caption_suffix_ids = text_ids + suffix  # what goes into packed sequence
                    caption_suffix_len = len(caption_suffix_ids)

                    # packed_text_ids: ALL tokens for text encoder (including prefix)
                    sequence_status['packed_text_ids'].extend(full_ids)
                    # condition_sample_lens: full length for text encoder input
                    sequence_status['condition_sample_lens'].append(len(full_ids))
                    # actual_txt_indexes: indexes into text encoder output to extract
                    # caption+suffix embeddings (skip prefix). These are offsets within
                    # the flat text encoder output tensor.
                    te_offset = sum(sequence_status['condition_sample_lens'][:-1])
                    sequence_status['actual_txt_indexes'].extend(
                        range(te_offset + prefix_len, te_offset + len(full_ids))
                    )

                    # packed_text_indexes: positions in the packed mmdit sequence
                    # (only caption+suffix, NO prefix — prefix is text-encoder-only)
                    sequence_status['packed_text_indexes'].extend(
                        range(curr, curr + caption_suffix_len)
                    )
                    curr += caption_suffix_len
                    curr_split_len += caption_suffix_len
                    # No CE loss for QwenImage text (it's conditioning, not generation)
                else:
                    # Original CausalFusion mode: [BOS] + caption + [EOS]
                    if text_dropped:
                        continue
                    shifted_text_ids = [self.bos_token_id] + text_ids
                    sequence_status['packed_text_ids'].extend(shifted_text_ids)
                    sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    if item['loss'] == 1:
                        sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                        sequence_status['ce_loss_weights'].extend(
                            [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                        )
                        sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                    curr += len(shifted_text_ids)
                    curr_split_len += len(shifted_text_ids)

                    # add a <|im_end|> token
                    sequence_status['packed_text_ids'].append(self.eos_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    if item['special_token_loss'] == 1: # <|im_end|> may have loss
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vit_patch_size
                w = W // self.data_config.vit_patch_size

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME (CD) fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                if self.qwenimage_template_prefix_ids is None:
                    # Original mode: add a <|startofimage|> token
                    sequence_status['packed_text_ids'].append(self.start_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')
                
                if self.data_config.timestep_shift == -1: # NOTE: -1 means using dynamic timestep shift
                    timestep_shift = np.sqrt(num_img_tokens/ 256) 
                    timestep = torch.sigmoid(torch.tensor(timestep))
                    timestep = timestep_shift * timestep / (1 + (timestep_shift - 1) * timestep)
                elif self.data_config.timestep_shift == -100: # NOTE: -100 means using random timestep
                    timestep = random.uniform(0, 1) if item['loss'] == 1 else 0

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                if self.qwenimage_template_prefix_ids is None:
                    # Original mode: add a <|endofimage|> token
                    sequence_status['packed_text_ids'].append(self.end_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    # <|endofimage|> may have loss
                    if item['special_token_loss'] == 1:
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                boundary_tokens = 0 if self.qwenimage_template_prefix_ids is not None else 2
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + boundary_tokens))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class PackedMaskboxDataset(PackedDataset):
    def get_watermark_loss_mask(
        watermark_boxes,
        patchified_vae_latent_shapes,
        device,
    ):
        """
        Loss mask in the watermark region will be set to 0.

        Args:
            watermark_boxes: List[List[List[float, float, float, float]]]
            patchified_vae_latent_shapes: List[Tuple[int, int]]
            device: torch.device

        Returns:
            loss_mask: torch.Tensor
        """

        num_tokens = sum([h * w for h, w in patchified_vae_latent_shapes])
        loss_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        if watermark_boxes is None or len(watermark_boxes) == 0:
            return loss_mask

        img_token_lens = torch.tensor(
            [h * w for h, w in patchified_vae_latent_shapes],
            device=device,
            dtype=torch.long,
        )
        img_token_offsets = torch.cat(
            [
                torch.tensor([0], device=device, dtype=torch.long),
                torch.cumsum(img_token_lens, 0)[:-1],
            ]
        )

        for i, boxes in enumerate(watermark_boxes):
            h, w = patchified_vae_latent_shapes[i]
            offset = img_token_offsets[i]

            # Create patch coordinates
            patch_y, patch_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            patch_x_min = patch_x.float() / w
            patch_y_min = patch_y.float() / h
            patch_x_max = (patch_x.float() + 1) / w
            patch_y_max = (patch_y.float() + 1) / h

            # Flatten patch coordinates
            patch_x_min = patch_x_min.flatten()
            patch_y_min = patch_y_min.flatten()
            patch_x_max = patch_x_max.flatten()
            patch_y_max = patch_y_max.flatten()

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                if x_min >= x_max or y_min >= y_max:
                    continue

                # Vectorized overlap check
                overlap_x = torch.max(torch.tensor(x_min, device=device), patch_x_min) < torch.min(
                    torch.tensor(x_max, device=device), patch_x_max
                )
                overlap_y = torch.max(torch.tensor(y_min, device=device), patch_y_min) < torch.min(
                    torch.tensor(y_max, device=device), patch_y_max
                )
                overlap = overlap_x & overlap_y

                # Update loss mask
                if torch.any(overlap):
                    overlapped_indices = torch.where(overlap)[0]
                    global_indices = offset + overlapped_indices
                    loss_mask[global_indices] = 0
        return loss_mask

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']
        caption_list = sample.get('caption_list', [])
        sequence_status['caption_list'].extend(caption_list)

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                text_dropped = (item['enable_cfg'] == 1 and
                                random.random() < self.data_config.text_cond_dropout_prob)

                if self.qwenimage_template_prefix_ids is not None:
                    # QwenImage mode: [template_prefix + caption + template_suffix]
                    # When dropped (CFG), use empty caption: just template prefix + suffix
                    if text_dropped:
                        text_ids = []  # empty caption
                    prefix = self.qwenimage_template_prefix_ids
                    suffix = self.qwenimage_template_suffix_ids
                    prefix_len = self.qwenimage_template_prefix_len
                    full_ids = prefix + text_ids + suffix
                    caption_suffix_ids = text_ids + suffix  # what goes into packed sequence
                    caption_suffix_len = len(caption_suffix_ids)

                    # packed_text_ids: ALL tokens for text encoder (including prefix)
                    sequence_status['packed_text_ids'].extend(full_ids)
                    # condition_sample_lens: full length for text encoder input
                    sequence_status['condition_sample_lens'].append(len(full_ids))
                    # actual_txt_indexes: indexes into text encoder output to extract
                    # caption+suffix embeddings (skip prefix). These are offsets within
                    # the flat text encoder output tensor.
                    te_offset = sum(sequence_status['condition_sample_lens'][:-1])
                    sequence_status['actual_txt_indexes'].extend(
                        range(te_offset + prefix_len, te_offset + len(full_ids))
                    )

                    # packed_text_indexes: positions in the packed mmdit sequence
                    # (only caption+suffix, NO prefix — prefix is text-encoder-only)
                    sequence_status['packed_text_indexes'].extend(
                        range(curr, curr + caption_suffix_len)
                    )
                    curr += caption_suffix_len
                    curr_split_len += caption_suffix_len
                    # No CE loss for QwenImage text (it's conditioning, not generation)
                else:
                    # Original CausalFusion mode: [BOS] + caption + [EOS]
                    if text_dropped:
                        continue
                    shifted_text_ids = [self.bos_token_id] + text_ids
                    sequence_status['packed_text_ids'].extend(shifted_text_ids)
                    sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    if item['loss'] == 1:
                        sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                        sequence_status['ce_loss_weights'].extend(
                            [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                        )
                        sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                    curr += len(shifted_text_ids)
                    curr_split_len += len(shifted_text_ids)

                    # add a <|im_end|> token
                    sequence_status['packed_text_ids'].append(self.eos_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    if item['special_token_loss'] == 1: # <|im_end|> may have loss
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vit_patch_size
                w = W // self.data_config.vit_patch_size

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME (CD) fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                if self.qwenimage_template_prefix_ids is None:
                    # Original mode: add a <|startofimage|> token
                    sequence_status['packed_text_ids'].append(self.start_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    curr += 1
                    curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                if self.qwenimage_template_prefix_ids is None:
                    # Original mode: add a <|endofimage|> token
                    sequence_status['packed_text_ids'].append(self.end_of_image)
                    sequence_status['packed_text_indexes'].append(curr)
                    # <|endofimage|> may have loss
                    if item['special_token_loss'] == 1:
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['ce_loss_weights'].append(1.0)
                        sequence_status['packed_label_ids'].append(item['special_token_label'])
                    curr += 1
                    curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                boundary_tokens = 0 if self.qwenimage_template_prefix_ids is not None else 2
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + boundary_tokens))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

        self.caption_list = data.get("caption_list", None)
        self.condition_sample_lens = data.get("condition_sample_lens", None)
        self.actual_txt_indexes = data.get("actual_txt_indexes", None)

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        if self.caption_list is not None:
            data['caption_list'] = self.caption_list

        if self.condition_sample_lens is not None:
            data['condition_sample_lens'] = self.condition_sample_lens
        if self.actual_txt_indexes is not None:
            data['actual_txt_indexes'] = self.actual_txt_indexes

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
