# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pyarrow.parquet as pq
import random
import traceback
from PIL import Image

import numpy as np

from .data_utils import pil_img2rgb, split_integer_exp_decay
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_hdfs_fs
from .video_utils import decode_video_byte, sample_mp4_frames
from .json_transform import sample_data_pipeline, sample_data_pipeline_qt

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2Iv2IterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        max_aspect_ratio=2, min_size=128,
        local_rank=0, world_size=1, num_workers=8, data_status=None, reverse_sample=False,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.max_aspect_ratio = max_aspect_ratio
        self.min_size = min_size
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, reverse_sample)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, reverse_sample):
        return get_parquet_data_paths(data_dir_list, num_used_data, reverse_sample=reverse_sample)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    print(f"The image ratio {image_ratio} is too large, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                long_side = max(image.size)
                                if max(image.size) < self.min_size:
                                    print(f"The long side {long_side} is too small, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue

                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption_dict = row['caption_dict']
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            en_caps, en_caps_token, en_caps_lens = [], [], []
                            cn_caps, cn_caps_token, cn_caps_lens = [], [], []
                            for k, v in caption_dict.items():
                                if 'text' in k and v is not None and v != '' and isinstance(v, str):
                                    if '_en_' in k:
                                        en_caps.append(v)
                                        cap_tokens = self.tokenizer.encode(v)
                                        en_caps_token.append(cap_tokens)
                                        en_caps_lens.append(len(cap_tokens))
                                    elif '_cn_' in k:
                                        cn_caps.append(v)
                                        cap_tokens = self.tokenizer.encode(v)
                                        cn_caps_token.append(cap_tokens)
                                        cn_caps_lens.append(len(cap_tokens))

                            if len(cn_caps) == 0 and len(en_caps) == 0:
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                caption_token = random.choice(en_caps_token + cn_caps_token)

                            sequence_plan, text_ids_list = [], []

                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                            
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2IFluxIterableDataset(T2Iv2IterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        max_aspect_ratio=2, min_size=400,
        prefix_tag=None, reverse_sample=False, discard_category=None,
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(
            dataset_name, transform, tokenizer, data_dir_list, num_used_data,
            data_status=data_status, reverse_sample=reverse_sample,
            max_aspect_ratio=max_aspect_ratio, min_size=min_size,
            local_rank=local_rank, world_size=world_size, num_workers=num_workers, 
        )
        self.prefix_tag = prefix_tag
        self.discard_category = discard_category

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                if isinstance(row['image'], dict):
                                    image_byte = row['image']['bytes']
                                else:
                                    image_byte = row['image'][0]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    print(f"The image ratio {image_ratio} is too large, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                long_side = max(image.size)
                                if max(image.size) < self.min_size:
                                    print(f"The long side {long_side} is too small, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption_dict = row['json_data']
                                if isinstance(caption_dict, str):
                                    caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            if 'category' in caption_dict.keys() and self.discard_category is not None:
                                if caption_dict['category'] in self.discard_category:
                                    continue

                            captions = []
                            short_caption = None
                            long_caption = None
                            for k in caption_dict.keys():
                                if "caption" in k:
                                    captions.append(caption_dict[k])
                                if "short_caption" in k:
                                    short_caption = caption_dict[k]
                                if "long_caption" in k:
                                    long_caption = caption_dict[k]

                            sequence_plan, text_ids_list = [], []
                            
                            if len(captions) > 0:
                                caption = random.choice(captions)
                            else:
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption = ' '
                            if self.prefix_tag is not None:
                                caption = f'[{self.prefix_tag}] {caption}'
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            
            
class T2ISeedreamIterableDataset(T2Iv2IterableDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    print(f"The image ratio {image_ratio} is too large, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                long_side = max(image.size)
                                if max(image.size) < self.min_size:
                                    print(f"The long side {long_side} is too small, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption = row['prompt']
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sequence_plan, text_ids_list = [], []
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2VIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        min_num_frames=8, max_num_frames=16, i2v_ratio=0.1, 
        use_causalfusion=True, exp_decay=0.95, random_delta=True,
        local_rank=0, world_size=1, num_workers=8, data_status=None, reverse_sample=False,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.i2v_ratio = i2v_ratio
        self.random_delta = random_delta
        self.use_causalfusion = use_causalfusion
        self.exp_decay = exp_decay
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, reverse_sample)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, reverse_sample):
        return get_parquet_data_paths(data_dir_list, num_used_data, reverse_sample=reverse_sample)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            row = row.to_dict()
                            try:
                                video_byte = row['image']
                                if isinstance(video_byte, np.ndarray):
                                    video_byte = video_byte[0]
                                vr = decode_video_byte(video_byte)
                                n_frames = random.randint(self.min_num_frames, self.max_num_frames)
                                frames, _, frame_indices = sample_mp4_frames(
                                    vr, n_frames=n_frames, return_frame_indices=True, random_sample=self.random_delta
                                )
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            try:
                                if 'extra_info' in row:
                                    extra_info = json.loads(row['extra_info'])
                                else:
                                    extra_info = json.loads(row['vgfm_info'])
                                try:
                                    all_captions = json.loads(extra_info['caption'])
                                    caption_list = []
                                    for k, v in all_captions.items():
                                        if 'ans' in k:
                                            caption_list.append(v)
                                    if caption_list:
                                        caption = random.choice(caption_list)
                                    else:
                                        continue
                                except:
                                    if extra_info is not None and 'caption' in extra_info.keys():
                                        caption = extra_info['caption']
                                    else:
                                        print(f"no caption in rg#{row_group_id}, {parquet_file_path}")
                                        continue
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            if caption is None:
                                print(f"no caption in rg#{row_group_id}, {parquet_file_path}")
                                continue

                            text_ids = self.tokenizer.encode(caption)

                            if self.use_causalfusion:
                                _, cumsum = split_integer_exp_decay(n_frames, self.exp_decay)
                            else:
                                cumsum = range(0, 17)

                            num_tokens += len(text_ids)
                            sequence_plan = [
                                {
                                    'type': 'text', 
                                    'enable_cfg': 1, 
                                    'loss': 0, 
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                },
                            ]

                            image_tensor_list = []
                            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indices)):
                                image_tensor = self.transform(image)
                                height, width = image_tensor.shape[1:]
                                num_tokens += width * height // transform_stride ** 2
                                image_tensor_list.append(image_tensor)
                                current_sequence_plan = {
                                    'type': 'vae_image', 
                                    'enable_cfg': 0, 
                                    'loss': 1, 
                                    'special_token_loss': 0,
                                    'special_token_label': None,
                                    'split_start': idx in cumsum,
                                    'split_end': idx + 1 in cumsum,
                                }
                                if idx < len(frame_indices) - 1:
                                    current_sequence_plan['frame_delta'] = frame_indices[idx + 1] - frame_idx
                                sequence_plan.append(current_sequence_plan)

                            if random.random() < self.i2v_ratio:
                                sequence_plan[1]['loss'] = 0
                                sequence_plan[1]['split_end'] = True
                                sequence_plan[2]['split_start'] = True

                            sample = dict(
                                image_tensor_list=image_tensor_list, 
                                text_ids_list=[text_ids],
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2IJsonPromptIterableDataset(T2Iv2IterableDataset):
    def __init__(self, sampling_config_name, json_single_quote=True, **kwargs):
        super().__init__(**kwargs)
        self.sampling_config_name = sampling_config_name
        self.json_single_quote = json_single_quote
        
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['images'][0]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    print(f"The image ratio {image_ratio} is too large, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                long_side = max(image.size)
                                if max(image.size) < self.min_size:
                                    print(f"The long side {long_side} is too small, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                raw_json_caption = json.loads(row['inputs'])[-5]["text"]
                                sampled_caption = sample_data_pipeline(json.loads(raw_json_caption), sampling_config_name=self.sampling_config_name)
                                if self.json_single_quote:
                                    caption = self._compact_single_quote_json(sampled_caption)
                                else:
                                    caption = json.dumps(
                                        sampled_caption,
                                        separators=(',', ':'), # for better compression
                                        ensure_ascii=False, # Chinese readable
                                    )
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sequence_plan, text_ids_list = [], []
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

    def _compact_single_quote_json(self, data):
        # 定义占位符
        PLACEHOLDER_SINGLE = "@@SP_SINGLE_QUOTE@@"
        PLACEHOLDER_DOUBLE = "@@SP_DOUBLE_QUOTE@@"

        def protect_single_quotes(obj):
            """递归预处理：把字符串中的 ' 变成 占位符"""
            if isinstance(obj, dict):
                # 只对字符串类型的 key 进行处理，避免对复杂类型（如元组）递归导致死循环
                result = {}
                for key, value in obj.items():
                    # 如果 key 是字符串，则处理它；否则直接使用原 key
                    processed_key = protect_single_quotes(key) if isinstance(key, str) else key
                    processed_value = protect_single_quotes(value)
                    result[processed_key] = processed_value
                return result
            elif isinstance(obj, list):
                return [protect_single_quotes(item) for item in obj]
            elif isinstance(obj, str):
                # 将 ' 替换为 占位符
                return obj.replace("'", PLACEHOLDER_SINGLE).replace('"', PLACEHOLDER_DOUBLE)
            else:
                return obj

        # 1. 预处理：保护单引号（Key 和 Value 里的单引号都会被藏起来）
        # "man's clothing" -> "man@@SQ@@s clothing"
        protected_data = protect_single_quotes(data)
        
        # 2. 序列化：生成最紧凑的 JSON
        json_str = json.dumps(protected_data, separators=(',', ':'), ensure_ascii=False)
        
        # 3. 关键置换
        
        # 第一步：外层双引号变单引号
        s_step1 = json_str.replace('"', "'")
        
        # 第二步：还原占位符为
        final_str = s_step1.replace(PLACEHOLDER_SINGLE, "\\'").replace(PLACEHOLDER_DOUBLE, "\"")
        
        return final_str

class T2IJsonMixedLevelIterableDataset(T2IJsonPromptIterableDataset):
    """Like T2IJsonPromptIterableDataset but samples from multiple degradation levels
    per iteration according to a weighted distribution.

    Args:
        sampling_strategy: dict mapping level name to weight, e.g. {"l10": 0.5, "l8": 0.25, "l5": 0.25}.
                           Weights are normalized automatically.
        json_single_quote: whether to use compact single-quote JSON encoding.
    """

    def __init__(self, sampling_strategy, json_single_quote=True, **kwargs):
        # Pass a dummy sampling_config_name to parent (we override the sampling logic)
        super().__init__(sampling_config_name="l10", json_single_quote=json_single_quote, **kwargs)
        # Normalize weights
        levels = list(sampling_strategy.keys())
        weights = [sampling_strategy[l] for l in levels]
        total = sum(weights)
        self._levels = levels
        self._weights = [w / total for w in weights]
        print(f"[{self.dataset_name}] Mixed-level sampling: "
              + ", ".join(f"{l}={w:.2%}" for l, w in zip(self._levels, self._weights)))

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['images'][0]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    continue
                                if max(image.size) < self.min_size:
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                raw_json_caption = json.loads(row['inputs'])[-5]["text"]
                                parsed_json = json.loads(raw_json_caption)

                                # Sample a level according to the weighted distribution
                                level = random.choices(self._levels, weights=self._weights, k=1)[0]
                                sampled_caption = sample_data_pipeline(parsed_json, sampling_config_name=level)

                                if self.json_single_quote:
                                    caption = self._compact_single_quote_json(sampled_caption)
                                else:
                                    caption = json.dumps(
                                        sampled_caption,
                                        separators=(',', ':'),
                                        ensure_ascii=False,
                                    )
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sequence_plan, text_ids_list = [], []
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2IDensePromptIterableDataset(T2Iv2IterableDataset):
    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['images'][0]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    print(f"The image ratio {image_ratio} is too large, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue
                                long_side = max(image.size)
                                if max(image.size) < self.min_size:
                                    print(f"The long side {long_side} is too small, skip it in rg#{row_group_id}, {parquet_file_path}")
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                caption = json.loads(row['inputs'])[-5]["text"]
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sequence_plan, text_ids_list = [], []
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor], 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class T2IQTJsonMixedLevelIterableDataset(T2IJsonPromptIterableDataset):
    """Like T2IJsonPromptIterableDataset but samples from multiple degradation levels
    per iteration according to a weighted distribution.

    Args:
        sampling_strategy: dict mapping level name to weight, e.g. {"l10": 0.5, "l8": 0.25, "l5": 0.25}.
                           Weights are normalized automatically.
        json_single_quote: whether to use compact single-quote JSON encoding.
    """

    def __init__(self, sampling_strategy, json_single_quote=True, **kwargs):
        # Pass a dummy sampling_config_name to parent (we override the sampling logic)
        super().__init__(sampling_config_name="l10", json_single_quote=json_single_quote, **kwargs)
        # Normalize weights
        levels = list(sampling_strategy.keys())
        weights = [sampling_strategy[l] for l in levels]
        total = sum(weights)
        self._levels = levels
        self._weights = [w / total for w in weights]
        print(f"[{self.dataset_name}] Mixed-level sampling: "
              + ", ".join(f"{l}={w:.2%}" for l, w in zip(self._levels, self._weights)))

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}, "
            f"total_files={len(data_paths_per_worker)}, "
            f"num_files_per_rank={self.num_files_per_rank}, "
            f"total_data_paths={len(self.data_paths) if self.data_paths else 0}"
        )

        sample_count = 0
        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_hdfs_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        try:
                            df = fr.read_row_group(row_group_id).to_pandas()
                            df = df.iloc[row_start_id:]
                        except Exception as e:
                            print(e)

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row['image_bytes']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))

                                image_ratio = max(image.size) / min(image.size)
                                if image_ratio > self.max_aspect_ratio:
                                    continue
                                if max(image.size) < self.min_size:
                                    continue

                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                parsed_json = json.loads(row['json_output'])

                                # Sample a level according to the weighted distribution
                                level = random.choices(self._levels, weights=self._weights, k=1)[0]
                                sampled_caption = sample_data_pipeline_qt(parsed_json, sampling_config_name=level)

                                if self.json_single_quote:
                                    caption = self._compact_single_quote_json(sampled_caption)
                                else:
                                    caption = json.dumps(
                                        sampled_caption,
                                        separators=(',', ':'),
                                        ensure_ascii=False,
                                    )
                            except Exception as e:
                                traceback.print_exc()
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            sequence_plan, text_ids_list = [], []
                            text_ids = self.tokenizer.encode(caption)
                            num_tokens += len(text_ids)

                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            sample_count += 1
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(
                f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}, "
                f"yielded {sample_count} samples from {len(data_paths_per_worker)} files"
            )
            sample_count = 0