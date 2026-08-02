# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import defaultdict
import numpy as np
import subprocess
import torch
import torch.distributed as dist


def is_hdfs_path(path: str) -> bool:
    """
    Detects whether a path is an hdfs path.
    A hdfs path must startswith "hdfs://" protocol prefix.
    """
    return path.lower().startswith("hdfs://")


def exists(path: str) -> bool:
    """
    Check whether a path exists.
    Returns True if exists, False otherwise.
    """
    if is_hdfs_path(path):
        process = subprocess.run(["hdfs", "dfs", "-test", "-e", path], capture_output=True)
        return process.returncode == 0
    return os.path.exists(path)


def _move_to_cuda(data, device=None):
    if device is None or device == torch.device("cpu"):
        device = f"cuda:{dist.get_rank() % torch.cuda.device_count()}"

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: _move_to_cuda(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [_move_to_cuda(elem, device) for elem in data]
    elif isinstance(data, tuple):
        return tuple(_move_to_cuda(elem, device) for elem in data)
    else:
        return data
        # raise TypeError("Unsupported data type")


def distributed_dict_reduce_and_log(
    stat_dict,
    wandb_log,
    prefix,
    device,
    clear=True,
    reduce_type="mean",  # "mean" or "sum"
    token_suffix="token",  # 用于mean型的第二个log名
):
    """
    stat_dict: dict, key -> tensor 或 [tensor, tensor]
        - 如果 value 是单个 tensor，则 all_reduce 后直接 log
        - 如果 value 是 [cnt, value]，则 all_reduce 后 log value/cnt
    wandb_log: dict, 用于 wandb.log
    prefix: str, log 前缀
    device: torch.device
    clear: 是否自动清零
    reduce_type: "mean"（均值）或 "sum"（直接 log 累加和）
    token_suffix: 均值型第二个 log 字段后缀
    """
    keys = list(stat_dict.keys())
    # 判断是单 tensor 还是 pair
    is_pair = isinstance(stat_dict[keys[0]], (list, tuple)) and len(stat_dict[keys[0]]) >= 2
    if is_pair:
        total_cnts = torch.stack([stat_dict[k][0] for k in keys])
        total_vals = torch.stack([stat_dict[k][1] for k in keys])
        dist.all_reduce(total_cnts, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vals, op=dist.ReduceOp.SUM)
        for i, key in enumerate(keys):
            cnt = total_cnts[i]
            val = total_vals[i]
            if reduce_type == "mean":
                if cnt > 0:
                    wandb_log[f"{prefix}/{key}"] = val / cnt
                    wandb_log[f"{prefix}_{token_suffix}/{key}"] = cnt
                else:
                    wandb_log[f"{prefix}/{key}"] = 0.0
                    wandb_log[f"{prefix}_{token_suffix}/{key}"] = 0.0
                if len(stat_dict[key]) > 2:
                    stat_dict[key][2] = stat_dict[key][2] + cnt
                    wandb_log[f"{prefix}_accum_{token_suffix}/{key}"] = stat_dict[key][2]
            elif reduce_type == "sum":
                wandb_log[f"{prefix}/{key}"] = val
            if clear:
                stat_dict[key][0] = torch.zeros_like(cnt, device=device)
                stat_dict[key][1] = torch.zeros_like(val, device=device)
    else:
        total_vals = torch.stack([stat_dict[k] for k in keys])
        dist.all_reduce(total_vals, op=dist.ReduceOp.SUM)
        for i, key in enumerate(keys):
            wandb_log[f"{prefix}/{key}"] = total_vals[i]
            if clear:
                stat_dict[key] = torch.zeros_like(total_vals[i], device=device)


def batch_log_to_wandb(log_dict):
    # 先收集所有 tensor
    keys, tensors = zip(*[(k, v) for k, v in log_dict.items() if torch.is_tensor(v)])
    stacked = torch.stack([v.detach() for v in tensors]).cpu().numpy()
    # 非 tensor 的直接保留
    out = {k: float(stacked[i]) for i, k in enumerate(keys)}
    for k, v in log_dict.items():
        if not torch.is_tensor(v):
            out[k] = v
    return out


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        root_logger = logging.getLogger()
        root_logger.handlers = [] # remove other loggers

        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/{filename}.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [
        d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


def bucket_mse_loss(loss_dict, timesteps, device):
    # 假设 loss_dict['mse'] shape: [N, ...]
    # 假设 data['packed_timesteps'] shape: [N]
    mse_per_token = loss_dict["mse"].detach().mean(-1)  # shape: [N]
    timesteps = timesteps * 1000.0  # 0~1000
    bucket_edges = np.arange(0, 1100, 100)  # [0, 100, ..., 1000]
    loss_bucket = {}
    for lower, upper in zip(bucket_edges[:-1], bucket_edges[1:]):
        bucket_name = f"{int(lower)}-{int(upper)}"
        mask = (timesteps >= lower) & (timesteps < upper)
        loss_i = mse_per_token[mask]
        if loss_i.numel() > 0:
            count_tensor = torch.tensor(loss_i.shape[0], device=device, dtype=torch.float32)
            sum_loss = loss_i.sum().to(device)
        else:
            count_tensor = torch.tensor(0, device=device, dtype=torch.float32)
            sum_loss = torch.tensor(0, device=device, dtype=torch.float32)
        loss_bucket[bucket_name] = [count_tensor, sum_loss]
    return loss_bucket


def get_name_to_idxs(token_dataset_name, indexes):
    # indexes = indexes.cpu().numpy()
    if indexes.numel() == 0:
        return {}
    name_to_idxs = defaultdict(list)
    for idx_in_indexes, token_idx in enumerate(indexes):
        name = token_dataset_name[int(token_idx)]
        name_to_idxs[name].append(idx_in_indexes)
    return name_to_idxs


def analyze_and_optimize_loss_aggregation(
    data_indexes,
    data,
    loss_dict,
    device,
    name_to_num_id,
    num_id_to_name,
    num_unique_datasets,
    dataset_mse,
    dataset_ce,
):
    """
    Analyzes and aggregates MSE and CE losses per dataset part, optimized for speed.

    Args:
        data_indexes: List of dictionaries, e.g., [{"dataset_name": "name1"}, ...].
                      Each entry corresponds to a segment of tokens.
        data: Dictionary containing:
              "sample_lens": List of integers, lengths for each segment in data_indexes.
              "mse_loss_indexes": Tensor of token indices for MSE loss calculation.
              "ce_loss_indexes": Tensor of token indices for CE loss calculation.
        loss_dict: Dictionary containing:
                   "mse": Tensor of MSE losses.
                   "ce": Tensor of CE losses.
        device: The PyTorch device (e.g., 'cuda:0' or 'cpu') where tensors should reside.

    Returns:
        (dict, dict): dataset_mse, dataset_ce dictionaries with aggregated results.
    """

    # --- Step 1: Create a tensor of numerical dataset IDs for ALL tokens in the batch ---
    # This tensor maps each token in the batch to its numerical dataset ID.

    # Efficiently get base numeric IDs for each segment in data_indexes
    # This list comprehension runs in Python. If len(data_indexes) is huge, this could be slow.
    # Typically, len(data_indexes) is the number of "samples" or "sequences" in a batch,
    # which is usually manageable.
    try:
        base_numeric_ids_list = [name_to_num_id[di["dataset_name"]] for di in data_indexes]
    except KeyError as e:
        raise ValueError(
            f"Dataset name '{e.args[0]}' from data_indexes not found in dataset_config. "
            "Ensure all dataset names are pre-registered in dataset_config.grouped_datasets."
        ) from e

    base_numeric_ids = torch.tensor(
        base_numeric_ids_list,
        dtype=torch.long,
        device=device,  # Create on target device if possible, or move later
    )

    # Get sample lengths as a tensor
    if len(data["sample_lens"]) != len(data_indexes):
        sample_lens = data["sample_lens"][:-1]
    else:
        sample_lens = data["sample_lens"]
    sample_lens_tensor = torch.tensor(
        sample_lens, dtype=torch.long, device=device  # Match device of base_numeric_ids
    )

    # Expand base_numeric_ids according to sample_lens to get IDs for all tokens
    # numeric_ids_for_all_tokens will have shape [total_num_tokens_in_batch]
    numeric_ids_for_all_tokens = base_numeric_ids.repeat_interleave(sample_lens_tensor)

    # --- Step 2: Process MSE Loss ---
    if 'mse' in loss_dict and loss_dict["mse"] is not None:
        mse_loss_values = loss_dict["mse"].detach().mean(-1)  # Shape: [N_mse_tokens]
        # Ensure mse_loss_indexes is LongTensor for indexing
        mse_loss_indices = data["mse_loss_indexes"].long()  # Shape: [N_mse_tokens]

        if mse_loss_indices.numel() > 0:
            # Get numerical dataset IDs for the specific tokens that have an MSE loss
            # These are indices into numeric_ids_for_all_tokens
            ids_for_mse_tokens = numeric_ids_for_all_tokens[
                mse_loss_indices
            ]  # Shape: [N_mse_tokens]

            # Initialize tensors for summing losses and counting occurrences per dataset ID
            sum_mse_losses_per_id = torch.zeros(
                num_unique_datasets, device=device, dtype=torch.float32
            )
            count_mse_per_id = torch.zeros(num_unique_datasets, device=device, dtype=torch.float32)

            # Use scatter_add_ to sum losses for each dataset ID
            # ids_for_mse_tokens provides the indices for scattering, mse_loss_values are the source values
            sum_mse_losses_per_id.scatter_add_(0, ids_for_mse_tokens, mse_loss_values)

            # Use scatter_add_ to count occurrences for each dataset ID
            ones_for_mse_counting = torch.ones_like(
                mse_loss_values
            )  # Or torch.ones(mse_loss_values.size(0), device=device)
            count_mse_per_id.scatter_add_(0, ids_for_mse_tokens, ones_for_mse_counting)

            # # Calculate mean loss per ID, avoiding division by zero
            valid_ids_mask_mse = count_mse_per_id > 0
            # mean_mse_loss_per_id = torch.zeros_like(sum_mse_losses_per_id)
            # mean_mse_loss_per_id[valid_ids_mask_mse] = (
            #     sum_mse_losses_per_id[valid_ids_mask_mse] / count_mse_per_id[valid_ids_mask_mse]
            # )

            # Update dataset_mse dictionary
            for i in range(num_unique_datasets):
                if valid_ids_mask_mse[i]:  # Check if this dataset ID had any MSE losses
                    part_name = num_id_to_name[i]
                    # add cnt
                    dataset_mse[part_name][0] = count_mse_per_id[i]
                    # Add the sum of loss for this part
                    dataset_mse[part_name][1] = sum_mse_losses_per_id[i]

    # --- Step 3: Process CE Loss (similarly) ---
    if 'ce' in loss_dict and loss_dict["ce"] is not None:
        ce_loss_values = loss_dict["ce"].detach()
        ce_loss_indices = data["ce_loss_indexes"].long()

        if ce_loss_indices.numel() > 0:
            ids_for_ce_tokens = numeric_ids_for_all_tokens[ce_loss_indices]

            sum_ce_losses_per_id = torch.zeros(
                num_unique_datasets, device=device, dtype=torch.float32
            )
            count_ce_per_id = torch.zeros(num_unique_datasets, device=device, dtype=torch.float32)

            sum_ce_losses_per_id.scatter_add_(0, ids_for_ce_tokens, ce_loss_values)
            ones_for_ce_counting = torch.ones_like(ce_loss_values)
            count_ce_per_id.scatter_add_(0, ids_for_ce_tokens, ones_for_ce_counting)

            valid_ids_mask_ce = count_ce_per_id > 0
            # mean_ce_loss_per_id = torch.zeros_like(sum_ce_losses_per_id)
            # mean_ce_loss_per_id[valid_ids_mask_ce] = (
            #     sum_ce_losses_per_id[valid_ids_mask_ce] / count_ce_per_id[valid_ids_mask_ce]
            # )

            for i in range(num_unique_datasets):
                if valid_ids_mask_ce[i]:
                    part_name = num_id_to_name[i]
                    dataset_ce[part_name][0] = count_ce_per_id[i]
                    dataset_ce[part_name][1] = sum_ce_losses_per_id[i]

    return dataset_mse, dataset_ce