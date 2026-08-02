# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import gc
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.causalfusion_navit.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.causalfusion_navit.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.causalfusion_navit.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer
from modeling.causalfusion_navit.siglip2_navit import Siglip2EncoderLayer, Siglip2VisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
                Siglip2EncoderLayer,
                Siglip2VisionTransformer,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE (kc): position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                if key in model_state_dict:
                    model_state_dict.pop(key)
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE (kc): position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                    if key in ema_state_dict:
                        ema_state_dict.pop(key)
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state_old(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status
    
    @staticmethod
    def fsdp_save_fsdp_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                # save_state_dict 自动按分片保存
                ema_writer = FileSystemWriter(os.path.join(save_path, "ema"))
                save_state_dict(ema_state_dict, ema_writer)
                del ema_state_dict
                gc.collect()
                torch.cuda.empty_cache()

        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)
        ):
            model_state_dict = model.state_dict()
            # save_state_dict 自动按分片保存
            model_writer = FileSystemWriter(os.path.join(save_path, "model"))
            save_state_dict(model_state_dict, model_writer)
            del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        # if dist.get_rank() == 0 and data_status is not None:
        #     torch.save(data_status, os.path.join(save_path, "data_status.pt"))
        if data_status is not None:
            os.makedirs(os.path.join(save_path, "data_status"), exist_ok=True)
            torch.save(
                data_status, os.path.join(save_path, "data_status", f"rank{dist.get_rank()}.pt")
            )
            del data_status
            gc.collect()
            torch.cuda.empty_cache()

        dist.barrier()
        return
    
    @staticmethod
    def try_load_fsdp_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_load_dir = os.path.join(resume_from, "ema")
            else:
                model_load_dir = os.path.join(resume_from, "model")

            assert isinstance(model, FSDP)
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),  # 关键参数
            ):
                model_state_dict = model.state_dict()
                model_reader = FileSystemReader(model_load_dir)
                load_state_dict(model_state_dict, model_reader)
                # 删除不需要的 key，避免 KeyError
                for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                    if key in model_state_dict:
                        model_state_dict.pop(key)
                msg = model.load_state_dict(model_state_dict, strict=False)
                logger.info(msg)
                del model_state_dict
                gc.collect()
                torch.cuda.empty_cache()

            if ema_model is not None:
                ema_load_dir = os.path.join(resume_from, "ema")
                assert isinstance(ema_model, FSDP)
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),  # 关键参数
                ):
                    ema_state_dict = ema_model.state_dict()
                    ema_reader = FileSystemReader(ema_load_dir)
                    load_state_dict(ema_state_dict, ema_reader)
                    # 删除不需要的 key，避免 KeyError
                    for key in ["latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"]:
                        if key in ema_state_dict:
                            ema_state_dict.pop(key)
                    msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                    logger.info(msg)
                    del ema_state_dict
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            logger.info("Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(
                optimizer_state_dict_path, map_location="cpu", weights_only=True
            )
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(
                scheduler_state_dict_path, weights_only=True, map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            # torch.save(data_status, os.path.join(save_path, "data_status", f'rank{dist.get_rank()}.pt'))
            data_status_path = os.path.join(
                resume_from, "data_status", f"rank{dist.get_rank()}.pt"
            )
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                # local_rank = dist.get_rank()
                # if local_rank < len(data_status):
                #     data_status = data_status[local_rank]
                # else:
                #     data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status

def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer,
        Siglip2EncoderLayer, 
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999, _logged=[False]):
    """In-place EMA update: ``ema = decay * ema + (1 - decay) * model``.

    Math runs in fp32 even when one side is bf16, because for decay close to 1
    the increment ``(1-decay) * model`` is small relative to ``ema`` and would
    be silently rounded to zero in bf16, causing EMA to monotonically decay
    toward 0 instead of tracking the model.
    """
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles), (
        f"EMA handle count mismatch: ema={len(ema_handles)}, model={len(new_handles)}"
    )
    ema_params_native = []   # original-dtype tensors we eventually write back to
    ema_params_fp32 = []     # fp32 copies for the actual mul/add
    new_params_fp32 = []

    skipped_none = 0
    skipped_frozen = 0
    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is None:
            skipped_none += 1
            continue
        if not new_handle.flat_param.requires_grad:
            skipped_frozen += 1
            continue
        ema_t = ema_handle.flat_param.data
        ema_params_native.append(ema_t)
        ema_params_fp32.append(ema_t.float() if ema_t.dtype != torch.float32 else ema_t)
        # Move model params to same device as ema (handles cpu_offload ema)
        # and upcast to fp32 for the additive blend.
        new_params_fp32.append(
            new_handle.flat_param.data.to(
                device=ema_t.device, dtype=torch.float32,
            )
        )

    if not _logged[0]:
        total_ema_numel = sum(p.numel() for p in ema_params_native)
        print(
            f"[EMA] handles: {len(ema_handles)}, "
            f"updating: {len(ema_params_native)}, "
            f"skipped_none: {skipped_none}, skipped_frozen: {skipped_frozen}, "
            f"total_params: {total_ema_numel}, decay: {decay}"
        )
        if len(ema_params_native) == 0:
            print("[EMA] WARNING: no parameters to update! EMA will NOT change.")
        _logged[0] = True

    if len(ema_params_fp32) > 0:
        torch._foreach_mul_(ema_params_fp32, decay)
        torch._foreach_add_(ema_params_fp32, new_params_fp32, alpha=1 - decay)

        # If the native dtype differs from fp32 (legacy bf16 EMA setup), copy back.
        for native_t, fp32_t in zip(ema_params_native, ema_params_fp32):
            if native_t is not fp32_t:
                native_t.copy_(fp32_t)


