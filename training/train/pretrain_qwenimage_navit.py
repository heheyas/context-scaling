# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import gc
import json
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from train.vis_utils import construct_vis_table
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed, AutoTokenizer
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.qwenimage.model import CausalFusionQwenImage, CausalFusionQwenImageConfig
from modeling.qwenimage.transformer import QwenImageTransformerBlock, QwenImageTransformer2DModel
from modeling.qwenimage.text_encoder_navit import PackedQwen2TextLayer, PackedQwen2TextEncoder
from modeling.causalfusion_navit.modeling_utils import TimestepEmbedder, PositionEmbedding
from train.train_utils import (
    exists,
    _move_to_cuda,
    analyze_and_optimize_loss_aggregation,
    batch_log_to_wandb,
    bucket_mse_loss,
    create_logger,
    distributed_dict_reduce_and_log,
    get_latest_ckpt,
)
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_ema_setup, fsdp_ema_update,
)



# ──────────────────────────────────────────────────────────────────────────────
# FSDP wrappers (QwenImage-specific)
# ──────────────────────────────────────────────────────────────────────────────

def _qwenimage_fsdp_wrapper(original_model, fsdp_config, ignored_modules=[], use_orig_params=False):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        device_mesh = None

    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                QwenImageTransformerBlock,
                PackedQwen2TextLayer,
                QwenImageTransformer2DModel,
                PackedQwen2TextEncoder,
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
        use_orig_params=use_orig_params,
    )


def _qwenimage_grad_checkpoint_check_fn(module):
    return isinstance(module, QwenImageTransformerBlock)


# ──────────────────────────────────────────────────────────────────────────────
# QwenImage VAE loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_qwenimage_vae(vae_path: str):
    """Load QwenImage VAE from safetensors + config.json.  Returns (model, config_dict)."""
    from modeling.qwenimage.vae import AutoencoderKLQwenImage
    from safetensors.torch import load_file

    config_path = os.path.join(vae_path, "config.json")
    with open(config_path, "r") as f:
        vae_config = json.load(f)

    vae = AutoencoderKLQwenImage(
        base_dim=vae_config.get("base_dim", 96),
        z_dim=vae_config.get("z_dim", 16),
        dim_mult=vae_config.get("dim_mult", [1, 2, 4, 4]),
        num_res_blocks=vae_config.get("num_res_blocks", 2),
        attn_scales=vae_config.get("attn_scales", []),
        temperal_downsample=vae_config.get("temperal_downsample", [False, True, True]),
        dropout=vae_config.get("dropout", 0.0),
        latents_mean=vae_config.get("latents_mean"),
        latents_std=vae_config.get("latents_std"),
    )
    sf = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
    state = load_file(sf, device="cpu")
    vae.load_state_dict(state, strict=False)
    vae.config_dict = vae_config
    return vae, vae_config


def load_qwenimage_transformer(dit_path: str):
    """Load QwenImage transformer from safetensors + config.json."""
    from safetensors.torch import load_file
    import glob as _glob

    config_path = os.path.join(dit_path, "config.json")
    with open(config_path, "r") as f:
        dit_config = json.load(f)

    # Collect all safetensors shards
    shard_files = sorted(_glob.glob(os.path.join(dit_path, "*.safetensors")))
    state_dict = {}
    for sf in shard_files:
        state_dict.update(load_file(sf, device="cpu"))

    return dit_config, state_dict


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    dit_path:                   str = field(
        default="<WEIGHTS_ROOT>/qwenimage/origin/raw_data/transformer"
    )
    text_encoder_path:          str = field(
        default="<WEIGHTS_ROOT>/qwenimage/origin/raw_data/text_encoder"
    )
    vae_path:                   str = field(
        default="<WEIGHTS_ROOT>/qwenimage/origin/raw_data/vae"
    )
    tokenizer_path:             str = field(
        default="<WEIGHTS_ROOT>/qwenimage/origin/raw_data/tokenizer"
    )

    max_latent_size:            int = field(default=32)
    latent_patch_size:          int = field(default=2)

    text_cond_dropout_prob:     float = field(default=0.1)
    vae_cond_dropout_prob:      float = field(default=0.3)


@dataclass
class DataArguments:
    dataset_config_file:        str = field(default="data/configs/sft/t2i_sanity_check.yaml")
    prefetch_factor:            int = field(default=2)
    num_workers:                int = field(default=4)
    max_num_tokens_per_sample:  int = field(default=16384)
    max_num_tokens:             int = field(default=36864)
    prefer_buffer_before:       int = field(default=16384)
    max_buffer_size:            int = field(default=50)
    data_seed:                  int = field(default=42)


@dataclass
class TrainingArguments:
    visual_gen:                 bool = field(default=True)
    visual_und:                 bool = field(default=False)

    results_dir:                str = field(default="results")
    checkpoint_dir:             str = field(default="results/checkpoints")
    wandb_project:              str = field(default="context-scaling-qwenimage")
    wandb_name:                 str = field(default="run")
    wandb_runid:                str = field(default="trial")
    wandb_resume:               str = field(default="allow")
    wandb_offline:              bool = field(default=False)
    global_seed:                int = field(default=4396)
    auto_resume:                bool = field(default=False)
    resume_from:                str = field(default=None)
    resume_model_only:          bool = field(default=False)
    finetune_from_ema:          bool = field(default=False)
    ignore_ema:                 bool = field(default=False)
    log_every:                  int = field(default=10)
    save_every:                 int = field(default=2000)
    total_steps:                int = field(default=500_000)

    warmup_steps:               int = field(default=2000)
    lr_scheduler:               str = field(default="constant")
    lr:                         float = field(default=1e-4)
    min_lr:                     float = field(default=1e-7)
    beta1:                      float = field(default=0.9)
    beta2:                      float = field(default=0.95)
    eps:                        float = field(default=1e-15)
    ema:                        float = field(default=0.9999)
    max_grad_norm:              int = field(default=1.0)
    timestep_shift:             float = field(default=1.0)
    mse_weight:                 float = field(default=1.0)
    ce_weight:                  float = field(default=1.0)
    ce_loss_reweighting:        bool = field(default=False)
    expected_num_tokens:        int = field(default=32768)

    num_replicate:              int = field(default=1)
    num_shard:                  int = field(default=8)
    sharding_strategy:          str = field(default="HYBRID_SHARD")
    backward_prefetch:          str = field(default="BACKWARD_PRE")
    cpu_offload:                bool = field(default=False)

    freeze_vae:                 bool = field(default=True)
    freeze_text_encoder:        bool = field(default=True)
    copy_init_moe:              bool = field(default=True)
    use_flex:                   bool = field(default=False)
    manual_load:                str = field(default=None)
    vis_every:                  int = field(default=250)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Logging ──────────────────────────────────────────────────────────
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # ── Auto resume logic ────────────────────────────────────────────────
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # ── Seed ─────────────────────────────────────────────────────────────
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # ── Load text encoder (NaviT PackedQwen2TextEncoder with flash_attn_varlen) ──
    logger.info(f"Loading NaviT text encoder from {model_args.text_encoder_path}")
    from modeling.qwenimage.text_encoder_navit import PackedQwen2TextEncoder
    text_encoder = PackedQwen2TextEncoder.from_pretrained(
        model_args.text_encoder_path, dtype=torch.bfloat16,
    )
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    logger.info("NaviT text encoder loaded and frozen.")

    # ── Setup tokenizer for the packing system ───────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    # QwenImage does not use LLM text token embeddings; skip resize_token_embeddings.

    # ── Load QwenImage transformer (DiT) ─────────────────────────────────
    logger.info(f"Loading QwenImage transformer from {model_args.dit_path}")
    dit_config, dit_state_dict = load_qwenimage_transformer(model_args.dit_path)
    model = CausalFusionQwenImage(dit_config, dit_state_dict, text_encoder=text_encoder)
    del dit_state_dict
    logger.info("QwenImage transformer loaded.")

    # ── Load QwenImage VAE ───────────────────────────────────────────────
    if training_args.visual_gen:
        logger.info(f"Loading QwenImage VAE from {model_args.vae_path}")
        vae_model, vae_config = load_qwenimage_vae(model_args.vae_path)
        logger.info("QwenImage VAE loaded.")

    # ── Freeze VAE ───────────────────────────────────────────────────────
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False

    # ── FSDP and checkpoint loading ──────────────────────────────────────
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    if training_args.manual_load is not None:
        # Load model weights first (without EMA to save memory)
        model, _ = FSDPCheckpoint.try_load_ckpt(
            training_args.manual_load, logger, model, None, resume_from_ema=finetune_from_ema
        )
    # Create EMA before FSDP wrapping (deepcopy must happen on raw modules).
    if training_args.ema > 0:
        ema_model = deepcopy(model)
    else:
        ema_model = None
    # Ensure uniform dtype before FSDP wrapping
    model = model.to(torch.bfloat16)
    # FSDP-wrap model
    fsdp_model = _qwenimage_fsdp_wrapper(model, fsdp_config, use_orig_params=True)
    # FSDP-wrap ema with cpu_offload — params stay on CPU, zero GPU memory for ema.
    # Uses the same auto_wrap_policy as model so FSDP handles align for fsdp_ema_update.
    if ema_model is not None:
        ema_model = ema_model.to(torch.bfloat16)
        for param in ema_model.parameters():
            param.requires_grad = False
        ema_fsdp_config = FSDPConfig(
            sharding_strategy=fsdp_config.sharding_strategy,
            backward_prefetch=fsdp_config.backward_prefetch,
            cpu_offload=True,
            num_replicate=fsdp_config.num_replicate,
            num_shard=fsdp_config.num_shard,
        )
        ema_model = _qwenimage_fsdp_wrapper(ema_model, ema_fsdp_config, use_orig_params=True)
    # New resume with FSDP
    if training_args.manual_load is None:
        if training_args.ignore_ema and ema_model is not None:
            # Load model weights only, skip EMA checkpoint.
            # After model is loaded, EMA will be re-initialized from model weights below.
            logger.info("ignore_ema=True: loading model only, will re-init EMA from model weights.")
            model, _ = FSDPCheckpoint.try_load_fsdp_ckpt(
                resume_from, logger, fsdp_model, None, resume_from_ema=finetune_from_ema
            )
        else:
            model, ema_model = FSDPCheckpoint.try_load_fsdp_ckpt(
                resume_from, logger, fsdp_model, ema_model, resume_from_ema=finetune_from_ema
            )
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=_qwenimage_grad_checkpoint_check_fn,
    )

    # Re-init EMA from loaded model weights when ignore_ema=True
    if training_args.ignore_ema and ema_model is not None:
        logger.info("Re-initializing EMA from current model weights (ignore_ema=True).")
        fsdp_ema_update(ema_model, fsdp_model, decay=0.0)  # decay=0 → ema = model

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # ── Optimizer and scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    # ── Maybe resume optimizer / scheduler / train_steps ─────────────────
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config,
        )

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    # visual_gen=True, visual_und=False: no VIT processing
    if training_args.visual_gen:
        # QwenImage VAE downsamples by 8
        vae_image_downsample = model_args.latent_patch_size * 8
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = 0.0  # no VIT

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=False,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    # Enable QwenImage template: text segments become [template_prefix+caption+suffix]
    # instead of [BOS+caption+EOS], and vae_image segments skip vision_start/end.
    train_dataset.setup_qwenimage_template(tokenizer)
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # ── Prepare models for training ──────────────────────────────────────
    if training_args.visual_gen:
        vae_model.to(device).eval()
    # text_encoder is inside FSDP (use_orig_params=True handles frozen params)
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # ── Train loop ───────────────────────────────────────────────────────
    gc.disable()
    gc.collect()
    start_time = time()

    # Loss recording
    all_dataset_names_sorted = sorted(list(dataset_config.grouped_datasets.keys()))
    name_to_num_id = {name: i for i, name in enumerate(all_dataset_names_sorted)}
    num_id_to_name = {i: name for name, i in name_to_num_id.items()}
    num_unique_datasets = len(all_dataset_names_sorted)
    dataset_num_sample = {
        name: torch.tensor(0.0, device=device, dtype=torch.float32)
        for name in all_dataset_names_sorted
    }
    dataset_mse = {
        name: [
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device, dtype=torch.float32),
        ]
        for name in all_dataset_names_sorted
    }
    dataset_ce = {
        name: [
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device, dtype=torch.float32),
        ]
        for name in all_dataset_names_sorted
    }

    # Try to resume loss tracking from the step checkpoint directory
    _loss_resume_dir = resume_from if resume_from is not None else training_args.checkpoint_dir
    if exists(os.path.join(_loss_resume_dir, "dataset_mse.pt")):
        logger.info(f"Resuming dataset mse from {_loss_resume_dir}")
        resume_dataset_mse = torch.load(
            os.path.join(_loss_resume_dir, "dataset_mse.pt"), map_location="cpu"
        )
        for key, value in resume_dataset_mse.items():
            dataset_mse[key] = _move_to_cuda(value, device)
    if exists(os.path.join(_loss_resume_dir, "dataset_ce.pt")):
        logger.info(f"Resuming dataset ce from {_loss_resume_dir}")
        resume_dataset_ce = torch.load(
            os.path.join(_loss_resume_dir, "dataset_ce.pt"), map_location="cpu"
        )
        for key, value in resume_dataset_ce.items():
            dataset_ce[key] = _move_to_cuda(value, device)

    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    for curr_step, data in enumerate(train_loader, start=train_step):
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        data.pop('caption_list', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        # condition_sample_lens stays in data — passed to model.forward()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                # ── QwenImage VAE encoding (5D input) ────────────────
                images = data.pop("padded_images")  # [N, C, H, W]
                with torch.no_grad():
                    images_5d = images.unsqueeze(2)  # [N, C, 1, H, W]
                    enc_out = vae_model.encode(images_5d)
                    posterior = enc_out["latent_dist"] if isinstance(enc_out, dict) else enc_out
                    latent = posterior.sample()
                    # Normalize using QwenImage VAE latent stats
                    latents_mean = torch.tensor(
                        vae_model.config_dict["latents_mean"]
                    ).view(1, -1, 1, 1, 1).to(latent)
                    latents_std = torch.tensor(
                        vae_model.config_dict["latents_std"]
                    ).view(1, -1, 1, 1, 1).to(latent)
                    latent = (latent - latents_mean) * (1.0 / latents_std)
                    data['padded_latent'] = latent.squeeze(2)  # [N, 16, H/8, W/8]

            loss_dict, extra_info = fsdp_model(**data)

        # Debug: save clean / noised / pred images to trash/train_debug_vis/
        if (dist.get_rank() == 0 and training_args.visual_gen
                and extra_info.get('latent_pred') is not None
                and curr_step % training_args.vis_every == 0):
            _debug_dir = os.path.join("trash", "train_debug_vis")
            os.makedirs(_debug_dir, exist_ok=True)
            with torch.no_grad():
                _indices = extra_info['latent_indices']
                _shapes = data['patchified_vae_latent_shapes']
                from einops import rearrange as _rearrange
                from PIL import Image as _Img
                _p, _C = model_args.latent_patch_size, 16
                _lm = torch.tensor(vae_model.config_dict["latents_mean"]).view(1,-1,1,1,1).to(device)
                _ls_inv = 1.0 / torch.tensor(vae_model.config_dict["latents_std"]).view(1,-1,1,1,1).to(device)

                def _decode_flat_latent(flat_lat, h, w):
                    """[h*w, 64] → PIL Image. QwenImage token layout: [C, p, q]."""
                    unp = flat_lat.reshape(h, w, _C, _p, _p)
                    unp = unp.permute(2, 0, 3, 1, 4).reshape(_C, h*_p, w*_p)
                    z = unp.float().unsqueeze(0).unsqueeze(2).to(device)
                    z = z / _ls_inv + _lm
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        dec = vae_model.decode(z)
                    if isinstance(dec, dict): dec = dec["sample"]
                    pix = dec[0, :, 0].clamp(-1, 1).float().cpu()
                    return _Img.fromarray((((pix+1)/2)*255).to(torch.uint8).permute(1,2,0).numpy())

                for _img_idx, (_h, _w) in enumerate(_shapes):
                    _mask = _indices == _img_idx
                    if _mask.sum() == 0 or _mask.sum() != _h * _w:
                        continue
                    # Save: clean, noised, pred
                    for _name, _tensor in [
                        ("clean", extra_info['latent_clean']),
                        ("noised", extra_info['latent_noised']),
                        ("pred", extra_info['latent_pred']),
                    ]:
                        _img = _decode_flat_latent(_tensor[_mask], _h, _w)
                        _img.save(os.path.join(_debug_dir, f"step{curr_step:05d}_{_name}.png"))
                    break  # first image only

        # Update dataset mse and ce loss
        dataset_mse, dataset_ce = analyze_and_optimize_loss_aggregation(
            data_indexes,
            data,
            loss_dict,
            device,
            name_to_num_id,
            num_id_to_name,
            num_unique_datasets,
            dataset_mse,
            dataset_ce,
        )

        # Bucket mse loss
        if training_args.visual_gen:
            loss_bucket = bucket_mse_loss(
                loss_dict, timesteps=extra_info['latent_timesteps'], device=device
            )

        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            # visual_und=False, so no CE expected
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if ema_model is not None:
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item["dataset_name"] not in data_status.keys():
                data_status[item["dataset_name"]] = {}
            data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]
            dataset_num_sample[item["dataset_name"]] += 1.0

        # ── Logging ──────────────────────────────────────────────────
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()

            wandb_log["latent_mean"] = extra_info.pop("latent_mean")
            wandb_log["latent_std"] = extra_info.pop("latent_std")

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            # Visualization
            if training_args.visual_gen and curr_step % training_args.vis_every == 0 and curr_step > 0:
                # QwenImage VAE expects 5D [B,C,T,H,W]; vis_utils passes 4D after squeeze.
                # Also need to denormalize before decoding.
                def _vis_vae_decode(z_4d):
                    z_5d = z_4d.unsqueeze(2)  # [B,C,H,W] → [B,C,1,H,W]
                    # Denormalize
                    lm = torch.tensor(vae_model.config_dict["latents_mean"]).view(1,-1,1,1,1).to(z_5d)
                    ls = 1.0 / torch.tensor(vae_model.config_dict["latents_std"]).view(1,-1,1,1,1).to(z_5d)
                    z_5d = z_5d / ls + lm
                    out = vae_model.decode(z_5d)
                    if isinstance(out, dict):
                        out = out["sample"]
                    return out[:, :, 0]  # [B,C,H,W]

                # packed_text_ids has prefix+caption+suffix; packed_text_indexes has only caption+suffix positions.
                # Use actual_txt_indexes to extract matching token IDs.
                _vis_token_ids = data["packed_text_ids"][data["actual_txt_indexes"].long()] \
                    if "actual_txt_indexes" in data else data["packed_text_ids"]
                vis_table = construct_vis_table(
                    token_ids=_vis_token_ids,
                    sample_len=data["sample_lens"],
                    vae_indices=data["packed_vae_token_indexes"],
                    txt_indices=data["packed_text_indexes"],
                    tokenizer=tokenizer,
                    vae_decoding=_vis_vae_decode,
                    **extra_info,
                    latent_shapes=data["patchified_vae_latent_shapes"],
                    patch_size=model_args.latent_patch_size,
                    latent_channel=16,  # QwenImage VAE z_channels
                )
                wandb_log[f"{curr_step:07d}"] = vis_table
                print('curr_step for vis_table', f"{curr_step:07d}")

            if training_args.visual_gen:
                distributed_dict_reduce_and_log(
                    dataset_mse,
                    wandb_log,
                    prefix="mse",
                    token_suffix="token",
                    device=device,
                    reduce_type="mean",
                )
                distributed_dict_reduce_and_log(
                    loss_bucket, wandb_log, prefix="mse_bucket", device=device, reduce_type="mean"
                )

            distributed_dict_reduce_and_log(
                dataset_num_sample,
                wandb_log,
                prefix="num_sample",
                device=device,
                reduce_type="sum",
            )
            if dist.get_rank() == 0:
                wandb_log = batch_log_to_wandb(wandb_log)
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        # ── Checkpointing ────────────────────────────────────────────
        if curr_step > 0 and curr_step % training_args.save_every == 0 and dist.get_rank() == 0:
            torch.cuda.empty_cache()
            gc.collect()
            step_dir = os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}")
            os.makedirs(step_dir, exist_ok=True)
            torch.save(dataset_ce, os.path.join(step_dir, "dataset_ce.pt"))
            torch.save(dataset_mse, os.path.join(step_dir, "dataset_mse.pt"))

        if (curr_step + 1) % 10 == 0:
            gc.collect()
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            torch.cuda.empty_cache()
            gc.collect()

            FSDPCheckpoint.fsdp_save_fsdp_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=data_status,
            )

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
