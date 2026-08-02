# Training: QwenImage DiT

The `training/` subproject contains a self-contained QwenImage text-to-image
DiT training and serving stack. It is independent of the other two
subprojects and can be used on its own.

## Architecture

QwenImage is a 60-layer Diffusion Transformer with joint image-text attention,
a causal 3D VAE, and a Qwen2.5 text encoder. The training stack:

```
training/
├── modeling/
│   ├── qwenimage/          # transformer.py, vae.py, text_encoder_navit.py, model.py
│   ├── qwen2/              # Qwen2 LLM modeling (text encoder backbone)
│   └── shared/             # TimestepEmbedder, PositionEmbedding
├── train/
│   ├── pretrain_qwenimage_navit.py   # main training script
│   ├── fsdp_utils.py                 # FSDP wrapper, checkpoint save/load
│   ├── train_utils.py                # loss, logging, distributed reduce
│   └── vis_utils.py                  # visualization for training samples
├── data/
│   ├── dataset_base.py / data_utils.py / distributed_iterable_dataset.py
│   ├── t2i_dataset_navit.py          # packed NaviT T2I datasets
│   ├── json_transform.py             # structured-prompt drop / sample utilities
│   ├── parquet_utils.py / transforms.py
│   ├── dataset_info.py               # dataset registry (HDFS paths placeholder)
│   └── configs/json/                 # dataset-mixture YAML configs
└── scripts/
    ├── pack_fsdp_ckpt.py             # consolidate FSDP shards to safetensors
    ├── inference.py                  # standalone inference
    ├── merge_weights.py              # delta-merge / model-soup
    ├── verify_merged_ckpt.py         # sanity check merged ckpt
    ├── serve.py                      # single-GPU FastAPI server
    ├── serve_multigpu.py             # multi-GPU queue-based serving
    └── serve_ray.py                  # Ray distributed serving
```

## Key design points

- **Dual-stream attention**: image and text are processed by parallel
  attention streams instead of unified causal attention.
- **3D multi-axis RoPE** `(16, 56, 56)` for spatial positions.
- **MSE-only loss** (no cross-entropy); timestep-shifted diffusion schedule.
- **NaviT packing**: variable-length samples packed into fixed-size token
  sequences for batching efficiency.
- **FSDP + bf16 + flash-attn + flex-attention**: hybrid sharded training.
- **EMA support** (use `use_orig_params=True` in FSDP).

## Training command

Set `${DATA_ROOT}` to your dataset root and `${WEIGHTS_ROOT}` to a directory
containing the base QwenImage weights (DiT, VAE, text encoder).

```bash
torchrun --nproc_per_node=8 train/pretrain_qwenimage_navit.py \
    --dit_path ${WEIGHTS_ROOT}/qwenimage/transformer \
    --text_enc_path ${WEIGHTS_ROOT}/qwen2.5-vl-7b \
    --vae_path ${WEIGHTS_ROOT}/qwenimage/vae \
    --dataset_config_file data/configs/json/ct2_j5t5_l10.yaml \
    --total_steps 500000 --lr 1e-4
```

## Inference / serving

```bash
# Single-GPU server (FastAPI on port 8899 by default)
python scripts/serve.py --ckpt-root ${WEIGHTS_ROOT}/qwenimage-finetuned

# Multi-GPU server
python scripts/serve_multigpu.py --ckpt-root ${WEIGHTS_ROOT}/qwenimage-finetuned \
    --num_gpus 8 --port 8899

# Ray distributed
python scripts/serve_ray.py --ckpt-root ${WEIGHTS_ROOT}/qwenimage-finetuned
```

POST a JSON payload to `/generate`:

```json
{
  "prompt": "a cat sitting on a wooden table, oil painting",
  "height": 1024,
  "width": 1024,
  "num_steps": 25,
  "cfg_scale": 4.0,
  "seed": 42
}
```

## Checkpoint workflow

After distributed training, consolidate FSDP shards into a single
`model.safetensors`:

```bash
python scripts/pack_fsdp_ckpt.py --input ${TRAIN_OUT}/step_22000 \
    --output ${WEIGHTS_ROOT}/qwenimage-22k
```

Combine multiple checkpoints via delta-merge or model-soup:

```bash
# Delta merge: original + weight * (finetuned - original)
python scripts/merge_weights.py \
    --base ${WEIGHTS_ROOT}/qwenimage-base \
    --finetuned ${WEIGHTS_ROOT}/qwenimage-22k \
    --alpha 0.7 \
    --output ${WEIGHTS_ROOT}/qwenimage-22k-d0p7

# Linear soup: weighted average of multiple checkpoints
python scripts/merge_weights.py --mode soup \
    --inputs ${WEIGHTS_ROOT}/qwenimage-{10k,15k,22k} \
    --weights 0.2 0.3 0.5 \
    --output ${WEIGHTS_ROOT}/qwenimage-soup
```

## Attribution

- `modeling/qwenimage/{transformer,vae,model}.py` and
  `modeling/shared/modeling_utils.py` are ported from
  [`huggingface/diffusers`](https://github.com/huggingface/diffusers).
  Parameter names match the diffusers checkpoint exactly so weights load
  directly.
- `modeling/qwen2/*` and `modeling/qwenimage/text_encoder_navit.py` are
  derived from [`huggingface/transformers`](https://github.com/huggingface/transformers)
  Qwen2 modeling code.

All derived files carry per-file copyright headers acknowledging the
original authors. See [`NOTICE`](../NOTICE) at the repository root for
a complete list.
