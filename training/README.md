# Training

QwenImage DiT training + serving stack. See [`../docs/training.md`](../docs/training.md)
for full documentation.

## Layout

```
training/
├── modeling/
│   ├── qwenimage/     # transformer.py, vae.py, model.py, text_encoder_navit.py
│   ├── qwen2/         # Qwen2 LLM (text encoder backbone)
│   └── shared/        # TimestepEmbedder, PositionEmbedding
├── train/             # FSDP training loop, loss utilities
├── data/              # PackedDataset, T2I NaviT loaders, JSON transforms
│   └── configs/json/  # dataset-mixture YAML configs
└── scripts/           # checkpoint packing, merging, serving
```

## Quick start

```bash
# Single-GPU serving
python scripts/serve.py --ckpt-root <path/to/weights> --port 8899

# Distributed training (8 GPUs)
torchrun --nproc_per_node=8 train/pretrain_qwenimage_navit.py \
    --dit_path <weights>/qwenimage/transformer \
    --text_enc_path <weights>/qwen2.5-vl-7b \
    --vae_path <weights>/qwenimage/vae \
    --dataset_config_file data/configs/json/ct2_j5t5_l10.yaml
```

## Attribution

`modeling/qwenimage/` and `modeling/qwen2/` are derived from
[`huggingface/diffusers`](https://github.com/huggingface/diffusers) and
[`huggingface/transformers`](https://github.com/huggingface/transformers)
respectively, both under Apache-2.0. See [`../NOTICE`](../NOTICE) for the
full attribution list.
