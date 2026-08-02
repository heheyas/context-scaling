# PE-RSFT

Data pipeline for training the prompt-expansion LLM. Two entry points,
usable independently:

- [`teacher_distill/`](teacher_distill/) — **cold start**: distill an
  initial SFT dataset from a strong vision-capable teacher (`rollout →
  ratio-clean → pack_sft`).
- [`src/`](src/) — **RFT loop**: distributed rejection-sampling on the
  cold-started prompter, coordinating an LLM host, a DiT host and a
  Gemini scoring host through a shared-filesystem file queue.

See [`../docs/pe-rsft.md`](../docs/pe-rsft.md) for full docs.

## Layout

```
pe-rsft/
├── teacher_distill/       # cold-start SFT via teacher distillation
│   ├── teacher_rollout.py / ratio_clean.py / pack_sft.py
│   └── system_prompts/{teacher,student}.txt
├── src/
│   ├── llm/               # Stage 1: LLM rollout (vLLM backend)
│   ├── dit/               # Stage 2: DiT image generation
│   ├── judge/             # Stage 3: Gemini pairwise + Bradley-Terry
│   ├── ops/               # file-queue primitives, janitor, monitor
│   ├── system_prompts/    # Gemini judge prompts
│   └── synth/             # optional synthetic data generation
├── scripts/               # host startup, SFT data export
└── configs/               # YAML iter-0 example
```

## Pipeline

```
LLM rollout host  →  shared FS queue  →  DiT render host  →  shared FS queue
                                                                    │
                                                                    ▼
                                                          Gemini scoring host
                                                                    │
                                                                    ▼
                                                            SFT data export
```

The three hosts coordinate via atomic-rename file queues on a shared
POSIX filesystem (NFS or HDFS via fuse). No RPC.

## Quick start — RFT loop

```bash
# 1. start the three hosts (separate machines / nodes)
bash scripts/start_llm_host.sh    --config configs/iter0_example.yaml
bash scripts/start_dit_host.sh    --config configs/iter0_example.yaml
bash scripts/start_gemini_host.sh --config configs/iter0_example.yaml

# 2. export SFT data once enough indices are scored
python scripts/export_sft_data.py --rankings rankings/results.jsonl \
    --output sft_data/iter0.jsonl
```

## Quick start — teacher distillation

See [`teacher_distill/README.md`](teacher_distill/README.md).
