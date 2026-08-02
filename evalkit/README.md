# EvalKit

Multi-benchmark T2I evaluation framework. See
[`../docs/evalkit.md`](../docs/evalkit.md) for full documentation.

## Layout

```
evalkit/
├── context-scaling/           # multi-benchmark eval
│   ├── benchmarks/            # 9 benchmarks (see below)
│   ├── rewrite_prompts/       # LLM-based prompt rewriting (7 backends)
│   ├── inference/             # HTTP image generation client
│   └── scripts/               # orchestration shell + python scripts
├── detailness/                # ED + GPG caption-quality metrics
└── utils/                     # parquet helpers
```

## Benchmarks

GenEval2 • GenEval • DPG-Bench (ELLA) • OneIG-Bench • TIIF-Bench •
T2I-CoReBench • GenEval++ (Echo-4o) • GenExam • WISE

Each `benchmarks/<name>/` is the upstream code with its own README and
LICENSE. See [`../NOTICE`](../NOTICE) for attributions.

## Quick start

```bash
cd context-scaling

# 1. rewrite prompts (optional)
python -m rewrite_prompts.rewrite --benchmark geneval2 --backend gemini \
    --data_path benchmarks/GenEval2/geneval2_data.jsonl \
    --output rewritten/geneval2.jsonl

# 2. generate images via a DiT serving endpoint
python -m inference.generate --rewritten_jsonl rewritten/geneval2.jsonl \
    --url http://localhost:8899 --output_dir outputs/geneval2

# 3. run the benchmark's native eval
cd benchmarks/GenEval2 && python evaluation.py ...
```
