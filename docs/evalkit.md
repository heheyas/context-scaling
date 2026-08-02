# EvalKit: Multi-Benchmark Eval

The `evalkit/` subproject contains:

1. **`evalkit/context-scaling/`** — multi-benchmark text-to-image evaluation
   framework spanning 9 public benchmarks, with a unified rewrite → inference
   → judge pipeline.
2. **`evalkit/detailness/`** — caption-quality metrics (ED + GPG). See
   [`../evalkit/detailness/README.md`](../evalkit/detailness/README.md).

For the SP-conditioned prompter pipeline (NL → SP → DiT), see the
paper-release demo at [`../demo/`](../demo/) and the RFT data pipeline
at [`../pe-rsft/`](../pe-rsft/).

## Benchmarks

| Benchmark | Path | Eval signal |
|---|---|---|
| GenEval2 | `benchmarks/GenEval2/` | Soft-TIFA VQA |
| GenEval | `benchmarks/geneval/` | Mask2Former + CLIP |
| DPG-Bench (ELLA) | `benchmarks/ELLA/dpg_bench/` | MPLUG VQA |
| OneIG-Bench | `benchmarks/OneIG-Benchmark/` | 6-dim VLM scoring |
| TIIF-Bench | `benchmarks/TIIF-Bench/` | TIIF long-prompt eval |
| T2I-CoReBench | `benchmarks/T2I-CoReBench/` | Compositional reasoning |
| GenEval++ (Echo-4o) | `benchmarks/Echo-4o/` | GPT-4.1 judge |
| GenExam | `benchmarks/GenExam/` | Exam-style |
| WISE | `benchmarks/WISE/` | World-knowledge |

Each subdirectory contains the original benchmark code (with its own
`LICENSE` and `README`). See [`NOTICE`](../NOTICE) for upstream attributions.

## Pipeline overview

```
benchmark prompts
        │
        ▼
┌──────────────────┐
│ rewrite_prompts/ │  (optional) LLM-based prompt rewriting
└──────────────────┘
        │
        ▼  rewritten/*.jsonl
┌──────────────────┐
│   inference/     │  HTTP POST to DiT serving endpoint
└──────────────────┘
        │
        ▼  outputs/{benchmark}/...
┌──────────────────┐
│ benchmark eval   │  per-benchmark scoring (CLIP / VQA / VLM judge)
└──────────────────┘
```

## Quick start

```bash
cd evalkit/context-scaling

# 1. (optional) rewrite prompts with an LLM
python -m rewrite_prompts.rewrite \
    --benchmark geneval2 \
    --data_path benchmarks/GenEval2/geneval2_data.jsonl \
    --backend gemini --output rewritten/geneval2_gemini.jsonl \
    --workers 16

# 2. generate images via your DiT serving endpoint
python -m inference.generate \
    --rewritten_jsonl rewritten/geneval2_gemini.jsonl \
    --url http://localhost:8899 \
    --output_dir outputs/geneval2 \
    --height 1024 --width 1024 --num_steps 25 --cfg_scale 4.0 --workers 8

# 3. run the benchmark's native evaluation
cd benchmarks/GenEval2 && python evaluation.py --image_dir ../../outputs/geneval2 ...
```

## Rewrite backends

`rewrite_prompts/` ships abstract `RewriteBase` + 7 concrete backends:

| Backend | Module | Endpoint |
|---|---|---|
| Claude | `claude.py` | Anthropic API |
| Gemini 3 Pro | `gemini3pro.py` | Google AI Studio |
| GPT | `gpt.py` | Azure OpenAI |
| Qwen (local) | `qwen.py` | local HF model |
| vLLM-served Qwen | `vllm_qwen.py` | OpenAI-compatible vLLM endpoint |
| Generic OpenAI | `openai_generic.py` | any OpenAI-compatible URL |
| Generic PSM | `seed.py` | parameterized HTTP backend |

`KeyPool` (in `base.py`) handles thread-safe API-key rotation with
rate-limit tracking.

## Configuration

All serving endpoints, API keys, and storage paths are parameterized via
CLI flags or environment variables. Common patterns:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` / `GEMINI_API_KEY_FILE` | Gemini LLM and judge calls |
| `OPENAI_API_KEY` | GPT / OpenAI-compatible backends |
| `ANTHROPIC_API_KEY` | Claude backend |
| `<RAY_PSM>` placeholder | replace with your Ray PSM or HTTP URL |
| `<HDFS_ROOT>` placeholder | replace with your dataset root |
