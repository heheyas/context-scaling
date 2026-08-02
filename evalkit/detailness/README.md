# Detailness metrics

Two complementary metrics for measuring how much an image-caption pair
"tells" — and a single `Evaluator` that runs either or both on arbitrary
inputs.

| Metric | Where it lives | What it needs | What it measures |
|---|---|---|---|
| **ED** (Effective Detailness) | [`ed/`](ed/) | LLM API only (no GPU) | Caption-side `F_{β=0.5}(P_A, R_A)` against an exhaustive image-grounded OARG enumeration produced by a vision LLM. Model-free, hyperparameter-free, matcher-robust. |
| **GPG** (Grounded Perplexity Gain) | [`gpg/`](gpg/) | Qwen3-VL-8B on a CUDA device | Per-token PMI `log p(t \| image, prefix) − log p(t \| prefix)` summed over the content tokens of a (canonicalized) caption. Model-conditional. |

Both validate against the same 16-cell paired-pool / Bagel MSE setup;
see the per-package READMEs for the fit numbers and paper figures.

## Layout

```
detailness/
├── judge.py          # shared LLM client (Gemini-3-pro + GPT-4o, KeyPool)
├── evaluator.py      # unified Evaluator(metrics=("ed", "gpg"))
├── ed/
│   ├── extract_image.py / extract_caption.py / match.py
│   ├── aggregate.py / bootstrap.py
│   ├── io.py
│   └── prompts/{image_source,extract_tuples,match_tuples}.txt
└── gpg/
    ├── score.py / analyze.py / metric_candidates.py / judge_robustness.py
    ├── content_mask.py / io.py
    ├── overnight.sh / cross_judge.sh
    └── prompts/structured_schema.txt
```

## Quick start

For an ad-hoc `(image, caption)` pair via the unified `Evaluator`:

```python
from detailness import Evaluator          # run from inside evalkit/

# ED only (no GPU)
ev = Evaluator(metrics=("ed",))
ev.score("/path/to/img.png", "A short caption.")
# {'ed': 0.27, 'ed_F05_A': 0.27, 'ed_P_A': 1.0, 'ed_R_A': 0.07, 'ed_n_src_A': 70, 'ed_n_cap_A': 2}

# ED + GPG
ev = Evaluator(
    metrics=("ed", "gpg"),
    gpg_model_path="<WEIGHTS_ROOT>/Qwen3-VL-8B-Instruct",
)
ev.score(img, caption)
# {'ed': ..., 'gpg': ..., 'gpg_total_nats': ..., ...}
```

Or via CLI on a JSONL dataset:

```bash
python -m detailness.evaluator \
    --jsonl    /path/to/dataset.jsonl \
    --metrics  ed,gpg \
    --gpg-model <WEIGHTS_ROOT>/Qwen3-VL-8B-Instruct \
    --out      /tmp/scored.jsonl
```

For reproducing the canonical 16-point paper plots, see the
per-package READMEs ([`ed/README.md`](ed/README.md),
[`gpg/README.md`](gpg/README.md)).

## Configuration

All cluster-specific values are env-var driven (no in-source defaults):

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY_FILE` | judge.py `DEFAULT_KEY_FILE` (one key per line) |
| `GEMINI_API_CONFIG` | ED matcher `--api_config_json` default (GPT-4o keys JSON) |
| `WANDB_ROOT` | `gpg.io._WANDB_ROOT` (Bagel MSE CSV root) |
| `FIGURES_DIR` | aggregate / analyze `--fig-dir` default |

## When to use which

- **Caption-only audit, no compute budget** — use ED. Scales linearly
  in LLM API calls; image source extraction is cached per uid.
- **Conditional on a specific VLM judge** — use GPG. Two forward
  passes per (image, caption); needs the Qwen3-VL-8B weights on disk.
- **Both, for cross-validation** — `Evaluator(metrics=("ed", "gpg"))`.
  They agree on the 16-cell rankings but read out different things
  about the caption.
