# ED (Effective Detailness)

A fully **model-free, caption-only, hyperparameter-free, matcher-agnostic**
metric of caption detailness for T2I training. Predicts Bagel image-MSE
across the same 16 caption variants × 300 paired uids that GPG validates
against.

Definition (paper-main, v4.1):

```
ED(condition) = trim20-mean over uids of  F_{β=0.5}(P_A, R_A)

where
  P_A, R_A    = precision and recall of the caption's Attribute-category
                OARG tuples against an image-grounded OARG source (Gemini-
                3-pro-vision exhaustive enumeration; caption never shown
                to the source extractor)
  F_{β=0.5}   = (1.25 · P · R) / (0.25 · P + R)
                Van Rijsbergen 1979 F-beta; precision weighted 2× recall
  trim20-mean = Hampel 1974: drop top 20% + bottom 20%, average middle 60%
  matcher     = GPT-4o-2024-11-20 (paper main) OR
                Gemini-3-pro-preview-new (cross-matcher robustness)
```

Single-sample ρ vs Bagel MSE = **−0.9087** (n=16 cells); bootstrap mean
**−0.853 ± 0.048**, 95 % CI **[−0.932, −0.744]** (B = 10,000).

## Files

| file | role |
|---|---|
| `extract_image.py` | image → OARG tuples via Gemini-vision (per-uid; ~5 min for N=300) |
| `extract_caption.py` | caption text → OARG tuples via GPT-4o (closed 16-attr schema) |
| `match.py` | per-(uid, caption) lenient text-only matcher (GPT-4o; `--matcher gemini` for the cross-matcher cache) |
| `aggregate.py` | per-cell trim20 of F<sub>0.5</sub>(P_A, R_A), 16-point CSV + main figure + power-law fit |
| `bootstrap.py` | B=10,000 resample CI of ρ; optional N-scaling experiment |
| `io.py` | shared loaders: caches, F-helpers, 16-point join (re-uses `gpg/io.py` for MSE-side loaders) |
| `prompts/image_source.txt` | exhaustive OARG enumeration prompt for the vision step |
| `prompts/extract_tuples.txt` | caption-side OARG (closed 16-attr schema, paper-final) |
| `prompts/match_tuples.txt` | paraphrase-tolerant lenient matcher prompt |

## Quick start

End-to-end on the canonical N=300 pool. Each step writes a cache; subsequent
runs only fill in missing entries.

```bash
# 1. Image → source OARG (Gemini-vision, ~5 min)
HOME=<RAY_SERVE_HOME> python -m detailness.ed.extract_image \
  --workers 32

# 2. Caption text → OARG (GPT-4o, ~10 min)
HOME=<RAY_SERVE_HOME> python -m detailness.ed.extract_caption \
  --workers 512

# 3a. Paper-main match (GPT-4o, ~90 min @ 1024 workers × 8 keys)
HOME=<RAY_SERVE_HOME> python -m detailness.ed.match \
  --matcher gpt --workers 1024

# 3b. Cross-matcher robustness (Gemini, ~50 min @ 512 workers × 4 keys)
HOME=<RAY_SERVE_HOME> python -m detailness.ed.match \
  --matcher gemini --workers 512

# 4. Aggregate → 16-point table + main figure + power-law fit
HOME=<RAY_SERVE_HOME> python -m detailness.ed.aggregate

# 5. Bootstrap CI (B=10,000, ~5 min CPU)
HOME=<RAY_SERVE_HOME> python -m detailness.ed.bootstrap --B 10000
```

Cross-matcher robustness aggregate (Gemini cache):

```bash
HOME=<RAY_SERVE_HOME> python -m detailness.ed.aggregate --matcher gemini
```

## Reproduce the paper figure numbers

```python
from detailness.ed.io import load_v41
pts, budget = load_v41()
# 16 points, single-sample ρ = -0.9087, inv = 15/120
# Power law: MSE = 0.384 · ED^-0.267, R² ≈ 0.79
# struct/l10 = abl/full are top (lowest MSE 0.43699, highest ED 0.6117)
# struct/l5 is bottom (highest MSE 0.44664, lowest ED 0.5746 among struct)
```

`pts` is a list of `(ed_trim20, ed_se, mse, label, category)` tuples
mirroring `gpg.io.load_v6_dropmore`. The `abl_no_depth` and
`abl_no_atmosphere_lighting` cells are excluded by default — their
Bagel MSE runs are pending rerun (same exclusion as GPG).

## Outputs

Per-row match records live under `/tmp/detailness_real_n100/cache/`:

```json
{"key": "<sha1(uid + \\0 + caption)>",
 "value": {
   "uid": "...", "kind": "structured", "level": "l10",
   "source": {"O": [...], "A": [...], "R": [...], "G": [...]},
   "caption": {"O": [...], "A": [...], "R": [...], "G": [...]},
   "source_recall":     {"O": ["YES","NO",...], "A": [...], "R": [...], "G": [...]},
   "caption_precision": {"O": [...], "A": [...], "R": [...], "G": [...]}
 }}
```

Per-cell aggregates live under `/tmp/detailness_n100_16cell/`:
- `per_cond_v41.csv`              — paper main (GPT-4o matcher)
- `per_cond_v24.csv`              — cross-matcher (Gemini)
- `bootstrap_v41.json`            — B=10,000 ρ distribution

Figures land under `docs/figures/`:
- `ed_v41.png`                    — paper-main fit figure
- `ed_v24.png`                    — Gemini-matcher fit (robustness)

## Unified evaluator (ED + GPG on arbitrary inputs)

For ad-hoc (image, caption) scoring outside the canonical pool — e.g.,
the `truck.png` adversarial demo or any other custom dataset — use the
top-level Evaluator that wraps both ED and GPG behind one API:

```python
from detailness import Evaluator
ev = Evaluator(metrics=("ed",))                            # ED only, no GPU
ev.score("/path/to/img.png", "A short caption.")
# {'ed': 0.27, 'ed_F05_A': 0.27, 'ed_P_A': 1.0, 'ed_R_A': 0.07,
#  'ed_n_src_A': 70, 'ed_n_cap_A': 2}

ev_gpg = Evaluator(
    metrics=("ed", "gpg"),
    gpg_model_path="<HDFS_ROOT>/.../Qwen3-VL-8B-Instruct",
)
ev_gpg.score(img, caption)
# {'ed': ..., 'gpg': ..., 'gpg_total_nats': ..., ...}
```

Or via CLI on a JSONL dataset:

```bash
HOME=<RAY_SERVE_HOME> PYTHONNOUSERSITE=1 \
python -m detailness.evaluator \
  --jsonl  /path/to/dataset.jsonl \
  --out    /tmp/scored.jsonl \
  --metrics ed,gpg \
  --gpg-model <HDFS_ROOT>/.../Qwen3-VL-8B-Instruct \
  --ed-cache-dir /tmp/my_cache       # optional, reuses image-source / caption-tuple results
```

See `evalkit/detailness/evaluator.py` for the full API.

## Notes

- `judge.py` (KeyPool, `_chat`, `extract_tuples`, `match_tuples`) is
  shared with GPG and lives at `evalkit/detailness/judge.py`.
- The 16-cell pool excludes `abl_no_depth` and `abl_no_atmosphere_lighting`
  — same exclusion convention as GPG (`gpg/io.py:build_16_points`).
- `struct/l5` is a known Bagel-side outlier (highest MSE despite ranking
  reasonably on the metric). Keep it in the per-cell table, do not drop
  by default.
- Matcher cache filenames retain historical version tags
  (`v4_image_match.jsonl`, `v24_match_gemini.jsonl`) to avoid breaking
  the paper-final caches; the public-facing version labels are `v41`
  and `v24` respectively.
