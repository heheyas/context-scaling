# GPG (Grounded Perplexity Gain)

PMI-based caption-grounding metric used to predict Bagel T2I MSE across
caption variants (struct / dense / spatial / abl). Definition:

```
GPG = sum_t [ log p(t | image, prefix) − log p(t | prefix) ]
                                              # over content tokens of a canonicalized caption
```

Judge: Qwen3-VL-8B-Instruct. Returns total nats per caption.

Recipe for the paper version (`v6_dropmore`, n=300 paired uids/cell):
- `--canonicalize-json` REMOVES non-visual fields from the JSON before tokenizing
- `--extra-drop-keys "atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect"`
- `--content-only` masks JSON scaffold tokens (`{ } " : ,`) out of the PMI sum
- No system prompt (schema prompt collapses spatial discrimination)
- No bbox conversion (Qwen3-VL is 0–1000 norm native)

## Files

| file | role |
|---|---|
| `score.py` | per-sample PMI scorer; sharded driver for an image+caption pool |
| `content_mask.py` | JSON value/scaffold per-token mask for `--content-only` |
| `io.py` | shared loaders: GPG shard records, Bagel MSE CSVs, 16-point join |
| `analyze.py` | aggregate GPG → MSE, summary table + per-experiment figures |
| `metric_candidates.py` | sweep alternative metric transforms (capacity-bounded `GPG_eff`, power, exp, etc.) |
| `overnight.sh` | run v5/v6/v7/v8 *recipe* settings back-to-back, 8 shards × 1 GPU |
| `cross_judge.sh` | run v6 recipe across multiple judge VLMs (Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-MoE 35B+122B / InternVL3-8B) for paper-appendix robustness |
| `judge_robustness.py` | discover all `gpg_v6_<tag>/` dirs, build cross-judge fit table + grid figure |
| `prompts/structured_schema.txt` | schema system prompt (tested in v5/v7/v8, dropped in v6_dropmore final) |

Supported judge model_types in `score.py`: `qwen2_vl`, `qwen2_5_vl`,
`qwen3_vl`, `qwen3_5_moe`, `internvl`/`internvl_chat`, plus a generic
`AutoModelForImageTextToText` fallback for anything else with a standard
HF chat-template processor.

## Quick start

Score a pool (single GPU, 8 shards launched manually):
```bash
HOME=<RAY_SERVE_HOME> CUDA_VISIBLE_DEVICES=0 python -m detailness.gpg.score \
  --pool /tmp/detailness_real_n100/pool.jsonl \
  --pool-dir /tmp/detailness_real_n100 \
  --model <HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct \
  --out /tmp/detailness_real_n100/gpg_v6_dropmore/shard0.jsonl \
  --shard 0 --num-shards 8 \
  --content-only --canonicalize-json \
  --extra-drop-keys "atmosphere,lighting,style,photography,layout,shot_type,camera_angle,lens_and_effect"
```

Or run the full overnight sweep:
```bash
POOL_DIR=/tmp/detailness_real_n100 \
MODEL=<HDFS_ROOT>/weights/Qwen3-VL-8B-Instruct \
  evalkit/detailness/gpg/overnight.sh
```

Aggregate to the 16-point table + figures:
```bash
python -m detailness.gpg.analyze \
  --gpg-dir /tmp/detailness_real_n100/gpg_v6_dropmore \
  --tag v6_dropmore
```

Sweep metric redefinitions (capacity-bounded `GPG_eff`, etc.):
```bash
python -m detailness.gpg.metric_candidates \
  --gpg-dir /tmp/detailness_real_n100/gpg_v6_dropmore
```

Cross-judge robustness sweep (writes `gpg_v6_<tag>/` per judge):
```bash
POOL_DIR=/tmp/detailness_real_n100 \
  evalkit/detailness/gpg/cross_judge.sh
# Then aggregate
python -m detailness.gpg.judge_robustness \
  --csv judge_robustness.csv
```

## Reproduce the paper figure numbers

```python
from detailness.gpg.io import load_v6_dropmore
pts, budget = load_v6_dropmore()
# 16 points, raw GPG: r = -0.9863, R² = 0.9727 vs Bagel MSE
# Capacity-bounded GPG_eff = -log(1 - GPG/250): linear r = -0.969,
#   saturation R² = 0.973, floor = 0.4343
```

`pts` is a list of `(gpg_mean, gpg_se, mse, label, category)` tuples,
one per (kind, level) cell. The `abl_no_depth` and
`abl_no_atmosphere_lighting` cells are excluded by default — their
Bagel MSE runs are flagged for rerun.

## Outputs

Per-sample GPG records live in `shard{0..7}.jsonl` files under
`/tmp/detailness_real_n100/gpg_<version>/`. Schema:

```json
{"uid": "...", "kind": "structured", "level": "l10",
 "n_tokens": 756, "n_tokens_prior": 756,
 "nll_grounded_sum": 1328.55, "nll_grounded_mean": 1.7573,
 "nll_prior_sum":    1617.89, "nll_prior_mean":    2.1401,
 "gpg": 0.3827}
```

`gpg` here is the per-token mean PMI; multiply by `n_tokens` for the
caption-total nats reported in the 16-point table.
