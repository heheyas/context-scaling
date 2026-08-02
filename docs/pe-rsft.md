# PE-RSFT: prompt-expansion training data

Two independent entry points, both under `pe-rsft/`:

1. [Cold start via teacher distillation](#cold-start--teacher-distillation-sft)
   — bootstraps an initial SFT dataset for the prompter from a strong
   vision-capable teacher LLM. Three-step scripted pipeline; use this
   before you have any prompter to rejection-sample from.
2. [RFT loop](#rft-loop) — distributed rejection-sampling on the cold-
   started prompter. Coordinates an LLM host, a DiT host, and a Gemini
   scoring host through a shared-filesystem file queue; produces
   preference-filtered SFT (and optionally DPO) data.

---

## Cold start — teacher-distillation SFT

Path: `pe-rsft/teacher_distill/`.

```
(image, short_prompt) manifest
            │
            ▼   teacher_rollout.py — vision LLM w/ teacher SP
rollout jsonl  (<analysis>Stage A..E</analysis>{SP JSON})
            │
            ▼   ratio_clean.py — drop orientation-flipped rows (~1–2%)
ratio-clean jsonl
            │
            ▼   pack_sft.py — <analysis>→<think>, student SP as system
sft.jsonl
```

### Input manifest

```
<dataset>/
├── images/
└── pairs_manifest.jsonl
```

```json
{"prompt": "a small orange cat on a windowsill",
 "image":  "images/xxx.jpg", "width": 1200, "height": 800}
```

### Run

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=<endpoint>   # if not api.openai.com

python -m pe-rsft.teacher_distill.teacher_rollout \
    --dataset-dir     /path/to/<dataset> \
    --system-prompt   pe-rsft/teacher_distill/system_prompts/teacher.txt \
    --out             /path/to/<dataset>/rollout_full.jsonl \
    --model           <teacher-model-id> --workers 500

python -m pe-rsft.teacher_distill.ratio_clean \
    --in  /path/to/<dataset>/rollout_full.jsonl \
    --out /path/to/<dataset>/rollout_full_ratioclean.jsonl

python -m pe-rsft.teacher_distill.pack_sft \
    --passed-in          /path/to/<dataset>/rollout_full_ratioclean.jsonl \
    --system-prompt-file pe-rsft/teacher_distill/system_prompts/student.txt \
    --out                /path/to/<dataset>/sft.jsonl
```

### System prompts

- `system_prompts/teacher.txt` — teacher SP. Instructs the teacher to
  emit `<analysis>Stage A..E</analysis>{JSON}`.
- `system_prompts/student.txt` — student SP. Installed as the
  assistant's system prompt during SFT so the student learns to
  condition on the shorter, non-teacher-flavored SP it will see at
  inference time.

### Gotchas

- **Timeouts.** The teacher call takes 1–8 min per row with long CoT.
  Client timeouts under 1800 s cascade-fail — clients abort, the
  server logs 499s, and the retry storm exhausts the vendor's
  per-account concurrency cap.
- **Concurrency is the binding constraint.** Documented RPM/TPM caps
  are usually not what you hit first — concurrent-in-flight
  connections is. Warm up worker count gradually (500 → 600 → up);
  crossing the cap RSTs new connections rather than 429-ing them.
- **Resume is append-only.** Every runner reloads uids whose
  `rollout_text` does not start with `__ERROR__` and skips them.
  Don't delete partial output files during a run.
- **Malformed teacher JSON** (~0.3% of rows) is handled by
  `json_repair` inside `pack_sft.py`; unfixable rows are counted and
  dropped.

---

## RFT loop

Path: `pe-rsft/src/`. Three stages, three machines, coordinated through
a shared-filesystem file queue. No RPC, no shared memory.

```
                                      shared filesystem
                                  ╔═══════════════════════╗
LLM rollout host  ────────►       ║  llm_out/pending/     ║
                                  ║  llm_out/claimed/     ║
                                  ║  llm_out/done/        ║
                                  ╚═══════════════════════╝
                                              │
                                              ▼
                                  DiT render host
                                              │
                                              ▼
                                  ╔═══════════════════════╗
                                  ║  rank_in/pending/     ║
                                  ║  rank_in/claimed/     ║
                                  ║  rank_in/done/        ║
                                  ╚═══════════════════════╝
                                              │
                                              ▼
                                  Gemini scoring host
                                              │
                                              ▼
                                       rankings/results.jsonl
                                              │
                                              ▼
                                   SFT data export
```

### Stage 1 — LLM rollout (`src/llm/`)

For each query, the LLM produces K=5 candidate structured prompts.
Determinism is anchored on `(index, image_idx)` via `compute_seed()`.
Per-host append files (`metadata_{hostname}_{pid}.jsonl`) avoid lock
contention on the shared FS.

### Stage 2 — DiT render (`src/dit/`)

The DiT consumer polls `llm_out/pending/`, renders 1 image per LLM
rollout, saves PNG to `images/{index}/{image_idx}.png`, and emits a
ranking task to `rank_in/pending/` once an index has all N images.

Backends:
- `backends/native.py` — local DiT inference
- `backends/http.py` — generic HTTP endpoint (parameterize `--endpoint`)

### Stage 3 — Gemini scoring (`src/judge/`)

For each index, the Gemini judge runs C(N,2)×2 pairwise comparisons,
fits a Bradley-Terry MLE (`reg=1e-2`), and appends per-host results to
`rankings/results.jsonl`.

Judge system prompts in `src/system_prompts/`:

| Prompt | Dimension |
|---|---|
| `mix_verify_thinking.txt` | reasoning quality (4-dim) |
| `mix_verify_alignment.txt` | prompt-image alignment (10-item checklist) |
| `mix_verify_aesthetic.txt` | visual quality |
| `mix_verify_structure.txt` | structural correctness (strict) |
| `mix_verify.txt` | unified multi-criterion |
| `single_score.txt` | single-axis template |
| `image2json_v1.txt` | image → structured JSON |
| `seed_grm.txt` | grounded reasoning |
| `rewrite_caption_to_instruction.txt` | caption rewriting |
| `synth_backward.txt` | backward synthesis |

### File-queue protocol

```
{queue}/
├── pending/        # tasks waiting for a worker
├── claimed/        # tasks currently being processed
└── done/           # completed tasks
```

Atomic rename is used for transitions:

```
pending/task.jsonl   ──[claim]──►   claimed/task.{hostname}.{pid}.jsonl
claimed/task...      ──[finish]─►   done/task.jsonl
```

Stale claims (worker died / host crashed) are re-parked into `pending/`
by `src/ops/janitor.py`. Queue depth is reported by `src/ops/monitor.py`.

### Quick start

```bash
cd pe-rsft

# 1. start the three hosts (one per machine, or one per GPU)
bash scripts/start_llm_host.sh    --config configs/iter0_example.yaml
bash scripts/start_dit_host.sh    --config configs/iter0_example.yaml
bash scripts/start_gemini_host.sh --config configs/iter0_example.yaml

# 2. queue depth / janitor (optional, on any host)
python -m src.ops.monitor --config configs/iter0_example.yaml
python -m src.ops.janitor --config configs/iter0_example.yaml --stale_minutes 15

# 3. once enough indices are scored, export SFT data
python scripts/export_sft_data.py \
    --rankings rankings/results.jsonl \
    --output sft_data/iter0.jsonl \
    --min_bt_gap 0.5
```

### Configuration

`configs/iter0_example.yaml` is the canonical iter-0 template. Key fields:

- `shared_root` — shared filesystem root (NFS, HDFS via fuse, etc.)
- `llm.endpoint`, `dit.endpoint` — replace `<RAY_PSM>` with your URL
- `judge.api_key_file` — path to a file containing Gemini API keys
  (one per line; the judge rotates on rate-limit errors)
- `iter.queries` — input queries JSONL
- `iter.rollouts_per_query` — K (default 5)
- `iter.dit_resolution` — `[h, w]` (default `[1024, 1024]`)
