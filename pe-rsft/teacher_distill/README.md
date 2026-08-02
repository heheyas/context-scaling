# Teacher-distillation SFT (cold start for the prompter)

Simple three-step data pipeline that produces initial SFT training data
for the prompt-expansion LLM by distilling a strong vision-capable
teacher model. Use this to bootstrap a prompter for the RFT loop in
[`../src/`](../src/), or as a standalone data pipeline if you don't
need rejection sampling.

```
(image, short_prompt) pairs
            │
            ▼   teacher_rollout.py — vision LLM w/ teacher SP
rollout jsonl (<analysis>…</analysis>{SP JSON})
            │
            ▼   ratio_clean.py — drop orientation-flipped rows (~1–2%)
ratio-clean jsonl
            │
            ▼   pack_sft.py — <analysis>→<think>, student SP as system
sft.jsonl
```

## Input manifest

One line per (image, short prompt) pair, under a dataset root:

```
<dataset>/
├── images/
└── pairs_manifest.jsonl
```

```json
{
  "prompt": "a small orange cat on a windowsill",
  "image":  "images/xxx.jpg",
  "source": "<optional provenance>",
  "width": 1200, "height": 800
}
```

## Run

```bash
export OPENAI_API_KEY=<your-key>          # or vendor equivalent
export OPENAI_BASE_URL=<endpoint>         # if not api.openai.com

# 1. Teacher rollout (resume-safe; append-only output).
python -m pe-rsft.teacher_distill.teacher_rollout \
    --dataset-dir     /path/to/<dataset> \
    --system-prompt   pe-rsft/teacher_distill/system_prompts/teacher.txt \
    --out             /path/to/<dataset>/rollout_v6_structured_full.jsonl \
    --model           <teacher-model-id> \
    --workers         500

# 2. Aspect-ratio clean (drops rows where teacher's chosen ratio flips
#    portrait vs landscape; usually 1–2% of rows).
python -m pe-rsft.teacher_distill.ratio_clean \
    --in  /path/to/<dataset>/rollout_v6_structured_full.jsonl \
    --out /path/to/<dataset>/rollout_v6_structured_full_ratioclean.jsonl

# 3. Pack into SFT training JSONL — <analysis> → <think>, student SP
#    installed as the assistant's system prompt.
python -m pe-rsft.teacher_distill.pack_sft \
    --passed-in         /path/to/<dataset>/rollout_v6_structured_full_ratioclean.jsonl \
    --system-prompt-file pe-rsft/teacher_distill/system_prompts/student.txt \
    --out               /path/to/<dataset>/sft_v6_$(date +%Y%m%d).jsonl
```

## System prompts

Two files ship with this pipeline:

- `system_prompts/teacher.txt` — the SP the teacher sees at rollout
  time. Instructs the teacher to emit `<analysis>Stage A..E</analysis>{JSON}`.
- `system_prompts/student.txt` — the SP the student sees at inference
  time. Installed as the assistant's system prompt during SFT so it
  learns to condition on this shorter, non-teacher-flavored SP.

## Notes

- The teacher call takes 1–8 min per row with long CoT. Set client
  timeouts to ≥1800 s per attempt — long-tail rows will otherwise
  cascade-fail as the client aborts them, the server logs 499s, and
  the retry storm exhausts the vendor's per-account concurrency cap.
- Documented rate limits (RPM / TPM) are rarely the binding
  constraint; concurrent-in-flight-connections is. Warm up worker
  count gradually (500 → 600 → up); crossing the cap RSTs new
  connections rather than 429-ing them.
- Resume is append-only: every runner reloads uids whose
  `rollout_text` does not start with `__ERROR__` and skips them.
  Don't delete partial output files during a run.
- Malformed teacher JSON (~0.3% of rows) is handled by `json_repair`
  in `pack_sft.py`; unfixable rows are counted and dropped.
