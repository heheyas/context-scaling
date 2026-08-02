# Context Scaling: Scaling Properties of Text Conditioning in Visual Generation

<p align="center">
  <a href="https://heheyas.github.io/context-scaling"><img src="https://img.shields.io/badge/Project-Page-2166ac?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/collections/heheyas/context-scaling"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-ffcc4d?style=flat-square" alt="HF Models"></a>
  <a href="https://github.com/heheyas/context-scaling"><img src="https://img.shields.io/badge/GitHub-Code-black?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-brightgreen?style=flat-square" alt="License"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
</p>

<p align="center">
  <b>Zilong Chen</b> · Chaorui Deng · Kunchang Li · Hongyi Yuan · Haoqi Fan<br>
  <em>ByteDance Seed</em><br>
  <sub><a href="mailto:jaysonabcchen@gmail.com">jaysonabcchen@gmail.com</a></sub>
</p>

<p align="center">
  <img src="assets/figure2_short.png" alt="Text prompt scaling law overview">
</p>

> We study empirical scaling properties for text conditioning in visual
> generation. Such properties have rarely been measured because diffusion
> loss does not scale with the number of tokens in natural-language prompts.
> Surprisingly, we find that the converged diffusion loss scales with the
> amount of **structured language** in the prompt. To quantify structured
> language, we adapt two complementary measures: a white-box likelihood
> metric (**GPG**) and a black-box attribute metric (**ED**). Across
> controlled training runs, the converged diffusion loss decreases
> approximately linearly with GPG and follows a power law with ED.
>
> Guided by these scaling properties, we improve **diffusability** by
> constructing structured prompts with semantic and geometric annotations
> derived from images, and improve **promptability** by training a prompter
> through supervised fine-tuning, cold-start, and verifier-gated on-policy
> distillation. The resulting system outperforms all evaluated open-weight
> models on nearly every compositional, reasoning, and world-knowledge
> benchmark, while matching or surpassing the strongest closed-weight models
> on most evaluations.

## Highlights

- **Scaling laws for text conditioning.** A controlled fifteen-cell sweep
  showing that converged diffusion loss tracks two caption-information
  metrics (**GPG**, likelihood-based; **ED**, attribute-based) much more
  tightly than it tracks caption length.
- **Structured prompts (SP)** as a diffusability upgrade — a JSON schema
  with semantic and geometric fields, annotated by expert models.
- **A trainable LLM prompter + captioner** that turns a user request or
  an image into SP.
- **Zero-shot structured editing** — SP exposes each factor as a named
  field, so targeted edits regenerate an image with the rest of the
  composition preserved.

## Installation

```bash
git clone https://github.com/heheyas/context-scaling.git
cd context-scaling
pip install -r requirements.txt
```

CUDA 12, `torch >= 2.4`, `flash-attn`. The demo, ED/GPG scoring, and
serving pipelines run on a single GPU (2 for the demo); DiT training
expects multi-GPU FSDP.

## Released models

Two Hugging Face Hub repos ship with this release:

| Model | HF repo | What it does | Size (bf16) |
|---|---|---|---|
| **PE / Captioner** | [`heheyas/SP-PE-Qwen3.5-35B-A3B`](https://huggingface.co/heheyas/SP-PE-Qwen3.5-35B-A3B) | SP-fine-tuned Qwen3.5-35B-A3B VLM. Serves **both** roles from the same weights: text-only chat template + [`rft_iter0_..._student.txt`](demo/system_prompts/rft_iter0_with_ratio_v6_synthesis_student.txt) → NL prompt → SP JSON (prompter); multimodal chat template + [`image2json.txt`](demo/system_prompts/image2json.txt) → image → SP JSON (captioner). | ~70 GB |
| **DiT** | [`heheyas/Qwen-Image-SP`](https://huggingface.co/heheyas/Qwen-Image-SP) | SP-fine-tuned QwenImage transformer, shipped as a **10-shard overlay** on top of [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) (which supplies the VAE + Qwen2.5-VL text encoder). Consumes SP JSON, emits an image. | ~40 GB overlay + ~17 GB base |

Model card: [`docs/model_card.md`](docs/model_card.md). All three roles
(PE, captioner, DiT) are wrapped in the browser demo below; for
standalone serving see [Serving the released models](#serving-the-released-models).

## Quickstart

### Try the demo — PE → DiT + image captioner in the browser

Boots one FastAPI process on a 2-GPU box: the released prompter LLM
([`heheyas/SP-PE-Qwen3.5-35B-A3B`](https://huggingface.co/heheyas/SP-PE-Qwen3.5-35B-A3B))
plus the fine-tuned DiT ([`heheyas/Qwen-Image-SP`](https://huggingface.co/heheyas/Qwen-Image-SP)),
both loaded resident in GPU memory, fronted by an HTML studio at
`http://localhost:7860/`.

```bash
pip install -r demo/requirements.txt
python demo/build_index.py            # one-off: assemble the studio HTML
export CUDA_VISIBLE_DEVICES=0,1
python -m demo.app                    # → http://localhost:7860/
```

Both `PE_CKPT` and `DIT_CKPT` default to the released HF Hub repos;
first launch downloads ~110 GB and merges the DiT overlay shards into a
cached file under `$HF_HOME/context-scaling/`. Subsequent launches take
seconds.

The demo exposes three JSON endpoints:

| Endpoint | Input | Output | System prompt |
|---|---|---|---|
| `POST /api/generate_sp` | `{user_prompt, width, height}` | `{sp: ...}` | [`rft_iter0_with_ratio_v6_synthesis_student.txt`](demo/system_prompts/rft_iter0_with_ratio_v6_synthesis_student.txt) |
| `POST /api/caption_image` | `{image_base64}` | `{sp: ...}` | [`image2json.txt`](demo/system_prompts/image2json.txt) — bboxes on a 1000×1000 grid |
| `POST /api/generate_image` | `{prompt, width, height, num_steps, seed, cfg_scale}` | `{image_base64}` | — |

`generate_sp` and `caption_image` share the same VLM weights (loaded
once); the difference is text-only vs multimodal chat template + the
corresponding system prompt. `caption_image` accepts either raw base64
or a full `data:image/png;base64,…` URL:

```bash
IMG=$(base64 -w0 /path/to/some_image.png)
curl -s -X POST http://localhost:7860/api/caption_image \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMG\"}" | jq .sp
```

Full env-var reference (GPU pinning, HF token for gated repos, PE memory
caps) and Hugging Face Docker Space deployment recipe live in
[`demo/README.md`](demo/README.md).

### Measure caption quality of a dataset — ED + GPG

Score one or both metrics over an `(image, caption)` JSONL. Use this to
compare captions from different sources (human, GPT-4o, Gemini, your
prompter) on the same image pool.

```json
{"image": "/abs/path/to/img.png", "caption": "A short caption.", "uid": "row-0001"}
```

```bash
# ED only — no GPU, LLM-bound (Gemini-3-pro + GPT-4o).
export GEMINI_API_KEY_FILE=<path>       # one Gemini key per line
export GEMINI_API_CONFIG=<path>         # GPT-4o key JSON (matcher)
python -m evalkit.detailness.evaluator \
    --jsonl /path/to/dataset.jsonl \
    --out   /tmp/scored_ed.jsonl \
    --metrics ed

# ED + GPG — needs one GPU for Qwen3-VL-8B-Instruct forward passes.
python -m evalkit.detailness.evaluator \
    --jsonl /path/to/dataset.jsonl \
    --out   /tmp/scored_both.jsonl \
    --metrics ed,gpg \
    --gpg-model  <WEIGHTS_ROOT>/Qwen3-VL-8B-Instruct \
    --gpg-device cuda:0
```

Each output row is the input row plus:

```json
{"ed": 0.27, "ed_F05_A": 0.27, "ed_P_A": 1.0, "ed_R_A": 0.07,
 "gpg": 0.382, "gpg_total_nats": 289.4, "gpg_n_tokens": 758, ...}
```

Higher is more informative. See
[`evalkit/detailness/README.md`](evalkit/detailness/README.md) for
metric definitions, cache options, and per-metric CLIs.

## Serving the released models

The all-in-one demo above wraps PE + captioner + DiT behind one process.
Below is how to serve **each model on its own** — useful for batch jobs,
benchmark runs, or splitting PE and DiT across separate machines.

### 1. PE as a prompter (NL → SP)

Serve the VLM with vLLM's OpenAI-compatible endpoint (two GPUs, ~70 GB
weights in bf16):

```bash
vllm serve heheyas/SP-PE-Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 16384 \
  --trust-remote-code
```

Call it with the student system prompt to turn an NL request into SP
JSON:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("demo/system_prompts/rft_iter0_with_ratio_v6_synthesis_student.txt") as f:
    system = f.read()

resp = client.chat.completions.create(
    model="heheyas/SP-PE-Qwen3.5-35B-A3B",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": "a small orange cat on a windowsill"},
    ],
    max_tokens=4096, temperature=0.7,
)
print(resp.choices[0].message.content)   # <think>...</think>{SP JSON}
```

### 2. Captioner (image → SP) — same weights, different chat template

The captioner reuses the same vLLM endpoint. Only the chat template
(multimodal instead of text-only) and the system prompt change:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("demo/system_prompts/image2json.txt") as f:
    system = f.read()
img_b64 = base64.b64encode(open("photo.jpg", "rb").read()).decode()

resp = client.chat.completions.create(
    model="heheyas/SP-PE-Qwen3.5-35B-A3B",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]},
    ],
    max_tokens=4096, temperature=0.7,
)
print(resp.choices[0].message.content)   # SP JSON with bboxes on a 1000×1000 grid
```

### 3. DiT (SP → image)

The demo's `DiTBackend` is the easiest way to serve DiT standalone — it
pulls the base pipeline + overlay shards from HF Hub on first launch,
merges the shards into a single safetensors cached under `$HF_HOME/context-scaling/`,
and exposes a `.render()` method. Single 80 GB GPU is enough.

```python
from demo.backend.dit import DiTBackend
import json

dit = DiTBackend(
    dit_ckpt="heheyas/Qwen-Image-SP",
    base_repo="Qwen/Qwen-Image",
    device="cuda:0",
    dtype="bfloat16",
)
sp = json.dumps({...})   # SP JSON from PE or captioner above
img = dit.render(
    prompt=sp, height=1024, width=1024,
    num_steps=25, seed=42, cfg_scale=4.0,
)
img.save("out.png")
```

Or launch a standalone HTTP server (same DiT code that `training/scripts/serve.py`
uses under the hood) and POST SP JSON to `/generate`:

```bash
# First launch: pulls + merges (~55 GB from HF, one-time)
DIT_CKPT=heheyas/Qwen-Image-SP DIT_BASE_REPO=Qwen/Qwen-Image \
  python -m demo.app     # → POST http://localhost:7860/api/generate_image
```

Full env-var reference (GPU pinning, HF token for gated repos, cache
dir) lives in [`demo/README.md`](demo/README.md).

## Repo layout

Everything past the quickstart lives in per-package docs.

| Path | What it does | Doc |
|---|---|---|
| [`training/`](training/) | QwenImage DiT training + serving stack (FSDP, NaviT-packed batches, EMA, delta-merge / model-soup checkpoint tools). | [docs/training.md](docs/training.md) |
| [`evalkit/`](evalkit/) | Multi-benchmark T2I eval (9 benchmarks), structured-prompt prompter pipeline, and the ED/GPG detailness metrics. | [docs/evalkit.md](docs/evalkit.md) |
| [`pe-rsft/`](pe-rsft/) | Distributed rejection-sampling pipeline that generates SFT (and optionally DPO) data for the prompter. | [docs/pe-rsft.md](docs/pe-rsft.md) |
| [`demo/`](demo/) | The in-process FastAPI + browser studio wrapping PE, captioner, and DiT. | [demo/README.md](demo/README.md) |

## Citation

```bibtex
@article{chen2026contextscaling,
  title   = {Context Scaling: Scaling Properties of Text Conditioning in Visual Generation},
  author  = {Chen, Zilong and Deng, Chaorui and Li, Kunchang and Yuan, Hongyi and Fan, Haoqi},
  journal = {Technical Report, ByteDance Seed},
  year    = {2026}
}
```

## Acknowledgements

This release stands on the shoulders of several open-source projects.
We are grateful to their authors and maintainers.

- **[Qwen-Image](https://github.com/QwenLM/Qwen-Image)** (Alibaba) — DiT
  architecture, tokenizer, and reference pipeline that
  `training/modeling/qwenimage/` and the demo's `QwenImagePipeline` are
  built on. Also [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for
  `training/modeling/qwen2/`.
- **[Bagel](https://github.com/bytedance-seed/Bagel)** — provided the
  Bagel training runs used as the ground-truth MSE side of the ED/GPG
  scaling-law measurements in `evalkit/detailness/`.
- **[ms-swift](https://github.com/modelscope/ms-swift)** (ModelScope) —
  used to serve and fine-tune the prompt-expansion LLM (PE) throughout
  the RFT loop in `pe-rsft/`.
- **[huggingface/diffusers](https://github.com/huggingface/diffusers)**
  and **[huggingface/transformers](https://github.com/huggingface/transformers)**
  — modeling primitives underpinning `training/modeling/`.
- **Evaluation benchmarks** (vendored under `evalkit/context-scaling/benchmarks/`):
  GenEval, GenEval2, GenEval++ (Echo-4o), DPG-Bench (ELLA), OneIG-Bench,
  TIIF-Bench, T2I-CoReBench, GenExam, WISE — each used under its own
  upstream license (see per-benchmark `LICENSE` files and
  [NOTICE](NOTICE) for the full attribution list).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
