# Model Card: QwenImage trained on structured prompts

## Model description

A QwenImage text-to-image diffusion transformer fine-tuned on
**structured-prompt JSON** inputs as part of the *Context-Scaling*
project. The model accepts long, JSON-formatted prompts (intent / scene
description / element list with bounding boxes / relationships / style)
and renders an image at 1024×1024 (with support up to 1536×1536 at
inference time).

- **Architecture** — 60-layer QwenImage DiT, Qwen2.5 text encoder
  (frozen), causal 3D VAE; dual-stream image/text attention; 3D
  multi-axis RoPE `(16, 56, 56)`.
- **Training framework** — FSDP + bf16 + flash-attn + flex-attention
  with EMA shadow weights, implemented in
  [`training/`](../training/).
- **Base weights** — Continued-trained from the public QwenImage
  release.

## Intended use

- Text-to-image generation from natural language **or** from a
  structured-prompt JSON produced by the paper's prompter LLM
  (see [`demo/`](../demo/) for the released prompter + DiT stack).
- Reproducing the benchmark results reported by the Context-Scaling
  project on the 9 benchmarks supported by
  [`evalkit/context-scaling/`](../evalkit/context-scaling/).

The model is **research / reference quality**. It is not intended for
production deployment without additional safety filtering.

## How to use

Serve the checkpoint with one of the entry points in
[`training/scripts/`](../training/scripts/):

```bash
python training/scripts/serve.py --ckpt-root <path/to/checkpoint>
```

then POST a JSON payload to `/generate` (see
[`docs/training.md`](training.md)). For the prompter + DiT pipeline
that exercises this checkpoint end-to-end with a natural-language
prompt, see [`docs/evalkit.md`](evalkit.md).

## Limitations

- Long-prompt fidelity degrades for prompts beyond the structured-prompt
  schema the model was trained on.
- Bounding-box adherence is approximate; the model is not a layout
  generator.
- Text rendering quality is uneven across scripts and font weights.
- No safety filter is bundled — apply your own content moderation.

## License and attribution

Apache License 2.0. Derivative of `huggingface/diffusers` QwenImage and
`huggingface/transformers` Qwen2 modeling code; full attribution in
[NOTICE](../NOTICE).

## Citation

```bibtex
@misc{context-scaling-2026,
  title  = {Text Prompt Scaling Law in Visual Generation},
  author = {ByteDance Ltd. and/or its affiliates},
  year   = {2026}
}
```
