# Security Advisories

This document records third-party CVE matches surfaced by automated
license / vulnerability scanning, and our assessment of whether each one
is reachable from this repository's code paths.

## Scope

The advisories below cover code in `training/modeling/qwen2/` that was
derived from `huggingface/transformers` (see [NOTICE](../NOTICE)). The
upstream `transformers` package historically shipped a vulnerable code
path that lives in the same module family we vendored.

## CVEs

### CVE-2025-6921 (transformers)

- **Upstream**: `huggingface/transformers`
- **Match**: `training/modeling/qwen2/{__init__,configuration_qwen2,tokenization_qwen2,tokenization_qwen2_fast}.py`
- **Scanner reachability**: `2` (not reachable — vulnerable functions are
  not present in the vendored snippet)
- **Assessment**: Not reachable. The vulnerable upstream behavior lives in
  code paths that are not part of the ported Qwen2 modeling slice. Our
  vendored files implement only the model definition, configuration, and
  tokenization that the QwenImage text encoder needs at training and
  inference time.
- **Action**: None required. We will re-validate this assessment when we
  next sync against a newer `transformers` upstream (≥ v4.45 has the fix
  upstream).

### CVE-2024-3568 (transformers)

- **Upstream**: `huggingface/transformers`
- **Match**: same files as above.
- **Scanner reachability**: `2` (not reachable)
- **Assessment**: Same as CVE-2025-6921 — the vulnerable behavior is in a
  feature path the vendored snippet does not implement.
- **Action**: None required. Same revalidation plan as above.

## Methodology

Reachability was assessed by the automated component scanner used at
release time. A `reachable: 2` value indicates the scanner walked the
import / call graph of the vulnerable functions and could not find a
path from a public entry point in this repository to those functions.

If you discover a reachable path that contradicts this assessment,
please report it via the channel described in [SECURITY.md](../SECURITY.md).

## Re-validation cadence

These advisories will be re-checked:

1. Whenever the vendored files under `training/modeling/qwen2/` are
   refreshed against a newer `transformers` upstream.
2. Whenever the scanner is re-run as part of a release-readiness check.

Any future change in reachability will be recorded as a new entry in
this document, and a corresponding patch will be issued.
