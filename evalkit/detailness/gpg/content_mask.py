# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Build a per-character content/scaffold mask for a caption string.

For JSON captions: VALUES (string contents, numeric values, array elements)
are content; KEYS, brackets, quotes, separators are scaffold.

For non-JSON captions: all chars are content (no masking).

Used for "content-only GPG" to remove cross-field interaction noise that
appears when only summing PMI over scaffold-laden full captions.
"""

from __future__ import annotations

import json
from typing import List, Tuple


# Fields whose VALUES are treated as scaffold (not content) regardless
# of where they appear. These are metadata-like fields that don't
# correspond to grounded visual content. Removing them from training
# captions has near-zero impact on Bagel MSE (verified by ablation runs),
# yet their tokens have ≈0 PMI under VLM judges — counting them adds
# noise to GPG without adding signal.
SCAFFOLD_KEYS = {"depth", "id"}


def _serialize_with_mask(obj) -> Tuple[str, List[bool]]:
    out: List[str] = []
    mask: List[bool] = []

    def emit(s: str, is_content: bool) -> None:
        out.append(s)
        mask.extend([is_content] * len(s))

    def walk(node, in_value: bool, parent_key: str = None):
        if isinstance(node, dict):
            emit("{", False)
            first = True
            for k, v in node.items():
                if not first:
                    emit(",", False)
                first = False
                key_str = json.dumps(str(k), ensure_ascii=False)
                emit(key_str, False)  # key is scaffold
                emit(":", False)
                # Values are content (True), unless the key is in SCAFFOLD_KEYS
                child_in_value = False if str(k) in SCAFFOLD_KEYS else True
                walk(v, child_in_value, parent_key=str(k))
            emit("}", False)
        elif isinstance(node, list):
            emit("[", False)
            first = True
            for x in node:
                if not first:
                    emit(",", False)
                first = False
                walk(x, in_value, parent_key=parent_key)
            emit("]", False)
        elif isinstance(node, str):
            emit('"', False)
            body = json.dumps(node, ensure_ascii=False)[1:-1]
            emit(body, in_value)
            emit('"', False)
        elif isinstance(node, bool):
            emit("true" if node else "false", in_value)
        elif node is None:
            emit("null", in_value)
        else:  # int / float
            emit(json.dumps(node), in_value)

    walk(obj, False)
    return "".join(out), mask


def content_mask_for_caption(caption: str) -> List[bool]:
    """Returns a list of bool, len == len(caption). True = content char.

    If the caption is valid compact JSON whose re-serialization matches the
    input exactly, returns a JSON-aware mask. Otherwise returns all-True.
    """
    try:
        obj = json.loads(caption)
    except Exception:
        return [True] * len(caption)
    rendered, mask = _serialize_with_mask(obj)
    if rendered == caption:
        return mask
    # Compact form differs (e.g., key ordering, escaping). Try to detect
    # what changed; for the safe case, fall back to all-content.
    return [True] * len(caption)


def token_content_mask(caption: str, tokenizer, content_threshold: float = 0.5) -> List[bool]:
    """Tokenize caption and return per-token content mask.

    A token is "content" if more than `content_threshold` of its character
    span is content. Tokens straddling a value/scaffold boundary get
    classified by majority overlap.
    """
    char_mask = content_mask_for_caption(caption)
    enc = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    for start, end in offsets:
        if end <= start:
            out.append(False)
            continue
        span = char_mask[start:end]
        if not span:
            out.append(False)
            continue
        frac = sum(1 for c in span if c) / len(span)
        out.append(frac >= content_threshold)
    return out


if __name__ == "__main__":
    # Smoke test
    s = '{"intent":"A woman holds a jar","style":"photo","elements":[{"id":1,"caption":"Woman in blue","position":[93,203,782,997]}]}'
    cm = content_mask_for_caption(s)
    print(f"caption: {s}")
    print(f"length: {len(s)}, content chars: {sum(cm)} ({sum(cm)/len(s)*100:.1f}%)")
    # Show which chars are content
    print("char    content?")
    for c, m in zip(s, cm):
        if m:
            print(f"  {c!r:6s} ✓")
