# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pack a teacher-rollout JSONL into an SFT training JSONL.

Format:
    {"messages": [{"role":"system","content":<student SP>},
                  {"role":"user","content":<original prompt>},
                  {"role":"assistant","content":"<think>\\n<analysis>\\n</think>\\n\\n<compact JSON>"}],
     "tools": []}

Reads:    ratio-cleaned rollout JSONL (output of ratio_clean.py)
Writes:   sft.jsonl — one row per successfully parsed rollout.

The system prompt is the STUDENT SP (`teacher_distill/system_prompts/student.txt`),
NOT the teacher SP — this is the contract the student will see at
inference time.
"""
import argparse
import json
import re
import sys
from typing import Tuple, Optional

try:
    from json_repair import repair_json, loads as repair_loads
    _HAS_REPAIR = True
except ImportError:
    _HAS_REPAIR = False
    print('[warn] json_repair not available; will skip malformed-JSON repair', file=sys.stderr)


def _compact_json(json_str: str) -> Optional[str]:
    """Parse JSON (try plain, then json_repair fallback) and re-emit compactly.
    Returns None on total failure."""
    try:
        obj = json.loads(json_str)
        return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
    except Exception:
        pass
    if _HAS_REPAIR:
        try:
            obj = repair_loads(json_str)
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            pass
    return None


def _split_analysis_and_json(rollout_text: str) -> Optional[Tuple[str, str]]:
    """rollout_text format: optional ``` fence + <analysis>...</analysis>{JSON}.
    Returns (analysis_content, json_str) or None on parse failure.
    """
    # strip optional outer markdown fence
    txt = rollout_text.strip()
    txt = re.sub(r'^```(?:json)?\s*', '', txt)
    txt = re.sub(r'\s*```\s*$', '', txt)

    # extract <analysis>...</analysis>
    m = re.search(r'<analysis>\s*(.*?)\s*</analysis>\s*(.*)', txt, re.DOTALL)
    if not m:
        return None
    analysis = m.group(1).strip()
    rest = m.group(2).strip()

    # rest should start with {; trim any trailing fence
    rest = re.sub(r'\s*```\s*$', '', rest)
    if not rest.startswith('{'):
        # find first '{'
        i = rest.find('{')
        if i < 0:
            return None
        rest = rest[i:]
    return analysis, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passed-in', required=True, help='passed_rollouts.jsonl')
    ap.add_argument('--system-prompt-file', required=True)
    ap.add_argument('--out', required=True, help='SFT JSONL output')
    ap.add_argument('--tools', default='[]',
                    help='JSON-encoded tools array (default: empty)')
    ap.add_argument('--max-rows', type=int, default=None)
    args = ap.parse_args()

    with open(args.system_prompt_file) as f:
        sp_text = f.read().rstrip()
    print(f'[SP] {args.system_prompt_file}: {len(sp_text)} chars')

    try:
        tools = json.loads(args.tools)
    except Exception:
        print(f'ERROR parsing --tools as JSON', file=sys.stderr)
        sys.exit(2)
    print(f'[tools] {len(tools)} tool(s) in array')

    n_in = 0
    n_out = 0
    n_split_fail = 0
    n_json_fail = 0
    n_json_repaired = 0
    with open(args.passed_in) as fin, open(args.out, 'w', encoding='utf-8') as fout:
        for line in fin:
            if args.max_rows is not None and n_out >= args.max_rows:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_in += 1
            rt = r.get('rollout_text', '')
            split = _split_analysis_and_json(rt)
            if split is None:
                n_split_fail += 1
                continue
            analysis, json_raw = split

            # normalize the JSON: compact, json_repair fallback if malformed
            try:
                json.loads(json_raw)
                compact = json.dumps(json.loads(json_raw), separators=(',', ':'), ensure_ascii=False)
            except Exception:
                compact = _compact_json(json_raw)
                if compact is None:
                    n_json_fail += 1
                    continue
                n_json_repaired += 1

            assistant_content = f'<think>\n{analysis}\n</think>\n\n{compact}'
            record = {
                'messages': [
                    {'role': 'system', 'content': sp_text},
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': assistant_content},
                ],
                'tools': tools,
                # extra metadata for traceability (downstream training pipeline usually ignores extras)
                '_meta': {
                    'uid': r.get('uid'),
                    'source': r.get('source'),
                    'data_source': r.get('data_source'),
                    'image_path': r.get('image_path'),
                    'image_mime': r.get('image_mime'),
                    'n_completion_tokens': r.get('n_completion_tokens'),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            n_out += 1

    print(f'\nread {n_in} rows, wrote {n_out}')
    print(f'  split fail (no <analysis> tag): {n_split_fail}')
    print(f'  json parse repaired via json_repair: {n_json_repaired}')
    print(f'  json parse total fail (dropped): {n_json_fail}')
    print(f'output: {args.out}')


if __name__ == '__main__':
    main()
