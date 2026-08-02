#!/usr/bin/env python3
"""
Count the total number of binary (yes/no) questions in all .jsonl files
under a given directory.  Each line in a .jsonl file is a JSON object
with a key "yn_question_list" that holds a list of binary questions.

Usage:
    python count_binary_questions.py \
        /Users/weixinyu/Desktop/Research/FITBench/TIIF-Bench/data/test_eval_prompts
"""

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    """Yield all .jsonl files under *root* (recursively)."""
    return root.rglob("*.jsonl")


def count_questions_in_file(fp: Path) -> int:
    """Return the total number of binary questions in one .jsonl file."""
    total = 0
    with fp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {fp} (line {line_no}): {e}"
                ) from e
            if "yn_question_list" not in obj or not isinstance(
                obj["yn_question_list"], list
            ):
                raise ValueError(
                    f'Missing or invalid "yn_question_list" in {fp} (line {line_no})'
                )
            total += len(obj["yn_question_list"])
    return total


def main(dir_path: Path) -> None:
    if not dir_path.is_dir():
        raise SystemExit(f"Error: {dir_path} is not a directory.")

    grand_total = 0
    for jf in iter_jsonl_files(dir_path):
        grand_total += count_questions_in_file(jf)

    print(
        f"Total binary questions found in '{dir_path}': {grand_total:,}"
    )  # comma-separated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count binary questions across .jsonl files."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory containing .jsonl files.",
    )
    args = parser.parse_args()
    main(args.directory)
