# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Output formatters: save generated images in benchmark-native evaluation format."""

import json
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

log = logging.getLogger(__name__)

# OneIG category name → directory name
ONEIG_CATEGORY_MAP = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


def make_grid(images: List[Image.Image], rows: int = 2, cols: int = 2) -> Image.Image:
    """Concatenate images into a rows x cols grid. Resizes all to same size."""
    if not images:
        raise ValueError("No images to grid")
    # Use first image's size as target
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows))
    for idx, img in enumerate(images):
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


class OutputFormatter(ABC):
    """Base class for benchmark-specific output formatting."""

    def __init__(self, output_dir: str, **kwargs):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def get_num_images(self, item: dict) -> int:
        """How many images to generate for this item."""

    @abstractmethod
    def get_save_path(self, item: dict, image_idx: int) -> str:
        """Return save path for the image_idx-th generated image."""

    @abstractmethod
    def get_final_path(self, item: dict) -> str:
        """Return the final output path (for checkpoint: if exists, skip item)."""

    def on_item_complete(self, item: dict, image_paths: List[str]):
        """Called after all images for an item are generated. Override for grid/metadata."""
        pass

    def finalize(self, items: List[dict]):
        """Called after ALL items are done. Override for global metadata files."""
        pass


class GenEval2Formatter(OutputFormatter):
    """GenEval2: flat images + image_filepath_data.json mapping original_prompt → path."""

    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir)
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def get_num_images(self, item):
        return 1

    def get_save_path(self, item, image_idx):
        return os.path.join(self.images_dir, f"{item['id']}.png")

    def get_final_path(self, item):
        return os.path.join(self.images_dir, f"{item['id']}.png")

    def finalize(self, items):
        mapping = {}
        for item in items:
            img_path = os.path.abspath(self.get_final_path(item))
            if os.path.exists(img_path):
                mapping[item["original_prompt"]] = img_path
        out_path = os.path.join(self.output_dir, "image_filepath_data.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        log.info("GenEval2: wrote %d entries to %s", len(mapping), out_path)


class GenEvalFormatter(OutputFormatter):
    """GenEval: {id}/samples/{i}.png + {id}/metadata.jsonl."""

    def __init__(self, output_dir: str, benchmark_data: str = None, **kwargs):
        super().__init__(output_dir)
        self.benchmark_data = benchmark_data
        self._metadata_lines = {}
        if benchmark_data:
            with open(benchmark_data, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line:
                        self._metadata_lines[str(idx).zfill(5)] = line

    def get_num_images(self, item):
        return 1

    def get_save_path(self, item, image_idx):
        return os.path.join(self.output_dir, item["id"], "samples", f"{image_idx}.png")

    def get_final_path(self, item):
        return os.path.join(self.output_dir, item["id"], "samples", "0.png")

    def on_item_complete(self, item, image_paths):
        # Write metadata.jsonl (single line JSON, matching evaluation_metadata.jsonl)
        meta_path = os.path.join(self.output_dir, item["id"], "metadata.jsonl")
        if item["id"] in self._metadata_lines:
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(self._metadata_lines[item["id"]] + "\n")


class OneIGFormatter(OutputFormatter):
    """OneIG-Bench: {category_dir}/{model_name}/{id}.webp (2x2 grid)."""

    def __init__(self, output_dir: str, model_name: str = "model", **kwargs):
        super().__init__(output_dir)
        self.model_name = model_name
        self.tmp_dir = os.path.join(output_dir, "_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def get_num_images(self, item):
        return 4

    def _parse_id(self, item):
        """Parse item id. Supports 'category/idx' format (new) and plain 'idx' (legacy)."""
        item_id = item["id"]
        if "/" in item_id:
            cat, idx = item_id.split("/", 1)
            cat_dir = ONEIG_CATEGORY_MAP.get(cat, cat.lower())
            return cat_dir, idx
        cat = item.get("category", "")
        cat_dir = ONEIG_CATEGORY_MAP.get(cat, cat.lower())
        return cat_dir, item_id

    def get_save_path(self, item, image_idx):
        cat_dir, idx = self._parse_id(item)
        return os.path.join(self.tmp_dir, f"{cat_dir}_{idx}_{image_idx}.png")

    def get_final_path(self, item):
        cat_dir, idx = self._parse_id(item)
        return os.path.join(self.output_dir, cat_dir, self.model_name, f"{idx}.webp")

    def on_item_complete(self, item, image_paths):
        final_path = self.get_final_path(item)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        images = []
        for p in sorted(image_paths):
            if os.path.exists(p):
                images.append(Image.open(p))
        if len(images) == 1:
            # Single image: save directly, no grid
            images[0].save(final_path, format="WEBP")
        else:
            if len(images) < 4:
                # Pad with black images
                w, h = images[0].size if images else (1024, 1024)
                while len(images) < 4:
                    images.append(Image.new("RGB", (w, h), (0, 0, 0)))
            grid = make_grid(images, rows=2, cols=2)
            grid.save(final_path, format="WEBP")
        # Clean up tmp files
        for p in image_paths:
            if os.path.exists(p):
                os.remove(p)


class DPGBenchFormatter(OutputFormatter):
    """DPG-Bench: {item_id}.png (2x2 grid)."""

    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir)
        self.tmp_dir = os.path.join(output_dir, "_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def get_num_images(self, item):
        return 4

    def get_save_path(self, item, image_idx):
        return os.path.join(self.tmp_dir, f"{item['id']}_{image_idx}.png")

    def get_final_path(self, item):
        return os.path.join(self.output_dir, f"{item['id']}.png")

    def on_item_complete(self, item, image_paths):
        final_path = self.get_final_path(item)
        images = []
        for p in sorted(image_paths):
            if os.path.exists(p):
                images.append(Image.open(p))
        if len(images) == 1:
            # Single image: save directly, no grid
            images[0].save(final_path, format="PNG")
        else:
            if len(images) < 4:
                w, h = images[0].size if images else (1024, 1024)
                while len(images) < 4:
                    images.append(Image.new("RGB", (w, h), (0, 0, 0)))
            grid = make_grid(images, rows=2, cols=2)
            grid.save(final_path, format="PNG")
        # Clean up tmp files
        for p in image_paths:
            if os.path.exists(p):
                os.remove(p)


class GenExamFormatter(OutputFormatter):
    """GenExam: flat {id}.png images, 1 per prompt."""

    def get_num_images(self, item):
        return 1

    def get_save_path(self, item, image_idx):
        return os.path.join(self.output_dir, f"{item['id']}.png")

    def get_final_path(self, item):
        return os.path.join(self.output_dir, f"{item['id']}.png")


class WISEFormatter(OutputFormatter):
    """WISE: flat {prompt_id}.png images, 1 per prompt."""

    def get_num_images(self, item):
        return 1

    def get_save_path(self, item, image_idx):
        return os.path.join(self.output_dir, f"{item['id']}.png")

    def get_final_path(self, item):
        return os.path.join(self.output_dir, f"{item['id']}.png")


class TIIFBenchFormatter(OutputFormatter):
    """TIIF-Bench: {type}/{model_name}/{desc_type}/{idx}.png, 1 per prompt.

    Matches eval_with_vlm.py expected path:
        image_dir/{attr_type}/{eval_model}/{short_description|long_description}/{idx}.png

    Item ID format: "{type}/{idx}".
    """

    def __init__(self, output_dir: str, model_name: str = "model",
                 desc_type: str = "short_description", **kwargs):
        super().__init__(output_dir)
        self.model_name = model_name
        self.desc_type = desc_type

    def get_num_images(self, item):
        return 1

    def _parse_id(self, item):
        # id format: "{type}/{idx}"
        type_name, idx = item["id"].rsplit("/", 1)
        return type_name, idx

    def get_save_path(self, item, image_idx):
        type_name, idx = self._parse_id(item)
        return os.path.join(
            self.output_dir, type_name, self.model_name, self.desc_type, f"{idx}.png"
        )

    def get_final_path(self, item):
        return self.get_save_path(item, 0)


class TextBenchFormatter(OutputFormatter):
    """X-Omni TextBench: {prompt_id:04d}_{repeat}.png, 4 per prompt.

    repeat is 1-indexed (1,2,3,4) matching eval.sh expectations.
    """

    def get_num_images(self, item):
        return 4

    def get_save_path(self, item, image_idx):
        pid = int(item["id"])
        return os.path.join(self.output_dir, f"{pid:04d}_{image_idx + 1}.png")

    def get_final_path(self, item):
        # Use 4th image as completion marker
        pid = int(item["id"])
        return os.path.join(self.output_dir, f"{pid:04d}_4.png")


def get_formatter(benchmark: str, **kwargs) -> OutputFormatter:
    """Factory: create the right formatter for the benchmark."""
    if benchmark in ("tiif_short", "tiif_testmini_short"):
        return TIIFBenchFormatter(desc_type="short_description", **kwargs)
    if benchmark in ("tiif_long", "tiif_testmini_long"):
        return TIIFBenchFormatter(desc_type="long_description", **kwargs)

    formatters = {
        "geneval2": GenEval2Formatter,
        "genevalpp": GenExamFormatter,  # flat {id}.png, 1 per prompt
        "sp_l10_200": GenEval2Formatter,  # same flat PNG layout as GenEval2
        "sp_sft_200": GenEval2Formatter,
        "sp_rl_200": GenEval2Formatter,
        "geneval": GenEvalFormatter,
        "oneig": OneIGFormatter,
        "dpgbench": DPGBenchFormatter,
        "genexam": GenExamFormatter,
        "wise": WISEFormatter,
        "textbench": TextBenchFormatter,
        "alignment_v5": GenExamFormatter,  # flat {id}.png, 1 per prompt
        "alignment_v5_long": GenExamFormatter,  # flat {id}.png, 1 per prompt
        "corebench": GenExamFormatter,  # flat {id}.png, 1 per prompt
        "infograph": GenExamFormatter,  # flat {id}.png, 1 per prompt
    }
    if benchmark not in formatters:
        # Default: flat {id}.png, 1 image per prompt
        log.warning("Unknown benchmark '%s', using flat PNG formatter (1 img/prompt)", benchmark)
        return GenExamFormatter(**kwargs)
    return formatters[benchmark](**kwargs)
