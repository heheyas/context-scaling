# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Parquet I/O + dataset path resolution.

The original implementation contained ByteDance-internal HDFS cluster
dispatch and Arnold platform paths. This open-source version exposes a
minimal facade: callers supply either a local directory or an explicit
`hdfs://<host>/<path>` URI, and `pyarrow.fs.HadoopFileSystem` resolves
the URI via the environment's `core-site.xml`.
"""

import io
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.fs as pf
import torch.distributed as dist
from decord import VideoReader
from PIL import Image


def get_hdfs_block_size() -> int:
    """Return HDFS block size (bytes). Defaults to 128MB."""
    return int(os.environ.get("HDFS_BLOCK_SIZE", 134217728))


def init_arrow_hdfs_fs(hdfs_path: str = "") -> pf.HadoopFileSystem:
    """Return a pyarrow HDFS filesystem.

    If `hdfs_path` starts with `hdfs://`, the host authority is parsed from
    it. Otherwise the env var `HDFS_DEFAULT_FS` is used, falling back to
    the empty host (pyarrow will then read `core-site.xml`).
    """
    host = ""
    if hdfs_path.startswith("hdfs://"):
        host = "hdfs://" + hdfs_path[len("hdfs://"):].split("/", 1)[0]
    else:
        host = os.environ.get("HDFS_DEFAULT_FS", "")
    return pf.HadoopFileSystem(
        host=host,
        port=0,
        buffer_size=get_hdfs_block_size(),
    )


def hdfs_ls_cmd(directory: str) -> list[str]:
    """List HDFS paths via the `hdfs dfs -ls` CLI."""
    result = subprocess.run(
        ["hdfs", "dfs", "-ls", directory],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [
        "hdfs://" + line.split("hdfs://")[-1].strip()
        for line in result.splitlines()
        if "hdfs://" in line
    ]


def read_image_from_hdfs(image_path: str) -> Image.Image:
    fs = init_arrow_hdfs_fs(image_path)
    with fs.open_input_file(image_path) as f:
        return Image.open(io.BytesIO(f.read()))


def read_video_from_hdfs(video_path: str):
    fs = init_arrow_hdfs_fs(video_path)
    with fs.open_input_file(video_path) as f:
        video_data = f.read()
    with tempfile.NamedTemporaryFile(suffix=Path(video_path).suffix.lower()) as temp:
        temp.write(video_data)
        temp.flush()
        return VideoReader(temp.name, num_threads=1)


def get_parquet_data_paths(
    data_dir_list,
    num_sampled_data_paths,
    rank: int = 0,
    world_size: int = 1,
    reverse_sample: bool = False,
):
    """Resolve and shard parquet file paths across distributed workers.

    Each entry in `data_dir_list` is either:
      - a local directory containing `*.parquet` files,
      - an `hdfs://` URI containing `*.parquet` files, or
      - a registered dataset name (resolved via env var
        `PARQUET_INDEX_ROOT`, expected to contain a `<name>.txt` file
        listing one parquet path per line).
    """
    num_dirs = len(data_dir_list)
    if world_size > 1:
        chunk = (num_dirs + world_size - 1) // world_size
        local_dirs = data_dir_list[rank * chunk : rank * chunk + chunk]
        local_nums = num_sampled_data_paths[rank * chunk : rank * chunk + chunk]
    else:
        local_dirs = data_dir_list
        local_nums = num_sampled_data_paths

    index_root = os.environ.get("PARQUET_INDEX_ROOT", "")
    local_paths: list[str] = []
    for data_dir, num_data_path in zip(local_dirs, local_nums):
        if data_dir.startswith("hdfs://"):
            files = hdfs_ls_cmd(data_dir)
            paths = [f for f in files if f.endswith(".parquet")]
        elif index_root and os.path.isfile(os.path.join(index_root, data_dir + ".txt")):
            with open(os.path.join(index_root, data_dir + ".txt"), encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip().endswith(".parquet")]
        else:
            files = os.listdir(data_dir)
            paths = [os.path.join(data_dir, n) for n in files if n.endswith(".parquet")]

        if not paths:
            raise RuntimeError(f"no parquet files resolved for data_dir={data_dir!r}")
        repeat = num_data_path // len(paths)
        paths = paths * (repeat + 1)
        if reverse_sample:
            paths = paths[::-1]
        local_paths.extend(paths[:num_data_path])

    if world_size > 1:
        gather: list = [None] * world_size
        dist.all_gather_object(gather, local_paths)
        combined: list[str] = []
        for chunk_list in gather:
            if chunk_list is not None:
                combined.extend(chunk_list)
        return combined
    return local_paths
