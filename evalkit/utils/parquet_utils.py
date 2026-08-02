# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""HDFS / parquet helpers.

The original implementation contained ByteDance-internal dispatch logic for
multiple HDFS clusters and Arnold platform conventions. This open-source
version provides a minimal facade: callers may either use a local filesystem
or pass an explicit `hdfs://...` URI and let `pyarrow.fs.HadoopFileSystem`
resolve it via the environment's `core-site.xml`.
"""

import io
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.fs as pf
from PIL import Image


def get_hdfs_host() -> str:
    """Return the HDFS default URI.

    Resolution order:
      1. `HDFS_DEFAULT_FS` environment variable
      2. `fs.defaultFS` from `core-site.xml` at `HADOOP_CONF_DIR`
      3. Empty string (caller should pass full `hdfs://...` URIs)
    """
    if "HDFS_DEFAULT_FS" in os.environ:
        return os.environ["HDFS_DEFAULT_FS"]
    conf_dir = os.environ.get("HADOOP_CONF_DIR", "/etc/hadoop/conf")
    core_site = Path(conf_dir) / "core-site.xml"
    if core_site.is_file():
        try:
            tree = ET.parse(core_site)
            for prop in tree.getroot():
                if prop.tag != "property":
                    continue
                name = prop.find("name")
                value = prop.find("value")
                if name is not None and name.text == "fs.defaultFS" and value is not None:
                    return value.text or ""
        except ET.ParseError:
            pass
    return ""


def get_hdfs_block_size() -> int:
    """Return HDFS block size (bytes). Defaults to 128MB."""
    return int(os.environ.get("HDFS_BLOCK_SIZE", 134217728))


def init_arrow_hdfs_fs(hdfs_path: str = "") -> pf.HadoopFileSystem:
    """Initialize a pyarrow HDFS filesystem handle.

    If `hdfs_path` starts with `hdfs://`, the host is parsed from it.
    Otherwise the default URI from `get_hdfs_host()` is used.
    """
    host = ""
    if hdfs_path.startswith("hdfs://"):
        # Take just the `hdfs://<authority>` prefix
        rest = hdfs_path[len("hdfs://"):]
        host = "hdfs://" + rest.split("/", 1)[0]
    else:
        host = get_hdfs_host()
    return pf.HadoopFileSystem(
        host=host,
        port=0,
        buffer_size=get_hdfs_block_size(),
    )


def hdfs_ls_cmd(directory: str) -> list[str]:
    """List HDFS paths under `directory` via the `hdfs dfs -ls` CLI."""
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
    """Read a single image file from HDFS into a PIL Image."""
    fs = init_arrow_hdfs_fs(image_path)
    with fs.open_input_file(image_path) as f:
        return Image.open(io.BytesIO(f.read()))
