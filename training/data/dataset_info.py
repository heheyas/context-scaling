# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Dataset registry.

This file imports dataset loader classes from sibling modules and exposes
two registries:

  - DATASET_REGISTRY: maps a dataset *kind* (e.g. "t2i_ct_json") to the
    loader class used to materialize samples.
  - DATASET_INFO:     maps a dataset *kind* to a dict of named datasets
    (parquet root, file count, sample count, etc.).

The open-source release ships only the T2I QwenImage-relevant loaders in
`t2i_dataset_navit.py`. Other loader classes used by the original
internal CausalFusion (Bagel) model — interleave editing, VLM
understanding, video, etc. — are NOT included; references to them remain
in DATASET_REGISTRY for completeness but resolve to a NotImplementedError
stub at runtime.
"""

from .t2i_dataset_navit import (
    T2Iv2IterableDataset,
    T2IFluxIterableDataset,
    T2ISeedreamIterableDataset,
    T2VIterableDataset,
    T2IJsonPromptIterableDataset,
    T2IJsonMixedLevelIterableDataset,
    T2IDensePromptIterableDataset,
    T2IQTJsonMixedLevelIterableDataset,
)


# --- Stub for loaders that are NOT part of this OSS release. ----------------
# The full implementations live in the internal codebase and are not shipped
# here. Looking these names up in DATASET_REGISTRY succeeds; *instantiating*
# them raises NotImplementedError with a clear message.

class _MissingDatasetStub:
    """Placeholder for dataset classes not shipped in the OSS release."""

    def __init__(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__} is not included in the open-source release. "
            "This loader is part of the internal CausalFusion (Bagel) codebase. "
            "Use one of the T2I*IterableDataset classes from t2i_dataset_navit.py "
            "for QwenImage training, or implement your own loader."
        )


def _make_stub(name: str) -> type:
    return type(name, (_MissingDatasetStub,), {})


# Interleave editing / video / multi-modal loaders (Bagel-only)
UnifiedEditIterableDataset = _make_stub("UnifiedEditIterableDataset")
SEditIterableDataset = _make_stub("SEditIterableDataset")
OmniEditAugIterableDataset = _make_stub("OmniEditAugIterableDataset")
HQEditIterableDataset = _make_stub("HQEditIterableDataset")
SeedXEditP1IterableDataset = _make_stub("SeedXEditP1IterableDataset")
UltraEditIterableDataset = _make_stub("UltraEditIterableDataset")
VideoWMv2BiIterableDataset = _make_stub("VideoWMv2BiIterableDataset")
VideoWMv3MultiIterableDataset = _make_stub("VideoWMv3MultiIterableDataset")
InstructionVideoWMIterableDataset = _make_stub("InstructionVideoWMIterableDataset")
VT2VIterableDataset = _make_stub("VT2VIterableDataset")
OmniGenIterableDataset = _make_stub("OmniGenIterableDataset")
MVImgNetIterableDataset = _make_stub("MVImgNetIterableDataset")
ObjaverseIterableDataset = _make_stub("ObjaverseIterableDataset")
WebOmniV1IterableDataset = _make_stub("WebOmniV1IterableDataset")
DepthIterableDataset = _make_stub("DepthIterableDataset")
InstructionWebIterableDataset = _make_stub("InstructionWebIterableDataset")
ThinkGenIterableDataset = _make_stub("ThinkGenIterableDataset")
DeAugmentationIteratbleDataset = _make_stub("DeAugmentationIteratbleDataset")
TicTacToeIterableDataset = _make_stub("TicTacToeIterableDataset")
WebOmniV2IterableDataset = _make_stub("WebOmniV2IterableDataset")
ThinkZoomIterableDataset = _make_stub("ThinkZoomIterableDataset")
ImgEditIterableDataset = _make_stub("ImgEditIterableDataset")
PCTEditIterableDataset = _make_stub("PCTEditIterableDataset")
UnifiedMultiIterableDataset = _make_stub("UnifiedMultiIterableDataset")
StyleIterableDataset = _make_stub("StyleIterableDataset")
ThinkGenNewIterableDataset = _make_stub("ThinkGenNewIterableDataset")
ReCAAugmentationIteratbleDataset = _make_stub("ReCAAugmentationIteratbleDataset")

# VLM understanding loaders (Bagel-only)
VLMIterableDataset = _make_stub("VLMIterableDataset")
SftJSONLIterableDataset = _make_stub("SftJSONLIterableDataset")
SftParquetIterableDataset = _make_stub("SftParquetIterableDataset")
TextIterableDataset = _make_stub("TextIterableDataset")
Text_VLMIterableDataset = _make_stub("Text_VLMIterableDataset")

# Unified T2I + edit loader (Bagel-only)
T2IMagusIterableDataset = _make_stub("T2IMagusIterableDataset")


DATASET_REGISTRY = {
    'unified_t2i_pretrain': T2IMagusIterableDataset,
    'unified_t2i_pretrain_merge': T2IMagusIterableDataset,
    'unified_t2i_ct': T2IMagusIterableDataset,
    'unified_editing_ct': T2IMagusIterableDataset,
    'unified_editing_ct_origin': T2IMagusIterableDataset,
    'unified_editing_ct_origin_v30l': T2IMagusIterableDataset,
    'unified_editing_ct_origin_global': T2IMagusIterableDataset,
    'unified_editing_ct_origin_local': T2IMagusIterableDataset,
    'unified_editing_ct_origin_none': T2IMagusIterableDataset,
    'unified_editing_ct_origin_restore': T2IMagusIterableDataset,
    'unified_interleaved': T2IMagusIterableDataset,
    'unified_editing_multi': T2IMagusIterableDataset,
    't2i_pretrain': T2Iv2IterableDataset,
    't2i_sft_flux': T2IFluxIterableDataset,
    't2i_sft_hq': T2IFluxIterableDataset,
    't2i_sft_civitai': T2IFluxIterableDataset,
    't2i_sft_seedream': T2ISeedreamIterableDataset,
    't2i_ct_dense': T2IDensePromptIterableDataset,
    't2i_ct_json': T2IJsonPromptIterableDataset,
    't2i_ct_json_mixed': T2IJsonMixedLevelIterableDataset,
    't2i_ct_qt_json_mixed': T2IQTJsonMixedLevelIterableDataset,
    't2i_sft_json': T2IJsonPromptIterableDataset,
    't2i_sft_json_mixed': T2IJsonMixedLevelIterableDataset,
    't2i_rl_json': T2IJsonPromptIterableDataset,
    't2i_rl_json_mixed': T2IJsonMixedLevelIterableDataset,
    't2i_cartoon': T2IFluxIterableDataset,
    't2v_pretrain': T2VIterableDataset,
    'vlm_pretrain': VLMIterableDataset,
    'vlm_pretrain_msv7': VLMIterableDataset,
    'vlm_pretrain_text': Text_VLMIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'vlm_sft_text': SftJSONLIterableDataset,
    'vlm_stage1': SftJSONLIterableDataset,
    'vlm_sft_parquet': SftParquetIterableDataset,
    'vlm_sft_text_parquet': SftParquetIterableDataset,
    'text': TextIterableDataset,
    'ultraedit': UltraEditIterableDataset,
    'seedxeditp1': SeedXEditP1IterableDataset,
    'hqedit': HQEditIterableDataset,
    'omniedit_aug': OmniEditAugIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'unified_multi': UnifiedMultiIterableDataset,
    'omnigen': OmniGenIterableDataset,
    'sedit': SEditIterableDataset,
    'videowmv2bi': VideoWMv2BiIterableDataset,
    'videowmv3multi': VideoWMv3MultiIterableDataset,
    'vt2v': VT2VIterableDataset,
    'mvimgnet': MVImgNetIterableDataset,
    'objaverse': ObjaverseIterableDataset,
    'webomni': WebOmniV1IterableDataset,
    'webomniv2': WebOmniV2IterableDataset,
    'depth': DepthIterableDataset,
    'instructvideowm': InstructionVideoWMIterableDataset, 
    'instruct_packed_thinkgen': InstructionWebIterableDataset, 
    'packed_thinkgen': ThinkGenIterableDataset,
    'thinkgen_new': ThinkGenNewIterableDataset,
    'deaugmentation': DeAugmentationIteratbleDataset,
    'tic-tac-toe': TicTacToeIterableDataset,
    'thinkzoom': ThinkZoomIterableDataset,
    'imgedit': ImgEditIterableDataset,
    'pct_edit_visual': PCTEditIterableDataset,
    'pct_edit_single': PCTEditIterableDataset,
    'pct_edit_multi': PCTEditIterableDataset,
    'style_ref': StyleIterableDataset,
    'reca_augmentation': ReCAAugmentationIteratbleDataset,
}

# ----------------------------------------------------------------------------
# DATASET_INFO — dataset registry (placeholders only)
# ----------------------------------------------------------------------------
#
# This is a *template*. Each entry maps:
#
#     <dataset-kind> ->
#         <dataset-name> ->
#             {
#                 'data_dir':          <path to the parquet root>,
#                 'num_files':         <int: number of parquet shards>,
#                 'num_total_samples': <int: total sample count>,
#             }
#
# `<dataset-kind>` must be a key registered in `DATASET_REGISTRY` above —
# this selects the loader class used to materialize samples.
#
# `data_dir` can be:
#   - a local directory containing `*.parquet` files,
#   - an HDFS URI starting with `hdfs://<authority>/<path>`,
#   - or a registered dataset name resolvable via the
#     `PARQUET_INDEX_ROOT` env var (see `parquet_utils.resolve_parquet_paths`).
#
# To use this repository for training:
#   1. Add one entry per dataset you want to train on, replacing
#      `<DATA_ROOT>` and `<DATASET_NAME>` with your real values.
#   2. Reference the dataset by its `<dataset-name>` key from your YAML
#      mixture config in `data/configs/json/*.yaml`.
# ----------------------------------------------------------------------------

DATASET_INFO: dict = {
    # Example: structured-prompt + dense-caption T2I pretrain
    't2i_ct_json': {
        '<DATASET_NAME>': {
            'data_dir': '<DATA_ROOT>/<dataset-name>',
            'num_files': 0,
            'num_total_samples': 0,
        },
    },
    # Example: structured-prompt + dense-caption T2I SFT
    't2i_sft_json': {
        '<DATASET_NAME>': {
            'data_dir': '<DATA_ROOT>/<dataset-name>',
            'num_files': 0,
            'num_total_samples': 0,
        },
    },
    # Example: dense-only T2I pretrain
    't2i_ct_dense': {
        '<DATASET_NAME>': {
            'data_dir': '<DATA_ROOT>/<dataset-name>',
            'num_files': 0,
            'num_total_samples': 0,
        },
    },
    # Example: Seedream / Flux T2I SFT
    't2i_sft_seedream': {
        '<DATASET_NAME>': {
            'data_dir': '<DATA_ROOT>/<dataset-name>',
            'num_files': 0,
            'num_total_samples': 0,
        },
    },
    # Example: QT (QwenImage thinking) T2I JSON mixed-level
    't2i_ct_qt_json_mixed': {
        '<DATASET_NAME>': {
            'data_dir': '<DATA_ROOT>/<dataset-name>',
            'num_files': 0,
            'num_total_samples': 0,
        },
    },
}
