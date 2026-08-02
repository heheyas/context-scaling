# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates on 2026-05-01
#
# Original file from huggingface/transformers was released under Apache-2.0,
# full license text available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.

"""
NaviT text encoder: packed Qwen2.5-VL language backbone using flash_attn_varlen.

Two implementations:
  1. forward_text_encoder_navit: HF-native batched forward (pad/unpad). Simple, exact.
  2. PackedQwen2TextEncoder: Standalone flash_attn_varlen implementation. Fast, training.

Weight-compatible with Qwen2_5_VLTextModel (same param names:
    embed_tokens, layers.N.self_attn.{q,k,v,o}_proj, layers.N.{input_layernorm,
    post_attention_layernorm}, layers.N.mlp.{gate,up,down}_proj, norm).
"""

from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func


# ---------------------------------------------------------------------------
# RoPE utilities (standard RoPE; for text-only M-RoPE all 3 axes are identical)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope(position_ids, head_dim, rope_theta, device, dtype):
    """Compute cos/sin for standard RoPE.

    Parameters
    ----------
    position_ids : [T] int64
    head_dim : int
    rope_theta : float
    device, dtype : torch device/dtype

    Returns
    -------
    cos : [T, head_dim]
    sin : [T, head_dim]
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    # [T] x [head_dim//2] -> [T, head_dim//2]
    freqs = position_ids.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    # Duplicate to full head_dim: [T, head_dim]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(q, k, cos, sin):
    """Apply RoPE to packed q, k tensors.

    q, k : [T, H, D]
    cos, sin : [T, D]
    """
    cos_u = cos.unsqueeze(1)  # [T, 1, D]
    sin_u = sin.unsqueeze(1)
    q = (q.float() * cos_u + rotate_half(q.float()) * sin_u).to(q.dtype)
    k = (k.float() * cos_u + rotate_half(k.float()) * sin_u).to(k.dtype)
    return q, k


# ---------------------------------------------------------------------------
# RMSNorm (matches Qwen2RMSNorm: param named `weight`, eps named `variance_epsilon`)
# ---------------------------------------------------------------------------

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# MLP (matches Qwen2MLP: gate_proj, up_proj, down_proj)
# ---------------------------------------------------------------------------

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention (matches Qwen2Attention: q/k/v/o_proj, q_norm, k_norm)
# ---------------------------------------------------------------------------

class PackedQwen2TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK norm (optional — Qwen2.5-VL-7B doesn't have it)
        qk_norm = getattr(config, 'qk_norm', False)
        if qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, hidden, cos, sin, cu_seqlens, max_seqlen):
        """
        hidden : [T, D]
        cos, sin : [T, head_dim]
        cu_seqlens : [N+1] int32
        max_seqlen : int
        """
        q = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(-1, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )  # [T, H, D_h]

        return self.o_proj(attn_out.reshape(-1, self.num_heads * self.head_dim))


# ---------------------------------------------------------------------------
# Decoder layer (matches Qwen2DecoderLayer: input_layernorm, self_attn,
#   post_attention_layernorm, mlp)
# ---------------------------------------------------------------------------

class PackedQwen2TextLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PackedQwen2TextAttention(config, layer_idx)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(self, hidden, cos, sin, cu_seqlens, max_seqlen):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, cos, sin, cu_seqlens, max_seqlen)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden


# ---------------------------------------------------------------------------
# Full text encoder
# ---------------------------------------------------------------------------

class PackedQwen2TextEncoder(nn.Module):
    """NaviT text encoder using Qwen2 architecture with flash_attn_varlen.

    Weight-compatible with Qwen2_5_VLTextModel (same param names:
        embed_tokens, layers.N.self_attn.{q,k,v,o}_proj,
        layers.N.self_attn.{q_norm,k_norm},
        layers.N.{input_layernorm, post_attention_layernorm},
        layers.N.mlp.{gate,up,down}_proj, norm).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [PackedQwen2TextLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(self, packed_text_ids, text_sample_lens):
        """
        Parameters
        ----------
        packed_text_ids : [total_text] int64
        text_sample_lens : List[int]

        Returns
        -------
        packed_hidden : [total_text, hidden_size]
        """
        device = packed_text_ids.device
        dtype = self.embed_tokens.weight.dtype

        hidden = self.embed_tokens(packed_text_ids)

        # Per-sample sequential position IDs
        position_ids = torch.cat(
            [torch.arange(n, device=device, dtype=torch.long) for n in text_sample_lens]
        )

        # RoPE cos/sin
        cos, sin = compute_rope(position_ids, self.head_dim, self.rope_theta, device, dtype)

        # cu_seqlens for flash_attn_varlen
        lens_t = torch.tensor(text_sample_lens, device=device, dtype=torch.int32)
        cu_seqlens = torch.zeros(len(text_sample_lens) + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lens_t, dim=0)
        max_seqlen = max(text_sample_lens)

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, cu_seqlens, max_seqlen)

        return self.norm(hidden)

    @staticmethod
    def from_pretrained(text_encoder_path, dtype=torch.bfloat16):
        """Load weights from a Qwen2.5-VL checkpoint.

        Extracts the language_model weights from Qwen2_5_VLForConditionalGeneration
        and maps them into this standalone encoder.
        """
        from transformers import Qwen2_5_VLForConditionalGeneration

        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path, torch_dtype=dtype
        )
        lm = full_model.model.language_model  # Qwen2_5_VLTextModel with embed_tokens, layers, norm

        lm_config = lm.config if hasattr(lm, 'config') else full_model.config
        config = SimpleNamespace(
            hidden_size=lm_config.hidden_size,           # 3584
            num_hidden_layers=lm_config.num_hidden_layers,  # 28
            num_attention_heads=lm_config.num_attention_heads,  # 28
            num_key_value_heads=lm_config.num_key_value_heads,  # 4
            intermediate_size=lm_config.intermediate_size,  # 18944
            rms_norm_eps=lm_config.rms_norm_eps,            # 1e-6
            rope_theta=lm_config.rope_theta,                # 1000000.0
            vocab_size=lm_config.vocab_size,                # 152064
        )

        model = PackedQwen2TextEncoder(config)

        # Build state dict mapping from HF model
        # HF Qwen2_5_VLModel has: embed_tokens, layers.N.*, norm
        src_sd = {}
        for k, v in lm.state_dict().items():
            # Skip visual/merger weights from the VL model
            if k.startswith("visual.") or k.startswith("merger."):
                continue
            src_sd[k] = v

        missing, unexpected = model.load_state_dict(src_sd, strict=False)
        if missing:
            print(f"[PackedQwen2TextEncoder] Missing keys: {len(missing)}")
            for k in missing[:10]:
                print(f"  {k}")
        if unexpected:
            print(f"[PackedQwen2TextEncoder] Unexpected keys: {len(unexpected)}")
            for k in unexpected[:10]:
                print(f"  {k}")

        model = model.to(dtype)
        del full_model
        return model


# ---------------------------------------------------------------------------
# Approach 1: HF-native packed forward (exact match guaranteed)
# Kept for backward compatibility and verification.
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_text_encoder_navit(
    qwen2_model: nn.Module,
    packed_text_ids: torch.Tensor,       # [total_tokens] int64
    text_sample_lens: List[int],         # per-sample token counts
) -> torch.Tensor:
    """
    Packed text encoding using HF's Qwen2_5_VLTextModel.

    Pads to max length, builds per-sample attention mask, runs batched HF forward,
    then extracts valid tokens back to packed format.

    Parameters
    ----------
    qwen2_model : Qwen2_5_VLTextModel
        model_full.model.language_model
    packed_text_ids : [total_tokens] int64
    text_sample_lens : List[int]

    Returns
    -------
    packed_hidden : [total_tokens, hidden_size]
    """
    device = packed_text_ids.device
    N = len(text_sample_lens)
    max_len = max(text_sample_lens)

    # Split packed IDs by sample
    id_splits = packed_text_ids.split(text_sample_lens)

    # Pad to max_len for batching
    pad_id = 0
    padded_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((N, max_len), dtype=torch.long, device=device)
    for i, (ids, length) in enumerate(zip(id_splits, text_sample_lens)):
        padded_ids[i, :length] = ids
        attn_mask[i, :length] = 1

    # Run HF forward
    out = qwen2_model(input_ids=padded_ids, attention_mask=attn_mask)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state  # [N, max_len, D]

    # Extract valid tokens back to packed format
    parts = []
    for i, length in enumerate(text_sample_lens):
        parts.append(hidden[i, :length])
    packed_hidden = torch.cat(parts, dim=0)  # [total_tokens, D]

    return packed_hidden
