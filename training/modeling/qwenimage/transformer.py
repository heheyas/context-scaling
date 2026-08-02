# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates on 2026-05-01
#
# Original file from huggingface/diffusers was released under Apache-2.0,
# full license text available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.
# Standalone port of QwenImage Transformer from diffusers.
# Parameter names match diffusers checkpoint exactly for direct weight loading.
# NO diffusers dependency.

import functools
import math
import numbers
from math import prod
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (matches diffusers implementation)."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings. For QwenImage we use the complex-number path (use_real=False)."""
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


def apply_rotary_emb_qwen_navit(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings for packed NaviT tensors.
    x: [T, H, D], freqs_cis: [T, D//2] (complex).
    """
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs_cis.unsqueeze(1).to(x.device)).flatten(-2)
    return out.type_as(x)


def compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor, encoder_hidden_states_mask: Optional[torch.Tensor]
):
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None, None

    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)

    position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
    active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return text_seq_len, per_sample_len, encoder_hidden_states_mask


# ---------------------------------------------------------------------------
# Normalization layers (matching diffusers parameter names)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Norm matching diffusers normalization.RMSNorm. Has `weight` parameter."""

    def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Matches diffusers normalization.AdaLayerNormContinuous. Has `silu`, `linear`, `norm` attributes."""

    def __init__(self, embedding_dim, conditioning_embedding_dim, elementwise_affine=True, eps=1e-5, bias=True, norm_type="layer_norm"):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if x.ndim == 3:  # batched [B, S, D]
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        else:  # packed [T, D]
            scale, shift = torch.chunk(emb, 2, dim=-1)
            x = self.norm(x) * (1 + scale) + shift
        return x


# ---------------------------------------------------------------------------
# Activation / FeedForward (matching diffusers naming: net.0.proj, net.1, net.2)
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    """GELU activation with `proj` attribute matching diffusers activations.GELU."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    """FeedForward matching diffusers attention.FeedForward.
    net is ModuleList: [GELU (net.0), Dropout (net.1), Linear (net.2)]
    So weight paths are: net.0.proj.weight, net.0.proj.bias, net.2.weight, net.2.bias
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 dropout: float = 0.0, activation_fn: str = "geglu", bias: bool = True,
                 inner_dim: Optional[int] = None):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Timestep embeddings (matching diffusers embeddings.Timesteps + TimestepEmbedding)
# ---------------------------------------------------------------------------

class Timesteps(nn.Module):
    """Matches diffusers embeddings.Timesteps."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    """Matches diffusers embeddings.TimestepEmbedding. Has `linear_1`, `act`, `linear_2`."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu",
                 out_dim: Optional[int] = None, sample_proj_bias=True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class QwenTimestepProjEmbeddings(nn.Module):
    """Matches diffusers transformer_qwenimage.QwenTimestepProjEmbeddings.
    Has `time_proj` (Timesteps) and `timestep_embedder` (TimestepEmbedding).
    """

    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class QwenEmbedRope(nn.Module):
    """3D complex RoPE matching diffusers transformer_qwenimage.QwenEmbedRope.
    Has `pos_freqs`, `neg_freqs` (non-buffer complex tensors), `axes_dim`, `scale_rope`.
    """

    def __init__(self, theta: int, axes_dim: list, scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(40960)
        neg_index = torch.arange(40960).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw,
        txt_seq_lens=None,
        device=None,
        max_txt_seq_len=None,
    ):
        # Handle deprecated txt_seq_lens parameter
        if txt_seq_lens is not None:
            if max_txt_seq_len is None:
                max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens

        if max_txt_seq_len is None:
            raise ValueError("Either `max_txt_seq_len` or `txt_seq_lens` must be provided.")

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs = self.pos_freqs.to(device)[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(self, frame, height, width, idx=0, device=None):
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# ---------------------------------------------------------------------------
# Joint Attention (matching diffusers Attention class parameter names exactly)
# ---------------------------------------------------------------------------

class _JointAttention(nn.Module):
    """Joint attention matching diffusers Attention parameter names exactly.

    Attributes match the diffusers Attention class created with:
        Attention(query_dim=dim, cross_attention_dim=None, added_kv_proj_dim=dim,
                  dim_head=attention_head_dim, heads=num_attention_heads, out_dim=dim,
                  context_pre_only=False, bias=True, qk_norm="rms_norm", eps=eps)

    Parameter paths:
        to_q.weight, to_q.bias
        to_k.weight, to_k.bias
        to_v.weight, to_v.bias
        to_out.0.weight, to_out.0.bias  (Linear)
        to_out.1  (nn.Dropout -> nn.Identity for inference)
        add_q_proj.weight, add_q_proj.bias
        add_k_proj.weight, add_k_proj.bias
        add_v_proj.weight, add_v_proj.bias
        to_add_out.weight, to_add_out.bias
        norm_q.weight
        norm_k.weight
        norm_added_q.weight
        norm_added_k.weight
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 qk_norm: str = "rms_norm", eps: float = 1e-6):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        # Image stream projections
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)

        # Image output: ModuleList [Linear, Identity] matching diffusers to_out
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, dim, bias=True),
            nn.Identity(),
        ])

        # Text stream projections
        self.add_q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.add_k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.add_v_proj = nn.Linear(dim, inner_dim, bias=True)

        # Text output
        self.to_add_out = nn.Linear(inner_dim, dim, bias=True)

        # QK norm
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(attention_head_dim, eps=eps)
            self.norm_k = RMSNorm(attention_head_dim, eps=eps)
            self.norm_added_q = RMSNorm(attention_head_dim, eps=eps)
            self.norm_added_k = RMSNorm(attention_head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        seq_txt = encoder_hidden_states.shape[1]

        # Image stream QKV
        img_query = self.to_q(hidden_states)
        img_key = self.to_k(hidden_states)
        img_value = self.to_v(hidden_states)

        # Text stream QKV
        txt_query = self.add_q_proj(encoder_hidden_states)
        txt_key = self.add_k_proj(encoder_hidden_states)
        txt_value = self.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention: [B, S, H, D]
        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        # QK normalization
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate [text, image] for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Transpose to [B, H, S, D] for SDPA
        joint_query = joint_query.transpose(1, 2)
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)

        # Build attention mask for SDPA if needed
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, S_total] bool mask
            # Need to expand to [B, 1, 1, S_total] for SDPA
            attn_mask = attention_mask[:, None, None, :].to(dtype=joint_query.dtype)
            attn_mask = attn_mask.masked_fill(~attention_mask[:, None, None, :], float('-inf'))
            attn_mask = attn_mask.masked_fill(attention_mask[:, None, None, :], 0.0)

        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query, joint_key, joint_value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Transpose back to [B, S, H, D] and flatten
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Output projections
        img_attn_output = self.to_out[0](img_attn_output.contiguous())
        img_attn_output = self.to_out[1](img_attn_output)

        txt_attn_output = self.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output

    def forward_navit(
        self,
        packed_img: torch.Tensor,
        packed_txt: torch.Tensor,
        img_lens: List[int],
        txt_lens: List[int],
        img_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
    ):
        """
        NaviT packed-sequence joint attention using flash_attn_varlen_func.

        packed_img: [total_img, D]
        packed_txt: [total_txt, D]
        img_lens: List[int] -- per sample image token counts
        txt_lens: List[int] -- per sample text token counts
        img_freqs: [total_img, D_h//2] complex
        txt_freqs: [total_txt, D_h//2] complex

        Returns: (packed_img_out [total_img, D], packed_txt_out [total_txt, D])
        """
        H = self.heads
        D_h = self.head_dim

        # Project -> [total, H, D_h]
        iq = self.to_q(packed_img).view(-1, H, D_h)
        ik = self.to_k(packed_img).view(-1, H, D_h)
        iv = self.to_v(packed_img).view(-1, H, D_h)
        tq = self.add_q_proj(packed_txt).view(-1, H, D_h)
        tk = self.add_k_proj(packed_txt).view(-1, H, D_h)
        tv = self.add_v_proj(packed_txt).view(-1, H, D_h)

        # QK norm
        iq = self.norm_q(iq)
        ik = self.norm_k(ik)
        tq = self.norm_added_q(tq)
        tk = self.norm_added_k(tk)

        # RoPE (NaviT version -- operates on [T, H, D])
        iq = apply_rotary_emb_qwen_navit(iq, img_freqs)
        ik = apply_rotary_emb_qwen_navit(ik, img_freqs)
        tq = apply_rotary_emb_qwen_navit(tq, txt_freqs)
        tk = apply_rotary_emb_qwen_navit(tk, txt_freqs)

        # Interleave per sample: [txt_i, img_i] for each sample i
        iq_s = iq.split(img_lens)
        ik_s = ik.split(img_lens)
        iv_s = iv.split(img_lens)
        tq_s = tq.split(txt_lens)
        tk_s = tk.split(txt_lens)
        tv_s = tv.split(txt_lens)

        joint_q, joint_k, joint_v = [], [], []
        joint_lens = []
        for _tq, _iq, _tk, _ik, _tv, _iv in zip(tq_s, iq_s, tk_s, ik_s, tv_s, iv_s):
            joint_q.append(torch.cat([_tq, _iq], dim=0))
            joint_k.append(torch.cat([_tk, _ik], dim=0))
            joint_v.append(torch.cat([_tv, _iv], dim=0))
            joint_lens.append(_tq.shape[0] + _iq.shape[0])

        jq = torch.cat(joint_q, dim=0)  # [total_joint, H, D_h]
        jk = torch.cat(joint_k, dim=0)
        jv = torch.cat(joint_v, dim=0)

        # cu_seqlens for flash_attn_varlen_func
        cu = torch.zeros(len(joint_lens) + 1, dtype=torch.int32, device=jq.device)
        cu[1:] = torch.cumsum(torch.tensor(joint_lens, device=jq.device), dim=0).to(torch.int32)
        max_seqlen = max(joint_lens)

        out = flash_attn_varlen_func(
            jq, jk, jv,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=0.0, causal=False,
        )  # [total_joint, H, D_h]

        # Split back to img and txt per sample
        out_splits = out.split(joint_lens, dim=0)
        img_parts, txt_parts = [], []
        for split, tl in zip(out_splits, txt_lens):
            txt_parts.append(split[:tl])
            img_parts.append(split[tl:])

        packed_img_out = torch.cat(img_parts, dim=0).reshape(-1, H * D_h)
        packed_txt_out = torch.cat(txt_parts, dim=0).reshape(-1, H * D_h)

        img_out = self.to_out[0](packed_img_out)
        txt_out = self.to_add_out(packed_txt_out)
        return img_out, txt_out

    def forward_navit2(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        seq_len: int,
        packed_txt_indexes: torch.LongTensor,
        packed_img_indexes: torch.LongTensor,
        packed_rope_freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """
        Joint attention accepting separate img/txt tensors directly.

        img : [total_img, D] -- image hidden states (already modulated)
        txt : [total_txt, D] -- text hidden states (already modulated)
        seq_len : int -- total packed sequence length (for allocating q/k/v)
        packed_txt_indexes : [total_txt] -- positions of text tokens
        packed_img_indexes : [total_img] -- positions of image tokens
        packed_rope_freqs : [seq_len, D_h//2] complex -- RoPE per position
        cu_seqlens : [N+1] int32
        max_seqlen : int
        """
        H = self.heads
        D_h = self.head_dim

        # Project separately, then scatter into packed layout for flash_attn
        img_q = self.to_q(img).view(-1, H, D_h)
        img_k = self.to_k(img).view(-1, H, D_h)
        img_v = self.to_v(img).view(-1, H, D_h)
        txt_q = self.add_q_proj(txt).view(-1, H, D_h)
        txt_k = self.add_k_proj(txt).view(-1, H, D_h)
        txt_v = self.add_v_proj(txt).view(-1, H, D_h)

        packed_q = img_q.new_zeros(seq_len, H, D_h)
        packed_k = img_k.new_zeros(seq_len, H, D_h)
        packed_v = img_v.new_zeros(seq_len, H, D_h)

        packed_q[packed_img_indexes] = img_q
        packed_k[packed_img_indexes] = img_k
        packed_v[packed_img_indexes] = img_v

        packed_q[packed_txt_indexes] = txt_q
        packed_k[packed_txt_indexes] = txt_k
        packed_v[packed_txt_indexes] = txt_v

        # QK norm (different norms for txt and img)
        packed_q[packed_img_indexes] = self.norm_q(packed_q[packed_img_indexes])
        packed_k[packed_img_indexes] = self.norm_k(packed_k[packed_img_indexes])
        packed_q[packed_txt_indexes] = self.norm_added_q(packed_q[packed_txt_indexes])
        packed_k[packed_txt_indexes] = self.norm_added_k(packed_k[packed_txt_indexes])

        # RoPE
        packed_q = apply_rotary_emb_qwen_navit(packed_q, packed_rope_freqs)
        packed_k = apply_rotary_emb_qwen_navit(packed_k, packed_rope_freqs)

        # One flash_attn_varlen call
        attn_out = flash_attn_varlen_func(
            packed_q, packed_k, packed_v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )  # [seq_len, H, D_h]

        # Separate output projections
        img_attn = self.to_out[0](attn_out[packed_img_indexes].reshape(-1, H * D_h))
        txt_attn = self.to_add_out(attn_out[packed_txt_indexes].reshape(-1, H * D_h))

        return img_attn, txt_attn


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class QwenImageTransformerBlock(nn.Module):
    """Matches diffusers QwenImageTransformerBlock.
    Attributes: img_mod, img_norm1, img_norm2, img_mlp, txt_mod, txt_norm1, txt_norm2, txt_mlp, attn
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = _JointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)

            shift_result = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale_result = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate_result = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(self, *args, **kwargs):
        """Dispatch to batched or navit2 forward based on argument types."""
        # NaviT2: called with keyword 'img_temb' or first arg is [total_img, D] (2D)
        # and second arg is also 2D (txt).
        # Batched: first is hidden_states [B, S, D] (3D).
        if 'img_temb' in kwargs or (len(args) >= 2 and args[0].ndim == 2 and args[1].ndim == 2):
            return self.forward_navit2(*args, **kwargs)
        return self.forward_batched(*args, **kwargs)

    def forward_batched(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        img_mod_params = self.img_mod(temb)

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Image norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Text norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Image norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Text norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_navit(
        self,
        packed_img: torch.Tensor,
        packed_txt: torch.Tensor,
        packed_img_temb: torch.Tensor,
        packed_txt_temb: torch.Tensor,
        img_lens: List[int],
        txt_lens: List[int],
        img_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
    ):
        """
        NaviT packed-sequence forward. All inputs are packed [total_tokens, D].
        temb is per-token (already expanded).
        """
        img_mod = self.img_mod(packed_img_temb)   # [total_img, 6D]
        txt_mod = self.txt_mod(packed_txt_temb)   # [total_txt, 6D]

        img_m1, img_m2 = img_mod.chunk(2, dim=-1)
        txt_m1, txt_m2 = txt_mod.chunk(2, dim=-1)

        # Modulate -- no unsqueeze needed for packed tensors
        def _mod(x, mod):
            shift, scale, gate = mod.chunk(3, dim=-1)
            return x * (1 + scale) + shift, gate

        img_normed, img_g1 = _mod(self.img_norm1(packed_img), img_m1)
        txt_normed, txt_g1 = _mod(self.txt_norm1(packed_txt), txt_m1)

        img_attn, txt_attn = self.attn.forward_navit(
            img_normed, txt_normed, img_lens, txt_lens, img_freqs, txt_freqs
        )

        packed_img = packed_img + img_g1 * img_attn
        packed_txt = packed_txt + txt_g1 * txt_attn

        img_normed2, img_g2 = _mod(self.img_norm2(packed_img), img_m2)
        packed_img = packed_img + img_g2 * self.img_mlp(img_normed2)

        txt_normed2, txt_g2 = _mod(self.txt_norm2(packed_txt), txt_m2)
        packed_txt = packed_txt + txt_g2 * self.txt_mlp(txt_normed2)

        if packed_txt.dtype == torch.float16:
            packed_txt = packed_txt.clamp(-65504, 65504)
        if packed_img.dtype == torch.float16:
            packed_img = packed_img.clamp(-65504, 65504)

        return packed_txt, packed_img

    def forward_navit2(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_temb: torch.Tensor,
        txt_temb: torch.Tensor,
        seq_len: int,
        packed_txt_indexes: torch.LongTensor,
        packed_img_indexes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_rope_freqs: torch.Tensor,
    ):
        """
        Transformer block operating on separate img/txt tensors.
        Only packs into seq_len layout inside attention for flash_attn q/k/v.

        img : [total_img, D]
        txt : [total_txt, D]
        img_temb : [total_img, D] -- per-token timestep embedding for image
        txt_temb : [total_txt, D] -- per-token timestep embedding for text
        seq_len : int -- total packed sequence length
        packed_txt_indexes : [total_txt] int64
        packed_img_indexes : [total_img] int64
        cu_seqlens : [N+1] int32
        max_seqlen : int
        packed_rope_freqs : [seq_len, D_h//2] complex
        """
        # Modulation (per-token, different for txt and img)
        img_mod = self.img_mod(img_temb)  # [total_img, 6D]
        txt_mod = self.txt_mod(txt_temb)  # [total_txt, 6D]
        img_m1, img_m2 = img_mod.chunk(2, dim=-1)
        txt_m1, txt_m2 = txt_mod.chunk(2, dim=-1)

        def _mod(x, m):
            shift, scale, gate = m.chunk(3, dim=-1)
            return x * (1 + scale) + shift, gate

        img_normed, img_g1 = _mod(self.img_norm1(img), img_m1)
        txt_normed, txt_g1 = _mod(self.txt_norm1(txt), txt_m1)

        # Joint attention — pack only q/k/v inside, not hidden states
        img_attn, txt_attn = self.attn.forward_navit2(
            img_normed, txt_normed, seq_len,
            packed_txt_indexes, packed_img_indexes,
            packed_rope_freqs, cu_seqlens, max_seqlen,
        )

        # Residual + gate
        img = img + img_g1 * img_attn
        txt = txt + txt_g1 * txt_attn

        # MLP (separate)
        img_n2, img_g2 = _mod(self.img_norm2(img), img_m2)
        img = img + img_g2 * self.img_mlp(img_n2)

        txt_n2, txt_g2 = _mod(self.txt_norm2(txt), txt_m2)
        txt = txt + txt_g2 * self.txt_mlp(txt_n2)

        # fp16 clamp
        if txt.dtype == torch.float16:
            txt = txt.clamp(-65504, 65504)
        if img.dtype == torch.float16:
            img = img.clamp(-65504, 65504)

        return img, txt


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class QwenImageTransformer2DModel(nn.Module):
    """Standalone port of diffusers QwenImageTransformer2DModel.
    Parameter names match exactly for direct weight loading.

    Attributes: pos_embed, time_text_embed, txt_norm, img_in, txt_in,
                transformer_blocks, norm_out, proj_out
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

    def _set_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    def forward(self, *args, **kwargs):
        """Auto-dispatch: if packed_img_indexes is provided, use NaviT path."""
        if 'packed_img_indexes' in kwargs:
            return self.forward_navit2(*args, **kwargs)
        return self.forward_batched(*args, **kwargs)

    def forward_batched(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states: [B, S_img, in_channels]
            encoder_hidden_states: [B, S_txt, joint_attention_dim]
            encoder_hidden_states_mask: [B, S_txt] optional bool mask
            timestep: [B]
            img_shapes: list of list of (frame, h, w) tuples for RoPE
            return_dict: if False, returns tuple (output,)
        Returns:
            tuple: (output,) where output is [B, S_img, patch_size^2 * out_channels]
        """
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        # Construct joint attention mask
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                    use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return (output,)

    def forward_navit(
        self,
        packed_img_tokens: torch.Tensor,
        packed_txt_embeds: torch.Tensor,
        per_sample_timesteps: torch.Tensor,
        img_sample_lens: List[int],
        txt_sample_lens: List[int],
        img_shapes: list,
    ):
        """
        NaviT forward: all samples packed into flat 1D tensors.

        packed_img_tokens: [total_img, in_channels]
        packed_txt_embeds: [total_txt, joint_attention_dim]
        per_sample_timesteps: [N]
        img_sample_lens: List[int], length N
        txt_sample_lens: List[int], length N
        img_shapes: [[(1,h,w)], ...] length N

        Returns: [total_img, patch_size^2 * out_channels]
        """
        device = packed_img_tokens.device

        # 1. Project
        packed_img = self.img_in(packed_img_tokens)
        packed_txt = self.txt_norm(packed_txt_embeds)
        packed_txt = self.txt_in(packed_txt)

        # 2. Timestep -> per-token via repeat_interleave
        ts = per_sample_timesteps.to(packed_img.dtype)
        temb = self.time_text_embed(ts, packed_img[:1].expand(len(ts), -1))  # [N, D]
        img_lens_t = torch.tensor(img_sample_lens, device=device)
        txt_lens_t = torch.tensor(txt_sample_lens, device=device)
        packed_img_temb = torch.repeat_interleave(temb, img_lens_t, dim=0)
        packed_txt_temb = torch.repeat_interleave(temb, txt_lens_t, dim=0)

        # 3. Per-sample RoPE, concatenated
        all_img_f, all_txt_f = [], []
        for i, shape in enumerate(img_shapes):
            s = [shape] if not isinstance(shape[0], (list, tuple)) else shape
            img_f, txt_f = self.pos_embed(s, max_txt_seq_len=txt_sample_lens[i], device=device)
            all_img_f.append(img_f)
            all_txt_f.append(txt_f)
        packed_img_freqs = torch.cat(all_img_f, dim=0)
        packed_txt_freqs = torch.cat(all_txt_f, dim=0)

        # 4. Transformer blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                packed_txt, packed_img = torch.utils.checkpoint.checkpoint(
                    block.forward_navit,
                    packed_img, packed_txt,
                    packed_img_temb, packed_txt_temb,
                    img_sample_lens, txt_sample_lens,
                    packed_img_freqs, packed_txt_freqs,
                    use_reentrant=False,
                )
            else:
                packed_txt, packed_img = block.forward_navit(
                    packed_img, packed_txt,
                    packed_img_temb, packed_txt_temb,
                    img_sample_lens, txt_sample_lens,
                    packed_img_freqs, packed_txt_freqs,
                )

        # 5. Output: AdaLayerNormContinuous (per-token conditioning)
        norm = self.norm_out
        emb = norm.linear(norm.silu(packed_img_temb))
        scale, shift = emb.chunk(2, dim=-1)
        packed_img = norm.norm(packed_img) * (1 + scale) + shift

        output = self.proj_out(packed_img)
        return output

    def forward_navit2(
        self,
        packed_img_tokens: torch.Tensor,
        packed_txt_embeds: torch.Tensor,
        packed_img_indexes: torch.LongTensor,
        packed_txt_indexes: torch.LongTensor,
        sample_lens: List[int],
        per_sample_timesteps: torch.Tensor,
        img_shapes: list,
        txt_sample_lens: List[int],
        img_sample_lens: List[int],
    ):
        """
        NaviT forward using a single packed_sequence [seq_len, D].

        The caller provides raw (unprojected) image and text tokens plus their
        index arrays into the packed sequence. This method:
          1. Projects img/txt into inner_dim
          2. Builds packed_sequence, packed_temb, packed_rope_freqs
          3. Runs transformer blocks via forward_navit2
          4. Returns predicted output only at image positions

        Parameters
        ----------
        packed_img_tokens : [total_img, in_channels]
        packed_txt_embeds : [total_txt, joint_attention_dim]
        packed_img_indexes : [total_img] int64 -- positions in packed_sequence
        packed_txt_indexes : [total_txt] int64 -- positions in packed_sequence
        sample_lens : List[int] -- total tokens (txt + img) per sample
        per_sample_timesteps : [N]
        img_shapes : [[(1,h,w)], ...] length N
        txt_sample_lens : List[int], length N
        img_sample_lens : List[int], length N

        Returns
        -------
        output : [total_img, patch_size^2 * out_channels]
        """
        device = packed_img_tokens.device
        seq_len = sum(sample_lens)
        N = len(sample_lens)

        # 1. Project
        img = self.img_in(packed_img_tokens)  # [total_img, inner_dim]
        txt = self.txt_in(self.txt_norm(packed_txt_embeds))  # [total_txt, inner_dim]

        # 2. Timestep embedding -> pre-extract per-token temb for img and txt
        ts = per_sample_timesteps.to(img.dtype)
        # time_text_embed expects [N, D] hidden for conditioning; use a dummy
        temb = self.time_text_embed(ts, img[:1].expand(N, -1))  # [N, D]
        img_lens_t = torch.tensor(img_sample_lens, device=device, dtype=torch.long)
        txt_lens_t = torch.tensor(txt_sample_lens, device=device, dtype=torch.long)
        img_temb = torch.repeat_interleave(temb, img_lens_t, dim=0)  # [total_img, D]
        txt_temb = torch.repeat_interleave(temb, txt_lens_t, dim=0)  # [total_txt, D]

        # 3. RoPE: per-sample, build packed_rope_freqs [seq_len, D_h//2] complex
        D_h = self.transformer_blocks[0].attn.head_dim
        packed_rope_freqs = torch.zeros(seq_len, D_h // 2, dtype=torch.cfloat, device=device)

        img_offset, txt_offset = 0, 0
        total_txt_in_indexes = packed_txt_indexes.shape[0]
        for i in range(N):
            shape = img_shapes[i]
            s = [shape] if not isinstance(shape[0], (list, tuple)) else shape
            n_img = img_sample_lens[i]
            n_txt = txt_sample_lens[i]
            img_f, txt_f = self.pos_embed(s, max_txt_seq_len=n_txt, device=device)

            assert txt_offset + n_txt <= total_txt_in_indexes, \
                f"Sample {i}: txt_offset({txt_offset})+n_txt({n_txt})={txt_offset+n_txt} > total_txt({total_txt_in_indexes}). " \
                f"sample_lens={sample_lens}, img_sample_lens={img_sample_lens}, txt_sample_lens={txt_sample_lens}"

            sample_img_idx = packed_img_indexes[img_offset:img_offset + n_img]
            sample_txt_idx = packed_txt_indexes[txt_offset:txt_offset + n_txt]

            packed_rope_freqs[sample_img_idx] = img_f[:n_img]
            packed_rope_freqs[sample_txt_idx] = txt_f[:n_txt]

            img_offset += n_img
            txt_offset += n_txt

        # 4. cu_seqlens
        sample_lens_t = torch.tensor(sample_lens, device=device, dtype=torch.long)
        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(sample_lens_t.to(torch.int32), dim=0)
        max_seqlen = max(sample_lens)

        # 5. Transformer blocks — pass img/txt directly, no packed_sequence
        #    Must call block() (not block.forward_navit2) so FSDP all-gathers params.
        #    The block.forward() dispatcher routes to forward_navit2 when args[0:2] are 2D.
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                img, txt = torch.utils.checkpoint.checkpoint(
                    block,
                    img, txt, img_temb, txt_temb,
                    seq_len, packed_txt_indexes, packed_img_indexes,
                    cu_seqlens, max_seqlen, packed_rope_freqs,
                    use_reentrant=False,
                )
            else:
                img, txt = block(
                    img, txt, img_temb, txt_temb,
                    seq_len, packed_txt_indexes, packed_img_indexes,
                    cu_seqlens, max_seqlen, packed_rope_freqs,
                )

        # 6. Output: only image positions
        norm = self.norm_out
        emb = norm.linear(norm.silu(img_temb))
        scale, shift = emb.chunk(2, dim=-1)
        img = norm.norm(img) * (1 + scale) + shift
        output = self.proj_out(img)
        return output
