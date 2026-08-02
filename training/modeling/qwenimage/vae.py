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
# Standalone port of QwenImage VAE from diffusers.
# Parameter names match diffusers checkpoint exactly for direct weight loading.
# NO diffusers dependency.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

CACHE_T = 2


# ---------------------------------------------------------------------------
# Causal Conv3d
# ---------------------------------------------------------------------------

class QwenImageCausalConv3d(nn.Conv3d):
    """Causal 3D convolution with feature caching support.
    Matches diffusers autoencoder_kl_qwenimage.QwenImageCausalConv3d.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


# ---------------------------------------------------------------------------
# RMS Norm (channel-first)
# ---------------------------------------------------------------------------

class QwenImageRMS_norm(nn.Module):
    """Channel-first RMS normalization.
    Matches diffusers autoencoder_kl_qwenimage.QwenImageRMS_norm.
    """

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


# ---------------------------------------------------------------------------
# Upsample / Resample
# ---------------------------------------------------------------------------

class QwenImageUpsample(nn.Upsample):
    """Upsample that preserves dtype. Matches diffusers QwenImageUpsample."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    """Resampling module for 2D/3D. Matches diffusers QwenImageResample."""

    def __init__(self, dim, mode="none"):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class QwenImageResidualBlock(nn.Module):
    """Matches diffusers QwenImageResidualBlock."""

    def __init__(self, in_dim, out_dim, dropout=0.0, non_linearity="silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = nn.SiLU() if non_linearity == "silu" else nn.ReLU()

        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


# ---------------------------------------------------------------------------
# Attention Block
# ---------------------------------------------------------------------------

class QwenImageAttentionBlock(nn.Module):
    """Single-head self-attention. Matches diffusers QwenImageAttentionBlock."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


# ---------------------------------------------------------------------------
# Mid Block
# ---------------------------------------------------------------------------

class QwenImageMidBlock(nn.Module):
    """Matches diffusers QwenImageMidBlock."""

    def __init__(self, dim, dropout=0.0, non_linearity="silu", num_layers=1):
        super().__init__()
        self.dim = dim

        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache, feat_idx)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)

        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class QwenImageEncoder3d(nn.Module):
    """Matches diffusers QwenImageEncoder3d."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        input_channels=3,
        non_linearity="silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = nn.SiLU() if non_linearity == "silu" else nn.ReLU()

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(input_channels, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# Up Block
# ---------------------------------------------------------------------------

class QwenImageUpBlock(nn.Module):
    """Matches diffusers QwenImageUpBlock."""

    def __init__(self, in_dim, out_dim, num_res_blocks, dropout=0.0, upsample_mode=None, non_linearity="silu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class QwenImageDecoder3d(nn.Module):
    """Matches diffusers QwenImageDecoder3d."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        input_channels=3,
        non_linearity="silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.nonlinearity = nn.SiLU() if non_linearity == "silu" else nn.ReLU()

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2

            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            if upsample_mode is not None:
                scale *= 2.0

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, input_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


# ---------------------------------------------------------------------------
# DiagonalGaussianDistribution
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    """Matches diffusers autoencoders.vae.DiagonalGaussianDistribution."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator=None):
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


# ---------------------------------------------------------------------------
# AutoencoderKLQwenImage
# ---------------------------------------------------------------------------

class AutoencoderKLQwenImage(nn.Module):
    """Standalone port of diffusers AutoencoderKLQwenImage.
    Parameter names match exactly for direct weight loading.

    Default config: base_dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
                    temperal_downsample=[False,True,True]
    """

    _supports_gradient_checkpointing = False

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: list = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_downsample: list = [False, True, True],
        dropout: float = 0.0,
        input_channels: int = 3,
        latents_mean: list = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
        latents_std: list = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
    ):
        super().__init__()

        # Store config as dict for compatibility
        self.config = {
            "base_dim": base_dim,
            "z_dim": z_dim,
            "dim_mult": dim_mult,
            "num_res_blocks": num_res_blocks,
            "attn_scales": attn_scales,
            "temperal_downsample": temperal_downsample,
            "dropout": dropout,
            "input_channels": input_channels,
            "latents_mean": latents_mean,
            "latents_std": latents_std,
        }

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = QwenImageEncoder3d(
            base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales,
            self.temperal_downsample, dropout, input_channels
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder3d(
            base_dim, z_dim, dim_mult, num_res_blocks, attn_scales,
            self.temperal_upsample, dropout, input_channels
        )

        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        self.use_slicing = False
        self.use_tiling = False

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def clear_cache(self):
        def _count_conv3d(model):
            count = 0
            for m in model.modules():
                if isinstance(m, QwenImageCausalConv3d):
                    count += 1
            return count

        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def enable_tiling(self, tile_sample_min_height=None, tile_sample_min_width=None,
                      tile_sample_stride_height=None, tile_sample_stride_width=None):
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return {"latent_dist": posterior}

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return {"sample": out}

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice)["sample"] if return_dict else self._decode(z_slice, return_dict=False)[0] for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            result = self._decode(z, return_dict=return_dict)
            decoded = result["sample"] if return_dict else result[0]

        if not return_dict:
            return (decoded,)
        return {"sample": decoded}

    def encode_to_latent(self, x: torch.Tensor):
        """Encode images to normalized latents."""
        enc_result = self.encode(x, return_dict=True)
        posterior = enc_result["latent_dist"]
        z = posterior.sample()
        # Normalize using config mean/std
        latents_mean = torch.tensor(self.config["latents_mean"], device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.config["latents_std"], device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        z = (z - latents_mean) / latents_std
        return z

    def decode_from_latent(self, z: torch.Tensor):
        """Decode normalized latents to images."""
        latents_mean = torch.tensor(self.config["latents_mean"], device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.config["latents_std"], device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        z = z * latents_std + latents_mean
        return self.decode(z, return_dict=False)[0]

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[:, :, 1 + 4 * (k - 1) : 1 + 4 * k, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.quant_conv(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z, return_dict=True):
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return {"sample": dec}

    def forward(self, sample, sample_posterior=False, return_dict=True, generator=None):
        x = sample
        enc_result = self.encode(x, return_dict=True)
        posterior = enc_result["latent_dist"]
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
