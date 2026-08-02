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
# CausalFusionQwenImage: NaviT adapter for QwenImage.
# forward() dispatches to forward_train() or forward_inference().
# All index/sample_lens handling is done in dataset_base.py.

import math

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .transformer import QwenImageTransformer2DModel
from .text_encoder_navit import PackedQwen2TextEncoder


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=8192,
    base_shift=0.5,
    max_shift=0.9,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class CausalFusionQwenImageConfig:
    def __init__(self, latent_patch_size=2, max_latent_size=32, timestep_shift=1.0):
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.timestep_shift = timestep_shift


class CausalFusionQwenImage(nn.Module):

    def __init__(self, dit_config_or_model, state_dict_or_config=None,
                 text_encoder=None, tokenizer=None):
        super().__init__()

        if isinstance(dit_config_or_model, dict):
            dit_config = dit_config_or_model
            state_dict = state_dict_or_config
            dit_model = QwenImageTransformer2DModel(
                patch_size=dit_config.get("patch_size", 2),
                in_channels=dit_config.get("in_channels", 64),
                out_channels=dit_config.get("out_channels", 16),
                num_layers=dit_config.get("num_layers", 60),
                attention_head_dim=dit_config.get("attention_head_dim", 128),
                num_attention_heads=dit_config.get("num_attention_heads", 24),
                joint_attention_dim=dit_config.get("joint_attention_dim", 3584),
                axes_dims_rope=tuple(dit_config.get("axes_dims_rope", [16, 56, 56])),
                guidance_embeds=dit_config.get("guidance_embeds", False),
            )
            if state_dict is not None:
                missing, unexpected = dit_model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[CausalFusionQwenImage] DiT missing keys: {len(missing)}")
                if unexpected:
                    print(f"[CausalFusionQwenImage] DiT unexpected keys: {len(unexpected)}")
            config = CausalFusionQwenImageConfig()
        else:
            dit_model = dit_config_or_model
            config = state_dict_or_config or CausalFusionQwenImageConfig()

        self.dit_model = dit_model
        self.text_encoder = text_encoder
        self.config = config
        self.latent_patch_size = config.latent_patch_size
        self.latent_channel = 16
        self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel

    # ── Shared helpers ────────────────────────────────────────────────

    def _encode_text(self, packed_text_ids, condition_sample_lens, actual_txt_indexes):
        """Text encoder → extract caption+suffix embeddings."""
        model_dtype = next(self.dit_model.parameters()).dtype
        packed_all = self.text_encoder(packed_text_ids.long(), condition_sample_lens)
        return packed_all[actual_txt_indexes].to(model_dtype)

    def _patchify_latents(self, padded_latent, patchified_vae_latent_shapes, device):
        """Patchify VAE latents using QwenImage layout: [C, p, q] per token.

        Matches the QwenImage pipeline's _pack_latents:
            view(B, C, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*4)
        """
        p = self.latent_patch_size
        C = self.latent_channel
        packed_list, rev_index, img_lens = [], [], []
        for idx, (latent, (h, w)) in enumerate(zip(padded_latent, patchified_vae_latent_shapes)):
            # latent: [C, H_full, W_full] (may be padded)
            latent = latent[:, :h*p, :w*p]  # [C, h*2, w*2]
            # Pack: [C, h, 2, w, 2] → [h, w, C, 2, 2] → [h*w, C*4]
            latent = latent.view(C, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(-1, C * p * p)
            packed_list.append(latent)
            rev_index.extend([idx] * len(latent))
            img_lens.append(h * w)
        packed_clean = torch.cat(packed_list, dim=0)
        rev_indices = torch.tensor(rev_index, device=device)
        return packed_clean, rev_indices, img_lens

    def _call_dit(self, x_t, packed_text_embeds, packed_vae_token_indexes,
                  packed_text_indexes, sample_lens, per_sample_timesteps,
                  img_shapes, text_sample_lens, img_sample_lens):
        """Single DiT forward (goes through FSDP __call__)."""
        return self.dit_model(
            packed_img_tokens=x_t,
            packed_txt_embeds=packed_text_embeds,
            packed_img_indexes=packed_vae_token_indexes,
            packed_txt_indexes=packed_text_indexes,
            sample_lens=sample_lens,
            per_sample_timesteps=per_sample_timesteps,
            img_shapes=img_shapes,
            txt_sample_lens=text_sample_lens,
            img_sample_lens=img_sample_lens,
        )

    # ── forward: dispatch ─────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        """Dispatch to forward_train or forward_inference.

        If padded_latent is provided → training (add noise, compute loss).
        Otherwise → inference (multi-step denoising).
        """
        if kwargs.get("padded_latent") is not None or (len(args) > 9 and args[9] is not None):
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    # ── forward_train ─────────────────────────────────────────────────

    def forward_train(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks=None, split_lens=None, attn_modes=None,
        ce_loss_indexes=None, packed_label_ids=None,
        packed_vit_tokens=None, packed_vit_token_indexes=None,
        packed_vit_position_ids=None, vit_latent_shapes=None, vit_token_seqlens=None,
        padded_latent=None,
        patchified_vae_latent_shapes=None,
        packed_latent_position_ids=None,
        packed_vae_token_indexes=None,
        packed_timesteps=None,
        mse_loss_indexes=None,
        condition_sample_lens=None,
        actual_txt_indexes=None,
    ):
        device = packed_text_ids.device
        N = len(patchified_vae_latent_shapes)

        # 1. Text
        packed_text_embeds = self._encode_text(packed_text_ids, condition_sample_lens, actual_txt_indexes)

        # 2. Patchify
        packed_latent_clean, packed_rev_indices, img_sample_lens = \
            self._patchify_latents(padded_latent, patchified_vae_latent_shapes, device)
        latent_mean = packed_latent_clean.flatten().mean()
        latent_std = packed_latent_clean.flatten().std()
        text_sample_lens = [sample_lens[i] - img_sample_lens[i] for i in range(N)]
        img_shapes = [[(1, h, w)] for h, w in patchified_vae_latent_shapes]

        # 3. Noise — QwenImage uses uniform timestep sampling (not logit-normal)
        noise = torch.randn_like(packed_latent_clean)
        # Ignore dataset's randn timesteps; sample uniform [0, 1] per image then shift
        packed_timesteps = torch.zeros(packed_latent_clean.shape[0], device=device)
        for idx, (h, w) in enumerate(patchified_vae_latent_shapes):
            mask = packed_rev_indices == idx
            t_uniform = torch.rand(1, device=device).item()  # one t per image
            mu = calculate_shift(h * w)
            exp_mu = math.exp(mu)
            t_shifted = exp_mu * t_uniform / (1 + (exp_mu - 1) * t_uniform)
            packed_timesteps[mask] = t_shifted
        t = packed_timesteps.to(packed_latent_clean.dtype)
        packed_latent_noised = (1 - t[:, None]) * packed_latent_clean + t[:, None] * noise

        # 4. Per-sample timestep
        per_sample_t = []
        off = 0
        for n in img_sample_lens:
            per_sample_t.append(packed_timesteps[off]); off += n
        per_sample_timesteps = torch.stack(per_sample_t)

        # 5. DiT forward
        packed_mse_preds = self._call_dit(
            packed_latent_noised, packed_text_embeds,
            packed_vae_token_indexes, packed_text_indexes,
            sample_lens, per_sample_timesteps,
            img_shapes, text_sample_lens, img_sample_lens,
        )

        # 6. Loss
        # NOTE: compute MSE on ALL tokens (including timestep==0).
        # Timestep==0 means x_t == clean, pred should be (noise - clean), loss is valid.
        # Filtering by has_mse would break analyze_and_optimize_loss_aggregation which
        # expects loss count == mse_loss_indexes count.
        target = noise - packed_latent_clean
        mse = (packed_mse_preds - target) ** 2
        out_preds = packed_latent_noised - packed_timesteps[:, None].to(packed_latent_noised.dtype) * packed_mse_preds

        loss_dict = dict(mse=mse, ce=None)

        extra_info = dict(
            latent_mean=latent_mean, latent_std=latent_std,
            latent_clean=packed_latent_clean, latent_noised=packed_latent_noised,
            latent_pred=out_preds.detach() if out_preds is not None else None,
            latent_indices=packed_rev_indices,
            latent_timesteps=packed_timesteps,
        )
        return loss_dict, extra_info

    # ── forward_inference ─────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        patchified_vae_latent_shapes: List[Tuple[int, int]] = None,
        packed_vae_token_indexes: torch.LongTensor = None,
        condition_sample_lens: List[int] = None,
        actual_txt_indexes: torch.LongTensor = None,
        num_timesteps: int = 50,
        seed: int = 42,
        **kwargs,
    ):
        """Multi-step denoising via NaviT. Uses same packed layout as training.

        Returns list of [h*w, patch_dim] tensors — one per image, in normalized latent space.
        """
        device = packed_text_ids.device
        model_dtype = next(self.dit_model.parameters()).dtype
        N = len(patchified_vae_latent_shapes)

        # 1. Text
        packed_text_embeds = self._encode_text(packed_text_ids, condition_sample_lens, actual_txt_indexes)

        # 2. Shapes
        img_sample_lens = [h * w for h, w in patchified_vae_latent_shapes]
        text_sample_lens = [sample_lens[i] - img_sample_lens[i] for i in range(N)]
        img_shapes = [[(1, h, w)] for h, w in patchified_vae_latent_shapes]
        total_img = sum(img_sample_lens)

        # 3. Initial noise
        gen = torch.Generator(device=device).manual_seed(seed)
        x_t = torch.randn(total_img, self.patch_latent_dim, generator=gen,
                           device=device, dtype=model_dtype)

        # 4. Per-token timestep shifts + schedule
        packed_shifts = []
        for h, w in patchified_vae_latent_shapes:
            n = h * w
            packed_shifts.extend([math.exp(calculate_shift(n))] * n)
        packed_shifts = torch.tensor(packed_shifts, device=device)

        base_t = torch.linspace(1, 0, num_timesteps + 1, device=device)
        shifts = packed_shifts.unsqueeze(1)
        token_t = shifts * base_t.unsqueeze(0) / (1 + (shifts - 1) * base_t.unsqueeze(0))
        token_dt = (token_t[:, :-1] - token_t[:, 1:]).to(model_dtype)
        token_t = token_t[:, :-1].to(model_dtype)

        # 5. Denoising loop
        for i in range(num_timesteps):
            per_token_t = token_t[:, i]
            per_sample_t = []
            off = 0
            for n in img_sample_lens:
                per_sample_t.append(per_token_t[off]); off += n

            v_t = self._call_dit(
                x_t, packed_text_embeds,
                packed_vae_token_indexes, packed_text_indexes,
                sample_lens, torch.stack(per_sample_t),
                img_shapes, text_sample_lens, img_sample_lens,
            )

            dt = token_dt[:, i:i+1]
            x_t = x_t - v_t * dt

        return x_t.split(img_sample_lens)
