# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Hugging Face Space image for the Context-Scaling Structured Prompt Studio.
#
# Uses NVIDIA's CUDA 12 base so torch, flash-attn, transformers, and
# safetensors can all use GPU acceleration. The PE model and the DiT are
# loaded once at container startup from paths defined by env vars — set
# them in the Space's "Variables and secrets" panel.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/.hf_home \
    TRANSFORMERS_CACHE=/data/.hf_home/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        git build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the demo needs — training/, evalkit/structure_prompt/ are
# imported at runtime; demo/ contains app.py + backend/ + static/.
COPY training/ /app/training/
COPY evalkit/structure_prompt/system_prompts/ /app/evalkit/structure_prompt/system_prompts/
COPY demo/ /app/demo/

RUN python -m pip install --upgrade pip \
 && python -m pip install -r /app/demo/requirements.txt

# HF Spaces set PORT=7860 by convention.
ENV PORT=7860
EXPOSE 7860

# The lifespan hook in app.py handles model loading.
CMD ["python", "-m", "demo.app"]
