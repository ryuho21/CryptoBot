# ==============================================================================
# PRODUCTION-READY TRADING BOT DOCKERFILE - ULTIMATE FIX (FINAL VERSION)
# Multi-stage build with proper permissions for Windows + Linux
# PyTorch 2.3.1 + CUDA 12.1 + Complete dependency stack
# ==============================================================================

# ==============================================================================
# STAGE 1: Builder - Compile dependencies
# ==============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./ 
COPY pandas_ta-0.3.14b.tar.gz ./

# Upgrade pip and install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
        -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ./pandas_ta-0.3.14b.tar.gz


# ==============================================================================
# STAGE 2: Runtime - Minimal production image
# ==============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="khalilzaldua@gmail.com"
LABEL version="2.1-production-final"
LABEL description="Advanced PPO Trading Bot - Final Production Build"

WORKDIR /app

# ------------------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_CUDA=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MPLBACKEND=Agg \
    NUMEXPR_MAX_THREADS=8 \
    TRADING_MODE=paper \
    LOG_LEVEL=INFO \
    ENABLE_REGIME_DETECTION=1 \
    ENABLE_MULTI_ASSET=0 \
    ENABLE_ORDER_BOOK=0

# ------------------------------------------------------------------------------
# Runtime system packages
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Create non-root user
# ------------------------------------------------------------------------------
RUN groupadd -g 1000 tradingbot && \
    useradd -m -u 1000 -g 1000 -s /bin/bash tradingbot

# ------------------------------------------------------------------------------
# ✅ Copy globally installed Python libs from builder
# ------------------------------------------------------------------------------
COPY --from=builder /opt/conda /opt/conda


RUN pip install --force-reinstall --no-cache-dir psutil
# ------------------------------------------------------------------------------
# Prepare application directory
# ------------------------------------------------------------------------------
RUN mkdir -p \
    /app/runs/data_cache \
    /app/runs/charts \
    /app/runs/episodes \
    /app/runs/checkpoints \
    /app/runs/plots \
    /app/runs/backtest \
    /app/runs/ab_test \
    /app/logs && \
    chmod -R 777 /app/runs /app/logs && \
    chown -R tradingbot:tradingbot /app

# ------------------------------------------------------------------------------
# Copy application code
# ------------------------------------------------------------------------------
COPY --chown=tradingbot:tradingbot *.py ./ 
COPY --chown=tradingbot:tradingbot requirements.txt ./ 

RUN touch /app/runs/.gitkeep /app/logs/.gitkeep && \
    chown tradingbot:tradingbot /app/runs/.gitkeep /app/logs/.gitkeep

# ------------------------------------------------------------------------------
# Switch to non-root user
# ------------------------------------------------------------------------------
USER tradingbot

# ------------------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "\
import sys, torch, ccxt, numpy as np, pandas as pd; \
try: import plotly; has_plotly=True; except: has_plotly=False; \
try: import torch_geometric; has_pyg=True; except: has_pyg=False; \
print('✅ Health check passed'); \
print(f'  PyTorch: {torch.__version__}'); \
print(f'  CUDA: {torch.cuda.is_available()}'); \
print(f'  CCXT: {ccxt.__version__}'); \
print(f'  Plotly: {has_plotly}'); \
print(f'  PyG: {has_pyg}'); \
sys.exit(0)" || exit 1

# ------------------------------------------------------------------------------
# Ports and Volumes
# ------------------------------------------------------------------------------
EXPOSE 8501 8080
VOLUME ["/app/runs", "/app/logs"]

# ------------------------------------------------------------------------------
# Entrypoint & Default CMD
# ------------------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "advanced_main.py", "--preset", "development", "--mode", "train"]
