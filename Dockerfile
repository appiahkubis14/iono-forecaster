# ── IonoForecaster Dockerfile ─────────────────────────────────────────────────
# Multi-stage build for a reproducible ESA-grade pipeline container.
#
# Usage:
#   docker build -t iono-forecaster .
#   docker run --rm -v $(pwd)/data:/app/data iono-forecaster python main.py --synthetic --step all
#
# GPU variant (uncomment FROM line below):
#   docker build --build-arg BASE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 -t iono-forecaster-gpu .
#
# Author: Samuel Appiah Kubi | Paris Lodron University Salzburg

ARG BASE=python:3.11-slim
FROM ${BASE}

LABEL maintainer="Samuel Appiah Kubi <s.appiah-kubi@stud.plus.ac.at>"
LABEL description="IonoForecaster: AI-based ionospheric scintillation forecasting over equatorial Africa"
LABEL version="1.0.0"
LABEL project="Copernicus Master's in Digital Earth — ESA Portfolio"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgeos-dev \
        libproj-dev \
        libhdf5-serial-dev \
        libnetcdf-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# CPU-only PyTorch to keep image size manageable
RUN pip install --no-cache-dir \
        torch==2.1.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Data directories ──────────────────────────────────────────────────────────
RUN mkdir -p \
        data/raw/gnss \
        data/raw/solar_wind \
        data/raw/geomagnetic \
        data/raw/swarm \
        data/raw/aurora \
        data/processed/gnss \
        data/processed/solar_wind \
        data/processed/geomagnetic \
        data/processed/swarm \
        data/processed/aurora \
        data/features \
        data/graphs \
        data/models \
        data/outputs/forecasts \
        data/outputs/events \
        data/outputs/alerts \
        data/outputs/validation \
        data/outputs/stac \
        data/outputs/dashboard \
        data/logs

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import pandas; import numpy; print('OK')" || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "main.py", "--synthetic", "--step", "all"]
