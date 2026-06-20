#!/bin/bash
# ── Emotion API — Pyxis/Enroot SLURM job ──────────────────────────────────────
# DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026
#
# Runs emotion-api inside a python:3.11-slim container on SLURM (avoids the
# broken Python 3.14 on the login node). Same pattern as kokoro_service.
#
# NOTE: KOKORO_SERVICE_URL below points to gammaweb07 — the node Kokoro is
# CURRENTLY running on. If you ever restart the Kokoro job, check its new
# node with `squeue -u qeso3721` and update the line below.

echo "Requesting compute node and starting Emotion API container..."

srun \
  --container-image=python:3.11-slim \
  --container-mounts="$(pwd)":/app \
  --container-workdir=/app \
  --container-writable \
  --container-remap-root \
  --mem=16GB \
  --cpus-per-task=4 \
  --time=02:00:00 \
  --job-name=emotion-api \
  --pty bash -c "
    echo '=== Running on node: '\$(hostname)' ===' && \
    export DEBIAN_FRONTEND=noninteractive && \
    mkdir -p /usr/share/man/man1 /usr/share/man/man7 && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential libsndfile1 ffmpeg > /dev/null && \
    pip install --no-cache-dir -q -r requirements.txt && \
    export KOKORO_SERVICE_URL=http://gammaweb07.medien.uni-weimar.de:8003 && \
    echo '=== Starting Emotion API on port 8000 ===' && \
    uvicorn main:app --host 0.0.0.0 --port 8000
  "