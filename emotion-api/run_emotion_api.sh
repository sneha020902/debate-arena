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

# Read current Kokoro node (written by run_kokoro.sh at startup)
KOKORO_NODE=$(cat "$HOME/kokoro_node.txt" 2>/dev/null || echo "")
if [ -z "$KOKORO_NODE" ]; then
  echo "WARNING: kokoro_node.txt not found — start Kokoro first, then restart this."
  echo "         Falling back to last known node. Audio synthesis may fail."
  KOKORO_NODE="gammaweb08"
fi
echo "Kokoro node: $KOKORO_NODE"

srun \
  --container-image=python:3.11-slim \
  --container-mounts="$(pwd)":/app,"$HOME":/userhome \
  --container-workdir=/app \
  --container-writable \
  --container-remap-root \
  --mem=16GB \
  --cpus-per-task=4 \
  --time=08:00:00 \
  --job-name=emotion-api \
  --pty bash -c "
    echo '=== Running on node: '\$(hostname)' ===' && \
    echo \$(hostname) > /userhome/emotion_node.txt && \
    export DEBIAN_FRONTEND=noninteractive && \
    mkdir -p /usr/share/man/man1 /usr/share/man/man7 && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential libsndfile1 ffmpeg > /dev/null && \
    python -m pip install --no-cache-dir -q -r requirements.txt && \
    export KOKORO_SERVICE_URL=http://${KOKORO_NODE}.medien.uni-weimar.de:18003 && \
    echo '=== Kokoro URL: '\$KOKORO_SERVICE_URL' ===' && \
    echo '=== Starting Emotion API on port 8000 ===' && \
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
  "