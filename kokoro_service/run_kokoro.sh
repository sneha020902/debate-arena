#!/bin/bash
# ── Kokoro TTS — Pyxis/Enroot SLURM job ───────────────────────────────────────
# DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026
#
# Follows the same pattern as the team's Ollama setup (srun --container-image=...)
# Uses a plain python:3.11-slim image, mounts kokoro_service/ in, installs deps,
# and runs the FastAPI service on port 8003.
#
# ── How to use ─────────────────────────────────────────────────────────────
# 1. Upload this folder to the cluster:
#      scp -r kokoro_service qeso3721@ssh.webis.de:~/debate-emotion/
#
# 2. SSH in, open a NEW tmux session (separate from the Ollama one):
#      ssh qeso3721@ssh.webis.de
#      tmux new -s kokoro
#
# 3. cd into the folder and run this script directly (NOT sbatch — same
#    interactive style as your Ollama srun command):
#      cd ~/debate-emotion/kokoro_service
#      bash run_kokoro.sh
#
# 4. Wait until you see: "Uvicorn running on http://0.0.0.0:8003"
#    Note the hostname printed at the top (e.g. gammaweb07) — you'll need it.
#
# 5. Detach: Ctrl+b then d  (keeps it running, just like the Ollama terminal)
#
# 6. From your Mac, tunnel port 8003:
#      ssh -N -L 8003:gammaweb07.medien.uni-weimar.de:8003 qeso3721@ssh.webis.de
#    (replace gammaweb07 with whatever hostname was printed)
#
# 7. Test from a third terminal:
#      curl http://localhost:8003/health

echo "Requesting compute node and starting Kokoro TTS container..."

srun \
  --container-image=python:3.11-slim \
  --container-mounts="$(pwd)":/app,"$HOME":/userhome \
  --container-workdir=/app \
  --container-writable \
  --container-remap-root \
  --mem=4GB \
  --cpus-per-task=2 \
  --time=24:00:00 \
  --job-name=kokoro-tts \
  --exclude=gammaweb05 \
  --pty bash -c "
    echo '=== Running on node: '\$(hostname)' ===' && \
    echo \$(hostname) > /userhome/kokoro_node.txt && \
    export DEBIAN_FRONTEND=noninteractive && \
    mkdir -p /usr/share/man/man1 /usr/share/man/man7 && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential libsndfile1 espeak-ng > /dev/null && \
    python -m pip install --no-cache-dir -q -r requirements.txt && \
    echo '=== Starting Kokoro TTS service on port 18003 ===' && \
    python -m uvicorn main:app --host 0.0.0.0 --port 18003
  "