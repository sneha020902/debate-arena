#!/bin/bash
# ── Judge API — Pyxis/Enroot SLURM job ────────────────────────────────────────
# DEBATE Project · Bauhaus-Universität Weimar · Webis Lab · SS 2026
#
# Runs judge-api on the cluster so heavy ML inference (NLI CrossEncoder,
# sentence-transformers ArgQuality) happens on cluster CPU, not your laptop.
#
# Start ORDER: Kokoro → Emotion API → Ollama → Judge API
# (Judge API reads Emotion and Ollama node files written by those scripts)
#
# ── How to use ─────────────────────────────────────────────────────────────
# 1. Upload judge-api to cluster:
#      scp -r judge-api joso1563@ssh.webis.de:~/debate-arena-integrated/
#
# 2. SSH in, open a new tmux session:
#      tmux new -s judge-api
#
# 3. Run:
#      cd ~/debate-arena-integrated/judge-api && bash run_judge.sh
#
# 4. Wait for: "Application startup complete"
#    Note the node name printed at the top.
#
# 5. Detach: Ctrl+B then D
#
# 6. Open tunnel locally:
#      ssh -N -L 8001:<NODE>.medien.uni-weimar.de:8001 joso1563@ssh.webis.de

echo "Requesting compute node and starting Judge API container..."

# Read Ollama node (set manually: echo <node> > ~/ollama_node.txt)
OLLAMA_NODE=$(cat "$HOME/ollama_node.txt" 2>/dev/null || echo "")
if [ -z "$OLLAMA_NODE" ]; then
  echo "WARNING: ollama_node.txt not found. Run: echo gammaweb07 > ~/ollama_node.txt"
  echo "         Falling back to gammaweb07."
  OLLAMA_NODE="gammaweb07"
fi
echo "Ollama node: $OLLAMA_NODE"

# Read Emotion API node (written automatically by run_emotion_api.sh)
EMOTION_NODE=$(cat "$HOME/emotion_node.txt" 2>/dev/null || echo "")
if [ -z "$EMOTION_NODE" ]; then
  echo "WARNING: emotion_node.txt not found — start Emotion API first."
  EMOTION_NODE="gammaweb08"
fi
echo "Emotion API node: $EMOTION_NODE"

srun \
  --container-image=python:3.11-slim \
  --container-mounts="$(pwd)":/app,"$HOME":/userhome \
  --container-workdir=/app \
  --container-writable \
  --container-remap-root \
  --mem=16GB \
  --cpus-per-task=4 \
  --time=24:00:00 \
  --job-name=judge-api \
  --exclude=gammaweb05 \
  --pty bash -c "
    echo '=== Running on node: '\$(hostname)' ===' && \
    echo \$(hostname) > /userhome/judge_node.txt && \
    export DEBIAN_FRONTEND=noninteractive && \
    mkdir -p /usr/share/man/man1 /usr/share/man/man7 && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential > /dev/null && \
    pip install --cache-dir /userhome/.cache/pip -q \
      fastapi 'uvicorn[standard]' pydantic httpx requests ollama \
      sentence-transformers datasets numpy python-dotenv && \
    export OLLAMA_HOST=http://${OLLAMA_NODE}.medien.uni-weimar.de:11437 && \
    export OLLAMA_MODEL=qwen2.5:7b && \
    export EMOTION_API=http://${EMOTION_NODE}.medien.uni-weimar.de:8000 && \
    echo '=== Ollama: '\$OLLAMA_HOST' ===' && \
    echo '=== Emotion API: '\$EMOTION_API' ===' && \
    echo '=== Starting Judge API on port 8001 ===' && \
    python -m uvicorn run_full:app --host 0.0.0.0 --port 8001
  "
