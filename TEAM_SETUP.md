# DebateArena — Team Setup Guide

This guide lets anyone on the team run the full system independently on their own Webis cluster session.

---

## What runs where

| Service | Runs on | Port |
|---|---|---|
| Orchestrator | Your laptop | 8002 |
| Frontend | Your laptop | (file, open in browser) |
| Kokoro TTS | SLURM cluster | 18003 |
| Emotion API | SLURM cluster | 8000 |
| Ollama (qwen2.5:7b) | SLURM cluster | 11437 |
| Judge API | SLURM cluster | 8001 |

Everything on the cluster runs under **your own SLURM account**. You do not share sessions with teammates.

---

## Step 0 — Get the code onto the cluster

SSH in and clone the repo to your home directory:

```bash
ssh <YOUR_USERNAME>@ssh.webis.de
git clone https://github.com/sneha020902/debate-arena.git ~/debate-arena
```

If you already have it cloned, just pull the latest:

```bash
cd ~/debate-arena && git pull
```

---

## Step 1 — Start Kokoro TTS (tmux session: kokoro)

```bash
tmux new -s kokoro
cd ~/debate-arena/kokoro_service
bash run_kokoro.sh
```

Wait for: `Uvicorn running on http://0.0.0.0:18003`

Note the node name printed at the top (e.g. `gammaweb09`). The script writes it to `~/kokoro_node.txt` automatically.

Detach: `Ctrl+B then D`

> First run takes 3–5 min while torch installs. Subsequent starts are fast.

---

## Step 2 — Start Emotion API (tmux session: emotion)

```bash
tmux new -s emotion
cd ~/debate-arena/emotion-api
bash run_emotion_api.sh
```

Wait for: `Application startup complete`

The script reads `~/kokoro_node.txt` automatically — no manual config needed.

Detach: `Ctrl+B then D`

---

## Step 3 — Start Ollama (tmux session: ollama)

```bash
tmux new -s ollama
srun --gres=gpu:ampere \
     --container-image=ollama/ollama:latest \
     --job-name=ollama-1 \
     --container-writable \
     --mem=32GB \
     --pty bash -c "echo \$(hostname) && export OLLAMA_HOST=0.0.0.0:11437 && ollama serve"
```

Wait for: `Listening on [::]:11437`

Note the node name (e.g. `gammaweb07`).

Detach: `Ctrl+B then D`

---

## Step 4 — Start Judge API (tmux session: judge-api)

First, write the Ollama node you noted above:

```bash
echo gammaweb07 > ~/ollama_node.txt   # replace with YOUR ollama node
```

Then start the judge-api:

```bash
tmux new -s judge-api
cd ~/debate-arena/judge-api
bash run_judge.sh
```

Wait for: `Application startup complete`

Note the node name printed at the top.

Detach: `Ctrl+B then D`

> First run takes ~5 min while NLI and sentence-transformer models download to the cluster. After that, subsequent starts are fast (models cached in `~/.cache/`).

---

## Step 5 — Check everything is running

```bash
squeue -u <YOUR_USERNAME>
```

You should see 4 jobs: `kokoro-tts`, `emotion-api`, `ollama-1`, `judge-api`.

---

## Step 6 — Open SSH tunnels (on your laptop)

Open **4 separate terminal windows** and run one command in each. Replace node names with what `squeue` showed you.

```bash
# Terminal 1 — Kokoro
ssh -N -L 18003:<KOKORO_NODE>.medien.uni-weimar.de:18003 <YOUR_USERNAME>@ssh.webis.de

# Terminal 2 — Emotion API
ssh -N -L 8000:<EMOTION_NODE>.medien.uni-weimar.de:8000 <YOUR_USERNAME>@ssh.webis.de

# Terminal 3 — Ollama
ssh -N -L 11437:<OLLAMA_NODE>.medien.uni-weimar.de:11437 <YOUR_USERNAME>@ssh.webis.de

# Terminal 4 — Judge API
ssh -N -L 8001:<JUDGE_NODE>.medien.uni-weimar.de:8001 <YOUR_USERNAME>@ssh.webis.de
```

These windows should show nothing — that's normal. Leave them open.

---

## Step 7 — Start the orchestrator on your laptop

Make sure you have Python 3.11+ and the repo cloned locally.

```powershell
# In PowerShell, from d:\path\to\debate-arena\
python -m venv venv-orchestrator
.\venv-orchestrator\Scripts\pip install fastapi "uvicorn[standard]" httpx pydantic websockets requests
.\run.ps1
```

If you already have `venv-orchestrator`, just run:

```powershell
.\run.ps1
```

This starts the orchestrator on port 8002 and opens the frontend automatically.

---

## Verify it's all working

Open the frontend, start a debate. At the end you should see a winner verdict like:

> **WINNER: Qwen-Pro** — LLM-A wins 0.772 to 0.753 (margin 0.019). LLM-A was carried by engagement (1.0) and rebuttal effectiveness (1.0)...

This means the full DL scoring pipeline ran on the cluster successfully.

---

## Common problems

| Problem | Fix |
|---|---|
| `cannot execute: required file not found` for pip | Already handled in scripts — uses `python -m pip` |
| `\r: command not found` when running .sh | Run: `sed -i 's/\r//' <script>.sh` (Windows line endings from git) |
| Judge API can't find Ollama | Check `~/ollama_node.txt` has the right node name |
| No corpus badge (purple ⚡) | Connect Webis VPN on your laptop |
| Tunnel drops | Re-run the ssh tunnel command in a new terminal |
| `tmux: duplicate session` | Run `tmux attach -t <name>` to check if it's still running |

---

## Node names change every restart

Every time a SLURM job is cancelled and restarted, it may land on a different node. Always run `squeue -u <YOUR_USERNAME>` after restarting any service, and update your tunnels accordingly.
