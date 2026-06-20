# HANDOFF — DebateArena SS26 (Week 7, Kokoro + SLURM Deployment Session)
**Bauhaus-Universität Weimar · Webis Lab · SS 2026**
**Sneha Agrawal · Emotion Track**

⚠️ **This supersedes any earlier handoff docs.** Kokoro is now fully deployed and working on SLURM — earlier docs marking it "not started" are outdated.

---

## What Changed This Session

| # | Task | Status |
|---|---|---|
| 1 | Kokoro TTS deployed on SLURM | ✅ Done, tested end-to-end |
| 2 | Emotion API deployed on SLURM (own account) | ✅ Done, tested end-to-end |
| 3 | Ollama deployed on SLURM (own account) | ✅ Done, using existing proven method |
| 4 | Full debate tested with live audio | ✅ Confirmed working |
| 5 | Frontend audio bugs (overlap, stop, speed) | ✅ Fixed |
| 6 | Composure vs reasoning density clarified | ✅ Done — composure stays with Emotion track |

---

## Architecture (Current, Real)

```
Frontend (index.html)         — browser, no server
      ↕ WebSocket
Orchestrator :8002             — runs LOCALLY on Mac (venv-orchestrator)
      ↕ HTTP (via 3 SSH tunnels)
Emotion API  :8000              — SLURM, qeso3721 account
Kokoro TTS   :8003              — SLURM, qeso3721 account
Ollama       :11437             — SLURM, qeso3721 account, GPU (A100)
Judge API    :8001              — NOT deployed, 404s, falls back to composure winner (fine)
```

**Important:** All 3 SLURM services currently run on the same node (`gammaweb07`), but this **can change** any time a job is restarted — always check `squeue -u qeso3721` for current hostname.

---

## Why Containers (Important Context)

- Login node Python is **3.14** — too new, breaks `blis`/`spacy`/`torch` installs (both locally on Mac AND on the cluster login node)
- Fix: run everything inside Pyxis/Enroot containers (`srun --container-image=...`) using `python:3.11-slim`
- Cluster confirmed to support Pyxis (NOT Singularity, despite earlier assumption)
- **Critical container flag:** `--container-remap-root` — without it, `apt-get`/`pip install` fail with "requested operation requires superuser privilege"
- Kokoro needs `espeak-ng` (system package) — add to `apt-get install` line
- Ollama uses the **official `ollama/ollama:latest` Docker image** (not installed in home dir) — Sneha already had a personal proven script for this (see "Sneha's Ollama Guide" below)

---

## Local Project Root
```
/Users/snehaagrawal/Desktop/Project/week-7/debate-arena-integrated/
```

## Cluster Folder Layout (qeso3721 account)
```
~/debate-emotion/
├── kokoro_service/        ← main.py, requirements.txt, run_kokoro.sh
└── emotion-api/           ← full emotion-api code + run_emotion_api.sh (new)
```

---

## How to Run — Full Restart Checklist

### 1. Check what's already alive
```bash
ssh qeso3721@ssh.webis.de
squeue -u qeso3721
```
Look for `kokoro-tts`, `emotion-api`, `ollama-1` (or `ollama_new` tmux session). Note the **current node hostname**.

### 2. If Kokoro is dead — restart
```bash
tmux new -s kokoro   # or: tmux attach -t kokoro if it exists
cd ~/debate-emotion/kokoro_service
bash run_kokoro.sh
```
Wait for `Uvicorn running on http://0.0.0.0:8003`. Detach: `Ctrl+B` then `D`.

### 3. If Emotion API is dead — restart
```bash
tmux new -s emotion-api
cd ~/debate-emotion/emotion-api
bash run_emotion_api.sh
```
This script sets `KOKORO_SERVICE_URL` internally — **must point to Kokoro's current node**, edit inside the script if Kokoro moved to a different node. Wait for `Application startup complete`. Detach.

### 4. If Ollama is dead — restart (Sneha's own proven command)
```bash
tmux new -s ollama_new
srun --gres=gpu:ampere --container-image=ollama/ollama:latest --job-name=ollama-11437 --container-writable --mem=32GB --pty bash -c "echo \$(hostname) && export OLLAMA_HOST=0.0.0.0:11437 && ollama serve"
```
Wait for `Listening on [::]:11437`. Detach.

### 5. Open 3 SSH tunnels on Mac (separate windows, leave all running silently)
```bash
ssh -N -L 8003:<NODE>.medien.uni-weimar.de:8003 qeso3721@ssh.webis.de
ssh -N -L 8000:<NODE>.medien.uni-weimar.de:8000 qeso3721@ssh.webis.de
ssh -N -L 11437:<NODE>.medien.uni-weimar.de:11437 qeso3721@ssh.webis.de
```
Replace `<NODE>` with whatever `squeue` showed.

### 6. Start orchestrator locally
```bash
cd /Users/snehaagrawal/Desktop/Project/week-7/debate-arena-integrated/orchestrator
source ../venv-orchestrator/bin/activate
uvicorn main:app --port 8002 --reload
```

### 7. Open frontend
```
frontend/index.html — double-click or `open frontend/index.html`
```

---

## Frontend Setup Screen Values (Use These, Not the Old Defaults)
| Field | Value |
|---|---|
| Ollama Model | `gemma:7b` |
| Ollama URL | `http://localhost:11437` |
| Total Turns | 6 (3 rounds) recommended |

---

## Known Issue — TTS Occasionally Silent (Real Cause Found)

**Symptom:** one argument's audio sometimes doesn't play, no Listen button either.

**Cause (confirmed via orchestrator logs):**
```
WARNING:debate_flow:TTS synthesis failed for role 'llm_a':
```
Empty message = timeout. `_synthesize()` in `debate_flow.py` only waits 30s — since Ollama, Kokoro, and emotion-api all share one GPU node, occasional load causes synthesis to exceed that.

**Fix (apply if not already done):** in `orchestrator/debate_flow.py`, inside `_synthesize()`:
```python
async with httpx.AsyncClient(timeout=30.0) as client:
```
→ change to:
```python
async with httpx.AsyncClient(timeout=60.0) as client:
```

---

## Frontend Fixes Applied (`frontend/index.html`)

| Fix | What it does |
|---|---|
| Unified audio state | Auto-play and manual "Listen" clicks now share one system — only one clip plays at a time, ever |
| Button reflects real state | Whichever card is playing (auto OR manual) shows **"■ Stop"** live, not just on manual click |
| Speed control | Small inline dropdown (0.25x–2x) next to status text, applies to whatever's currently playing |
| Removed | Old separate global "Stop Audio" button (redundant once state was unified) |
| Safari autoplay fix | A silent `Audio()` play is fired inside the "START DEBATE" click handler to unlock background audio for the session |
| Emotion label | Shows raw classifier output only — do NOT override with composure-derived labels (tried this, reverted — redundant with composure bar, and hides real model output) |

---

## Composure vs Reasoning Density (Settled, Don't Re-litigate)

- **Composure** = Sneha's Emotion API output. Formula: `composure = 1 − intensity`
- This IS what's currently given to the Logic team
- Logic team is **not currently using it** — that's fine, not a bug
- **Reasoning density** = Logic team's own separate internal metric — Sneha does not produce or own this

---

## Things NOT Done / Future Work
- Judge API still not deployed on SLURM (404 fallback works fine for demos)
- `debate_flow.py` timeout fix — confirm it was actually applied
- Consider running Kokoro + emotion-api + Ollama on separate nodes if resource contention keeps causing TTS timeouts (would require updating `KOKORO_SERVICE_URL` dynamically)

---

*DebateArena · Sneha Agrawal · Bauhaus-Universität Weimar · SS 2026*