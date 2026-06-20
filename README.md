# DebateArena — Integrated System
**Bauhaus-Universität Weimar · Webis Lab · SS 2026**
**Team:** Sneha Agrawal (Emotion + TTS) · Rosenmeet (Orchestrator/Frontend) · Manoj (Logic/Judge)
**Supervisors:** Maximilian Heinrich · Midhun Kanadan

---

## What This Is

A full LLM-vs-LLM debate system. Two language models argue a topic across multiple rounds. A host announcer introduces the debate with real spoken audio (Kokoro TTS), each argument is scored live by the Emotion API, a coach can steer either side mid-debate, and a winner is declared at the end.

**Status: Fully working end-to-end, including live voice synthesis.**

---

## Architecture

```
Frontend (index.html)          — open in browser, no server needed
      ↕ WebSocket
Orchestrator (port 8002)       — runs locally on Mac
      ↕ HTTP (via SSH tunnels)
Emotion API  (port 8000)       — SLURM cluster, scoring + TTS proxy
Kokoro TTS   (port 8003)       — SLURM cluster, voice synthesis
Ollama       (port 11437)      — SLURM cluster, GPU, LLM generation
Judge API    (port 8001)       — not deployed; falls back to composure-based winner
```

All three SLURM services run inside isolated containers (`python:3.11-slim` for Emotion API/Kokoro, official `ollama/ollama:latest` for Ollama) to avoid the cluster's broken default Python (3.14).

---

## Folder Structure

```
debate-arena-integrated/
├── emotion-api/
│   ├── main.py
│   ├── synthesis.py         ← calls Kokoro over HTTP
│   ├── run_emotion_api.sh   ← SLURM container launch script
│   └── ...
├── kokoro_service/
│   ├── main.py              ← standalone Kokoro microservice
│   ├── requirements.txt
│   └── run_kokoro.sh        ← SLURM container launch script
├── orchestrator/
│   ├── main.py
│   ├── debate_flow.py
│   └── host_script.py
├── frontend/
│   └── index.html
└── venv-orchestrator/
```

---

## How to Run

### Step 1 — Check / Start SLURM Services

```bash
ssh qeso3721@ssh.webis.de
squeue -u qeso3721
```

Should show `kokoro-tts`, `emotion-api`, `ollama-1`. If any are missing:

**Kokoro:**
```bash
tmux new -s kokoro
cd ~/debate-emotion/kokoro_service && bash run_kokoro.sh
```

**Emotion API:**
```bash
tmux new -s emotion-api
cd ~/debate-emotion/emotion-api && bash run_emotion_api.sh
```

**Ollama:**
```bash
tmux new -s ollama_new
srun --gres=gpu:ampere --container-image=ollama/ollama:latest --job-name=ollama-11437 --container-writable --mem=32GB --pty bash -c "echo \$(hostname) && export OLLAMA_HOST=0.0.0.0:11437 && ollama serve"
```

Detach each with `Ctrl+B` then `D` once running.

### Step 2 — Tunnel All 3 Ports (separate terminal windows on Mac)

```bash
ssh -N -L 8003:<NODE>.medien.uni-weimar.de:8003 qeso3721@ssh.webis.de
ssh -N -L 8000:<NODE>.medien.uni-weimar.de:8000 qeso3721@ssh.webis.de
ssh -N -L 11437:<NODE>.medien.uni-weimar.de:11437 qeso3721@ssh.webis.de
```
`<NODE>` = whatever `squeue` shows (e.g. `gammaweb07`). These tunnels print nothing — that's normal, leave them open.

### Step 3 — Start Orchestrator (local Mac)

```bash
cd orchestrator
source ../venv-orchestrator/bin/activate
uvicorn main:app --port 8002 --reload
```

### Step 4 — Open Frontend

```bash
open frontend/index.html
```

---

## Frontend Settings

| Field | Value |
|---|---|
| Ollama Model | `gemma:7b` |
| Ollama URL | `http://localhost:11437` |
| Total Turns | 6 (3 rounds) recommended |
| Coach Opening Strategy | optional, sets tone for Round 1 |

---

## Coach Steering

Appears automatically once a debate starts.

| Control | What it does |
|---|---|
| Coach A input + Send | Instruction for LLM-A (Pro), used once on its next turn |
| Coach B input + Send | Instruction for LLM-B (Con), used once on its next turn |
| Pause / Resume | Debate auto-pauses after every full round for coach input |
| New Debate | Resets everything |

---

## Audio Controls

- Each argument card has a **▶ Listen / ■ Stop** button — reflects real playback state whether audio started automatically or you clicked it
- Only one clip ever plays at a time (no overlapping voices)
- Speed control (0.25x–2x) next to the status line, applies to current + future playback

---

## Composure Score

```
composure = 1 − intensity
intensity = 0.55 × keyword_score + 0.45 × (anger + disgust + 0.5 × fear)
```

Produced by the Emotion API per argument. Shown live on each card and the top score bar. Given to the Logic team (not currently used by them — fine). Reasoning density is a separate metric owned by the Logic team, not produced here.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| "Ollama generation failed" | Tunnel down or Ollama job expired | Check `squeue`, restart if needed |
| Composure always 50% | Emotion API tunnel not connected | `curl http://localhost:8000/health` to test |
| Some arguments have no audio / no Listen button | TTS request timed out (shared GPU node under load) | Increase timeout in `debate_flow.py`'s `_synthesize()` from 30.0 to 60.0 |
| WebSocket error | Orchestrator not running | Start uvicorn on port 8002 |
| `tmux: duplicate session` | Session already exists | `tmux attach -t <name>` to check before creating a new one |
| Container `apt-get` fails with "superuser privilege" | Missing container flag | Add `--container-remap-root` to the `srun` command |
| Audio overlapping | Old cached page | Hard-reload the browser tab (not just the in-app "New Debate" button) |

---

## Known Limitations

- Judge API not deployed — system falls back to composure-based winner (works fine)
- All 3 SLURM services currently share one GPU node — occasional TTS timeouts under load
- Node hostname changes whenever a job restarts — always verify with `squeue` before tunneling

---

*DebateArena · Sneha Agrawal · Rosenmeet · Bauhaus-Universität Weimar · SS 2026*