# DebateArena — Integrated System
**Bauhaus-Universität Weimar · Webis Lab · SS 2026**
**Team:** Sneha Agrawal (Emotion + TTS) · Rosenmeet (Orchestrator/Frontend) · Manoj (Logic/Judge)
**Supervisors:** Maximilian Heinrich · Midhun Kanadan

---

## What This Is

A full LLM-vs-LLM debate system. Two language models argue a topic across multiple rounds. A host announcer introduces the debate with real spoken audio (Kokoro TTS), each argument is scored live by the Emotion API, a coach can steer either side mid-debate, and a winner is declared at the end. Arguments are enriched with real debate references from the CMV (ChangeMyView) Reddit corpus via Elasticsearch.

**Status: Fully working end-to-end, including live voice synthesis and ES corpus integration.**

---

## Architecture

```
Frontend (index.html)          — open in browser, no server needed
      ↕ WebSocket
Orchestrator (port 8002)       — runs locally on your machine
      ↕ HTTP (via SSH tunnels)
Emotion API  (port 8000)       — SLURM cluster, scoring + TTS proxy
Kokoro TTS   (port 18003)      — SLURM cluster, voice synthesis
Ollama       (port 11437)      — SLURM cluster, GPU, LLM generation
ES API       (141.54.159.66)   — webislab37, corpus reference search (requires Webis VPN)
Judge API    (port 8001)       — not deployed; falls back to composure-based winner
```

Each team member runs their **own** Kokoro, Emotion API, and Ollama on SLURM under their own account.

---

## Folder Structure

```
debate-arena-1/
├── emotion-api/
│   ├── main.py
│   ├── synthesis.py         ← calls Kokoro over HTTP
│   ├── run_emotion_api.sh   ← SLURM container launch script (auto-reads Kokoro node)
│   └── ...
├── kokoro_service/
│   ├── main.py              ← standalone Kokoro microservice
│   ├── requirements.txt
│   └── run_kokoro.sh        ← SLURM container launch script (writes node to ~/kokoro_node.txt)
├── orchestrator/
│   ├── main.py
│   ├── debate_flow.py       ← ES integration, coach steering, prompts
│   └── host_script.py
└── frontend/
    └── index.html
```

---

## How to Run

### Step 1 — SSH into cluster and check running jobs

```bash
ssh <your-username>@ssh.webis.de
squeue -u <your-username>
```

You need 3 jobs: `kokoro-tts`, `emotion-api`, `ollama-1`. Note node names — they change on every restart.

---

### Step 2 — Start SLURM services (if not running)

**Important: start Kokoro first — Emotion API reads its node automatically.**

**Kokoro TTS:**
```bash
tmux new -s kokoro
cd ~/debate-arena-integrated/kokoro_service && bash run_kokoro.sh
```
Wait for: `Uvicorn running on http://0.0.0.0:18003`
Note the node name printed at the top. Detach: `Ctrl+B then D`

**Emotion API (start AFTER Kokoro):**
```bash
tmux new -s emotion-api
cd ~/debate-arena-integrated/emotion-api && bash run_emotion_api.sh
```
The script reads Kokoro's node automatically from `~/kokoro_node.txt` — no manual editing needed.
Wait for: `Application startup complete`. Detach: `Ctrl+B then D`

**Ollama:**
```bash
tmux new -s ollama
srun --gres=gpu:ampere --container-image=ollama/ollama:latest --job-name=ollama-1 --container-writable --mem=32GB --pty bash -c "echo $(hostname) && export OLLAMA_HOST=0.0.0.0:11437 && ollama serve"
```
Wait for: `Listening on [::]:11437`. Note node name. Detach: `Ctrl+B then D`

---

### Step 3 — Open SSH tunnels (3 separate local terminals, leave open)

```bash
ssh -N -L 18003:<KOKORO_NODE>.medien.uni-weimar.de:18003 <your-username>@ssh.webis.de
ssh -N -L 8000:<EMOTION_NODE>.medien.uni-weimar.de:8000 <your-username>@ssh.webis.de
ssh -N -L 11437:<OLLAMA_NODE>.medien.uni-weimar.de:11437 <your-username>@ssh.webis.de
```

Replace `<NODE>` with whatever `squeue` shows. These print nothing — that's normal.

---

### Step 4 — Start Orchestrator (local)

```bash
cd debate-arena-1/orchestrator
uvicorn main:app --port 8002 --reload
```

---

### Step 5 — Open Frontend

Double-click `frontend/index.html` in your file explorer, or `open frontend/index.html` on Mac.

Connect **Webis VPN** before starting a debate to enable the ES corpus badge.

---

## Frontend Settings

| Field | Value |
|---|---|
| Ollama Model | `qwen2.5:7b` |
| Ollama URL | `http://localhost:11437` |
| Total Turns | 6 (3 rounds) recommended |

---

## Coach Steering

The debate auto-pauses after every full round. Type a keyword into the Coach A or Coach B box and click Resume.

| Keyword | What the LLM does |
|---|---|
| `statistics` | Backs every claim with numbers, percentages, or named studies |
| `examples` | Grounds every point in a concrete real-world example |
| `empathy` | Appeals to human suffering, real people, real lives — personal and emotional |
| `aggressive` | Attacks opponent's logic head-on, no mercy |
| `calm` | Cold, measured authority — logic over emotion |
| `realistic` | Only grounded real-world arguments, no hypotheticals |
| `simple` | Plain language, short sentences, no jargon |
| `technical` | Deep expertise, precise terminology |
| `rhetorical` | Rhetorical questions only — never state, always ask |

Typos are tolerated (75% similarity threshold). Free-form instructions also work and pass through directly to the LLM.

---

## ES Corpus Badge

Each argument card shows a **⚡ N corpus refs provided** badge when Elasticsearch references were sent to the LLM's prompt. Hover to see the actual claims. Requires Webis VPN.

> Note: the badge shows claims *provided* to the LLM, not necessarily *used*. The LLM may or may not have drawn on them.

---

## Composure Score

```
composure = 1 − intensity
intensity  = 0.55 × keyword_score + 0.45 × (anger + disgust + 0.5 × fear)
```

Produced by the Emotion API per argument. Shown live on each card and the top score bar.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| "Ollama generation failed" | Tunnel down or job expired | Check `squeue`, restart if needed |
| Composure always 50% | Emotion API tunnel not connected | `curl http://localhost:8000/health` |
| No audio / no Listen button | TTS timeout | Already fixed to 60s — check tunnel to port 18003 |
| No corpus badge | Webis VPN not connected | Connect VPN and restart debate |
| WebSocket error | Orchestrator not running | Start uvicorn on port 8002 |
| `tmux: duplicate session` | Session already exists | `tmux attach -t <name>` |
| `\r: command not found` | Windows line endings in script | `sed -i 's/\r//' <script>.sh` on cluster |
| Emotion API can't reach Kokoro | Wrong node in KOKORO_SERVICE_URL | Start Kokoro first so `~/kokoro_node.txt` exists, then restart Emotion API |
| LLM generating Chinese characters | Qwen multilingual model | Already fixed — "Respond in English only" in prompts |

---

## Known Limitations

- Judge API not deployed — falls back to composure-based winner (works fine for demos)
- Node hostname changes on every job restart — always check `squeue` before tunneling
- ES corpus badge requires Webis VPN — debate runs normally without it, just no badge

---

*DebateArena · Bauhaus-Universität Weimar · SS 2026*
