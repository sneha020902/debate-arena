# HANDOFF — DebateArena SS26
**Bauhaus-Universität Weimar · Webis Lab · SS 2026**
**Rosenmeet · Orchestrator + Frontend + Emotion Track**

⚠️ **This supersedes all earlier handoff docs.**

---

## What Changed This Session

| # | Task | Status |
|---|---|---|
| 1 | ES corpus integration in argument generation | ✅ Done |
| 2 | Corpus badge UI (⚡ N corpus refs provided) | ✅ Done |
| 3 | Aggressive base prompts by default | ✅ Done |
| 4 | Coach steering keyword system (9 keywords + typo tolerance) | ✅ Done |
| 5 | Coach instruction moved to END of prompt (overrides sentence limit) | ✅ Done |
| 6 | Kokoro port changed 8003 → 18003 (conflict avoidance) | ✅ Done |
| 7 | TTS timeout increased 30s → 60s | ✅ Done |
| 8 | Emotion API auto-reads Kokoro node from ~/kokoro_node.txt | ✅ Done |
| 9 | English-only output fix (Qwen switching to Chinese) | ✅ Done |
| 10 | emotional + empathy merged into one, rhetorical keyword added | ✅ Done |

---

## Architecture (Current)

```
Frontend (index.html)          — browser, no server needed
      ↕ WebSocket
Orchestrator :8002             — runs locally on your machine
      ↕ HTTP (via SSH tunnels)
Emotion API  :8000             — SLURM, your account
Kokoro TTS   :18003            — SLURM, your account (port changed from 8003)
Ollama       :11437            — SLURM, your account, GPU (A100)
ES API       :8000             — webislab37 (141.54.159.66), requires Webis VPN
Judge API    :8001             — NOT deployed, falls back to composure winner (fine)
```

**Each team member runs their own services under their own SLURM account.**
Shared services across accounts do not work due to cluster networking restrictions.

---

## SLURM Startup Order

**Always start Kokoro first — Emotion API reads its node automatically.**

### Kokoro TTS
```bash
tmux new -s kokoro
cd ~/debate-arena-integrated/kokoro_service && bash run_kokoro.sh
```
- Writes current node to `~/kokoro_node.txt` automatically
- Wait for: `Uvicorn running on http://0.0.0.0:18003`
- Detach: `Ctrl+B then D`

### Emotion API (after Kokoro)
```bash
tmux new -s emotion-api
cd ~/debate-arena-integrated/emotion-api && bash run_emotion_api.sh
```
- Reads `~/kokoro_node.txt` automatically — no manual sed needed
- Writes current node to `~/emotion_node.txt`
- Wait for: `=== Kokoro URL: http://gammaweb0X... ===` then `Application startup complete`
- Detach: `Ctrl+B then D`

### Ollama
```bash
tmux new -s ollama
srun --gres=gpu:ampere --container-image=ollama/ollama:latest --job-name=ollama-1 --container-writable --mem=32GB --pty bash -c "echo $(hostname) && export OLLAMA_HOST=0.0.0.0:11437 && ollama serve"
```
- Wait for: `Listening on [::]:11437`
- Detach: `Ctrl+B then D`

---

## SSH Tunnels (local machine, 3 separate terminals)

```bash
ssh -N -L 18003:<KOKORO_NODE>.medien.uni-weimar.de:18003 <username>@ssh.webis.de
ssh -N -L 8000:<EMOTION_NODE>.medien.uni-weimar.de:8000 <username>@ssh.webis.de
ssh -N -L 11437:<OLLAMA_NODE>.medien.uni-weimar.de:11437 <username>@ssh.webis.de
```

Node names change every restart — always check `squeue -u <username>` first.

---

## ES Corpus Integration

- Before each argument, orchestrator queries ES relay at `141.54.159.66:8000/semantic-search`
- Query: opponent's last argument (or topic for opening turns)
- Returns top-3 claims from CMV Reddit corpus
- All 3 claims provided to Ollama prompt as reference points
- Requires **Webis VPN** — fails silently if unreachable, debate continues normally
- ES server runs on webislab37: `ssh lab` → `nohup python app.py > app.log 2>&1 &`

### Corpus Badge
- Shows `⚡ N corpus refs provided` on each argument card
- Hover reveals the actual claims sent to the LLM
- Says *provided* not *used* — LLM may or may not have drawn on them
- No badge = VPN off or ES unreachable

---

## Coach Steering Keywords

| Keyword | Directive |
|---|---|
| `statistics` | Back every claim with a number, percentage, or named study |
| `examples` | Ground every point in a concrete real-world example |
| `empathy` | Appeal to human suffering and real lives — personal and emotional |
| `aggressive` | Attack opponent's logic head-on, show no mercy |
| `calm` | Cold measured authority — let logic demolish them |
| `realistic` | Only grounded real-world arguments, no hypotheticals |
| `simple` | Plain language, short sentences, no jargon |
| `technical` | Deep expertise, precise terminology |
| `rhetorical` | Rhetorical questions only — never state, always ask |

- Typo tolerant via `difflib` (75% similarity threshold)
- Coach instruction appended AFTER base prompt so it overrides sentence limits
- Free-form instructions (no keyword match) pass through directly to LLM

---

## Emotion Scoring

```
composure  = 1 − intensity
intensity  = 0.55 × keyword_score + 0.45 × (anger + disgust + 0.5 × fear)
```

- Text emotion model: `j-hartmann/emotion-english-distilroberta-base` (7 labels)
- Keyword matching: aggressive vs calm word lists (55% weight)
- Speech emotion module exists (`speech_emotion_module.py`) but not wired into scoring pipeline — text-only currently
- Composure shown live on each card and top score bar

---

## Known Issues / Future Work

- Judge API not deployed — composure fallback works fine for demos
- Speech emotion module built but unused — could be integrated for richer scoring
- Corpus badge shows refs *provided* not *used* — detecting actual LLM usage would require semantic similarity comparison
- Keyword matching (55% of intensity) is context-blind — "not outrageous" still triggers aggressive hit
- `emotional` and `empathy` were merged — were too similar to produce distinct LLM behaviour

---

## Files NOT in Git (gitignored)

- `PERSONAL_SETUP.md` — personal cluster setup notes
- `PRESENTATION_SCRIPT.md` — presentation script
- `generate_ppt.py` / `DebateArena_Changes.pptx` — presentation files
- `team_connect.sh` — teammate tunnel helper (doesn't work across accounts)

---

*DebateArena · Bauhaus-Universität Weimar · SS 2026*
