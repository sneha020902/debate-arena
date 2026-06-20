# Kokoro TTS Microservice — README
**DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026**

This folder is a **standalone, self-contained service**. Everything you need to understand, run, or debug it lives in this one document.

---

## What This Is

A small FastAPI microservice that wraps **Kokoro** (an open-weight, ~82M-parameter text-to-speech model) and exposes it over HTTP. Your main `emotion-api` calls this service to turn debate arguments into spoken audio.

---

## Why Kokoro (and Why a Separate Service)

### Why Kokoro, specifically
Three TTS options were evaluated:

| Option | Why not chosen |
|---|---|
| **ElevenLabs** | Best voice quality, but the free tier blocks API access to the Voice Library voices we needed — would require a paid plan |
| **edge-tts** | Free and was already working, but it's an unofficial wrapper around Microsoft Edge's browser voice service — not a real API, could break or get blocked anytime |
| **Kokoro** ✅ | Free, open-weight, self-hosted — no API key, no rate limits, no dependency on any company's service staying available. Right fit for a reproducible research project. |

### Why a separate microservice (not built into emotion-api directly)
Kokoro's dependencies (`misaki`, `espeak-ng`, specific `torch`/`transformers` versions) conflicted with the packages already required by `emotion-api`'s own models (text emotion, speech emotion, whisper transcription). Running Kokoro as its **own isolated service** — called over HTTP instead of imported directly — avoids that conflict entirely. `emotion-api/synthesis.py` calls this service; it never imports `kokoro` itself.

---

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "service": "kokoro-tts"}` |
| `/synthesize` | POST | Body: `{"text": str, "voice": str, "speed": float}` → returns raw WAV bytes (24kHz, mono) |

---

## How To Run (SLURM)

### Why a container at all
The cluster's login node (and compute nodes) run **Python 3.14**, which is too new for some of Kokoro's dependencies (`blis`/`spacy`-related build failures). Running inside a `python:3.11-slim` container sidesteps this entirely.

### Step 1 — Upload this folder
```bash
scp -r kokoro_service qeso3721@ssh.webis.de:~/debate-emotion/
```

### Step 2 — SSH in, new tmux session
```bash
ssh qeso3721@ssh.webis.de
tmux new -s kokoro
```

### Step 3 — Run it
```bash
cd ~/debate-emotion/kokoro_service
bash run_kokoro.sh
```
Wait for: `Uvicorn running on http://0.0.0.0:8003`
**Note the node hostname printed at the top** — you'll need it for tunneling and for telling emotion-api where to find this service.

### Step 4 — Detach (keeps it running)
`Ctrl+B` then `D`

### Step 5 — Tunnel from your Mac (separate terminal, leave open)
```bash
ssh -N -L 8003:<NODE>.medien.uni-weimar.de:8003 qeso3721@ssh.webis.de
```

### Step 6 — Test
```bash
curl http://localhost:8003/health
```

---

## The Debugging Story (so the next person doesn't repeat it)

These were real errors hit and fixed, in order — kept here so future debugging is faster:

| Error | Cause | Fix |
|---|---|---|
| `Failed building wheel for blis` (local Mac install) | Python 3.14 too new, no precompiled wheel | Moved to SLURM container with Python 3.11 instead of fighting it locally |
| `dpkg: error: requested operation requires superuser privilege` | Pyxis container doesn't grant root by default | Added `--container-remap-root` to the `srun` command |
| Missing pronunciation / phonemizer issues (anticipated) | Kokoro's `misaki` library needs `espeak-ng` as a system package, even for English | Added `espeak-ng` to the `apt-get install` line in `run_kokoro.sh` |

`run_kokoro.sh` in this folder already has all three fixes applied — you shouldn't need to rediscover them.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **Node hostname changes on every restart** | SLURM assigns whatever node is free — if you restart this job, `emotion-api`'s `KOKORO_SERVICE_URL` env var must be updated to match the new hostname, or emotion-api won't be able to reach it |
| **Shares the GPU node with Ollama + emotion-api** | Under load (e.g. Ollama actively generating), Kokoro can respond slowly, occasionally causing `emotion-api`'s synthesis call to time out (see main project README's troubleshooting table) |
| **CPU-only** | Kokoro-82M is small enough to run fine on CPU — no `--gpus` requested, intentional, not a bug |
| **Fresh pip install every restart** | The container isn't a saved image — every restart re-installs `torch`/`kokoro`/etc. from scratch, taking a few minutes. Normal, not stuck. |
| **`bf_emma` voice + American phonemization** | The pipeline is loaded once with `lang_code="a"` (American English), but one of the mapped voices (`bf_emma`, used for LLM-B) is a British voice — minor pronunciation mismatch, not fixed, low priority |

---

## How It Connects to the Rest of the Project

```
orchestrator (debate_flow.py)
      ↓ calls /synthesize on emotion-api
emotion-api (synthesis.py)
      ↓ calls THIS service over HTTP
kokoro_service (this folder)
      ↓ runs the actual Kokoro model
```

`emotion-api/synthesis.py` reads the URL for this service from the `KOKORO_SERVICE_URL` environment variable (defaults to `http://localhost:8003` if unset). When `emotion-api` runs on SLURM, this **must** be explicitly set to this service's real node address — see main project `README.md` for the exact export command.

---

*Kokoro TTS Microservice · DEBATE Project · Bauhaus-Universität Weimar · SS 2026*
