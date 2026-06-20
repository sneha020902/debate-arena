# DebateArena — Decision Log
**Bauhaus-Universität Weimar · Webis Lab · SS 2026**
**Team:** Rosenmeet · Sneha Agrawal

This file records every significant decision made during development — what was done, why, and what was considered but rejected. Updated after every change.

---

## 2026-06-17 — Initial Integration

### Goal
Integrate two existing codebases into a single frontend for the LLM-vs-LLM debate project:
- `debate-arena-enhanced/` — Rosen's arena UI + judge backend
- `presentations/Engineering-day/agrawal/` — Sneha's Emotion API with TTS

### Architecture Decision: Service-Oriented with Orchestrator
**Decision:** Add a new Orchestrator service (port 8002) that manages debate state and calls the existing Emotion API and Judge API.

**Why:** The existing code had debate logic scattered across Streamlit apps. A dedicated orchestrator separates concerns cleanly — the frontend doesn't need to know about scoring logic, and the Emotion API doesn't need to know about debate turns.

**Rejected:** Putting all logic in the frontend (JavaScript) — too complex, CORS issues with direct API calls from browser.

---

### TTS Decision: Kokoro over edge-tts
**Decision:** Replace edge-tts with Kokoro (local ONNX model) as the primary TTS provider.

**Why:**
- edge-tts requires network access to Microsoft servers — breaks offline/SLURM use
- Kokoro runs fully locally, no API key, ~82MB model, natural voices
- edge-tts only simulates emotion via rate/pitch hacks; Kokoro has distinct voice characters

**Rejected:** ElevenLabs — requires paid account. Free tier blocks Voice Library voices (confirmed via 402 error in Sneha's README). Integration code kept dormant but removed from this integrated version.

**Status:** Kokoro implemented in `emotion-api/synthesis.py` and `emotion-api/main.py`. However, the Emotion API running on SLURM is still Sneha's original code (edge-tts). TTS will return 404 until the SLURM code is updated.

---

### Deployment Decision: SLURM for heavy services, local for orchestrator only
**Decision:** Run Emotion API + Ollama + Judge API on SLURM. Run only the orchestrator locally.

**Why:**
- Emotion API needs torch, transformers, wav2vec2 (~4-5GB) — too heavy for a laptop
- C: drive on dev machine only had ~14GB free — Docker/local install would fill it
- Ollama needs GPU for fast inference — SLURM has GPU nodes
- Orchestrator is lightweight (5 packages) — fine to run locally

**Rejected:** Docker locally — C: drive space issue + user unfamiliar with Docker.
**Rejected:** Docker on SLURM — SLURM uses Singularity, not Docker. Noted for future proper deployment.

**Supervisor note:** Supervisor recommended virtual containers for dependency isolation. Singularity on SLURM is the right path for final deployment — noted as future work.

---

### Frontend Decision: Plain HTML/CSS/JS
**Decision:** Single `index.html` file, no build system, no React.

**Why:**
- All heavy logic stays in Python APIs
- No npm, no webpack, no build step — Sneha can open it directly in a browser
- Faster to build, easier to share

**Style:** Rose (#F43F5E) vs Teal (#2DD4BF) split-screen, Space Grotesk font, dark background (#080810) — ported from `debate_arena_style.py`.

---

### Communication: WebSocket for live streaming
**Decision:** Frontend connects to orchestrator via WebSocket (`ws://localhost:8002/ws/debate`).

**Why:** Debate turns take time (Ollama inference ~5-10s per turn). WebSocket streams each argument as it's generated rather than waiting for the full debate to finish. Better UX.

**Event types streamed:**
- `host_intro` — announcer text + audio
- `argument` — speaker, text, scores, audio
- `scores_update` — running composure averages
- `winner` — final result + announcer outro

---

## 2026-06-17 — Bug Fixes

### Fix: "Blue/Red corner" mismatch
**Problem:** Host script said "Blue corner / Red corner" but UI uses Rose and Teal.
**Fix:** Changed to "left corner / right corner" to match the visual split-screen layout.
**File:** `orchestrator/host_script.py`

---

### Fix: Composure always 50% / always tie
**Root cause:** Emotion API was not running when tested locally. The fallback in `debate_flow.py` returns hardcoded `0.5` when the API is unreachable.
**Fix:** Started Emotion API on SLURM (`~/debate-arena/arena/`) and set up SSH tunnel for port 8000.
**Note:** This is expected behavior when Emotion API is unreachable — not a bug in the orchestrator.

**SLURM Emotion API location:** `~/debate-arena/arena/main.py` (Sneha's original, not the Kokoro version)
**SLURM venv:** `~/debate-arena/venv/`
**Start command:**
```bash
source ~/debate-arena/venv/bin/activate
cd ~/debate-arena/arena
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Known Issues / Future Work

| Issue | Status | Notes |
|---|---|---|
| TTS returns 404 on SLURM | Open | SLURM runs original Emotion API without Kokoro. Need to update SLURM code. |
| Judge API not integrated | Open | Orchestrator falls back to composure-based winner. Judge API endpoints need wiring. |
| Singularity container | Future | Supervisor recommended for proper SLURM deployment |
| Ollama GPU time limit | Known | SLURM kills Ollama jobs at walltime limit. Use `srun --gpus=1 --time=04:00:00` |
| Login node reboots | Known | tmux sessions lost on reboot. Restart services manually. |

---

## File Locations Reference

| Service | Local path | SLURM path |
|---|---|---|
| Emotion API | `emotion-api/` | `~/debate-arena/arena/` |
| Judge API | `judge-api/` | TBD |
| Orchestrator | `orchestrator/` | not on SLURM yet |
| Frontend | `frontend/index.html` | not needed on SLURM |
| Emotion venv | not local | `~/debate-arena/venv/` |
| Orchestrator venv | `venv-orchestrator/` | not on SLURM yet |
