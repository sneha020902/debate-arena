"""
run_full.py — full Debate Judge service: Pavan's five Part-2 endpoints PLUS
the Part 1 (/score-unknown) and Part 3 (/judge) endpoints.

Imports Pavan's already-configured app and attaches the winner router:

    python -m uvicorn run_full:app --reload --port 8001

(main.py still runs the Part-2-only service exactly as before.)

Ollama routing: set OLLAMA_HOST (and optionally OLLAMA_MODEL) in your shell
BEFORE launching. LogicDetector now reads those env vars, so the same Ollama
drives both Pavan's Part 2 and our Part 1/3 — no code edit needed per run.
"""

from main import app                # Pavan's app (all 5 Part-2 routes)
from routes_winner import router as winner_router

app.include_router(winner_router)
