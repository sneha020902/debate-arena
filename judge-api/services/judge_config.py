"""
judge_config.py — central configuration for Part 1 (unknown-argument scoring)
and Part 3 (winner determination).

Everything that used to be hard-coded in scattered places (Ollama node name,
ES base URL, score thresholds, composite weights) lives here so the demo and
the API can change behaviour without editing logic. All values are
environment-overridable, which also fixes the brittle hard-coded SLURM node
name in services/logic_modules.py (gammaweb09 changes every job) — set
OLLAMA_HOST once and every module picks it up.
"""

import os
from pathlib import Path

# Load .env from debate_judge/.env (one level up from this services/ file).
# override=False → values already set in the shell take priority over .env,
# so you can still do  $env:OLLAMA_HOST="..."  on the command line to override.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed — silently fall back to system env vars

# ── External services ─────────────────────────────────────────────────────────
# Rosen's Elasticsearch (Webis VPN required). Contract: TEAM_LOGIC_HANDOFF.
ES_API = os.getenv("ES_API", "http://141.54.159.66:8000")

# Sneha's Emotion Track delivery API (composure / delivery vector).
EMOTION_API = os.getenv("EMOTION_API", "http://localhost:8000")

# Ollama for the LLM-as-judge fallback. OLLAMA_HOST overrides the default node.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ── ES score handling ───────────────────────────────────────────────────────
# Rosen's scores are cosine + 1.0, i.e. in [1.0, 2.0]. Normalise to [0, 1] with
# (score - 1.0). Per the score guide: >= 1.5 is a "good match"; below 1.5 is
# treated as no match — which is exactly the trigger for the Part 1 fallback.
ES_MIN_MATCH = float(os.getenv("ES_MIN_MATCH", "1.5"))   # raw scale (1.0–2.0)
ES_STRONG_MATCH = float(os.getenv("ES_STRONG_MATCH", "1.65"))  # "strong match"


def es_norm(raw_score: float) -> float:
    """Map an ES raw score (1.0–2.0) to a 0.0–1.0 quality signal."""
    return round(min(max(raw_score - 1.0, 0.0), 1.0), 3)


# ── Part 1: unknown-argument strategy ─────────────────────────────────────────
# How to score an argument that has NO close match in the reference set:
#   "llm"            → LLM-as-judge only
#   "extrapolation"  → team-average prior only
#   "blend"          → weighted mix (BLEND_LLM_WEIGHT * llm + rest * prior)
DEFAULT_UNKNOWN_STRATEGY = os.getenv("UNKNOWN_STRATEGY", "blend")
BLEND_LLM_WEIGHT = float(os.getenv("BLEND_LLM_WEIGHT", "0.6"))

# ── Part 3: composite weights ─────────────────────────────────────────────────
# Seven argument/debate-level components, each already on a 0–1 scale. The
# defaults reflect the brief's emphasis that engagement-vs-parallel is "the
# central question": engagement + rebuttal_coverage together carry the most
# weight, with quality/coverage and density filling out the rest. Weights are
# exposed via the API so different scoring philosophies can be explored
# (e.g. a "breadth & engagement" profile vs a "fewer but higher-quality" one).
# They are renormalised to sum to 1.0 at scoring time, so callers can pass any
# positive numbers.
DEFAULT_WEIGHTS = {
    "quality":            0.18,   # argument-level (Part 1): ES claim+premise / fallback
    "coverage":           0.12,   # argument-level (Part 1): corpus grounding
    "engagement":         0.20,   # debate-level  (Part 2): engaged vs parallel  ("the central question")
    "rebuttal_coverage":  0.20,   # debate-level  (Part 2): % of opponent answered
    "information_density":0.15,   # debate-level  (Part 2): new content vs rephrasing
    "response_quality":   0.10,   # debate-level  (Part 2): substantive vs deflecting
    "new_point_balance":  0.05,   # debate-level  (Part 2): small — refines engagement/density, not an independent axis
}

# Delivery (Emotion × Logic) is kept OUT of the weighted sum by default and
# used as a tie-breaker only — per the brief, "incorporate ... as a tiebreaker
# or as a separate scoring dimension." Set DELIVERY_AS_DIMENSION=1 to fold a
# composure component into the weighted sum instead.
DELIVERY_AS_DIMENSION = os.getenv("DELIVERY_AS_DIMENSION", "0") == "1"
DELIVERY_WEIGHT = float(os.getenv("DELIVERY_WEIGHT", "0.10"))
TIE_EPSILON = float(os.getenv("TIE_EPSILON", "0.02"))   # margin below which delivery breaks the tie

# Soft 500-token-per-turn budget (briefing). We don't enforce it, but we report
# an estimate per turn so over-budget arguments are visible as a strategic note.
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "500"))
