"""
routes_winner.py — Part 1 + Part 3 endpoints, as a self-contained APIRouter.

Kept separate from Pavan's main.py so it can be added with a single
`app.include_router(...)` line (or via run_full.py, which touches nothing).

Endpoints
  POST /score-unknown   Part 1 — per-argument quality+coverage for one team,
                        showing whether each was matched in ES, extrapolated,
                        or LLM-judged.
  POST /judge           Part 3 — the full pipeline: Part 1 + Part 2 + delivery
                        -> composite score, winner, margin, and explanation.
  GET  /weights/default Default composite weights (handy for demo sliders).

Input tolerance
  The canonical turn schema is {"turn": int, "speaker": str, "argument": str}
  (same as sample_transcript.json and Pavan's Part-2 functions). To avoid the
  classic boundary KeyError, the Turn model below ALSO accepts "text" or
  "content" in place of "argument", and auto-fills the "turn" index if it is
  omitted — so a slightly different payload no longer crashes the pipeline.
"""

from typing import Optional, List

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from services.judge_config import DEFAULT_WEIGHTS, DEFAULT_UNKNOWN_STRATEGY
from services import unknown_arguments
from services import winner_engine

router = APIRouter()


class Turn(BaseModel):
    """One debate turn. `argument` may arrive as `text` or `content`; `turn` optional."""
    turn: Optional[int] = None
    speaker: str
    argument: str
    # Pre-computed by the arena (lexical novelty, 0–1). When present for all
    # turns, winner_engine uses these values directly and skips the LLM
    # compute_information_density pass — faster and avoids double-counting.
    info_density: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data):
        if isinstance(data, dict):
            if "argument" not in data:
                for alt in ("text", "content", "message"):
                    if alt in data:
                        data = {**data, "argument": data[alt]}
                        break
            if "speaker" not in data and "speaker_id" in data:
                data = {**data, "speaker": data["speaker_id"]}
            if "turn" not in data and "round" in data:
                data = {**data, "turn": data["round"]}
        return data

    # Example shown in Swagger (replaces the unhelpful ["string"]).
    model_config = {
        "json_schema_extra": {
            "example": {"turn": 1, "speaker": "Team A",
                        "argument": "ID verification reduces anonymous harassment and bots.",
                        "info_density": 0.82}
        }
    }


class JudgeRequest(BaseModel):
    topic: str
    turns: List[Turn]
    weights: Optional[dict] = None              # override DEFAULT_WEIGHTS
    strategy: Optional[str] = None              # "llm" | "extrapolation" | "blend"
    blend_llm_weight: Optional[float] = None
    use_delivery: bool = True
    # Pre-computed per-speaker composure averages from the arena. When present,
    # winner_engine uses these directly and skips re-calling Sneha's API.
    delivery_signals: Optional[dict] = None     # {"LLM-A": 0.72, "LLM-B": 0.65}


class UnknownArg(BaseModel):
    turn: Optional[int] = None
    argument: str

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data):
        if isinstance(data, dict) and "argument" not in data:
            for alt in ("text", "content", "message"):
                if alt in data:
                    data = {**data, "argument": data[alt]}
                    break
        return data


class ScoreUnknownRequest(BaseModel):
    arguments: List[UnknownArg]                 # for ONE team
    strategy: Optional[str] = None
    blend_llm_weight: Optional[float] = None


def _canonical(turns: List[Turn]) -> list:
    """List[Turn] -> list of canonical dicts with turn indices filled in order."""
    out = []
    for i, t in enumerate(turns, start=1):
        entry = {"turn": t.turn if t.turn is not None else i,
                 "speaker": t.speaker, "argument": t.argument}
        if t.info_density is not None:
            entry["info_density"] = t.info_density
        out.append(entry)
    return out


@router.get("/weights/default")
def default_weights():
    return {"weights": DEFAULT_WEIGHTS, "default_strategy": DEFAULT_UNKNOWN_STRATEGY}


@router.post("/score-unknown")
def score_unknown(request: ScoreUnknownRequest):
    """Part 1: score a single team's arguments (KNOWN via ES vs UNKNOWN via strategy)."""
    args = [{"turn": a.turn if a.turn is not None else i, "argument": a.argument}
            for i, a in enumerate(request.arguments, start=1)]
    return unknown_arguments.score_team_arguments(
        args, request.strategy, request.blend_llm_weight)


@router.post("/judge")
def judge(request: JudgeRequest):
    """Part 3: full composite winner determination across both teams."""
    try:
        return winner_engine.judge_debate(
            topic=request.topic,
            turns=_canonical(request.turns),
            weights=request.weights,
            strategy=request.strategy,
            blend_llm_weight=request.blend_llm_weight,
            use_delivery=request.use_delivery,
            delivery_signals=request.delivery_signals,
        )
    except (httpx.TransportError, ConnectionError) as e:
        # Part 2 needs the LLM (qwen2.5) via Ollama. If it's unreachable, give a
        # clear, actionable message instead of a raw 500 traceback on demo day.
        raise HTTPException(
            status_code=503,
            detail=("Ollama is unreachable for Part-2 scoring "
                    f"({type(e).__name__}). Check that OLLAMA_HOST points at a "
                    "reachable Ollama serving qwen2.5:7b (VPN / SSH tunnel / local), "
                    "then retry."),
        )


# ── Auto-generate: one turn at a time ─────────────────────────────────────────

class GenerateTurnRequest(BaseModel):
    topic:          str
    speaker:        str          # "Team A" or "Team B"
    side:           str          # "FOR" or "AGAINST"
    previous_turns: List[Turn] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic":   "Social media should be regulated by governments",
                "speaker": "Team A",
                "side":    "FOR",
                "previous_turns": [],
            }
        }
    }


@router.post("/generate-turn")
def generate_turn(request: GenerateTurnRequest):
    """
    Generate one debate argument using Ollama, enriched with Elasticsearch
    reference arguments when the VPN is on.

    Returns:
      {text: str|None, references: list, reference_count: int}

    If Ollama is unreachable, text is None and a 503 is raised so the UI
    can show a clear error instead of a silent empty argument.
    """
    try:
        from services.debate_generator import generate_argument
        prev = [
            {"turn": t.turn or (i + 1), "speaker": t.speaker, "argument": t.argument}
            for i, t in enumerate(request.previous_turns)
        ]
        result = generate_argument(
            request.topic, request.speaker, request.side, prev)
        if result["text"] is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama is unreachable — cannot generate the argument. "
                    "Check OLLAMA_HOST / VPN and that the model is loaded."
                ),
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
