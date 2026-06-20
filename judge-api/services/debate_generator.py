"""
debate_generator.py — automatic debate generation service.

Orchestrates ES reference retrieval → Ollama argument generation.
The generated transcript is in the canonical {turn, speaker, argument} format
so it passes directly into the existing scoring pipeline (winner_engine /
/judge endpoint) without any modification to those components.

Pattern adapted from the reference debate_arena.py implementation
(fetch_reference_args + generate_argument), with three adjustments:
  • uses our es_client module (consistent error handling, same ES contract)
  • reads OLLAMA_HOST / OLLAMA_MODEL from judge_config (single source of truth)
  • enforces the 500-token per-turn budget from the Task 6 spec
"""

import requests

from services import es_client
from services.judge_config import OLLAMA_HOST, OLLAMA_MODEL

# Fixed debate schedule: 6 turns total (3 per side), alternating
DEBATE_SCHEDULE = [
    ("Team A", "FOR"),
    ("Team B", "AGAINST"),
    ("Team A", "FOR"),
    ("Team B", "AGAINST"),
    ("Team A", "FOR"),
    ("Team B", "AGAINST"),
]

_OPPONENT = {"Team A": "Team B", "Team B": "Team A"}


# ── Reference retrieval ────────────────────────────────────────────────────────

def fetch_reference_args(topic: str, argument: str = "") -> list:
    """
    Fetch reference arguments from Elasticsearch to enrich the generation prompt.

    Uses the opponent's last argument (or the topic itself for the opening turn)
    as the query so the retrieved references are relevant to the current
    point of the debate, not just the overarching topic.

    Returns [] if the VPN is off or ES is unreachable — generation continues
    without reference context (the fallback path, per spec).
    """
    query = argument.strip() or topic
    results = es_client.semantic_search(
        query, top_n=3, quality_only=False, min_score=1.3)
    return results or []


# ── Argument generation ────────────────────────────────────────────────────────

def generate_argument(topic: str, speaker: str, side: str,
                      previous_turns: list,
                      ollama_host: str = None,
                      model: str = None) -> dict:
    """
    Generate one debate turn via Ollama, with ES references injected into
    the prompt when available.

    Parameters
    ----------
    topic          : str   The debate motion.
    speaker        : str   "Team A" or "Team B".
    side           : str   "FOR" or "AGAINST".
    previous_turns : list  Already-generated turns [{turn, speaker, argument}].
    ollama_host    : str   Overrides OLLAMA_HOST env if provided.
    model          : str   Overrides OLLAMA_MODEL env if provided.

    Returns
    -------
    {
        "text":            str | None,   # the argument; None if Ollama unreachable
        "references":      list,         # ES results used (empty when VPN off)
        "reference_count": int,
    }
    """
    ollama_host = ollama_host or OLLAMA_HOST
    model       = model or OLLAMA_MODEL
    opponent    = _OPPONENT.get(speaker, "")

    # ── Rebuttal context: find opponent's most recent argument ────────────────
    last_opponent_arg = None
    for turn in reversed(previous_turns):
        if turn.get("speaker") == opponent:
            last_opponent_arg = (turn.get("argument") or "").strip()
            break

    # ── ES reference retrieval ────────────────────────────────────────────────
    query      = last_opponent_arg or topic
    references = fetch_reference_args(topic, query)

    # ── Prompt assembly ───────────────────────────────────────────────────────
    is_opening = len(previous_turns) < 2

    rebuttal_block = ""
    if last_opponent_arg and not is_opening:
        rebuttal_block = f"""
Your opponent just argued:
\"{last_opponent_arg[:400]}\"

In your response, do two things:
1. Rebut the weakest point of their argument (1–2 sentences only).
2. Advance a NEW argument your opponent has not yet addressed.
"""

    reference_block = ""
    if references:
        reference_block = "\nRelevant reference points from the debate corpus (use as inspiration — do not copy verbatim):\n"
        for ref in references[:2]:
            claim = ref.get("claim", "").strip()
            if claim:
                reference_block += f"- {claim}\n"

    prompt = (
        f"You are {speaker}, a skilled competitive debater.\n"
        f"Motion: \"{topic}\"\n"
        f"Your side: {side}\n"
        f"{rebuttal_block}"
        f"{reference_block}\n"
        "Requirements:\n"
        "- Maximum 500 tokens — be sharp and concise\n"
        "- One clear main claim supported by 2–3 premises\n"
        "- Plain prose only: no markdown, no bullet points, no headers\n"
        "- No meta-commentary, no \"As an AI\" phrases\n"
        "- Do NOT repeat the motion or your team name as a heading\n\n"
        "Write your argument now:"
    )

    try:
        r = requests.post(
            f"{ollama_host}/api/chat",
            json={
                "model":    model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"num_predict": 500, "temperature": 0.75},
            },
            timeout=180,
        )
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        return {"text": text, "references": references, "reference_count": len(references)}
    except Exception:
        return {"text": None, "references": references, "reference_count": len(references)}
