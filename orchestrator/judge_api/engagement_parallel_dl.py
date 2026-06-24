"""
engagement_parallel.py — Debate-level engagement vs parallel monologue scorer.

Responsibility
--------------
Computes engagement_score for each speaker as a debate-level score.
This is one of the four debate-level components in the composite formula:

    Debate_Level_Score =
        0.30 * rebuttal_effectiveness
        0.27 * argument_quality
        0.25 * engagement_score          <- this file
        0.18 * information_density

What engagement_score measures
---------------------------------
Whether a speaker actively engages with the opponent's arguments or
delivers a parallel monologue — advancing their own points while
ignoring what the opponent actually said.

    engagement_score = 0.50 * engagement_ratio
                     + 0.50 * mean_engagement_depth

engagement_ratio
    Fraction of scoreable turns classified as "engaged" or "partial".
    A speaker who responds to 4 of 5 opponent arguments scores 0.80 here.
    Denominatoris only scoreable turns (first turns excluded).

mean_engagement_depth
    HOW deeply each turn engages, on a 0-1 scale.
    Derived from semantic similarity + LLM confidence.
    High similarity + high LLM confidence = deep engagement (close to 1.0).
    Low similarity = clear parallel monologue (close to 0.0).
    This prevents two speakers both with 0.80 engagement_ratio from getting
    identical scores when one engages shallowly and the other substantively.

Three-level classification (replaces original binary)
------------------------------------------------------
    "engaged"  : directly addresses opponent content     depth >= 0.60
    "partial"  : references topic but misses core        depth 0.30-0.59
    "parallel" : ignores opponent, pursues own point     depth < 0.30

In engagement_ratio: engaged=1.0, partial=0.5, parallel=0.0
This gives partial credit rather than penalising partial engagement equally
with complete parallel monologue.

Best-match opponent pairing (not just most recent)
---------------------------------------------------
A speaker may engage with an opponent argument from 3 turns ago, not the
most recent one. The scorer finds the MOST SIMILAR prior opponent argument
across ALL prior opponent turns — not just the immediately preceding one.
This correctly identifies engagement even when speakers don't respond to
the most recent point.

Detection pipeline per turn
----------------------------
Step 1: Collect all prior opponent turns (before current turn number)
Step 2: Compute cosine similarity between current turn and each prior opponent turn
Step 3: Find max similarity and the best-matching prior opponent turn
Step 4: If max_sim < PARALLEL_THRESHOLD (0.15): parallel immediately (no LLM)
Step 5: If max_sim >= PARALLEL_THRESHOLD: call LLM classify_engagement
Step 6: Compute depth_score combining similarity and LLM confidence
Step 7: Apply confidence gating for low-confidence LLM calls
Step 8: Aggregate all per-turn scores to final engagement_score

Depth score formula
--------------------
    depth_score = confidence_gate(
        label_base * llm_confidence,
        similarity
    )

    label_base:
        "engaged"  -> 1.0
        "partial"  -> 0.55
        "parallel" -> 0.10

    confidence_gate: blends depth toward similarity when confidence is low
        high confidence (>= 0.75) -> use depth as-is
        low confidence  (<  0.40) -> use similarity as proxy
        middle          -> linear interpolation

First-turn exclusion
---------------------
Each speaker's first (lowest-numbered) turn is their opening statement.
Excluded from BOTH roles:
    - Cannot BE a scorer turn (no prior opponent argument to engage with)
    - Cannot BE the "prior opponent" target (no context established yet)

Fallback
---------
If Ollama unreachable: similarity alone determines classification.
    sim >= 0.55 -> engaged,  depth = similarity
    sim >= 0.25 -> partial,  depth = similarity * 0.70
    else        -> parallel, depth = similarity * 0.30

Dependencies
------------
    pip install sentence-transformers
    Ollama running locally
"""

from __future__ import annotations

import json
import logging
import re
import requests
from typing import Optional
from .judge_config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ── Optional sentence-transformers ────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — similarity pre-filter disabled. "
        "Install: pip install sentence-transformers"
    )

# ── Constants ─────────────────────────────────────────────────────────────────

# Weights for engagement_score formula
_W_RATIO = 0.50
_W_DEPTH = 0.50

# Similarity thresholds
PARALLEL_THRESHOLD = 0.15   # below this: parallel without LLM call
ENGAGED_THRESHOLD  = 0.55   # above this in fallback: engaged

# Confidence gate boundaries
_CONF_HIGH = 0.75   # above: use depth_score as-is
_CONF_LOW  = 0.40   # below: use similarity as proxy

# Label base scores (depth_score is scaled by LLM confidence)
_LABEL_BASE = {
    "engaged":  1.00,
    "partial":  0.55,
    "parallel": 0.10,
}

# Engagement ratio contribution per classification
_RATIO_CONTRIBUTION = {
    "engaged":  1.0,
    "partial":  0.5,
    "parallel": 0.0,
}

# Depth classification boundaries
_DEPTH_ENGAGED  = 0.60
_DEPTH_PARTIAL  = 0.30

# Embedding model
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_embed_model():
    global _embed_model
    if _embed_model is None and _ST_AVAILABLE:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _cosine_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts. Returns 0.5 if model unavailable."""
    model = _get_embed_model()
    if model is None:
        return 0.5
    embs = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(embs[0], embs[1]))


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ── Best prior opponent turn finder ──────────────────────────────────────────

def _find_best_opponent_turn(
    current_turn:   dict,
    all_turns:      list,
    opponent:       str,
) -> tuple[Optional[dict], float]:
    """
    Find the prior opponent turn most semantically related to current_turn.

    Checks ALL prior opponent turns (not just most recent) to correctly
    identify engagement even when speakers respond to older arguments.

    Returns (best_opponent_turn, max_similarity) or (None, 0.0).
    """
    prior_opponent = [
        t for t in all_turns
        if t["speaker"] == opponent and t["turn"] < current_turn["turn"]
    ]
    if not prior_opponent:
        return None, 0.0

    best_turn = None
    best_sim  = 0.0

    for opp_turn in prior_opponent:
        sim = _cosine_sim(current_turn["argument"], opp_turn["argument"])
        if sim > best_sim:
            best_sim  = sim
            best_turn = opp_turn

    return best_turn, round(best_sim, 4)


# ── Depth score calculator ────────────────────────────────────────────────────

def _compute_depth_score(
    label:      str,
    confidence: float,
    similarity: float,
) -> float:
    """
    Compute per-turn engagement depth score (0-1).

    Combines label base score with LLM confidence, then applies
    confidence gating to handle unreliable LLM outputs.

    High confidence -> trust the label-based score
    Low confidence  -> fall back to similarity as a proxy
    """
    base  = _LABEL_BASE.get(label, 0.10)
    depth = base * confidence   # raw depth from LLM

    # Confidence gate: blend depth toward similarity when confidence is low
    if confidence >= _CONF_HIGH:
        gated = depth
    elif confidence <= _CONF_LOW:
        gated = similarity   # low confidence: use similarity as proxy
    else:
        # Linear blend between similarity and depth
        blend = (confidence - _CONF_LOW) / (_CONF_HIGH - _CONF_LOW)
        gated = (1.0 - blend) * similarity + blend * depth

    return round(max(0.0, min(1.0, gated)), 3)


def _depth_to_label(depth: float) -> str:
    """Derive final classification label from depth score."""
    if depth >= _DEPTH_ENGAGED:
        return "engaged"
    if depth >= _DEPTH_PARTIAL:
        return "partial"
    return "parallel"


# ── LLM engagement classifier ─────────────────────────────────────────────────

def _classify_engagement_llm(
    arg1_claim:    str,
    arg1_premises: list,
    arg2_claim:    str,
    arg2_premises: list,
    similarity:    float
) -> dict:
    """
    Classify whether argument_2 engages with argument_1 or runs in parallel.

    Three-level classification: engaged / partial / parallel.
    Confidence gating applied in _compute_depth_score.
    Full try/except with similarity-based fallback.

    Returns
    -------
    {
        "label":      str,    "engaged" | "partial" | "parallel"
        "confidence": float,
        "reasoning":  str,
        "source":     str,    "llm" | "similarity_fallback"
    }
    """
    def _fmt(claim: str, premises: list) -> str:
        if premises:
            prems = "\n".join(f"  - {p}" for p in premises)
            return f"Premises:\n{prems}\nConclusion: {claim}"
        return f"Argument: {claim}"

    prompt = f"""You are an argumentation theory expert.

Argument 1 (by the opponent):
{_fmt(arg1_claim, arg1_premises)}

Argument 2 (current speaker's response):
{_fmt(arg2_claim, arg2_premises)}

Classify how Argument 2 relates to Argument 1. Choose EXACTLY ONE:
- "engaged"  : Argument 2 directly addresses, responds to, or builds on the
               actual content of Argument 1.
- "partial"  : Argument 2 references the same topic or theme as Argument 1
               but does not meaningfully engage with its specific claims
               or reasoning.
- "parallel" : Argument 2 largely ignores Argument 1 and pursues its own
               separate point without addressing what Argument 1 said.

CONFIDENCE: Your honest certainty 0.0 (pure guess) to 1.0 (certain).
Derive it from this specific pair — do not copy any number from this prompt.

Reply with ONLY this JSON (no markdown):
{{"label": "<engaged|partial|parallel>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 120, "temperature": 0.0},
            },
            timeout=45,
        )
        r.raise_for_status()
        raw    = r.json().get("response", "")
        result = _extract_json(raw)

        label = str(result.get("label", "parallel")).lower().strip()
        if label not in ("engaged", "partial", "parallel"):
            label = "parallel"

        confidence = float(result.get("confidence", 0.5))
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        return {
            "label":      label,
            "confidence": confidence,
            "reasoning":  str(result.get("reasoning", "No reasoning provided.")),
            "source":     "llm",
        }

    except Exception as e:
        logger.warning(f"Ollama unavailable for engagement classification: {e}")

        # ── Similarity-based fallback ─────────────────────────────────────────
        if similarity >= ENGAGED_THRESHOLD:
            label      = "engaged"
            confidence = round(similarity, 3)
        elif similarity >= PARALLEL_THRESHOLD:
            label      = "partial"
            confidence = round(similarity * 0.80, 3)
        else:
            label      = "parallel"
            confidence = round(1.0 - similarity, 3)

        return {
            "label":      label,
            "confidence": confidence,
            "reasoning":  "Ollama unavailable — engagement inferred from similarity.",
            "source":     "similarity_fallback",
        }


# ── Per-turn scorer ───────────────────────────────────────────────────────────

def _score_turn_engagement(
    current_turn: dict,
    opponent:     str,
    all_turns:    list
) -> dict:
    """
    Score one turn for engagement depth against the best prior opponent argument.

    Returns per-turn result dict including depth_score and final label.
    """
    # Step 1: Find best matching prior opponent turn
    best_opp, sim = _find_best_opponent_turn(current_turn, all_turns, opponent)

    snippet = (
        current_turn["argument"][:120] + "..."
        if len(current_turn["argument"]) > 120
        else current_turn["argument"]
    )

    # No prior opponent argument exists yet (defensive guard)
    if best_opp is None:
        return {
            "turn":             current_turn["turn"],
            "argument_snippet": snippet,
            "best_opponent_turn": None,
            "similarity":       0.0,
            "label":            "parallel",
            "depth_score":      0.0,
            "confidence":       None,
            "reasoning":        "No prior opponent argument found.",
            "source":           "no_opponent",
            "ratio_contribution": 0.0,
        }

    opp_snippet = (
        best_opp["argument"][:80] + "..."
        if len(best_opp["argument"]) > 80
        else best_opp["argument"]
    )

    # Step 2: Similarity pre-filter
    if sim < PARALLEL_THRESHOLD:
        # Clearly parallel — skip LLM
        return {
            "turn":               current_turn["turn"],
            "argument_snippet":   snippet,
            "best_opponent_turn": best_opp["turn"],
            "opponent_snippet":   opp_snippet,
            "similarity":         sim,
            "label":              "parallel",
            "depth_score":        round(sim * 0.30, 3),
            "confidence":         None,
            "reasoning":          f"Similarity {sim:.3f} below threshold — parallel.",
            "source":             "similarity_prefilter",
            "ratio_contribution": _RATIO_CONTRIBUTION["parallel"],
        }

    # Step 3: LLM classification
    llm_result = _classify_engagement_llm(
        arg1_claim=best_opp.get("claim",    best_opp["argument"]),
        arg1_premises=best_opp.get("premises", []),
        arg2_claim=current_turn.get("claim",    current_turn["argument"]),
        arg2_premises=current_turn.get("premises", []),
        similarity=sim,
    )

    # Step 4: Compute depth score with confidence gating
    depth_score = _compute_depth_score(
        label=llm_result["label"],
        confidence=llm_result["confidence"],
        similarity=sim,
    )

    # Step 5: Derive final label from depth score (not raw LLM label)
    # This ensures label is consistent with depth_score
    final_label = _depth_to_label(depth_score)

    return {
        "turn":               current_turn["turn"],
        "argument_snippet":   snippet,
        "best_opponent_turn": best_opp["turn"],
        "opponent_snippet":   opp_snippet,
        "similarity":         sim,
        "label":              final_label,
        "llm_label":          llm_result["label"],    # raw LLM output for transparency
        "depth_score":        depth_score,
        "confidence":         llm_result["confidence"],
        "reasoning":          llm_result["reasoning"],
        "source":             llm_result["source"],
        "ratio_contribution": _RATIO_CONTRIBUTION[final_label],
    }


# ── Per-speaker aggregation ───────────────────────────────────────────────────

def _aggregate_speaker(per_turn: list) -> dict:
    """
    Aggregate per-turn scores into final engagement_score for one speaker.

    engagement_score = 0.50 * engagement_ratio + 0.50 * mean_depth
    engagement_ratio = sum(ratio_contributions) / n_turns
    mean_depth       = mean(depth_scores)
    """
    n = len(per_turn)
    if n == 0:
        return {
            "engagement_score":   0.0,
            "engagement_ratio":   0.0,
            "mean_depth":         0.0,
            "n_turns_scored":     0,
            "engaged":            0,
            "partial":            0,
            "parallel":           0,
        }

    ratio_sum  = sum(t["ratio_contribution"] for t in per_turn)
    depth_sum  = sum(t["depth_score"]        for t in per_turn)

    engagement_ratio = round(ratio_sum / n,    3)
    mean_depth       = round(depth_sum / n,    3)
    engagement_score = round(
        _W_RATIO * engagement_ratio + _W_DEPTH * mean_depth, 3
    )

    return {
        "engagement_score":   engagement_score,
        "engagement_ratio":   engagement_ratio,
        "mean_depth":         mean_depth,
        "n_turns_scored":     n,
        "engaged":   sum(1 for t in per_turn if t["label"] == "engaged"),
        "partial":   sum(1 for t in per_turn if t["label"] == "partial"),
        "parallel":  sum(1 for t in per_turn if t["label"] == "parallel"),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def compute_engagement_parallel(
    turns:       list
) -> dict:
    """
    Compute engagement_score for both speakers across a full debate.

    Parameters
    ----------
    turns : list of dicts, each with:
        {
            "turn":      int,     debate turn number (1-indexed)
            "speaker":   str,     speaker identifier
            "argument":  str,     full argument text
            "claim":     str,     optional — from extract_claim_and_premises
            "premises":  list,    optional
        }

        First turn of each speaker MUST be their opening statement —
        it is automatically excluded from engagement scoring.

    ollama_host : str   Ollama base URL
    model       : str   Ollama model name

    Returns
    -------
    {
        "<speaker_a>": {
            "engagement_score":  float,   0.0-1.0  (for composite formula)
            "engagement_ratio":  float,   fraction of turns that engaged
            "mean_depth":        float,   average engagement depth
            "n_turns_scored":    int,
            "engaged":           int,
            "partial":           int,
            "parallel":          int,
            "per_turn":          list,    per-turn details
        },
        "<speaker_b>": { ... },
        "summary": {
            "total_turns":           int,
            "first_turns_excluded":  list,
            "parallel_threshold":    float,
            "depth_boundaries":      dict,
            "weights":               dict,
            "embed_model_available": bool,
        }
    }
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers."}

    speaker_a, speaker_b = speakers[0], speakers[1]
    opponent_map = {speaker_a: speaker_b, speaker_b: speaker_a}

    # ── Identify first turns ──────────────────────────────────────────────────
    first_turns = {}
    for sp in speakers:
        sp_turns = [t for t in turns if t["speaker"] == sp]
        if sp_turns:
            first_turns[sp] = min(t["turn"] for t in sp_turns)

    first_turn_log = [
        {"speaker": sp, "turn": tn}
        for sp, tn in first_turns.items()
    ]

    def _scoreable_turns(sp: str) -> list:
        """Turns for speaker excluding their first turn."""
        first = first_turns.get(sp, -1)
        return [
            t for t in turns
            if t["speaker"] == sp and t["turn"] != first
        ]

    # ── Score each speaker ────────────────────────────────────────────────────
    output = {}

    for speaker in speakers:
        opponent     = opponent_map[speaker]
        score_turns  = _scoreable_turns(speaker)
        per_turn     = []

        for turn in score_turns:
            result = _score_turn_engagement(
                current_turn=turn,
                opponent=opponent,
                all_turns=turns,
            )
            per_turn.append(result)

        aggregated = _aggregate_speaker(per_turn)

        output[speaker] = {
            **aggregated,
            "per_turn": per_turn,
        }

    return {
        **output,
        "summary": {
            "total_turns":           len(turns),
            "first_turns_excluded":  first_turn_log,
            "parallel_threshold":    PARALLEL_THRESHOLD,
            "depth_boundaries": {
                "engaged":  f">= {_DEPTH_ENGAGED}",
                "partial":  f"{_DEPTH_PARTIAL} - {_DEPTH_ENGAGED}",
                "parallel": f"< {_DEPTH_PARTIAL}",
            },
            "weights": {
                "engagement_ratio": _W_RATIO,
                "mean_depth":       _W_DEPTH,
            },
            "embed_model_available": _ST_AVAILABLE,
        },
    }