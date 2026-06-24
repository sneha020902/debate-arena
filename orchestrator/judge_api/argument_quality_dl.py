"""
argument_quality.py — Debate-level argument quality scorer.

Responsibility
--------------
Computes argument_quality for each speaker as a debate-level score.
This is one of the four debate-level components in the composite formula:

    Debate_Level_Score =
        0.30 * rebuttal_effectiveness
        0.27 * argument_quality          ← this file
        0.25 * engagement_score
        0.18 * information_density

What argument_quality measures
--------------------------------
The overall quality of ALL arguments a speaker produced across the debate —
both their standalone arguments AND how well they responded to the opponent.

    argument_quality = 0.60 * intrinsic_quality
                     + 0.40 * response_quality

intrinsic_quality
    Mean of the individual argument scores already computed by the
    scoring pipeline (final_score.py / unknown_arguments.py).
    Covers EVERY argument the speaker made — opening statements,
    new points, rebuttals — not just response pairs.
    Source: pre-scored per_argument list passed in by the caller.

response_quality
    How well the speaker responds to opponent arguments.
    Only scored for turns that are genuinely responding (similarity-detected).
    Prevents pairing unrelated consecutive turns as fake response pairs.
    Source: LLM pair scorer (this file).

Why 0.60 / 0.40 split
-----------------------
Intrinsic quality covers all arguments and is grounded in the IBM corpus
and NLI CrossEncoder — reliable signals. Response quality is LLM-based
and covers only a subset of turns. Giving intrinsic quality the majority
weight makes the score more stable and comprehensive.

Response pair detection
------------------------
NOT every turn is a response to the immediately prior opponent turn.
A speaker may address an argument from 3 turns ago.
Detection uses semantic similarity: if a turn has cosine similarity
>= RESPONSE_SIM_THRESHOLD with ANY prior opponent turn, it is considered
a response pair. The most similar prior opponent turn is chosen as the
"original being responded to."

First-turn exclusion
---------------------
Each speaker's first turn is their opening statement.
It is excluded from response_quality scoring on both sides:
    - First turns cannot BE a response (no prior opponent context)
    - First turns cannot be RESPONDED TO in the response_quality metric
      (they are standalone introductions, not argument claims to rebut)

Verdicts
---------
    substantive : response directly engages substance of original   score > 0.60
    partial     : engages some aspects but not core claim           score 0.35-0.60
    deflecting  : sidesteps, changes subject, surface-level only    score < 0.35

These replace the original binary substantive/deflecting with three levels
for better granularity. The quality_score (0-1) is still the primary metric.

Fallback
---------
If Ollama is unreachable: similarity score is used as quality proxy.
    similarity >= 0.70 → substantive,  quality = similarity
    similarity >= 0.40 → partial,      quality = similarity * 0.75
    else               → deflecting,   quality = similarity * 0.5

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

from judge_api.judge_config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ── Optional sentence-transformers ────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — similarity detection disabled. "
        "Install: pip install sentence-transformers"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

# Intrinsic vs response weight split
_W_INTRINSIC = 0.60
_W_RESPONSE  = 0.40

# Similarity threshold to consider a turn a "response" to an opponent turn
# Below this: the turn introduces a new point, not a response
RESPONSE_SIM_THRESHOLD = 0.35

# Verdict boundaries based on quality_score
_VERDICT_SUBSTANTIVE_MIN = 0.60   # score >= this → substantive
_VERDICT_PARTIAL_MIN     = 0.35   # score >= this → partial, else deflecting

# Embedding model
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_embed_model():
    global _embed_model
    if _embed_model is None and _ST_AVAILABLE:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _similarity(text_a: str, text_b: str) -> float:
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


# ── Response pair detector ────────────────────────────────────────────────────

def _find_best_prior_opponent_turn(
    current_turn:   dict,
    prior_turns:    list,
    opponent:       str,
) -> tuple[Optional[dict], float]:
    """
    Find the prior opponent turn that the current turn is most likely
    responding to, based on semantic similarity.

    Returns (best_opponent_turn, similarity) or (None, 0.0) if no turn
    clears RESPONSE_SIM_THRESHOLD.

    This replaces _find_prior_opponent_turn (which found the immediately
    prior opponent turn regardless of relevance).
    """
    opponent_prior = [
        t for t in prior_turns
        if t["speaker"] == opponent and t["turn"] < current_turn["turn"]
    ]
    if not opponent_prior:
        return None, 0.0

    best_turn = None
    best_sim  = 0.0

    for opp_turn in opponent_prior:
        sim = _similarity(current_turn["argument"], opp_turn["argument"])
        if sim > best_sim:
            best_sim  = sim
            best_turn = opp_turn

    if best_sim < RESPONSE_SIM_THRESHOLD:
        return None, best_sim   # not a response — introduces new point

    return best_turn, best_sim


# ── LLM response quality scorer ──────────────────────────────────────────────

def _score_response_quality_llm(
    original_claim:    str,
    original_premises: list,
    response_claim:    str,
    response_premises: list,
    similarity:        float
) -> dict:
    """
    Score how effectively response_argument addresses original_argument.

    Three-level verdict (substantive / partial / deflecting) replaces
    the original binary substantive / deflecting.

    Returns
    -------
    {
        "verdict":       str,    "substantive" | "partial" | "deflecting"
        "quality_score": float,  0.0-1.0
        "reasoning":     str,
        "source":        str,    "llm" | "similarity_fallback"
    }
    """
    def _fmt(claim: str, premises: list) -> str:
        if premises:
            prems = "\n".join(f"  - {p}" for p in premises)
            return f"Premises:\n{prems}\nConclusion: {claim}"
        return f"Argument: {claim}"

    prompt = f"""You are an expert debate judge assessing response quality.

Original argument:
{_fmt(original_claim, original_premises)}

Response:
{_fmt(response_claim, response_premises)}

Score HOW EFFECTIVELY the response addresses the original argument.

QUALITY_SCORE: Your honest rating 0.0–1.0 of how well the response engages.
  0.0–0.2 : Complete deflection — ignores substance, changes subject
  0.3–0.5 : Partial — touches the topic but misses the core claim
  0.6–0.7 : Substantive — engages clearly with reasoning or evidence
  0.8–1.0 : Thorough — dismantles or advances the original claim convincingly

VERDICT (derived from score):
  "substantive" : quality_score >= 0.60
  "partial"     : quality_score 0.35–0.59
  "deflecting"  : quality_score < 0.35

Derive your score from this specific pair. Do not copy any number from this prompt.

Reply with ONLY this JSON (no markdown):
{{"quality_score": <0.0-1.0>, "verdict": "<substantive|partial|deflecting>", "reasoning": "<one sentence>"}}"""

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

        quality_score = float(result.get("quality_score", similarity))
        quality_score = round(max(0.0, min(1.0, quality_score)), 3)

        # Derive verdict from score — do not trust LLM verdict blindly
        if quality_score >= _VERDICT_SUBSTANTIVE_MIN:
            verdict = "substantive"
        elif quality_score >= _VERDICT_PARTIAL_MIN:
            verdict = "partial"
        else:
            verdict = "deflecting"

        return {
            "verdict":       verdict,
            "quality_score": quality_score,
            "reasoning":     str(result.get("reasoning", "No reasoning provided.")),
            "source":        "llm",
        }

    except Exception as e:
        logger.warning(f"Ollama unavailable for response quality scoring: {e}")

        # ── Similarity-based fallback ─────────────────────────────────────────
        if similarity >= 0.70:
            quality = round(similarity, 3)
            verdict = "substantive"
        elif similarity >= 0.40:
            quality = round(similarity * 0.75, 3)
            verdict = "partial"
        else:
            quality = round(similarity * 0.5, 3)
            verdict = "deflecting"

        return {
            "verdict":       verdict,
            "quality_score": quality,
            "reasoning":     "Ollama unavailable — quality inferred from similarity.",
            "source":        "similarity_fallback",
        }


# ── Per-speaker response quality ──────────────────────────────────────────────

def _compute_response_quality(
    speaker_turns:   list,
    opponent_turns:  list,
    all_prior_turns: list,
    speaker:         str,
    opponent:        str
) -> dict:
    """
    Score the response quality of ONE speaker across the debate.

    Only turns that similarity-detect as responses to opponent arguments
    are scored. Turns that introduce new points are not included
    (those are captured by intrinsic_quality instead).

    Returns
    -------
    {
        "average_response_quality": float,
        "total_responses_scored":   int,
        "substantive":              int,
        "partial":                  int,
        "deflecting":               int,
        "per_pair":                 list,
    }
    """
    per_pair     = []
    scores       = []
    substantive  = 0
    partial      = 0
    deflecting   = 0

    for turn in speaker_turns:
        # Find the best prior opponent turn this is responding to
        best_opp_turn, sim = _find_best_prior_opponent_turn(
            current_turn=turn,
            prior_turns=all_prior_turns,
            opponent=opponent,
        )

        if best_opp_turn is None:
            # This turn introduces a new point — not a response
            # Do not score it here (captured by intrinsic_quality)
            continue

        # Score the response quality
        scored = _score_response_quality_llm(
            original_claim=best_opp_turn.get("claim",    best_opp_turn["argument"]),
            original_premises=best_opp_turn.get("premises", []),
            response_claim=turn.get("claim",    turn["argument"]),
            response_premises=turn.get("premises", []),
            similarity=sim,
        )

        scores.append(scored["quality_score"])

        if scored["verdict"] == "substantive":
            substantive += 1
        elif scored["verdict"] == "partial":
            partial += 1
        else:
            deflecting += 1

        snippet = (
            turn["argument"][:120] + "..."
            if len(turn["argument"]) > 120
            else turn["argument"]
        )
        orig_snippet = (
            best_opp_turn["argument"][:80] + "..."
            if len(best_opp_turn["argument"]) > 80
            else best_opp_turn["argument"]
        )

        per_pair.append({
            "turn":                turn["turn"],
            "responding_to_turn":  best_opp_turn["turn"],
            "argument_snippet":    snippet,
            "original_snippet":    orig_snippet,
            "similarity":          round(sim, 3),
            "verdict":             scored["verdict"],
            "quality_score":       scored["quality_score"],
            "reasoning":           scored["reasoning"],
            "source":              scored["source"],
        })

    total  = len(scores)
    avg_rq = round(sum(scores) / total, 3) if total > 0 else 0.0

    return {
        "average_response_quality": avg_rq,
        "total_responses_scored":   total,
        "substantive":              substantive,
        "partial":                  partial,
        "deflecting":               deflecting,
        "per_pair":                 per_pair,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def compute_argument_quality(
    turns:                list,
    scored_arguments:     dict,
    w_intrinsic:          float = _W_INTRINSIC,
    w_response:           float = _W_RESPONSE,
) -> dict:
    """
    Compute debate-level argument_quality for each speaker.

    argument_quality = w_intrinsic * intrinsic_quality
                     + w_response  * response_quality

    Parameters
    ----------
    turns : list of dicts, each with:
        {
            "turn":      int,
            "speaker":   str,
            "argument":  str,
            "claim":     str,      optional — from extract_claim_and_premises
            "premises":  list,     optional
        }

    scored_arguments : dict mapping speaker → list of per-argument score dicts
        Each dict must have a "quality" field (0.0-1.0).
        These come from final_score.py (KNOWN/BORDERLINE args) and
        unknown_arguments.py (UNKNOWN/ES_DOWN args).

        Example:
        {
            "Alice": [
                {"turn": 1, "quality": 0.82, "source": "full_pipeline"},
                {"turn": 3, "quality": 0.74, "source": "full_pipeline"},
                {"turn": 5, "quality": 0.61, "source": "partial_pipeline_llm_mode"},
            ],
            "Bob": [...]
        }

    ollama_host : str    Ollama base URL
    model       : str    Ollama model name
    w_intrinsic : float  Weight for intrinsic quality (default 0.60)
    w_response  : float  Weight for response quality  (default 0.40)

    Returns
    -------
    {
        "<speaker_a>": {
            "argument_quality":         float,   final debate-level score 0-1
            "intrinsic_quality":        float,   mean of individual arg scores
            "response_quality":         float,   mean response quality score
            "n_arguments_total":        int,     all arguments scored
            "n_responses_scored":       int,     turns identified as responses
            "substantive_responses":    int,
            "partial_responses":        int,
            "deflecting_responses":     int,
            "per_argument_scores":      list,    from scored_arguments input
            "per_response_pair":        list,    response pair details
        },
        "<speaker_b>": { ... },
        "summary": {
            "weights":                  dict,
            "response_sim_threshold":   float,
            "total_turns":              int,
            "first_turns_excluded":     list,
            "embed_model_available":    bool,
        }
    }
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers."}

    speaker_a, speaker_b = speakers[0], speakers[1]
    opponent_map = {speaker_a: speaker_b, speaker_b: speaker_a}

    # ── Identify first turns (excluded from response scoring) ─────────────────
    first_turns = {}
    for sp in speakers:
        sp_turns = [t for t in turns if t["speaker"] == sp]
        if sp_turns:
            first_turns[sp] = min(t["turn"] for t in sp_turns)

    first_turn_log = [
        {"speaker": sp, "turn": tn}
        for sp, tn in first_turns.items()
    ]

    def _without_first(sp: str) -> list:
        first = first_turns.get(sp, -1)
        return [t for t in turns if t["speaker"] == sp and t["turn"] != first]

    # ── Build result for each speaker ─────────────────────────────────────────
    output = {}

    for speaker in speakers:
        opponent = opponent_map[speaker]

        # ── Intrinsic quality: mean of pre-scored individual arguments ────────
        sp_scores = scored_arguments.get(speaker, [])
        intrinsic_quality = (
            round(sum(a["quality"] for a in sp_scores) / len(sp_scores), 3)
            if sp_scores else 0.0
        )

        # ── Response quality: LLM pair scoring ───────────────────────────────
        # Exclude first turns from response scoring
        speaker_turns_no_first = _without_first(speaker)
        opponent_turns_no_first = _without_first(opponent)

        rq_result = _compute_response_quality(
            speaker_turns=speaker_turns_no_first,
            opponent_turns=opponent_turns_no_first,
            all_prior_turns=turns,   # full list for prior-turn lookup
            speaker=speaker,
            opponent=opponent
        )

        response_quality = rq_result["average_response_quality"]

        # ── Final weighted score ──────────────────────────────────────────────
        # If speaker made no responses (all new points), response weight
        # falls back to intrinsic — avoids penalising for debate structure
        if rq_result["total_responses_scored"] == 0:
            argument_quality = intrinsic_quality
            effective_w_int  = 1.0
            effective_w_resp = 0.0
        else:
            argument_quality = round(
                w_intrinsic * intrinsic_quality + w_response * response_quality,
                3,
            )
            effective_w_int  = w_intrinsic
            effective_w_resp = w_response

        output[speaker] = {
            "argument_quality":      argument_quality,
            "intrinsic_quality":     intrinsic_quality,
            "response_quality":      response_quality,
            "weights_used": {
                "intrinsic": effective_w_int,
                "response":  effective_w_resp,
            },
            "n_arguments_total":     len(sp_scores),
            "n_responses_scored":    rq_result["total_responses_scored"],
            "substantive_responses": rq_result["substantive"],
            "partial_responses":     rq_result["partial"],
            "deflecting_responses":  rq_result["deflecting"],
            "per_argument_scores":   sp_scores,
            "per_response_pair":     rq_result["per_pair"],
        }

    return {
        **output,
        "summary": {
            "weights": {
                "intrinsic": w_intrinsic,
                "response":  w_response,
            },
            "response_sim_threshold": RESPONSE_SIM_THRESHOLD,
            "total_turns":            len(turns),
            "first_turns_excluded":   first_turn_log,
            "embed_model_available":  _ST_AVAILABLE,
        },
    }