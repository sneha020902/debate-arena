"""
rebuttal_effectiveness.py — Debate-level rebuttal effectiveness scorer.

Responsibility
--------------
Computes rebuttal_effectiveness for each speaker as a debate-level score.
This is one of the four debate-level components in the composite formula:

    Debate_Level_Score =
        0.30 * rebuttal_effectiveness   ← this file
        0.27 * argument_quality_aggregate
        0.25 * engagement_score
        0.18 * information_density

What rebuttal_effectiveness measures
--------------------------------------
Not just WHETHER each opponent argument was addressed (coverage),
but HOW WELL it was addressed (quality of each rebuttal).

    rebuttal_effectiveness = coverage_score * mean_rebuttal_quality

    coverage_score      = weighted fraction of opponent args addressed
                          (rebuttal counts more than undercut)
    mean_rebuttal_quality = average quality of the rebuttals that were made

Rebuttal types and weights
---------------------------
    rebuttal : attacks the CONCLUSION of argument_1        weight = 1.0
    undercut : attacks the EVIDENCE/REASONING of arg_1    weight = 0.7
    unrelated: argument_2 ignores argument_1 entirely      weight = 0.0

Coverage denominator = only arguments the opponent had a CHANCE to rebut
(arguments that appeared before the opponent's last turn).
Arguments after the opponent's last turn are marked "unreachable" and
excluded from the denominator — not penalised.

First-turn exclusion
---------------------
The first turn of each speaker is their opening statement (introduction).
It is excluded from BOTH roles:
    - First turns are NOT in the defender pool (cannot be rebutted —
      no prior context for the opponent to respond to)
    - First turns are NOT in the attacker pool (cannot be a rebuttal —
      they were made before the opponent even spoke)

Pipeline: similarity pre-filter → LLM classifier → quality scorer
------------------------------------------------------------------
Step 1: Semantic similarity pre-filter (SentenceTransformer)
    Compute cosine similarity between defender arg and attacker arg.
    If similarity < SIM_PREFILTER_THRESHOLD (0.20): skip LLM call entirely.
    This eliminates 60-80% of LLM calls for clearly unrelated pairs.

Step 2: LLM relation classifier (Ollama)
    For pairs that pass the pre-filter:
    classify as rebuttal / undercut / unrelated with confidence.
    Uses claim + premises (not just raw text) for accuracy.

Step 3: Find BEST rebuttal per defender argument
    Among all attacker turns that classify as rebuttal/undercut,
    pick the one with the highest confidence (not the first one found).

Step 4: Rebuttal quality scoring
    For each matched rebuttal pair, score HOW WELL the rebuttal was made:
    Does it engage substantively or just deflect?
    Returns quality score 0.0-1.0.

Step 5: Aggregate to rebuttal_effectiveness
    coverage_score = sum(type_weights) / total_rebuttable_args
    mean_quality   = mean of quality scores for matched rebuttals
    effectiveness  = coverage_score * mean_quality

Fallback
---------
If Ollama is unreachable: similarity score alone determines relation.
    similarity >= 0.60 → rebuttal (confidence = similarity)
    similarity >= 0.35 → undercut (confidence = similarity * 0.8)
    else               → unrelated

If SentenceTransformer unavailable: LLM-only mode (no pre-filter).

Dependencies
------------
    pip install sentence-transformers
    Ollama running locally with your configured model
"""

from __future__ import annotations

import json
import logging
import re
import requests
from dataclasses import dataclass, field
from typing import Optional

from judge_api.judge_config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ── Try to import sentence-transformers (optional but strongly recommended) ───
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Similarity pre-filter disabled — all pairs sent to LLM. "
        "Install with: pip install sentence-transformers"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

# Similarity pre-filter: pairs below this skip LLM entirely (marked unrelated)
SIM_PREFILTER_THRESHOLD = 0.20

# LLM confidence threshold: below this, treat as unrelated
LLM_CONFIDENCE_THRESHOLD = 0.55

# Rebuttal type weights for coverage score
# Full rebuttal (attacks conclusion) > undercut (attacks premise)
TYPE_WEIGHTS = {
    "rebuttal": 1.0,
    "undercut":  0.7,
    "unrelated": 0.0,
}

# Embedding model — small, fast, good for short text
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None   # loaded once on first call


# ── Embedding model ───────────────────────────────────────────────────────────

def _get_embed_model():
    global _embed_model
    if _embed_model is None and _ST_AVAILABLE:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two argument texts.
    Returns 0.0 if sentence-transformers is unavailable.
    """
    model = _get_embed_model()
    if model is None:
        return 0.5   # neutral — cannot pre-filter, let LLM decide

    embs = model.encode(
        [text_a, text_b],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(embs[0], embs[1]))


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Pull the first {...} block from LLM output and parse it."""
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ── LLM relation classifier ───────────────────────────────────────────────────

def _classify_relation(
    arg1_claim:    str,
    arg1_premises: list,
    arg2_claim:    str,
    arg2_premises: list,
    similarity:    float,
) -> dict:
    """
    Classify whether argument_2 rebuts or undercuts argument_1.

    Uses claim + premises for both arguments (not just raw text).
    Falls back to similarity-based classification if Ollama is unreachable.

    Returns
    -------
    {
        "relation":   "rebuttal" | "undercut" | "unrelated",
        "confidence": float,
        "reasoning":  str,
        "source":     "llm" | "similarity_fallback",
    }
    """
    def _fmt(claim: str, premises: list) -> str:
        if premises:
            prems = "\n".join(f"  - {p}" for p in premises)
            return f"Premises:\n{prems}\nConclusion: {claim}"
        return f"Argument: {claim}"

    prompt = f"""You are an argumentation theory expert.

Argument 1 (being defended):
{_fmt(arg1_claim, arg1_premises)}

Argument 2 (potential rebuttal):
{_fmt(arg2_claim, arg2_premises)}

Classify how Argument 2 relates to Argument 1.
Choose EXACTLY ONE:
- "rebuttal" : Argument 2 directly contradicts the CONCLUSION of Argument 1.
- "undercut" : Argument 2 attacks the EVIDENCE or REASONING of Argument 1
               without denying its conclusion.
- "unrelated": Argument 2 does not respond to Argument 1 at all.

CONFIDENCE: Give your honest estimate of certainty (0.0=pure guess, 1.0=certain).
Derive it from this specific pair — do not copy any number from this prompt.

Reply with ONLY this JSON (no markdown, no extra text):
{{"relation": "<rebuttal|undercut|unrelated>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 120,
                    "temperature": 0.0,   # deterministic
                },
            },
            timeout=45,
        )
        r.raise_for_status()
        raw    = r.json().get("response", "")
        result = _extract_json(raw)

        relation = str(result.get("relation", "unrelated")).lower().strip()
        if relation not in ("rebuttal", "undercut", "unrelated"):
            relation = "unrelated"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "relation":   relation,
            "confidence": round(confidence, 3),
            "reasoning":  str(result.get("reasoning", "No reasoning provided.")),
            "source":     "llm",
        }

    except Exception as e:
        logger.warning(f"Ollama unreachable for relation classification: {e}")

        # ── Similarity-based fallback ─────────────────────────────────────────
        # High similarity → likely rebuttal, medium → possible undercut
        if similarity >= 0.60:
            return {
                "relation":   "rebuttal",
                "confidence": round(similarity, 3),
                "reasoning":  "Ollama unavailable — inferred from high similarity.",
                "source":     "similarity_fallback",
            }
        if similarity >= 0.35:
            return {
                "relation":   "undercut",
                "confidence": round(similarity * 0.8, 3),
                "reasoning":  "Ollama unavailable — inferred from moderate similarity.",
                "source":     "similarity_fallback",
            }
        return {
            "relation":   "unrelated",
            "confidence": round(1.0 - similarity, 3),
            "reasoning":  "Ollama unavailable — low similarity suggests unrelated.",
            "source":     "similarity_fallback",
        }


# ── Rebuttal quality scorer ───────────────────────────────────────────────────

def _score_rebuttal_quality(
    original_claim:    str,
    original_premises: list,
    rebuttal_claim:    str,
    rebuttal_premises: list,
    relation:          str,
    similarity:        float,
) -> float:
    """
    Score HOW WELL argument_2 addresses argument_1.

    This is the quality dimension of rebuttal_effectiveness.
    A shallow deflection scores low; a substantive refutation scores high.

    Returns float 0.0-1.0.
    Fallback: similarity score if Ollama unavailable.
    """
    def _fmt(claim: str, premises: list) -> str:
        if premises:
            prems = "\n".join(f"  - {p}" for p in premises)
            return f"Premises:\n{prems}\nConclusion: {claim}"
        return f"Argument: {claim}"

    prompt = f"""You are an expert debate judge assessing rebuttal quality.

Original argument (Argument 1):
{_fmt(original_claim, original_premises)}

Response (Argument 2, classified as "{relation}"):
{_fmt(rebuttal_claim, rebuttal_premises)}

Score HOW WELL Argument 2 addresses Argument 1 on a scale of 0.0 to 1.0.

SCORING GUIDE:
0.0-0.2 : Mere deflection — ignores the substance, changes subject, or personal attack
0.3-0.4 : Superficial — mentions the topic but does not engage the specific claim
0.5-0.6 : Partial — addresses some aspects but leaves core claim intact
0.7-0.8 : Substantive — clearly engages with the claim/evidence with counter-reasoning
0.9-1.0 : Devastating — dismantles the original argument with evidence and tight logic

Derive your score from this specific pair. Do not copy any number from this prompt.

Reply with ONLY this JSON:
{{"quality": <0.0-1.0>, "reasoning": "<one sentence explaining the score>"}}"""

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 100, "temperature": 0.0},
            },
            timeout=45,
        )
        r.raise_for_status()
        raw    = r.json().get("response", "")
        result = _extract_json(raw)
        quality = float(result.get("quality", similarity))
        return round(max(0.0, min(1.0, quality)), 3)

    except Exception as e:
        logger.warning(f"Rebuttal quality scorer unavailable: {e}")
        # Fallback: use similarity as a proxy for quality
        return round(min(similarity, 1.0), 3)


# ── Per-pair evaluation ───────────────────────────────────────────────────────

def _evaluate_pair(
    defender_turn:  dict,
    attacker_turns: list
) -> dict:
    """
    Find the BEST rebuttal among all attacker turns for one defender argument.

    FIX vs original:
    - Finds BEST rebuttal (highest confidence), not FIRST rebuttal
    - Uses similarity pre-filter before LLM call
    - Populates claim + premises, not just raw text
    - Scores rebuttal quality separately

    Returns per-argument status dict.
    """
    d_arg      = defender_turn["argument"]
    d_claim    = defender_turn.get("claim",    d_arg)
    d_premises = defender_turn.get("premises", [])

    best_relation   = "unrelated"
    best_confidence = 0.0
    best_quality    = 0.0
    best_turn_num   = None
    best_reasoning  = None
    best_source     = None
    best_similarity = 0.0

    for at in attacker_turns:
        a_arg      = at["argument"]
        a_claim    = at.get("claim",    a_arg)
        a_premises = at.get("premises", [])

        # ── Step 1: Similarity pre-filter ─────────────────────────────────────
        similarity = _cosine_similarity(d_arg, a_arg)

        if similarity < SIM_PREFILTER_THRESHOLD:
            # Clearly unrelated — skip LLM call entirely
            continue

        # ── Step 2: LLM relation classifier ──────────────────────────────────
        rel_result = _classify_relation(
            arg1_claim=d_claim,    arg1_premises=d_premises,
            arg2_claim=a_claim,    arg2_premises=a_premises,
            similarity=similarity,
        )

        relation   = rel_result["relation"]
        confidence = rel_result["confidence"]

        # Only consider actual rebuttals/undercuts above confidence threshold
        if relation == "unrelated" or confidence < LLM_CONFIDENCE_THRESHOLD:
            continue

        # ── Step 3: Keep BEST match (highest confidence) ──────────────────────
        if confidence > best_confidence:
            best_relation   = relation
            best_confidence = confidence
            best_turn_num   = at["turn"]
            best_reasoning  = rel_result["reasoning"]
            best_source     = rel_result["source"]
            best_similarity = similarity

            # ── Step 4: Score rebuttal quality ────────────────────────────────
            best_quality = _score_rebuttal_quality(
                original_claim=d_claim,    original_premises=d_premises,
                rebuttal_claim=a_claim,    rebuttal_premises=a_premises,
                relation=relation,
                similarity=similarity,
            )

    # ── Build result for this defender argument ───────────────────────────────
    addressed = best_relation in ("rebuttal", "undercut")

    snippet = (
        d_arg[:120] + "..."
        if len(d_arg) > 120 else d_arg
    )

    return {
        "turn":             defender_turn["turn"],
        "argument_snippet": snippet,
        "status":           best_relation if addressed else "unanswered",
        "rebutted_by_turn": best_turn_num,
        "relation":         best_relation if addressed else None,
        "confidence":       round(best_confidence, 3) if addressed else None,
        "quality":          round(best_quality,    3) if addressed else None,
        "similarity":       round(best_similarity, 3) if addressed else None,
        "reasoning":        best_reasoning,
        "source":           best_source if addressed else None,
        "type_weight":      TYPE_WEIGHTS.get(best_relation, 0.0),
    }


# ── Per-speaker coverage + effectiveness ──────────────────────────────────────

def _compute_speaker_effectiveness(
    attacker_turns: list,
    defender_turns: list
) -> dict:
    """
    Compute rebuttal_effectiveness for one speaker (as attacker).

    Parameters
    ----------
    attacker_turns : turns belonging to the speaker being evaluated
    defender_turns : turns belonging to the opponent
    Both lists already have first turns removed.

    Returns
    -------
    {
        "coverage_score":          float,   weighted fraction addressed
        "mean_rebuttal_quality":   float,   avg quality of matched rebuttals
        "rebuttal_effectiveness":  float,   coverage * quality
        "arguments_rebuttable":    int,
        "arguments_addressed":     int,
        "arguments_unanswered":    int,
        "per_argument":            list,
    }
    """
    # Last attacker turn determines what the attacker had a CHANCE to rebut
    last_attacker_turn = max(
        (t["turn"] for t in attacker_turns), default=0
    )

    # Rebuttable = defender args that appeared BEFORE attacker's last turn
    rebuttable   = [t for t in defender_turns if t["turn"] < last_attacker_turn]
    unreachable  = [t for t in defender_turns if t["turn"] >= last_attacker_turn]

    per_argument = []
    weighted_sum = 0.0
    quality_scores = []

    for def_turn in rebuttable:
        # Only attacker turns AFTER this defender turn are valid rebuttals
        later_attacker = [
            t for t in attacker_turns
            if t["turn"] > def_turn["turn"]
        ]

        result = _evaluate_pair(def_turn, later_attacker)
        per_argument.append(result)

        if result["status"] in ("rebuttal", "undercut"):
            weighted_sum += result["type_weight"]
            if result["quality"] is not None:
                quality_scores.append(result["quality"])

    # Unreachable arguments — informational only, not in denominator
    for def_turn in unreachable:
        snippet = (
            def_turn["argument"][:120] + "..."
            if len(def_turn["argument"]) > 120
            else def_turn["argument"]
        )
        per_argument.append({
            "turn":             def_turn["turn"],
            "argument_snippet": snippet,
            "status":           "unreachable",
            "rebutted_by_turn": None,
            "relation":         None,
            "confidence":       None,
            "quality":          None,
            "similarity":       None,
            "reasoning":        "After attacker's last turn — no rebuttal opportunity.",
            "source":           None,
            "type_weight":      0.0,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = len(rebuttable)

    # coverage_score: weighted (rebuttal=1.0, undercut=0.7) fraction
    coverage_score = round(weighted_sum / total, 3) if total > 0 else 0.0

    # mean_rebuttal_quality: average quality of rebuttals that were made
    mean_quality = round(
        sum(quality_scores) / len(quality_scores), 3
    ) if quality_scores else 0.0

    # rebuttal_effectiveness: coverage * quality
    effectiveness = round(coverage_score * mean_quality, 3)

    addressed = sum(
        1 for a in per_argument
        if a["status"] in ("rebuttal", "undercut")
    )

    return {
        "coverage_score":         coverage_score,
        "mean_rebuttal_quality":  mean_quality,
        "rebuttal_effectiveness": effectiveness,
        "arguments_rebuttable":   total,
        "arguments_addressed":    addressed,
        "arguments_unanswered":   total - addressed,
        "per_argument":           per_argument,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def compute_rebuttal_effectiveness(
    turns:       list
) -> dict:
    """
    Compute rebuttal_effectiveness for both speakers across a full debate.

    Parameters
    ----------
    turns : list of dicts, each with:
        {
            "turn":      int,      debate turn number (1-indexed)
            "speaker":   str,      speaker identifier
            "argument":  str,      full argument text
            "claim":     str,      optional — from extract_claim_and_premises
            "premises":  list,     optional — from extract_claim_and_premises
        }

        IMPORTANT: First turn of each speaker MUST be their opening statement.
        It is automatically excluded from rebuttal scoring on both sides.

    ollama_host : str   Ollama base URL
    model       : str   Ollama model name

    Returns
    -------
    {
        "<speaker_a>": {
            "coverage_score":         float,   weighted fraction addressed
            "mean_rebuttal_quality":  float,   avg quality of made rebuttals
            "rebuttal_effectiveness": float,   coverage * quality (0.0-1.0)
            "arguments_rebuttable":   int,
            "arguments_addressed":    int,
            "arguments_unanswered":   int,
            "per_argument":           list,
        },
        "<speaker_b>": { ... },
        "summary": {
            "total_turns":            int,
            "first_turns_excluded":   list,    [{"speaker": str, "turn": int}]
            "embed_model_available":  bool,
        }
    }
    """
    print("Turns received for rebuttal effectiveness:", turns)
    # ── Validate input ────────────────────────────────────────────────────────
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers."}

    speaker_a, speaker_b = speakers[0], speakers[1]
    print("Turns received ", turns)

    # ── Identify and exclude first turns ─────────────────────────────────────
    # First turn = turn with the lowest turn number for each speaker.
    # These are opening statements — excluded from BOTH attacker and defender roles.
    first_turns = {}
    for sp in speakers:
        sp_turns = [t for t in turns if t["speaker"] == sp]
        if sp_turns:
            first_turns[sp] = min(t["turn"] for t in sp_turns)

    first_turn_log = [
        {"speaker": sp, "turn": turn_num}
        for sp, turn_num in first_turns.items()
    ]

    def _without_first(sp: str) -> list:
        """Return turns for speaker, excluding their first turn."""
        first = first_turns.get(sp, -1)
        return [t for t in turns if t["speaker"] == sp and t["turn"] != first]

    turns_a = _without_first(speaker_a)
    turns_b = _without_first(speaker_b)

    # ── Compute effectiveness for each speaker ────────────────────────────────
    # Speaker A as attacker: how well did A rebut B's arguments?
    result_a = _compute_speaker_effectiveness(
        attacker_turns=turns_a,
        defender_turns=turns_b,
    )

    # Speaker B as attacker: how well did B rebut A's arguments?
    result_b = _compute_speaker_effectiveness(
        attacker_turns=turns_b,
        defender_turns=turns_a,
    )

    return {
        speaker_a: result_a,
        speaker_b: result_b,
        "summary": {
            "total_turns":           len(turns),
            "first_turns_excluded":  first_turn_log,
            "embed_model_available": _ST_AVAILABLE,
            "sim_prefilter_threshold":  SIM_PREFILTER_THRESHOLD,
            "llm_confidence_threshold": LLM_CONFIDENCE_THRESHOLD,
            "type_weights":             TYPE_WEIGHTS,
        },
    }