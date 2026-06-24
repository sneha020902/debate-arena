"""
unknown_arguments.py — Handles ONLY arguments with no Elasticsearch match.

Responsibility
--------------
This file deals exclusively with arguments the ES index cannot confidently
match to a reference argument (score below ES_MIN_MATCH, or ES unreachable).

It does NOT score known arguments — those go through final_score.py.

For unknown arguments it runs a partial scoring pipeline using the two
model-based scorers that do NOT require an ES match:
    - NLI score     (CrossEncoder — only needs claim + premises)
    - ArgQuality    (IBM corpus interpolation — independent of debate ES index)
    - Evidence grounding (text heuristic — replaces Delta/Votes which need ES)

Then blends the partial score with an adaptive team prior using one of three
configurable strategies: llm | extrapolation | blend.

Match Classification
--------------------
KNOWN       best_overall >= ES_MIN_MATCH + BORDERLINE_UPPER  full pipeline
BORDERLINE  ES_MIN_MATCH <= best_overall < upper             full pipeline, reduced ES trust
UNKNOWN     best_overall < ES_MIN_MATCH, results exist       partial pipeline (this file)
ES_DOWN     ES unreachable entirely                          partial pipeline (this file)

Only UNKNOWN and ES_DOWN flow through this file.
KNOWN and BORDERLINE are handled by final_score.py.

Unknown Weights (Delta redistributed to NLI + ArgQuality)
----------------------------------------------------------
Full pipeline:    NLI=0.25  ArgQuality=0.40  Delta=0.20  Votes=0.15
Unknown pipeline: NLI=0.31  ArgQuality=0.54  Evidence=0.15
Delta's 0.20 redistributed proportionally to NLI and ArgQuality.

Adaptive Prior
--------------
Built from the team's already-scored KNOWN arguments.
Uses recency weighting (recent arguments matter more than early ones)
and reliability scoring (more known arguments = more reliable prior).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from ..judge_config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    ES_MIN_MATCH,
    DEFAULT_UNKNOWN_STRATEGY,
    BLEND_LLM_WEIGHT,
)
from utils.es_client import es_client
from utils.argument_miner import extract_claim_and_premises

# Partial pipeline scorers — both independent of ES
from .nli_score         import score_nli
from .argquality_score  import score_argquality
from .delta_votes_score import score_delta_votes

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Weights when Delta is unavailable (redistributed proportionally)
# Original: NLI=0.25, ArgQuality=0.40, Delta=0.20, Votes=0.15
# Delta's 0.20 redistributed: NLI gets 0.0625, ArgQuality gets 0.125
# Result (rounded): NLI=0.31, ArgQuality=0.54, Evidence=0.15 (sum=1.0)
_W_NLI_UNKNOWN        = 0.31
_W_ARGQUALITY_UNKNOWN = 0.54
_W_EVIDENCE_UNKNOWN   = 0.15

# Borderline band above ES_MIN_MATCH
_BORDERLINE_UPPER = 0.15

# Prior reliability thresholds
_PRIOR_MIN_RELIABLE = 2     # below this: prior barely exists
_PRIOR_MAX_RELIABLE = 6     # at this count: prior is fully reliable
_RECENT_DECAY       = 0.85  # exponential decay per turn (recency weight)

# Confidence constants
_CONF_MEDIUM  = 0.70
_CONF_LOW     = 0.50
_CONF_VERYLOW = 0.30


# ── Match type ────────────────────────────────────────────────────────────────

class MatchType(str, Enum):
    KNOWN      = "known"
    BORDERLINE = "borderline"
    UNKNOWN    = "unknown"
    ES_DOWN    = "es_down"


@dataclass
class ClassifiedArgument:
    argument:   str
    claim:      str
    premises:   list
    match_type: MatchType
    es_results: list  = field(default_factory=list)
    best_score: float = 0.0
    reason:     str   = ""
    turn:       int   = 0


# ── Step 1: Classify ──────────────────────────────────────────────────────────

def classify_argument(argument: str, turn: int = 0) -> ClassifiedArgument:
    """
    Extract claim/premises, query ES, classify match type.
    Called by the orchestrating layer to route each argument.
    Never raises — ES errors return ES_DOWN.
    """
    try:
        arg_result = extract_claim_and_premises(argument)
        claim      = arg_result.claim
        premises   = arg_result.premises or []
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        claim    = argument[:200]
        premises = []

    try:
        results = es_client.granular_similarity(
            argument, claim, premises, top_n=5
        )
    except Exception as e:
        logger.warning(f"ES unreachable: {e}")
        return ClassifiedArgument(
            argument=argument, claim=claim, premises=premises,
            match_type=MatchType.ES_DOWN,
            es_results=[], best_score=0.0,
            reason=f"ES unreachable: {e}", turn=turn,
        )

    if not results:
        return ClassifiedArgument(
            argument=argument, claim=claim, premises=premises,
            match_type=MatchType.UNKNOWN,
            es_results=[], best_score=0.0,
            reason="ES returned no results.", turn=turn,
        )

    best_overall   = results[0].get("overall_score", 1.0)
    borderline_top = ES_MIN_MATCH + _BORDERLINE_UPPER

    if best_overall >= borderline_top:
        return ClassifiedArgument(
            argument=argument, claim=claim, premises=premises,
            match_type=MatchType.KNOWN,
            es_results=results, best_score=best_overall,
            reason=f"Strong ES match: {best_overall:.3f}", turn=turn,
        )

    if best_overall >= ES_MIN_MATCH:
        return ClassifiedArgument(
            argument=argument, claim=claim, premises=premises,
            match_type=MatchType.BORDERLINE,
            es_results=results, best_score=best_overall,
            reason=f"Borderline ES match: {best_overall:.3f}", turn=turn,
        )

    return ClassifiedArgument(
        argument=argument, claim=claim, premises=premises,
        match_type=MatchType.UNKNOWN,
        es_results=results, best_score=best_overall,
        reason=f"Below threshold: {best_overall:.3f} < {ES_MIN_MATCH}",
        turn=turn,
    )


# ── Step 2: Partial pipeline ──────────────────────────────────────────────────

def score_partial_pipeline(ca: ClassifiedArgument) -> dict:
    """
    NLI + ArgQuality + evidence grounding for an argument with no ES match.
    Delta is unavailable — its weight is redistributed to NLI and ArgQuality.

    partial_score is on the same 0-1 scale as full pipeline because both
    use the same NLI and ArgQuality scorers.
    """
    # NLI — CrossEncoder, needs only claim + premises
    nli_result = score_nli(
        argument=ca.argument,
        claim=ca.claim,
        premises=ca.premises,
    )
    nli = nli_result["nli_score"]

    # ArgQuality — IBM 30k corpus, fully independent of debate ES index
    aq_result = score_argquality(argument=ca.argument)
    aq = aq_result["argquality_score"]

    # Evidence grounding — text heuristic, replaces Delta+Votes
    dv_result = score_delta_votes(
        es_results=[],
        argument_text=ca.argument,
        use_evidence_grounding=True,
    )
    evidence = dv_result["evidence_grounding"]

    partial_score = round(
        _W_NLI_UNKNOWN        * nli
        + _W_ARGQUALITY_UNKNOWN * aq
        + _W_EVIDENCE_UNKNOWN   * evidence,
        3,
    )

    # Coverage proxy: IBM corpus similarity (no ES cluster signal available)
    top_sim  = aq_result.get("top_k_similarity") or 0.0
    coverage = round(0.10 + 0.40 * top_sim, 3)

    return {
        "partial_score":    partial_score,
        "coverage":         coverage,
        "matched":          False,
        "match_type":       ca.match_type.value,
        "match_reason":     ca.reason,
        "es_down":          (ca.match_type == MatchType.ES_DOWN),
        "score_breakdown": {
            "nli":        round(nli,      3),
            "argquality": round(aq,       3),
            "delta":      None,
            "evidence":   round(evidence, 3),
            "weights": {
                "nli":        _W_NLI_UNKNOWN,
                "argquality": _W_ARGQUALITY_UNKNOWN,
                "delta":      "N/A — redistributed to NLI and ArgQuality",
                "evidence":   _W_EVIDENCE_UNKNOWN,
            },
        },
        "nli_detail":  nli_result,
        "aq_detail":   aq_result,
        "dv_detail":   dv_result,
    }


# ── Step 3: Adaptive prior ────────────────────────────────────────────────────

def build_adaptive_prior(known_results: list) -> dict:
    """
    Build team prior from already-scored KNOWN arguments (from final_score.py).

    Recency weighting: newest argument = weight 1.0, each step back * 0.85.
    Reliability: increases with sample size, caps at _PRIOR_MAX_RELIABLE.

    Parameters
    ----------
    known_results : list of {"turn": int, "quality": float, "coverage": float}

    Returns
    -------
    {"quality": float|None, "coverage": float, "reliability": float, "n_known": int}
    """
    if not known_results:
        return {"quality": None, "coverage": 0.30, "reliability": 0.0, "n_known": 0}

    n = len(known_results)
    sorted_r = sorted(known_results, key=lambda x: x.get("turn", 0))

    weights = [_RECENT_DECAY ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights) or 1e-9

    w_quality  = sum(weights[i] * sorted_r[i]["quality"]  for i in range(n)) / total_w
    w_coverage = sum(weights[i] * sorted_r[i]["coverage"] for i in range(n)) / total_w

    if n >= _PRIOR_MAX_RELIABLE:
        reliability = 1.0
    elif n >= _PRIOR_MIN_RELIABLE:
        reliability = (n - _PRIOR_MIN_RELIABLE) / (_PRIOR_MAX_RELIABLE - _PRIOR_MIN_RELIABLE)
    else:
        reliability = (n / _PRIOR_MIN_RELIABLE) * 0.3

    return {
        "quality":     round(w_quality,             3),
        "coverage":    round(w_coverage,            3),
        "reliability": round(max(reliability, 0.0), 3),
        "n_known":     n,
    }


# ── Step 4: Blend partial with prior ─────────────────────────────────────────

def resolve_unknown(
    partial:          dict,
    team_prior:       dict,
    strategy:         str,
    blend_llm_weight: float,
) -> dict:
    """
    Combine partial_score with team prior using chosen strategy.

    "llm"           Trust partial_score fully (NLI + ArgQuality ARE the signals)
    "extrapolation" Use prior; blend partial in when prior is unreliable
    "blend"         Adaptive weights based on prior reliability
    """
    partial_score = partial["partial_score"]
    prior_q       = team_prior.get("quality")
    prior_c       = team_prior.get("coverage", 0.30)
    reliability   = team_prior.get("reliability", 0.0)

    # ── llm: trust partial pipeline directly ─────────────────────────────────
    if strategy == "llm":
        confidence = (
            _CONF_MEDIUM
            if partial["score_breakdown"]["nli"] > 0.4
            else _CONF_LOW
        )
        return {
            "quality":      partial_score,
            "coverage":     partial["coverage"],
            "confidence":   confidence,
            "matched":      False,
            "match_type":   partial["match_type"],
            "match_reason": partial["match_reason"],
            "source":       "partial_pipeline_llm_mode",
            "score_breakdown": partial["score_breakdown"],
            "nli_detail":      partial["nli_detail"],
            "aq_detail":       partial["aq_detail"],
            "dv_detail":       partial["dv_detail"],
        }

    # ── extrapolation: use prior, blend if unreliable ─────────────────────────
    if strategy == "extrapolation":
        if prior_q is None:
            return {
                "quality":       partial_score,
                "coverage":      partial["coverage"],
                "confidence":    _CONF_LOW,
                "matched":       False,
                "match_type":    partial["match_type"],
                "match_reason":  partial["match_reason"],
                "source":        "extrapolation_no_prior_fallback",
                "prior_quality": None,
                "partial_score": partial_score,
                "score_breakdown": partial["score_breakdown"],
            }

        if reliability >= 0.8:
            q   = prior_q
            src = "extrapolation_high_reliability"
            confidence = _CONF_MEDIUM
        else:
            partial_weight = 1.0 - reliability
            q   = partial_weight * partial_score + (1 - partial_weight) * prior_q
            src = f"extrapolation_blended_reliability={reliability:.2f}"
            confidence = _CONF_LOW

        return {
            "quality":           round(q, 3),
            "coverage":          round(prior_c, 3),
            "confidence":        confidence,
            "matched":           False,
            "match_type":        partial["match_type"],
            "match_reason":      partial["match_reason"],
            "source":            src,
            "prior_quality":     prior_q,
            "prior_reliability": reliability,
            "partial_score":     partial_score,
            "score_breakdown":   partial["score_breakdown"],
        }

    # ── blend (default): adaptive weights based on reliability ────────────────
    # reliability=0.0 → partial=blend_w,       prior=1-blend_w
    # reliability=0.5 → partial=blend_w*0.75,  prior grows
    # reliability=1.0 → partial=blend_w*0.50,  prior dominates
    eff_partial_weight = 1.0
    if prior_q is None:
        q   = partial_score
        src = "blend_no_prior"
        confidence = _CONF_LOW
    else:
        eff_partial_weight = blend_llm_weight * (1.0 - reliability * 0.5)
        eff_prior_weight   = 1.0 - eff_partial_weight
        q   = eff_partial_weight * partial_score + eff_prior_weight * prior_q
        src = (
            f"blend_partial={eff_partial_weight:.2f}"
            f"_prior={eff_prior_weight:.2f}"
            f"_reliability={reliability:.2f}"
        )
        confidence = _CONF_MEDIUM if reliability >= 0.5 else _CONF_LOW

    return {
        "quality":             round(q, 3),
        "coverage":            round(prior_c, 3),
        "confidence":          confidence,
        "matched":             False,
        "match_type":          partial["match_type"],
        "match_reason":        partial["match_reason"],
        "source":              src,
        "prior_quality":       prior_q,
        "prior_reliability":   reliability,
        "partial_score":       partial_score,
        "partial_weight_used": round(eff_partial_weight, 3),
        "score_breakdown":     partial["score_breakdown"],
        "nli_detail":          partial["nli_detail"],
        "aq_detail":           partial["aq_detail"],
        "dv_detail":           partial["dv_detail"],
    }


# ── Public entry point ────────────────────────────────────────────────────────

def score_unknown_arguments(
    unknown_arguments: list,
    known_results:     list,
    strategy:          str   = None,
    blend_llm_weight:  float = None,
) -> dict:
    """
    Score arguments that had no close ES match.

    Receives ONLY UNKNOWN/ES_DOWN arguments (already classified by
    classify_argument). Known and borderline arguments are NOT passed here.

    Parameters
    ----------
    unknown_arguments : list of ClassifiedArgument
        Each with match_type == UNKNOWN or ES_DOWN.

    known_results : list of {"turn": int, "quality": float, "coverage": float}
        Scored KNOWN arguments from final_score.py — used to build prior.

    strategy : "llm" | "extrapolation" | "blend"

    blend_llm_weight : float  base partial-score weight in blend

    Returns
    -------
    {
        "per_argument": [
            {
                "turn":             int,
                "argument_snippet": str,
                "quality":          float,
                "coverage":         float,
                "confidence":       float,
                "matched":          False,
                "match_type":       "unknown" | "es_down",
                "match_reason":     str,
                "source":           str,
                "score_breakdown":  dict,
                "nli_detail":       dict,
                "aq_detail":        dict,
                "dv_detail":        dict,
            }
        ],
        "mean_quality":          float,
        "mean_coverage":         float,
        "conf_weighted_quality": float,
        "n_unknown":             int,
        "n_es_down":             int,
        "strategy_used":         str,
        "team_prior":            dict,
    }
    """
    strategy         = strategy         or DEFAULT_UNKNOWN_STRATEGY
    blend_llm_weight = blend_llm_weight or BLEND_LLM_WEIGHT

    team_prior = build_adaptive_prior(known_results)
    logger.info(
        f"Unknown scorer — prior: quality={team_prior['quality']} "
        f"reliability={team_prior['reliability']} n_known={team_prior['n_known']}"
    )

    per_argument = []
    for ca in unknown_arguments:
        partial = score_partial_pipeline(ca)
        final   = resolve_unknown(partial, team_prior, strategy, blend_llm_weight)

        snippet = (
            ca.argument[:120] + "..."
            if len(ca.argument) > 120
            else ca.argument
        )
        per_argument.append({
            "turn":             ca.turn,
            "argument_snippet": snippet,
            **final,
        })

    n = len(per_argument)
    if n == 0:
        return {
            "per_argument": [], "mean_quality": 0.0, "mean_coverage": 0.0,
            "conf_weighted_quality": 0.0,
            "n_unknown": 0, "n_es_down": 0,
            "strategy_used": strategy, "team_prior": team_prior,
        }

    mean_quality  = round(sum(a["quality"]  for a in per_argument) / n, 3)
    mean_coverage = round(sum(a["coverage"] for a in per_argument) / n, 3)
    total_conf    = sum(a["confidence"] for a in per_argument) or 1e-9
    conf_w_q      = round(
        sum(a["quality"] * a["confidence"] for a in per_argument) / total_conf, 3
    )

    return {
        "per_argument":          per_argument,
        "mean_quality":          mean_quality,
        "mean_coverage":         mean_coverage,
        "conf_weighted_quality": conf_w_q,
        "n_unknown":  sum(1 for a in per_argument if a["match_type"] == "unknown"),
        "n_es_down":  sum(1 for a in per_argument if a["match_type"] == "es_down"),
        "strategy_used": strategy,
        "team_prior":    team_prior,
    }