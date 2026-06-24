"""
final_score.py — Integrates all 4 scorers for KNOWN and BORDERLINE arguments.

Responsibility
--------------
This file scores arguments that DID get an Elasticsearch match
(match_type == KNOWN or BORDERLINE). It assembles all four individual scorer
outputs into a single weighted final score.

It does NOT handle unknown arguments — those go through unknown_arguments.py.

Updated Weights (changed from original)
----------------------------------------
Original:  NLI=0.30  ArgQuality=0.35  Delta=0.20  Votes=0.15
Updated:   NLI=0.25  ArgQuality=0.40  Delta=0.20  Votes=0.15

Why ArgQuality increased (0.35 → 0.40) and NLI decreased (0.30 → 0.25):
  - ArgQuality now uses IBM 30k corpus interpolation (human-annotated scores)
    rather than a bare LLM call — its signal is now more reliable
  - NLI (CrossEncoder) measures logical structure only, not factual grounding
    or persuasive strength — over-weighting it penalises valid emotional appeals
  - ArgQuality encompasses cogency, effectiveness, and clarity — three
    dimensions that together better capture overall argument strength

Borderline Handling
-------------------
For BORDERLINE matches (best_score between ES_MIN_MATCH and ES_MIN_MATCH+0.15),
the Delta and Votes signals derived from ES are blended toward neutral (0.5)
proportionally to how close the match is to the threshold.

  trust = (best_score - ES_MIN_MATCH) / BORDERLINE_UPPER
  delta_adjusted = trust * delta_raw + (1 - trust) * 0.5
  votes_adjusted = trust * votes_raw + (1 - trust) * 0.5

NLI and ArgQuality are NOT adjusted — they are independent of match quality.

Integration with unknown_arguments.py
--------------------------------------
The caller (score_team_arguments in the orchestrator) routes:
  KNOWN/BORDERLINE  → compute_full_score()  (this file)
  UNKNOWN/ES_DOWN   → score_unknown_arguments()  (unknown_arguments.py)

Both files return the same output schema:
  { quality, coverage, confidence, matched, score_breakdown,
    nli_detail, aq_detail, dv_detail }

so the orchestrator can merge results without special-casing.
"""

from __future__ import annotations

import logging
from typing import Optional

from .judge_config import ES_MIN_MATCH, es_norm, INDIVIDUAL_ARG_WEIGHTS

from .nli_score         import score_nli
from .argquality_score  import score_argquality
from .delta_votes_score import score_delta_votes
from .es_client import granular_similarity

logger = logging.getLogger(__name__)


# ── Updated weights ───────────────────────────────────────────────────────────

W_NLI        = INDIVIDUAL_ARG_WEIGHTS["nli"]        # was 0.30 — reduced: NLI measures structure only
W_ARGQUALITY = INDIVIDUAL_ARG_WEIGHTS["arg_quality"] # was 0.35 — increased: now IBM corpus, more reliable
W_DELTA      = INDIVIDUAL_ARG_WEIGHTS["delta"]      # unchanged
W_VOTES      = INDIVIDUAL_ARG_WEIGHTS["votes"]      # unchanged

# Borderline band width above ES_MIN_MATCH
_BORDERLINE_UPPER = 0.15

# Confidence levels
_CONF_HIGH   = 0.90   # strong ES match, full pipeline
_CONF_MEDIUM = 0.70   # borderline match, reduced ES trust


# ── Core function ─────────────────────────────────────────────────────────────

def compute_individual_arg_score(
    argument:   str,
    claim:      str  = "",
    premises:   list = None,
    es_results: Optional[list] = None,  # pass ES results to avoid redundant calls
    best_score: float = 0.0,
    is_borderline: bool = False,
) -> dict:
    """
    Run all 4 scorers and assemble the weighted final score.

    Called for KNOWN and BORDERLINE arguments only.
    All four scorers always run — the difference for BORDERLINE is that
    Delta and Votes are blended toward neutral based on match confidence.

    Parameters
    ----------
    argument      : str    Full debate argument text.
    es_results    : list   Top-k results from es_client.granular_similarity().
    claim         : str    Central assertion (from extract_claim_and_premises).
    premises      : list   Supporting statements.
    best_score    : float  Best ES overall_score (used for borderline blending).
    is_borderline : bool   If True, reduce ES signal trust proportionally.

    Returns
    -------
    {
        "final_score":      float,   0.0-1.0 weighted composite
        "quality":          float,   alias for final_score (schema compatibility)
        "coverage":         float,   ES cluster-based coverage signal
        "confidence":       float,   reliability of this score
        "matched":          True,
        "match_type":       "known" | "borderline",
        "score_breakdown":  dict,    per-scorer raw + adjusted scores + weights
        "nli_detail":       dict,    full output of score_nli()
        "aq_detail":        dict,    full output of score_argquality()
        "dv_detail":        dict,    full output of score_delta_votes()
    }
    """
    premises = premises or []

    es_results = granular_similarity(
        argument=argument,
        claim=claim,
        premises=premises,
        top_n=5
)

    # ── 1. NLI score ──────────────────────────────────────────────────────────
    # CrossEncoder on claim/premises.
    # Not affected by borderline status — purely structural.
    nli_result = score_nli(
        argument=argument,
        claim=claim,
        premises=premises,
    )
    nli = nli_result["nli_score"]

    # ── 2. ArgQuality score ───────────────────────────────────────────────────
    # IBM 30k corpus interpolation.
    # Not affected by borderline status — independent of debate ES index.
    aq_result = score_argquality(argument=argument)
    aq = aq_result["argquality_score"]

    # ── 3. Delta + Votes ──────────────────────────────────────────────────────
    # Derived from ES results — affected by borderline status.
    dv_result = score_delta_votes(
        es_results=es_results   # use cluster_quality for known args
    )
    delta_raw = dv_result["delta_score"]
    votes_raw = dv_result["votes_score"]

    # ── 4. Borderline adjustment ──────────────────────────────────────────────
    # For borderline matches, blend ES-derived signals toward neutral (0.5).
    # NLI and ArgQuality are NOT adjusted.
    if is_borderline and best_score > 0.0:
        # trust = 1.0 at borderline_top, 0.0 at ES_MIN_MATCH
        trust = (best_score - ES_MIN_MATCH) / _BORDERLINE_UPPER
        trust = max(0.0, min(1.0, trust))
        delta = round(trust * delta_raw + (1.0 - trust) * 0.5, 3)
        votes = round(trust * votes_raw + (1.0 - trust) * 0.5, 3)
    else:
        delta = delta_raw
        votes = votes_raw
        trust = 1.0

    # ── 5. Weighted final score ───────────────────────────────────────────────
    final_score = round(
        W_NLI        * nli
        + W_ARGQUALITY * aq
        + W_DELTA      * delta
        + W_VOTES      * votes,
        3,
    )

    # ── 6. Coverage signal ────────────────────────────────────────────────────
    # How well-supported is this argument in the debate corpus?
    # Combination of ES match strength and cluster size (number of similar args)
    best = es_results[0] if es_results else {}
    support      = best.get("cluster_member_count", 1) or 1
    support_norm = min(support / 6.0, 1.0)
    coverage     = round(
        0.6 * es_norm(best_score) + 0.4 * support_norm, 3
    )

    match_type = "borderline" if is_borderline else "known"
    confidence = _CONF_MEDIUM if is_borderline else _CONF_HIGH

    return {
        "final_score": final_score,
        "quality":     final_score,   # alias — same schema as unknown_arguments output
        "coverage":    coverage,
        "confidence":  confidence,
        "matched":     True,
        "match_type":  match_type,
        "score_breakdown": {
            "nli":              round(nli,       3),
            "argquality":       round(aq,        3),
            "delta_raw":        round(delta_raw, 3),
            "votes_raw":        round(votes_raw, 3),
            "delta_adjusted":   round(delta,     3),
            "votes_adjusted":   round(votes,     3),
            "borderline_trust": round(trust,     3),
            "weights": {
                "nli":        W_NLI,
                "argquality": W_ARGQUALITY,
                "delta":      W_DELTA,
                "votes":      W_VOTES,
            },
        },
        "nli_detail":  nli_result,
        "aq_detail":   aq_result,
        "dv_detail":   dv_result,
    }
