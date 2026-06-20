"""
final_score.py — Weighted combination of the four argument-quality metrics.

Formula (from the Individual Argument Scoring Requirement spec):

    final_score = 0.30 × NLI Score
                + 0.20 × Delta Score
                + 0.15 × Votes Score
                + 0.35 × ArgQuality Score

Call compute_full_score() when Elasticsearch has returned reference arguments.
It orchestrates the four scorers, assembles the complete breakdown, and
returns the final weighted score alongside every sub-score.

Call is_es_match() first: if False, skip this module and fall back to the
existing unknown-argument logic in unknown_arguments.py (unchanged).
"""

from services.judge_config import ES_MIN_MATCH

# ── Formula weights (per spec) ─────────────────────────────────────────────────
WEIGHT_NLI        = 0.30
WEIGHT_DELTA      = 0.20
WEIGHT_VOTES      = 0.15
WEIGHT_ARGQUALITY = 0.35
_WEIGHTS = {
    "nli":        WEIGHT_NLI,
    "delta":      WEIGHT_DELTA,
    "votes":      WEIGHT_VOTES,
    "argquality": WEIGHT_ARGQUALITY,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def is_es_match(es_results: list) -> bool:
    """
    True when Elasticsearch returned at least one result above the match
    threshold — i.e. the argument is KNOWN and we should run the 4-score
    pipeline. Mirrors the threshold check in unknown_arguments._score_known.
    """
    if not es_results:
        return False
    return es_results[0].get("overall_score", 1.0) >= ES_MIN_MATCH


def compute_full_score(argument: str, es_results: list,
                       claim: str = None, premises: list = None,
                       ollama_host: str = None, model: str = None) -> dict:
    """
    Run all four scorers and return the final weighted score + full breakdown.

    This function is called ONLY when is_es_match() is True (i.e. the argument
    has a reference in Elasticsearch). The fallback path for unknown arguments
    is handled separately in unknown_arguments.py and is not touched here.

    Parameters
    ----------
    argument   : str        Full argument text.
    es_results : list       Results from es_client.granular_similarity().
    claim      : str        Pre-extracted claim (optional; auto-split if absent).
    premises   : list[str]  Pre-extracted premises (optional).

    Returns
    -------
    {
        "final_score":    float,          # 0.0–1.0  ← replaces old quality field
        "nli_score":      float,
        "delta_score":    float,
        "votes_score":    float,
        "argquality_score": float,
        "argquality_breakdown": {
            "overall_quality":  float,
            "cogency":          float,
            "effectiveness":    float,
            "reasoning":        str,
        },
        "nli_reasoning":  str,
        "weights_used":   dict,
    }
    """
    from services.nli_score         import score_nli
    from services.delta_votes_score import score_delta_votes
    from services.argquality_score  import score_argquality

    # 1 — NLI: does the claim follow from the premises?
    nli_result = score_nli(argument, claim=claim, premises=premises,
                           ollama_host=ollama_host, model=model)

    # 2 — Delta + Votes: from ES reference data (no LLM call)
    dv_result = score_delta_votes(es_results)

    # 3 — ArgQuality: holistic LLM assessment (3 sub-dimensions)
    aq_result = score_argquality(argument, ollama_host=ollama_host, model=model)

    # 4 — Weighted final score
    fs = round(
        WEIGHT_NLI        * nli_result["nli_score"] +
        WEIGHT_DELTA      * dv_result["delta_score"] +
        WEIGHT_VOTES      * dv_result["votes_score"] +
        WEIGHT_ARGQUALITY * aq_result["argquality_score"],
        3,
    )

    return {
        "final_score":      fs,
        "nli_score":        nli_result["nli_score"],
        "delta_score":      dv_result["delta_score"],
        "votes_score":      dv_result["votes_score"],
        "argquality_score": aq_result["argquality_score"],
        "argquality_breakdown": {
            "overall_quality": aq_result["overall_quality"],
            "cogency":         aq_result["cogency"],
            "effectiveness":   aq_result["effectiveness"],
            "reasoning":       aq_result["reasoning"],
        },
        "nli_reasoning":  nli_result["reasoning"],
        "weights_used":   dict(_WEIGHTS),
        # pass-through detail for explainability
        "delta_detail": {
            "delta_flagged_count": dv_result["delta_flagged_count"],
            "total_matches":       dv_result["total_matches"],
            "top_cluster_quality": dv_result["top_match_cluster_quality"],
        },
    }
