"""
delta_votes_score.py — Hardened Delta and Votes scorers.

Fixes applied vs original:
  1. Delta formula uses top-k capped scoring to reduce k-sensitivity
  2. Votes score aggregates across top-3 neighbours, not just top-1
  3. _norm() is ES-score-range agnostic (auto-detects range)
  4. Fallback returns 0.5 (neutral) not 0.0 (punitive)
  5. Confidence score returned alongside both metrics
  6. Delta score capped contribution: one strong match can't dominate
"""

_FALLBACK_DELTA = 0.5   # neutral — unknown, not bad
_FALLBACK_VOTES = 0.5
_VOTES_TOP_K    = 3     # aggregate cluster quality across top-3, not just top-1


def _safe_norm(score: float, all_scores: list) -> float:
    """
    Normalise an ES score to 0-1 without hardcoding the range.
    Auto-detects whether scores are in BM25 range (>2), cosine range (0-1),
    or the 1-2 range your current code assumes.
    """
    if not all_scores:
        return 0.0

    min_s = min(all_scores)
    max_s = max(all_scores)

    if max_s == min_s:
        return 1.0  # all scores identical — treat as full match

    # Min-max normalisation — range agnostic
    return round((score - min_s) / (max_s - min_s), 4)


def score_delta_votes(es_results: list) -> dict:
    """
    Compute Delta Score and Votes Score from /granular-similarity results.

    Parameters
    ----------
    es_results : list of dicts
        Top-k results from es_client.granular_similarity(). Each must have:
          - overall_score       : float  (ES relevance score, any range)
          - quality_signal      : bool   (True = delta-flagged on CMV)
          - cluster_quality_max : float  (log-scaled votes proxy, 0-1)

    Returns
    -------
    {
        "delta_score":         float,   # 0.0-1.0  (formula weight: 0.20)
        "votes_score":         float,   # 0.0-1.0  (formula weight: 0.15)
        "delta_confidence":    float,   # how reliable the delta score is
        "votes_confidence":    float,   # how reliable the votes score is
        "delta_flagged_count": int,
        "total_matches":       int,
        "top_match_cluster_quality": float,
        "fallback_used":       bool,
    }
    """
    if not es_results:
        return {
            "delta_score":              _FALLBACK_DELTA,
            "votes_score":              _FALLBACK_VOTES,
            "delta_confidence":         0.0,
            "votes_confidence":         0.0,
            "delta_flagged_count":      0,
            "total_matches":            0,
            "top_match_cluster_quality": 0.0,
            "fallback_used":            True,
            "note": "No ES results — neutral fallback used.",
        }

    raw_scores = [r.get("overall_score", 0.0) for r in es_results]

    # FIX 1: Range-agnostic normalisation
    norm_sims = [_safe_norm(s, raw_scores) for s in raw_scores]
    total_sim = sum(norm_sims) or 1e-9

    # ── Delta Score ───────────────────────────────────────────────────────────
    # FIX 2: Cap each delta match's contribution at 1/k
    # Prevents one extremely strong delta match from dominating
    # and rewards consistent delta-alignment across multiple matches
    k = len(es_results)
    cap = 1.0 / k  # maximum contribution per match

    delta_contributions = []
    for i, r in enumerate(es_results):
        if r.get("quality_signal", False):
            # Contribution = normalised sim, capped at 1/k
            contrib = min(norm_sims[i] / total_sim, cap)
            delta_contributions.append(contrib)

    # Sum capped contributions, then rescale to 0-1
    raw_delta = sum(delta_contributions)
    delta_score = round(min(raw_delta * k, 1.0), 3)

    delta_flagged = sum(1 for r in es_results if r.get("quality_signal", False))

    # Delta confidence: how strong are the top matches overall?
    # High average sim = we can trust the delta signal
    avg_sim = sum(norm_sims) / len(norm_sims)
    delta_confidence = round(avg_sim, 3)

    # ── Votes Score ───────────────────────────────────────────────────────────
    # FIX 3: Aggregate across top-3 instead of just top-1
    top_results = es_results[:_VOTES_TOP_K]
    top_sims    = norm_sims[:_VOTES_TOP_K]
    top_sim_sum = sum(top_sims) or 1e-9

    cluster_qualities = [
        r.get("cluster_quality_max", 0.0) or 0.0
        for r in top_results
    ]

    # FIX 4: Similarity-weighted average of cluster quality across top-3
    votes_score = sum(
        top_sims[i] * cluster_qualities[i]
        for i in range(len(top_results))
    ) / top_sim_sum
    votes_score = round(min(max(votes_score, 0.0), 1.0), 3)

    # Votes confidence: driven by top match similarity
    # If best match sim < 0.40, cluster quality is unreliable
    top_sim = norm_sims[0] if norm_sims else 0.0
    votes_confidence = round(top_sim, 3)

    return {
        "delta_score":               delta_score,
        "votes_score":               votes_score,
        "delta_confidence":          delta_confidence,
        "votes_confidence":          votes_confidence,
        "delta_flagged_count":       delta_flagged,
        "total_matches":             len(es_results),
        "top_match_cluster_quality": round(cluster_qualities[0] if cluster_qualities else 0.0, 3),
        "fallback_used":             False,
    }