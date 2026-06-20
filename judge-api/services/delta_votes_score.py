"""
delta_votes_score.py — Delta and Votes scorers for the Debate Judge.

Both scores are computed directly from the Elasticsearch reference results
returned by /granular-similarity (Rosen's API). No LLM call is required.

Delta Score (0.0–1.0)
    Measures similarity to arguments that successfully changed someone's view
    in Reddit r/ChangeMyView discussions. The ES index marks these with
    quality_signal=True (delta-flagged by DeltaBot).

    Formula:
        Σ (norm_sim_i × quality_signal_i)          <- weighted delta mass
      ──────────────────────────────────────────   <- normalised by total mass
              Σ norm_sim_i

    A result of 1.0 means every matched reference argument was delta-flagged
    and matched strongly; 0.0 means no delta-flagged argument was found in
    the top matches.

Votes Score (0.0–1.0)
    Measures community approval via the upvote signal of the most similar
    reference argument's cluster. Rosen's index stores cluster_quality_max
    (highest votes_normalized in the argument cluster), which is already
    log-scaled. We use the best-matching result's cluster_quality_max as a
    conservative upper-bound proxy for the votes signal.

    cluster_quality_max is on a 0–1 scale (per TEAM_LOGIC_HANDOFF), so no
    further normalisation is needed.

Fallback
    If es_results is empty or None: both scores return 0.0 with a note.
    The final_score module handles this gracefully.
"""


def score_delta_votes(es_results: list) -> dict:
    """
    Compute Delta Score and Votes Score from /granular-similarity results.

    Parameters
    ----------
    es_results : list of dicts
        Top-k results from es_client.granular_similarity(), each containing
        at minimum: overall_score, quality_signal (bool), cluster_quality_max.

    Returns
    -------
    {
        "delta_score":         float,   # 0.0–1.0  (formula weight: 0.20)
        "votes_score":         float,   # 0.0–1.0  (formula weight: 0.15)
        "delta_flagged_count": int,     # how many top matches are delta-flagged
        "total_matches":       int,
        "top_match_cluster_quality": float,
    }
    """
    if not es_results:
        return {
            "delta_score": 0.0,
            "votes_score": 0.0,
            "delta_flagged_count": 0,
            "total_matches": 0,
            "top_match_cluster_quality": 0.0,
            "note": "No ES results — delta/votes computed as 0.",
        }

    # Normalise raw ES scores (1.0–2.0) → 0.0–1.0
    def _norm(s):
        return round(min(max(s - 1.0, 0.0), 1.0), 4)

    norm_sims = [_norm(r.get("overall_score", 1.0)) for r in es_results]
    total_sim = sum(norm_sims) or 1e-9   # avoid div/0

    # Delta Score: similarity-weighted fraction of delta-flagged matches
    delta_mass = sum(
        norm_sims[i]
        for i, r in enumerate(es_results)
        if r.get("quality_signal", False)
    )
    delta_score = round(delta_mass / total_sim, 3)
    delta_flagged = sum(1 for r in es_results if r.get("quality_signal", False))

    # Votes Score: cluster_quality_max of the best-matching result (log-scaled, 0–1)
    # cluster_quality_max = max votes_normalized in the argument cluster (HANDOFF p.7)
    top_cluster_q = es_results[0].get("cluster_quality_max", 0.0) or 0.0
    votes_score = round(min(max(top_cluster_q, 0.0), 1.0), 3)

    return {
        "delta_score":              delta_score,
        "votes_score":              votes_score,
        "delta_flagged_count":      delta_flagged,
        "total_matches":            len(es_results),
        "top_match_cluster_quality": round(top_cluster_q, 3),
    }
