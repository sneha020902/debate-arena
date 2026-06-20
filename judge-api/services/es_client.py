"""
es_client.py — thin client for Rosen's Debate Arguments API (Elasticsearch).

Implements exactly the contract in TEAM_LOGIC_HANDOFF: /semantic-search,
/granular-similarity, /topic-cluster, /health. Every call degrades gracefully
— if the Webis VPN is down or the server is unreachable, the function returns
None (never raises), so the scoring layer can cleanly fall back to the Part 1
LLM/extrapolation path instead of crashing the judge.

Scores from the server are raw (1.0–2.0); callers normalise with
judge_config.es_norm().
"""

import requests

from services.judge_config import ES_API, ES_MIN_MATCH

_TIMEOUT = 8


def es_available() -> bool:
    """True if Rosen's API answers /health (i.e. VPN connected, server up)."""
    try:
        r = requests.get(f"{ES_API}/health", timeout=4)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def semantic_search(argument: str, top_n: int = 5,
                    quality_only: bool = False,
                    min_score: float = ES_MIN_MATCH):
    """
    POST /semantic-search — vector search over the CMV index.
    Returns the list of similar arguments (each with similarity_score,
    1.0–2.0) or None if the server is unreachable.
    """
    try:
        r = requests.post(
            f"{ES_API}/semantic-search",
            json={"argument": argument, "top_n": top_n,
                  "quality_only": quality_only, "min_score": min_score},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("similar_arguments", [])
    except Exception:
        return None


def granular_similarity(argument: str, claim: str = "", premises=None,
                        top_n: int = 5, min_score: float = ES_MIN_MATCH):
    """
    POST /granular-similarity — claim-level and premise-level match scores.
    Returns the `results` list (each with overall_score, claim_score,
    best_premise_score, cluster_member_count, ...) or None if unreachable.
    """
    payload = {"argument": argument, "top_n": top_n, "min_score": min_score}
    if claim:
        payload["claim"] = claim
    if premises:
        payload["premises"] = premises
    try:
        r = requests.post(f"{ES_API}/granular-similarity", json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return None


def topic_cluster(topic: str, quality_aggregation: str = "max"):
    """
    GET /topic-cluster — the broader reference set for a topic. Returns the
    full response dict (with argument_count, arguments[...]) or None.
    """
    try:
        r = requests.get(
            f"{ES_API}/topic-cluster",
            params={"topic": topic, "quality_aggregation": quality_aggregation},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
