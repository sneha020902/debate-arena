"""
unknown_arguments.py — Part 1 of the Debate Judge: scoring individual
arguments whose quality is unknown.

Flow per argument:
  1. Ask Elasticsearch (granular-similarity) for the best claim+premise match.
  2. If the best match clears the threshold (>= ES_MIN_MATCH on the 1.0–2.0
     scale) the argument is KNOWN — score it directly from the ES match.
  3. Otherwise it is UNKNOWN (no close reference) — resolve it with the
     configured strategy:
        - "llm"           : LLM-as-judge rubric (claim clarity, evidence,
                            coherence, specificity)
        - "extrapolation" : the team's own average quality so far (a prior)
        - "blend"         : BLEND_LLM_WEIGHT * llm + (1 - w) * extrapolation

Scoring is two-pass: first pass collects the KNOWN matches so the team prior
is real before any UNKNOWN argument is extrapolated from it.

When ES is unreachable (no VPN), every argument is UNKNOWN and the strategy
takes over — so the judge still works end-to-end offline, it just leans on the
LLM / prior instead of the corpus.
"""

import re
import requests

from services.judge_config import (
    OLLAMA_HOST, OLLAMA_MODEL, ES_MIN_MATCH, es_norm,
    DEFAULT_UNKNOWN_STRATEGY, BLEND_LLM_WEIGHT,
)
from services import es_client
from services import final_score as _final_score


# ── Claim / premise split (same heuristic Rosen uses in debate_arena) ─────────
def split_claim_premises(argument: str):
    """First substantive sentence = claim; next few = premises."""
    sentences = [s.strip() for s in argument.replace("—", ".").split(".")
                 if len(s.strip()) > 20]
    claim = sentences[0] if sentences else argument[:200]
    premises = sentences[1:4] if len(sentences) > 1 else []
    return claim, premises


# ── LLM-as-judge rubric (consistent with Rosen's llm_judge_quality) ───────────
def llm_judge_quality(argument: str,
                      ollama_host: str = OLLAMA_HOST,
                      model: str = OLLAMA_MODEL):
    """
    Independent quality estimate from an LLM when there is no ES match.
    Returns a float 0.0–1.0, or None if Ollama is unreachable.
    """
    prompt = f"""You are an expert debate judge. Score this argument from 0.0 to 1.0.

Criteria:
- Claim clarity: is there a clear, specific central assertion?
- Evidence: does it cite facts, data, or expert reasoning?
- Logical coherence: do the supporting points actually back the claim?
- Specificity: concrete details rather than vague generalities?

0.0 = very weak (no clear claim, no evidence, incoherent)
0.5 = average (clear claim, weak support)
1.0 = excellent (sharp claim, strong evidence, tight logic)

Argument:
\"{argument[:800]}\"

Reply with ONLY a decimal number between 0.00 and 1.00. Nothing else."""
    try:
        r = requests.post(
            f"{ollama_host}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"num_predict": 8, "temperature": 0.05}},
            timeout=45,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        m = re.search(r"(?:0?\.\d+|[01](?:\.\d+)?)", content)
        if m:
            return round(min(max(float(m.group()), 0.0), 1.0), 3)
    except Exception:
        pass
    return None


# ── KNOWN: score with the 4-metric pipeline when ES reference is found ────────
def _score_known(argument: str):
    """
    Returns a dict with quality/coverage (+ full score breakdown), or None if
    no ES match clears the threshold (=> caller treats the argument as UNKNOWN).

    When a reference IS found the quality field is now the weighted final score:
        final_score = 0.30 × NLI + 0.20 × Delta + 0.15 × Votes + 0.35 × ArgQuality
    Coverage is still derived from the ES cluster signal (unchanged).

    The fallback path (_resolve_unknown / score_team_arguments) is NOT touched.
    """
    claim, premises = split_claim_premises(argument)
    results = es_client.granular_similarity(argument, claim, premises, top_n=5)
    if not results:                       # ES down OR no results at all
        return None

    best = results[0]
    best_overall = best.get("overall_score", 1.0)
    if best_overall < ES_MIN_MATCH:       # below "good match" => treat as unknown
        return None

    # ── Coverage (unchanged): corpus support signal ───────────────────────────
    support = best.get("cluster_member_count", 1) or 1
    support_norm = min(support / 6.0, 1.0)
    coverage = round(0.6 * es_norm(best_overall) + 0.4 * support_norm, 3)

    # ── Quality: run the full 4-score pipeline (NLI, Delta, Votes, ArgQuality) ─
    score_detail = _final_score.compute_full_score(
        argument, results, claim=claim, premises=premises)
    quality = score_detail["final_score"]

    return {
        "quality":  quality,
        "coverage": coverage,
        "source":   "es_match",
        "matched":  True,
        # pass-through for explainability and transparency
        "best_overall":       round(best_overall, 3),
        "matched_claim":      best.get("claim", ""),
        "score_breakdown":    score_detail,
    }


def _resolve_unknown(argument: str, team_prior: dict, strategy: str,
                     blend_llm_weight: float):
    """Score an argument that had no ES match, using the chosen strategy."""
    prior_q = team_prior.get("quality")
    prior_c = team_prior.get("coverage", 0.30)

    if strategy == "extrapolation":
        q = prior_q if prior_q is not None else 0.45
        return {"quality": round(q, 3), "coverage": round(prior_c, 3),
                "source": "extrapolation", "matched": False}

    if strategy == "llm":
        llm = llm_judge_quality(argument)
        q = llm if llm is not None else (prior_q if prior_q is not None else 0.45)
        src = "llm" if llm is not None else "extrapolation_fallback"
        return {"quality": round(q, 3), "coverage": round(prior_c, 3),
                "source": src, "matched": False}

    # blend (default)
    llm = llm_judge_quality(argument)
    if llm is None:
        q = prior_q if prior_q is not None else 0.45
        return {"quality": round(q, 3), "coverage": round(prior_c, 3),
                "source": "extrapolation_fallback", "matched": False}
    if prior_q is None:
        return {"quality": round(llm, 3), "coverage": round(prior_c, 3),
                "source": "llm", "matched": False}
    q = blend_llm_weight * llm + (1 - blend_llm_weight) * prior_q
    return {"quality": round(q, 3), "coverage": round(prior_c, 3),
            "source": "blend", "matched": False,
            "llm_component": round(llm, 3), "prior_component": round(prior_q, 3)}


def score_team_arguments(team_arguments: list, strategy: str = None,
                         blend_llm_weight: float = None) -> dict:
    """
    Score every argument a team made, mixing KNOWN (ES) and UNKNOWN (strategy).

    team_arguments: list of {"turn": int, "argument": str}
    Returns per-argument scores (with provenance) + team aggregates.
    """
    strategy = strategy or DEFAULT_UNKNOWN_STRATEGY
    blend_llm_weight = BLEND_LLM_WEIGHT if blend_llm_weight is None else blend_llm_weight

    # Pass 1 — try ES on everything; remember which were KNOWN.
    prelim = []
    known_q, known_c = [], []
    for t in team_arguments:
        known = _score_known(t["argument"])
        prelim.append((t, known))
        if known:
            known_q.append(known["quality"])
            known_c.append(known["coverage"])

    team_prior = {
        "quality": round(sum(known_q) / len(known_q), 3) if known_q else None,
        "coverage": round(sum(known_c) / len(known_c), 3) if known_c else 0.30,
    }

    # Pass 2 — resolve UNKNOWN arguments against the (now real) prior.
    per_argument = []
    for t, known in prelim:
        scored = known if known else _resolve_unknown(
            t["argument"], team_prior, strategy, blend_llm_weight)
        per_argument.append({
            "turn": t["turn"],
            "argument_snippet": (t["argument"][:120] + "...")
                                if len(t["argument"]) > 120 else t["argument"],
            **scored,
        })

    n = len(per_argument)
    mean_quality = round(sum(a["quality"] for a in per_argument) / n, 3) if n else 0.0
    mean_coverage = round(sum(a["coverage"] for a in per_argument) / n, 3) if n else 0.0
    return {
        "per_argument": per_argument,
        "mean_quality": mean_quality,
        "mean_coverage": mean_coverage,
        "n_arguments": n,
        "n_known": sum(1 for a in per_argument if a["matched"]),
        "n_unknown": sum(1 for a in per_argument if not a["matched"]),
        "strategy_used": strategy,
        "team_prior": team_prior,
    }
