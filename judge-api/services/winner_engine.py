"""
winner_engine.py — Part 3 of the Debate Judge: winner determination.

This is the convergence point. It pulls together:
  • Part 1 (unknown_arguments.py) → per-team argument QUALITY and corpus COVERAGE
  • Part 2 (services.debate_judge, Pavan) → ENGAGEMENT, REBUTTAL COVERAGE,
        NEW-POINT balance, RESPONSE QUALITY, INFORMATION DENSITY
  • Emotion × Logic (Sneha's API) → a DELIVERY/composure signal, used as a
        configurable tie-breaker (or a full dimension)

…into a single configurable composite score per team, declares a winner with a
margin of victory, and — crucially — explains WHY, by showing each team's top
contributing components and where it lost points.

Every component is on a 0–1 scale before weighting, and the weights are
renormalised to sum to 1.0, so a human can read the breakdown directly.
"""

import requests

from services.judge_config import (
    EMOTION_API, DEFAULT_WEIGHTS, DELIVERY_AS_DIMENSION, DELIVERY_WEIGHT,
    TIE_EPSILON, TOKEN_BUDGET,
)
from services import unknown_arguments
from services.debate_judge import (
    compute_engagement,
    compute_rebuttal_coverage,
    compute_new_point_detection,
    compute_response_quality,
    compute_information_density,
)

# Human-readable labels for the explanation text.
_LABELS = {
    "quality":             "argument quality",
    "coverage":            "corpus coverage",
    "engagement":          "engagement with the opponent",
    "rebuttal_coverage":   "rebutting the opponent's points",
    "information_density": "fresh information per turn",
    "response_quality":    "substantive responses",
    "new_point_balance":   "balance of new vs reactive points",
    "delivery":            "delivery / composure",
}


# ── Emotion Track delivery (optional, graceful fallback) ──────────────────────
def _delivery_signal(argument: str, speaker: str):
    """Composure (1 - intensity) from Sneha's API; neutral 0.6 if unreachable."""
    try:
        r = requests.post(f"{EMOTION_API}/emotion/delivery",
                          json={"text": argument, "speaker_id": speaker}, timeout=20)
        r.raise_for_status()
        dv = r.json().get("delivery_vector", {})
        return round(1.0 - dv.get("intensity", 0.4), 3)
    except Exception:
        return 0.60


def _new_point_balance(team_npd: dict) -> float:
    """
    A balanced debater both rebuts AND advances new points. Reward 'mixed'
    arguments and an even original/reactive split; punish pure monologue or
    pure rebuttal. Range 0–1.
    """
    mixed = team_npd.get("mixed_ratio", 0.0)
    orig = team_npd.get("original_ratio", 0.0)
    react = team_npd.get("reactive_ratio", 0.0)
    return round(0.5 * mixed + 0.5 * (1.0 - abs(orig - react)), 3)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~0.75 words/token) for the 500-token budget note."""
    return int(len(text.split()) / 0.75)


def _normalize_turns(turns: list) -> list:
    """
    Defensive boundary normalisation so a slightly-off payload never crashes
    Pavan's Part-2 functions with KeyError. Accepts 'argument' OR 'text'/
    'content'/'message', fills the 'turn' index in order, and requires a
    'speaker'. (The API router already does this via the Turn model; this
    protects direct callers and tests too.)
    """
    norm = []
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict):
            raise ValueError(f"Turn {i} is not an object: {t!r}")
        arg = (t.get("argument") or t.get("text") or t.get("content")
               or t.get("message"))
        speaker = t.get("speaker") or t.get("team") or t.get("speaker_id")
        if arg is None or speaker is None:
            raise ValueError(
                f"Turn {i} needs a speaker and an argument/text field; got keys {list(t.keys())}")
        norm.append({"turn": t.get("turn", i), "speaker": speaker, "argument": arg})
    return norm


def judge_debate(topic: str, turns: list, weights: dict = None,
                 strategy: str = None, blend_llm_weight: float = None,
                 use_delivery: bool = True,
                 delivery_signals=None, deterministic_components=None) -> dict:
    """
    Run the full Part 1 + Part 2 + Part 3 pipeline and declare a winner.

    turns: [{"turn": int, "speaker": str, "argument": str}]
    weights: optional override of DEFAULT_WEIGHTS (renormalised to sum 1.0)
    strategy / blend_llm_weight: Part 1 unknown-argument behaviour
    use_delivery: include the Emotion-track tie-breaker
    delivery_signals / deterministic_components: test hooks (inject scores to
        run fully offline without Ollama / ES / the Emotion API)
    """
    turns = _normalize_turns(turns)
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}
    a, b = speakers[0], speakers[1]

    weights = dict(weights or DEFAULT_WEIGHTS)
    if use_delivery and DELIVERY_AS_DIMENSION:
        weights["delivery"] = weights.get("delivery", DELIVERY_WEIGHT)
    wsum = sum(v for v in weights.values() if v > 0) or 1.0
    norm_w = {k: v / wsum for k, v in weights.items() if v > 0}

    # ── Part 2 (Pavan): the five debate-level signals, computed once ──────────
    if deterministic_components is not None:
        comp = deterministic_components            # injected for offline tests
    else:
        eng = compute_engagement(turns)
        rc = compute_rebuttal_coverage(turns)
        npd = compute_new_point_detection(turns)
        resp = compute_response_quality(turns)

        # Use pre-computed info_density from the arena when all turns carry it.
        # The arena scores lexical novelty per-turn during generation (no extra
        # LLM call). When those values arrive here we average them per speaker
        # and skip compute_information_density's LLM pass entirely.
        precomputed_density: dict = {sp: [] for sp in (a, b)}
        for t in turns:
            if "info_density" in t and t["speaker"] in precomputed_density:
                precomputed_density[t["speaker"]].append(t["info_density"])

        sp_turn_counts = {sp: sum(1 for t in turns if t["speaker"] == sp) for sp in (a, b)}
        all_precomputed = all(
            len(precomputed_density[sp]) == sp_turn_counts[sp] and sp_turn_counts[sp] > 0
            for sp in (a, b)
        )

        if all_precomputed:
            info = {
                sp: {"average_new_information": round(
                    sum(precomputed_density[sp]) / len(precomputed_density[sp]), 3)}
                for sp in (a, b)
            }
        else:
            info = compute_information_density(turns)

        # IMPORTANT mapping: in Pavan's /rebuttal-coverage, team X's
        # coverage_score is the fraction of X's OWN arguments that the opponent
        # answered (defence). For a team's OFFENSIVE rebuttal coverage — "what
        # fraction of the OPPONENT's arguments did this team rebut?", which is
        # what Part 3 rewards — we read the OPPONENT's coverage_score.
        comp = {
            a: {
                "engagement":          eng[a]["engagement_ratio"],
                "rebuttal_coverage":   rc[b]["coverage_score"],
                "information_density": info[a]["average_new_information"],
                "response_quality":    resp[a]["average_quality"],
                "new_point_balance":   _new_point_balance(npd[a]),
            },
            b: {
                "engagement":          eng[b]["engagement_ratio"],
                "rebuttal_coverage":   rc[a]["coverage_score"],
                "information_density": info[b]["average_new_information"],
                "response_quality":    resp[b]["average_quality"],
                "new_point_balance":   _new_point_balance(npd[b]),
            },
        }

    # ── Part 1 (this module): argument-level quality + coverage per team ──────
    teams = {}
    for sp in (a, b):
        sp_turns = [{"turn": t["turn"], "argument": t["argument"]}
                    for t in turns if t["speaker"] == sp]
        p1 = unknown_arguments.score_team_arguments(sp_turns, strategy, blend_llm_weight)

        # Delivery (mean composure across the team's turns)
        if delivery_signals is not None:
            delivery = delivery_signals.get(sp, 0.6)
        elif use_delivery:
            sigs = [_delivery_signal(t["argument"], sp) for t in turns if t["speaker"] == sp]
            delivery = round(sum(sigs) / len(sigs), 3) if sigs else 0.6
        else:
            delivery = None

        components = {
            "quality":            p1["mean_quality"],
            "coverage":           p1["mean_coverage"],
            "engagement":         comp[sp]["engagement"],
            "rebuttal_coverage":  comp[sp]["rebuttal_coverage"],
            "information_density":comp[sp]["information_density"],
            "response_quality":   comp[sp]["response_quality"],
            "new_point_balance":  comp[sp]["new_point_balance"],
        }
        if use_delivery and DELIVERY_AS_DIMENSION and delivery is not None:
            components["delivery"] = delivery

        # Weighted contributions (only components that have a weight)
        contributions = {k: round(norm_w[k] * components[k], 4)
                         for k in components if k in norm_w}
        composite = round(sum(contributions.values()), 3)

        # 500-token budget audit
        over_budget = [t["turn"] for t in turns
                       if t["speaker"] == sp and _estimate_tokens(t["argument"]) > TOKEN_BUDGET]

        teams[sp] = {
            "composite_score": composite,
            "components": {k: round(v, 3) for k, v in components.items()},
            "contributions": contributions,
            "new_point_balance": comp[sp]["new_point_balance"],
            "delivery": delivery,
            "argument_count": p1["n_arguments"],
            "arguments_known": p1["n_known"],
            "arguments_unknown": p1["n_unknown"],
            "strategy_used": p1["strategy_used"],
            "over_token_budget_turns": over_budget,
            "argument_detail": p1["per_argument"],
        }

    # ── Winner + margin (+ delivery tie-breaker) ──────────────────────────────
    ca, cb = teams[a]["composite_score"], teams[b]["composite_score"]
    margin = round(abs(ca - cb), 3)
    tiebreak_note = None
    if margin >= TIE_EPSILON:
        winner = a if ca > cb else b
    else:
        # Effective tie on the weighted sum → break with delivery if available
        da = teams[a]["delivery"] if teams[a]["delivery"] is not None else 0.6
        db = teams[b]["delivery"] if teams[b]["delivery"] is not None else 0.6
        if abs(da - db) < 1e-6:
            winner = a if ca >= cb else b
            tiebreak_note = "Scores effectively tied; delivery also level — edge to the higher composite."
        else:
            winner = a if da > db else b
            tiebreak_note = (f"Composite scores within {TIE_EPSILON} ({ca} vs {cb}); "
                             f"tie broken on delivery/composure ({da} vs {db}).")
    loser = b if winner == a else a

    # ── Explainability ────────────────────────────────────────────────────────
    def _explain(sp):
        contribs = teams[sp]["contributions"]
        comps = teams[sp]["components"]
        ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        top = [(_LABELS.get(k, k), comps[k], v) for k, v in ranked[:2]]
        weak = sorted(comps.items(), key=lambda kv: kv[1])[:2]
        return {
            "carried_by": [{"component": lbl, "value": val, "points_added": pts}
                           for lbl, val, pts in top],
            "lost_points_on": [{"component": _LABELS.get(k, k), "value": v} for k, v in weak],
        }

    w_top = _explain(winner)["carried_by"]
    l_weak = _explain(loser)["lost_points_on"]
    verdict = (
        f"{winner} wins {teams[winner]['composite_score']} to "
        f"{teams[loser]['composite_score']} (margin {margin}). "
        f"{winner} was carried by "
        + " and ".join(f"{c['component']} ({c['value']})" for c in w_top)
        + f". {loser} lost ground on "
        + " and ".join(f"{c['component']} ({c['value']})" for c in l_weak)
        + "."
    )
    if tiebreak_note:
        verdict += " " + tiebreak_note

    return {
        "topic": topic,
        "winner": winner,
        "margin": margin,
        "verdict": verdict,
        "explanation": verdict,
        "weights_used": {k: round(v, 3) for k, v in norm_w.items()},
        "teams": {
            a: {**teams[a], "explanation": _explain(a)},
            b: {**teams[b], "explanation": _explain(b)},
        },
        "summary": {
            "total_turns": len(turns),
            "delivery_used": bool(use_delivery),
            "delivery_mode": "dimension" if DELIVERY_AS_DIMENSION else "tiebreaker",
        },
    }
