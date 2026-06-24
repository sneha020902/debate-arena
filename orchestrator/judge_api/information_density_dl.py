"""
information_density.py — Debate-level information density scorer.

Responsibility
--------------
Computes information_density for each speaker as a debate-level score.
This is one of the four debate-level components in the composite formula:

    Debate_Level_Score =
        0.30 * rebuttal_effectiveness
        0.27 * argument_quality
        0.25 * engagement_score
        0.18 * information_density       <- this file

What information_density measures
-----------------------------------
How much genuinely new reasoning a speaker introduces across the debate.
Penalises repetition and rephrasing of already-made points.
Rewards opening multiple distinct lines of reasoning.

    information_density = 0.60 * novelty_score
                        + 0.40 * breadth_score

novelty_score
    Mean per-turn new_information_score across all the speaker's arguments.
    Measures how much each argument adds beyond what the team already said.
    Turn 1 correctly scores 1.0 (no prior same-team arguments — everything
    is new) as a natural result of the formula, not a hardcoded default.

breadth_score
    How many DISTINCT lines of reasoning the speaker opened.
    Computed by clustering the speaker's argument embeddings.
    If 5 arguments cluster into 3 semantic groups → breadth = 3/5 = 0.60.
    Rewards teams that consistently open new fronts rather than elaborating
    on a single theme.
    Normalised to 0-1: breadth = n_clusters / n_arguments.

Why these two components
-------------------------
novelty_score alone rewards consistently novel content but does not
distinguish between a team that opens 5 genuinely different topics (high
breadth) and one that opens 5 minor variations of the same theme
(low breadth but each step slightly novel).

breadth_score captures the strategic dimension: are they exploring the
debate space or drilling deep into a single lane? Both are valid strategies
but the balance matters for debate-level evaluation.

Per-turn novelty pipeline
--------------------------
Step 1: Embed current argument with SentenceTransformer
Step 2: Compute cosine similarity against EACH prior same-team argument
Step 3: max_sim = highest similarity to any single prior argument
        most_similar_prior = the prior argument with that max similarity

Step 4: Threshold routing (avoids LLM for clear-cut cases):
    max_sim >= REPETITIVE_THRESHOLD (0.88) -> obviously repetitive
        new_information_score = 1 - max_sim   (no LLM call)
    max_sim <= NEW_THRESHOLD (0.15)        -> obviously new
        new_information_score = 1 - max_sim   (no LLM call)
    else (0.15 < max_sim < 0.88)           -> ambiguous, call LLM

Step 5: LLM receives ONLY the current argument + ONE most-similar prior
    (not all prior arguments — prevents prompt size from growing
     unboundedly as the debate progresses)

Step 6: Combine similarity signal and LLM score
    final_score = llm_weight * llm_score + (1 - llm_weight) * (1 - max_sim)
    llm_weight = llm_confidence (high confidence LLM → trust LLM more)

Breadth computation
--------------------
Uses agglomerative clustering on the speaker's argument embeddings with
a similarity threshold: two arguments in the same cluster if their
cosine similarity >= CLUSTER_THRESHOLD (0.65).

n_clusters computed by greedy assignment:
    For each argument (sorted by turn), assign to existing cluster if
    similarity to cluster centroid >= threshold, else create new cluster.

Fallback
---------
If Ollama unreachable: similarity score alone used (1 - max_sim).
If SentenceTransformer unavailable: LLM-only mode for novelty,
breadth_score defaults to 0.5 (unknown).

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
        "sentence-transformers not installed — similarity routing disabled. "
        "Install: pip install sentence-transformers"
    )

# ── Constants ─────────────────────────────────────────────────────────────────

# Weights for information_density formula
_W_NOVELTY  = 0.60
_W_BREADTH  = 0.40

# Similarity thresholds for LLM routing
REPETITIVE_THRESHOLD = 0.88   # above: clearly repetitive, skip LLM
NEW_THRESHOLD        = 0.15   # below: clearly new, skip LLM

# Breadth clustering threshold
CLUSTER_THRESHOLD    = 0.65   # arguments with sim >= this share a cluster

# Embedding model
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model      = None
_embed_cache: dict[str, "np.ndarray"] = {}   # argument text -> embedding


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _get_embed_model():
    global _embed_model
    if _embed_model is None and _ST_AVAILABLE:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _embed(text: str) -> Optional["np.ndarray"]:
    """
    Embed text, using in-memory cache to avoid re-embedding the same argument.
    Returns None if sentence-transformers unavailable.
    """
    if text in _embed_cache:
        return _embed_cache[text]
    model = _get_embed_model()
    if model is None:
        return None
    emb = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    _embed_cache[text] = emb
    return emb


def _cosine(emb_a: "np.ndarray", emb_b: "np.ndarray") -> float:
    return float(np.dot(emb_a, emb_b))


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ── Similarity routing ────────────────────────────────────────────────────────

def _max_similarity_to_prior(
    current_emb:  "np.ndarray",
    prior_embs:   list,
    prior_turns:  list,
) -> tuple[float, Optional[dict]]:
    """
    Find the maximum cosine similarity between current argument and any
    prior same-team argument.

    Returns (max_sim, most_similar_prior_turn) or (0.0, None) if no priors.
    """
    if not prior_embs:
        return 0.0, None

    best_sim   = 0.0
    best_prior = None

    for emb, turn in zip(prior_embs, prior_turns):
        sim = _cosine(current_emb, emb)
        if sim > best_sim:
            best_sim   = sim
            best_prior = turn

    return round(best_sim, 4), best_prior


# ── LLM novelty scorer ────────────────────────────────────────────────────────

def _score_novelty_llm(
    current_argument: str,
    most_similar_prior: str,
    max_sim: float
) -> dict:
    """
    Ask LLM how much new information the current argument adds compared to
    the SINGLE most similar prior argument (not all priors).

    This fixes the unbounded prompt growth problem: instead of concatenating
    all prior arguments into the prompt, we only send the one the current
    argument most resembles. If the current argument is novel compared to
    its closest match, it is novel compared to all prior arguments.

    Returns
    -------
    {
        "new_information_score": float,   0.0-1.0
        "confidence":            float,
        "reasoning":             str,
        "source":                str,     "llm" | "similarity_fallback"
    }
    """
    prompt = f"""You are a debate analyst assessing argument novelty.

Most similar prior argument from this team:
\"{most_similar_prior[:400]}\"

Current argument from the SAME team:
\"{current_argument[:400]}\"

Rate how much NEW information the current argument introduces compared to
the prior argument shown above.

SCORING GUIDE:
0.0-0.2 : Nearly identical — rephrases or repeats the same point
0.3-0.4 : Mostly overlapping — adds minor detail but same core claim
0.5-0.6 : Partial novelty — same theme but meaningfully different angle
0.7-0.8 : Substantially new — opens a different line of reasoning
0.9-1.0 : Entirely new ground — completely different claim or evidence

Note: similarity={max_sim:.2f} has already been detected. Let this guide
your rating but derive your score independently from the actual content.

Reply with ONLY this JSON (no markdown):
{{"new_information_score": <0.0-1.0>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""

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

        score = float(result.get("new_information_score", 1.0 - max_sim))
        score = round(max(0.0, min(1.0, score)), 3)

        confidence = float(result.get("confidence", 0.5))
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        # Blend LLM score with similarity signal weighted by LLM confidence
        # High confidence LLM → trust LLM score more
        # Low confidence LLM  → fall back toward 1 - max_sim
        blended = round(
            confidence * score + (1 - confidence) * (1.0 - max_sim),
            3
        )

        return {
            "new_information_score": blended,
            "confidence":            confidence,
            "reasoning":             str(result.get("reasoning", "No reasoning provided.")),
            "source":                "llm",
        }

    except Exception as e:
        logger.warning(f"Ollama unavailable for novelty scoring: {e}")
        return {
            "new_information_score": round(1.0 - max_sim, 3),
            "confidence":            None,
            "reasoning":             "Ollama unavailable — novelty from similarity inversion.",
            "source":                "similarity_fallback",
        }


# ── Per-turn novelty scorer ───────────────────────────────────────────────────

def _score_turn_novelty(
    current_turn:  dict,
    prior_turns:   list,
    prior_embs:    list
) -> dict:
    """
    Score how much new information one turn introduces compared to all
    prior same-team turns.

    Pipeline:
        1. Embed current argument (cached)
        2. Find max similarity to any prior
        3. Route: obvious cases skip LLM, ambiguous cases use LLM
        4. Return per-turn result with full provenance
    """
    current_arg = current_turn["argument"]
    snippet     = (
        current_arg[:120] + "..."
        if len(current_arg) > 120 else current_arg
    )

    # Embed current argument
    current_emb = _embed(current_arg)

    # No prior arguments — first turn is always 1.0 (mathematically correct)
    if not prior_turns:
        return {
            "turn":                  current_turn["turn"],
            "argument_snippet":      snippet,
            "new_information_score": 1.0,
            "max_prior_similarity":  0.0,
            "most_similar_turn":     None,
            "confidence":            1.0,
            "reasoning":             "First argument from this speaker — all content is new.",
            "source":                "no_prior",
        }

    # Similarity unavailable — use LLM with concatenated priors (graceful degradation)
    if current_emb is None:
        prior_text = "\n".join(
            f"- {t['argument'][:200]}" for t in prior_turns[-3:]
        )   # limit to last 3 to bound prompt size
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"""You are a debate analyst.

Prior arguments from this team (last {min(3, len(prior_turns))}):
{prior_text}

Current argument:
\"{current_arg[:400]}\"

Rate how much NEW information the current argument introduces (0.0-1.0).
Reply ONLY: {{"new_information_score": <0.0-1.0>, "confidence": <0.0-1.0>, "reasoning": "<sentence>"}}""",
                    "stream": False,
                    "options": {"num_predict": 100, "temperature": 0.0},
                },
                timeout=45,
            )
            raw    = r.json().get("response", "")
            result = _extract_json(raw)
            score  = round(max(0.0, min(1.0, float(result.get("new_information_score", 0.5)))), 3)
            conf   = round(max(0.0, min(1.0, float(result.get("confidence", 0.5)))), 3)
        except Exception:
            score, conf = 0.5, None

        return {
            "turn":                  current_turn["turn"],
            "argument_snippet":      snippet,
            "new_information_score": score,
            "max_prior_similarity":  None,
            "most_similar_turn":     None,
            "confidence":            conf,
            "reasoning":             "No embeddings — LLM-only mode.",
            "source":                "llm_no_embeddings",
        }

    # Find max similarity to any prior argument
    max_sim, best_prior = _max_similarity_to_prior(
        current_emb, prior_embs, prior_turns
    )

    # Route based on thresholds
    if max_sim >= REPETITIVE_THRESHOLD:
        # Obviously repetitive — skip LLM
        score = round(1.0 - max_sim, 3)
        return {
            "turn":                  current_turn["turn"],
            "argument_snippet":      snippet,
            "new_information_score": score,
            "max_prior_similarity":  max_sim,
            "most_similar_turn":     best_prior["turn"] if best_prior else None,
            "confidence":            None,
            "reasoning":             f"High similarity ({max_sim:.2f}) to turn {best_prior['turn'] if best_prior else '?'} — clearly repetitive.",
            "source":                "similarity_repetitive",
        }

    if max_sim <= NEW_THRESHOLD:
        # Obviously new — skip LLM
        score = round(1.0 - max_sim, 3)
        return {
            "turn":                  current_turn["turn"],
            "argument_snippet":      snippet,
            "new_information_score": score,
            "max_prior_similarity":  max_sim,
            "most_similar_turn":     None,
            "confidence":            None,
            "reasoning":             f"Low similarity ({max_sim:.2f}) to all prior arguments — clearly new ground.",
            "source":                "similarity_new",
        }

    # Ambiguous range — call LLM with only the most similar prior
    llm_result = _score_novelty_llm(
        current_argument=current_arg,
        most_similar_prior=best_prior["argument"] if best_prior else "",
        max_sim=max_sim,
    )

    return {
        "turn":                  current_turn["turn"],
        "argument_snippet":      snippet,
        "new_information_score": llm_result["new_information_score"],
        "max_prior_similarity":  max_sim,
        "most_similar_turn":     best_prior["turn"] if best_prior else None,
        "confidence":            llm_result["confidence"],
        "reasoning":             llm_result["reasoning"],
        "source":                llm_result["source"],
    }


# ── Breadth scorer ────────────────────────────────────────────────────────────

def _compute_breadth_score(
    speaker_turns: list,
    embs:          list,
) -> dict:
    """
    Measure how many DISTINCT lines of reasoning the speaker opened.

    Uses greedy similarity-based clustering:
        For each argument in turn order, check similarity to each
        existing cluster centroid. Assign to the closest cluster if
        similarity >= CLUSTER_THRESHOLD, otherwise create a new cluster.

    breadth_score = n_clusters / n_arguments

    This rewards teams that consistently open new fronts.
    A team with 5 arguments in 5 distinct clusters scores 1.0.
    A team with 5 arguments all on the same theme scores close to 0.2.

    Returns
    -------
    {
        "breadth_score":    float,
        "n_clusters":       int,
        "n_arguments":      int,
        "cluster_labels":   list,   which cluster each argument belongs to
    }
    """
    n = len(speaker_turns)
    if n == 0:
        return {"breadth_score": 0.0, "n_clusters": 0,
                "n_arguments": 0, "cluster_labels": []}

    if not embs or len(embs) != n:
        # No embeddings available — breadth unknown
        return {"breadth_score": 0.5, "n_clusters": None,
                "n_arguments": n, "cluster_labels": None}

    # Greedy clustering
    centroids     = []    # list of np.ndarray — one per cluster
    cluster_sizes = []    # number of members per cluster
    labels        = []    # cluster index for each argument

    for i, emb in enumerate(embs):
        if not centroids:
            # First argument starts cluster 0
            centroids.append(emb.copy())
            cluster_sizes.append(1)
            labels.append(0)
            continue

        # Find best matching centroid
        sims = [_cosine(emb, c) for c in centroids]
        best_idx  = int(np.argmax(sims))
        best_sim  = sims[best_idx]

        if best_sim >= CLUSTER_THRESHOLD:
            # Assign to existing cluster — update centroid (running mean)
            n_members = cluster_sizes[best_idx]
            centroids[best_idx] = (
                centroids[best_idx] * n_members + emb
            ) / (n_members + 1)
            # Re-normalise centroid to unit length
            norm = np.linalg.norm(centroids[best_idx])
            if norm > 0:
                centroids[best_idx] /= norm
            cluster_sizes[best_idx] += 1
            labels.append(best_idx)
        else:
            # New cluster
            centroids.append(emb.copy())
            cluster_sizes.append(1)
            labels.append(len(centroids) - 1)

    n_clusters    = len(centroids)
    breadth_score = round(n_clusters / n, 3)

    return {
        "breadth_score":  breadth_score,
        "n_clusters":     n_clusters,
        "n_arguments":    n,
        "cluster_labels": labels,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def compute_information_density(
    turns:       list,
    w_novelty:   float = _W_NOVELTY,
    w_breadth:   float = _W_BREADTH,
) -> dict:
    """
    Compute information_density for both speakers across a full debate.

    Parameters
    ----------
    turns : list of dicts, each with:
        {
            "turn":      int,     debate turn number (1-indexed)
            "speaker":   str,     speaker identifier
            "argument":  str,     full argument text
        }
        NOTE: Unlike other scorers, information_density includes ALL turns
        (no first-turn exclusion). The first turn scoring 1.0 is mathematically
        correct — the first argument IS completely novel relative to prior
        same-team arguments (there are none).

    ollama_host : str    Ollama base URL
    model       : str    Ollama model name
    w_novelty   : float  Weight for novelty_score (default 0.60)
    w_breadth   : float  Weight for breadth_score (default 0.40)

    Returns
    -------
    {
        "<speaker_a>": {
            "information_density":      float,   0.0-1.0  (composite formula)
            "novelty_score":            float,   mean per-turn novelty
            "breadth_score":            float,   distinct lines / total args
            "n_clusters":               int,     number of distinct topics
            "n_arguments":              int,
            "per_argument":             list,    per-turn details
        },
        "<speaker_b>": { ... },
        "summary": {
            "total_turns":              int,
            "repetitive_threshold":     float,
            "new_threshold":            float,
            "cluster_threshold":        float,
            "weights":                  dict,
            "embed_model_available":    bool,
        }
    }
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers."}

    # ── Score each speaker ────────────────────────────────────────────────────
    output = {}

    for speaker in speakers:
        sp_turns = [t for t in turns if t["speaker"] == speaker]

        per_argument  = []
        prior_turns   = []
        prior_embs    = []

        for turn in sp_turns:
            # Score novelty against all prior same-team turns
            result = _score_turn_novelty(
                current_turn=turn,
                prior_turns=prior_turns,
                prior_embs=prior_embs,
            )
            per_argument.append(result)

            # Update rolling prior lists
            prior_turns.append(turn)
            emb = _embed(turn["argument"])
            if emb is not None:
                prior_embs.append(emb)

        # Novelty score: mean of all per-turn scores
        n = len(per_argument)
        novelty_score = round(
            sum(a["new_information_score"] for a in per_argument) / n, 3
        ) if n > 0 else 0.0

        # Breadth score: distinct clusters / total arguments
        all_embs = [_embed(t["argument"]) for t in sp_turns]
        valid_embs = [e for e in all_embs if e is not None]

        breadth_result = _compute_breadth_score(
            speaker_turns=sp_turns,
            embs=valid_embs if len(valid_embs) == len(sp_turns) else [],
        )
        breadth_score = breadth_result["breadth_score"]

        # Final information_density score
        information_density = round(
            w_novelty * novelty_score + w_breadth * breadth_score, 3
        )

        # Attach cluster labels to per_argument for explainability
        cluster_labels = breadth_result.get("cluster_labels") or []
        for i, arg_result in enumerate(per_argument):
            arg_result["cluster"] = (
                cluster_labels[i] if i < len(cluster_labels) else None
            )

        output[speaker] = {
            "information_density":  information_density,
            "novelty_score":        novelty_score,
            "breadth_score":        breadth_score,
            "n_clusters":           breadth_result["n_clusters"],
            "n_arguments":          n,
            "per_argument":         per_argument,
        }

    return {
        **output,
        "summary": {
            "total_turns":           len(turns),
            "repetitive_threshold":  REPETITIVE_THRESHOLD,
            "new_threshold":         NEW_THRESHOLD,
            "cluster_threshold":     CLUSTER_THRESHOLD,
            "weights": {
                "novelty":  w_novelty,
                "breadth":  w_breadth,
            },
            "embed_model_available": _ST_AVAILABLE,
        },
    }