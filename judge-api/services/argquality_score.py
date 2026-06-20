"""
argquality_score.py — Argument Quality scorer for the Debate Judge.

Evaluates the overall quality of a debate argument on three sub-dimensions
using the LLM (qwen2.5:7b), then derives the combined ArgQuality Score.

Sub-Dimensions
--------------
overall_quality   Combined assessment of logic, relevance, clarity, and
                  persuasiveness. This is the holistic judge's-eye view.
cogency           Quality of evidence, logical consistency, and the
                  premise-to-conclusion reasoning chain.
effectiveness     Persuasive strength, rhetorical impact, and likely
                  audience influence.

ArgQuality Score (used in the weighted formula)
    = mean(overall_quality, cogency, effectiveness)

Averaging the three targeted sub-scores is more stable than a single holistic
number and directly reflects the three aspects of argument quality the spec
describes.

Fallback: if Ollama is unreachable, returns 0.45 for all sub-scores
(slightly below average, conservative default) so the pipeline runs.
"""

import json
import re
import requests

from services.judge_config import OLLAMA_HOST, OLLAMA_MODEL

_FALLBACK_Q = 0.45   # conservative below-average default when LLM unavailable


def _parse_quality_json(text: str) -> dict:
    """Extract the JSON block and parse the three sub-scores robustly."""
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def score_argquality(argument: str,
                     ollama_host: str = None,
                     model: str = None) -> dict:
    """
    Assess the argument on three quality sub-dimensions and return the
    combined ArgQuality Score.

    Parameters
    ----------
    argument : str   Full debate argument text (max 800 chars used in prompt).

    Returns
    -------
    {
        "argquality_score":  float,   # 0.0–1.0  (used in formula, weight: 0.35)
        "overall_quality":   float,   # logic + relevance + clarity + persuasiveness
        "cogency":           float,   # evidence + consistency + premise-to-conclusion
        "effectiveness":     float,   # persuasive strength + rhetorical impact
        "reasoning":         str,     # brief LLM explanation
    }
    """
    ollama_host = ollama_host or OLLAMA_HOST
    model = model or OLLAMA_MODEL

    prompt = f"""You are an expert debate judge evaluating a single argument.

ARGUMENT:
\"{argument[:800]}\"

Rate this argument on three dimensions. For each, provide a score from 0.0 to 1.0.
Do NOT copy any score from this prompt — derive each score independently from
the argument text alone.

DIMENSIONS:
1. overall_quality  — Combined logic, relevance, clarity, and persuasiveness.
   (0.0 = deeply flawed on all axes | 1.0 = outstanding on all axes)

2. cogency          — Quality of evidence, logical consistency, how well
   premises support the conclusion.
   (0.0 = no evidence, contradictory | 1.0 = strong evidence, tight reasoning)

3. effectiveness    — Persuasive strength, rhetorical skill, likely impact on
   an undecided listener.
   (0.0 = unconvincing, poor rhetoric | 1.0 = highly persuasive, compelling)

Respond with ONLY this JSON (no markdown, no extra fields):
{{
  "overall_quality": <score>,
  "cogency": <score>,
  "effectiveness": <score>,
  "reasoning": "<one sentence summarising your assessment>"
}}"""

    try:
        r = requests.post(
            f"{ollama_host}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"num_predict": 120, "temperature": 0.1}},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        parsed = _parse_quality_json(raw)

        def _clamp(key):
            v = parsed.get(key, _FALLBACK_Q)
            try:
                return round(min(max(float(v), 0.0), 1.0), 3)
            except (TypeError, ValueError):
                return _FALLBACK_Q

        oq = _clamp("overall_quality")
        cg = _clamp("cogency")
        ef = _clamp("effectiveness")
        aq = round((oq + cg + ef) / 3.0, 3)   # ArgQuality Score = mean of 3
        reasoning = parsed.get("reasoning", "")

    except Exception:
        oq = cg = ef = aq = _FALLBACK_Q
        reasoning = "ArgQuality scorer unavailable (Ollama unreachable); fallback used."

    return {
        "argquality_score":  aq,
        "overall_quality":   oq,
        "cogency":           cg,
        "effectiveness":     ef,
        "reasoning":         reasoning,
    }
