"""
nli_score.py — NLI (Natural Language Inference) scorer for the Debate Judge.

Measures whether the claim logically follows from the premises of a given
argument. Returns a float 0.0–1.0, where:
  0.0 = premises contradict or are entirely irrelevant to the claim
  0.5 = premises are tangentially related (neutral / inconclusive)
  1.0 = premises directly and strongly entail the claim

Implementation
--------------
Uses the Ollama LLM already running in this environment (qwen2.5:7b), so no
additional model downloads are required. For higher accuracy in a research
context, the LLM call can be replaced with a dedicated cross-encoder NLI
model (e.g. cross-encoder/nli-deberta-v3-small from sentence-transformers),
but the LLM approach is demo-safe and consistent with the rest of the pipeline.

Fallback: if Ollama is unreachable, returns 0.5 (neutral / no information)
so the final_score formula still runs with a conservative estimate.
"""

import json
import re
import requests

from services.judge_config import OLLAMA_HOST, OLLAMA_MODEL

_FALLBACK_NLI = 0.5   # neutral when LLM unavailable


def _extract_json(text: str) -> dict:
    """Pull the first {...} block from an LLM response and parse it."""
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    # Last-resort: look for a bare decimal
    m2 = re.search(r"(?:0?\.\d+|[01](?:\.\d+)?)", text)
    return {"nli_score": float(m2.group()) if m2 else _FALLBACK_NLI}


def score_nli(argument: str, claim: str = None, premises: list = None,
              ollama_host: str = None, model: str = None) -> dict:
    """
    Rate how well the premises logically entail the claim.

    Parameters
    ----------
    argument : str
        Full debate argument text. Used as fallback if claim/premises not split.
    claim    : str, optional
        The central assertion. Auto-extracted if absent.
    premises : list[str], optional
        Supporting statements. Auto-extracted if absent.

    Returns
    -------
    {
        "nli_score":     float,   # 0.0–1.0 (used in final_score formula)
        "claim":         str,
        "premise_count": int,
        "reasoning":     str,     # one-sentence LLM explanation
    }
    """
    from services.unknown_arguments import split_claim_premises   # lazy: avoids circular

    ollama_host = ollama_host or OLLAMA_HOST
    model = model or OLLAMA_MODEL

    if not claim:
        claim, premises = split_claim_premises(argument)
    premises = premises or []

    premises_text = "\n".join(f"- {p}" for p in premises) if premises \
        else f"- {argument[:400]}"

    prompt = f"""You are an expert in formal logic and argumentation.

TASK: Determine how well the PREMISES logically support and entail the CLAIM.

CLAIM: {claim}

PREMISES:
{premises_text}

SCORING GUIDE (do NOT copy these numbers — derive your own score):
• Premises directly contradict or are completely irrelevant to the claim → close to 0.0
• Premises are somewhat related but do not conclusively support the claim → around 0.4–0.6
• Premises strongly and directly support / entail the claim → close to 1.0

Respond with ONLY this JSON (no markdown, no extra text):
{{"nli_score": <your_score>, "reasoning": "<one sentence>"}}"""

    try:
        r = requests.post(
            f"{ollama_host}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"num_predict": 80, "temperature": 0.05}},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        parsed = _extract_json(raw)
        score = round(min(max(float(parsed.get("nli_score", _FALLBACK_NLI)), 0.0), 1.0), 3)
        reasoning = parsed.get("reasoning", "")
    except Exception:
        score = _FALLBACK_NLI
        reasoning = "NLI scorer unavailable (Ollama unreachable); neutral fallback used."

    return {
        "nli_score":     score,
        "claim":         claim,
        "premise_count": len(premises),
        "reasoning":     reasoning,
    }
