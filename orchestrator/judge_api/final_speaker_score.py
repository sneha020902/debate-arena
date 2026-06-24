"""
final_speaker_score.py — Final debate score aggregator for a single speaker.

Responsibility (AGGREGATION ONLY)
----------------------------------
This module does ONE thing: combine already-computed scores from the
individual-argument, debate-level, and emotional-composure layers into a
single final speaker score, with a transparent breakdown and an LLM-generated
explanation of why the speaker won or lost.

It NEVER recomputes any underlying metric. All component scores are produced
by their own modules and passed in. This keeps the aggregator thin, testable,
and free of duplicated scoring logic.

Where the inputs come from (actual project modules)
----------------------------------------------------
    /judging/final_score.py
        compute_full_score()         -> individual argument score (KNOWN args)
    app/judging/unknown_arguments.py
        score_unknown_arguments()    -> individual argument score (UNKNOWN args)
    app/judging/argument_quality.py
        compute_argument_quality()   -> debate-level argument_quality
    app/judging/engagement_parallel.py
        compute_engagement_parallel()-> debate-level engagement_score
    app/judging/rebuttal_effectiveness.py
        compute_rebuttal_effectiveness() -> debate-level rebuttal_effectiveness
    app/judging/information_density.py
        compute_information_density() -> debate-level information_density
    app/judging/<emotion module>
        delivery / composure scoring -> emotional_composure_score

This module imports NONE of those at module load time. It receives their
OUTPUTS via a typed input object. This is deliberate: the aggregator must not
depend on the availability of every scorer to run, and must be unit-testable
without standing up the whole pipeline. A convenience builder
(`from_pipeline_outputs`) maps raw module dicts into the typed input.

Final formula
-------------
    Final Speaker Score (0-100) =
        0.50 * IndividualArgumentScore
      + 0.40 * DebateLevelScore
      + 0.10 * EmotionalComposureScore

    All three component scores are normalised to 0-100 before aggregation.

    DebateLevelScore is itself a weighted blend of the four debate-level
    metrics (weights configurable, defaulting to the values the built
    modules were designed around):
        0.30 * rebuttal_effectiveness
        0.27 * argument_quality
        0.25 * engagement_score
        0.18 * information_density

Weights are configurable
-------------------------
All weights live in dataclasses with defaults. Pass a custom ScoreWeights to
explore different scoring philosophies without touching the aggregation logic.

Author: Debate Scoring Platform
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

from .individual_arg_final_score import compute_individual_arg_score
from .judge_config import OLLAMA_HOST, OLLAMA_MODEL

import requests

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IndividualWeights:
    """
    Weights for the individual argument score components.

    Defaults reflect the UPDATED weights the built modules use
    (final_score.py), NOT the original spec. To use the original spec
    (NLI=0.30, ArgQuality=0.35), construct with those values explicitly.
    """
    nli:        float = 0.25
    argquality: float = 0.40
    delta:      float = 0.20
    votes:      float = 0.15

    def validate(self) -> None:
        total = self.nli + self.argquality + self.delta + self.votes
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Individual weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class DebateLevelWeights:
    """
    Weights for the four debate-level metrics.

    Defaults match the values the built debate-level modules were
    designed around (4-metric scheme, no clash weighing).
    """
    rebuttal_effectiveness: float = 0.30
    argument_quality:       float = 0.27
    engagement:             float = 0.25
    information_density:    float = 0.18

    def validate(self) -> None:
        total = (self.rebuttal_effectiveness + self.argument_quality
                 + self.engagement + self.information_density)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Debate-level weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class FinalWeights:
    """Top-level weights: individual vs debate-level vs composure."""
    individual: float = 0.50
    debate:     float = 0.40
    composure:  float = 0.10

    def validate(self) -> None:
        total = self.individual + self.debate + self.composure
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Final weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ScoreWeights:
    """Bundle of all weight groups. Pass a custom instance to reconfigure."""
    final:      FinalWeights        = field(default_factory=FinalWeights)
    individual: IndividualWeights   = field(default_factory=IndividualWeights)
    debate:     DebateLevelWeights  = field(default_factory=DebateLevelWeights)

    def validate(self) -> None:
        self.final.validate()
        self.individual.validate()
        self.debate.validate()


# ════════════════════════════════════════════════════════════════════════════
#  Input / Output data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class IndividualArgumentInputs:
    """
    Pre-computed individual-argument component scores for one speaker.

    All values are on a 0-1 scale (as the scorer modules output them).
    Any field left as None is treated as 'missing' and handled gracefully
    (excluded from the weighted average with weight renormalisation).
    """
    nli:        Optional[float] = None
    argquality: Optional[float] = None
    delta:      Optional[float] = None
    votes:      Optional[float] = None


@dataclass
class DebateLevelInputs:
    """
    Pre-computed debate-level metric scores for one speaker.
    All values on a 0-1 scale. None = missing.
    """
    rebuttal_effectiveness: Optional[float] = None
    argument_quality:       Optional[float] = None
    engagement:             Optional[float] = None
    information_density:    Optional[float] = None


@dataclass
class EmotionalComposureInput:
    """
    Pre-computed emotional composure score for one speaker.
    Value on a 0-1 scale. None = missing.
    """
    composure: Optional[float] = None


@dataclass
class SpeakerScoreInputs:
    """Complete set of pre-computed inputs for one speaker."""
    speaker_id:  str
    individual:  IndividualArgumentInputs
    debate:      DebateLevelInputs
    composure:   EmotionalComposureInput


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _to_100(value: Optional[float]) -> Optional[float]:
    """
    Normalise a 0-1 score to 0-100. Pass-through for None.
    Defensive: if a value already looks like 0-100 (>1.0), it is assumed
    to already be on the 100 scale and returned clamped.
    """
    if value is None:
        return None
    if value > 1.0:
        # Already on 0-100 scale — clamp and return
        return round(max(0.0, min(100.0, value)), 2)
    return round(max(0.0, min(1.0, value)) * 100.0, 2)


def _validate_100(value: Optional[float], name: str) -> Optional[float]:
    """Validate a 0-100 score; log and clamp if out of range."""
    if value is None:
        return None
    if value < 0.0 or value > 100.0:
        logger.warning(f"{name}={value} out of 0-100 range — clamping.")
        return round(max(0.0, min(100.0, value)), 2)
    return round(value, 2)


def _weighted_average_with_missing(
    values_weights: list[tuple[Optional[float], float]],
) -> tuple[Optional[float], dict]:
    """
    Compute a weighted average that gracefully handles missing (None) values.

    Missing components are dropped and the remaining weights are renormalised
    so they still sum to 1.0. This prevents a single missing metric from
    silently dragging the score toward zero.

    Parameters
    ----------
    values_weights : list of (value_or_None, weight)

    Returns
    -------
    (weighted_average, provenance)
        weighted_average : float on same scale as inputs, or None if ALL missing
        provenance       : dict describing which were used / dropped + renorm
    """
    present = [(v, w) for v, w in values_weights if v is not None]
    missing_weight = sum(w for v, w in values_weights if v is None)

    if not present:
        return None, {
            "used": 0, "dropped": len(values_weights),
            "renormalised": False, "missing_weight": round(missing_weight, 4),
        }

    total_present_weight = sum(w for _, w in present)
    if total_present_weight <= 0:
        return None, {"used": 0, "dropped": len(values_weights),
                      "renormalised": False, "missing_weight": 1.0}

    # Renormalise present weights to sum to 1.0
    weighted_sum = sum(v * (w / total_present_weight) for v, w in present)

    return round(weighted_sum, 3), {
        "used": len(present),
        "dropped": len(values_weights) - len(present),
        "renormalised": missing_weight > 1e-9,
        "missing_weight": round(missing_weight, 4),
        "present_weight_before_renorm": round(total_present_weight, 4),
    }


def _extract_json(text: str) -> dict:
    """Pull the first {...} block from LLM output and parse it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


# ════════════════════════════════════════════════════════════════════════════
#  Component score computation
# ════════════════════════════════════════════════════════════════════════════



def compute_debate_level_score(
    inputs:  DebateLevelInputs,
    weights: DebateLevelWeights,
) -> tuple[Optional[float], dict]:
    """
    Aggregate the four debate-level metrics into a 0-100 score.

    Returns (score_0_100, breakdown).
    """
    raw_avg, prov = _weighted_average_with_missing([
        (inputs.rebuttal_effectiveness, weights.rebuttal_effectiveness),
        (inputs.argument_quality,       weights.argument_quality),
        (inputs.engagement,             weights.engagement),
        (inputs.information_density,    weights.information_density),
    ])

    breakdown = {
        "components_0_1": {
            "rebuttal_effectiveness": inputs.rebuttal_effectiveness,
            "argument_quality":       inputs.argument_quality,
            "engagement":             inputs.engagement,
            "information_density":    inputs.information_density,
        },
        "weights": asdict(weights),
        "provenance": prov,
    }
    return _to_100(raw_avg), breakdown


def compute_composure_score(
    inputs: EmotionalComposureInput,
) -> tuple[Optional[float], dict]:
    """Normalise the emotional composure score to 0-100."""
    score = _to_100(inputs.composure)
    return score, {"composure_0_1": inputs.composure}


# ════════════════════════════════════════════════════════════════════════════
#  LLM explanation
# ════════════════════════════════════════════════════════════════════════════

def _generate_reasoning_llm(
    speaker_id:         str,
    individual_score:   Optional[float],
    debate_score:       Optional[float],
    composure_score:    Optional[float],
    final_score:        Optional[float],
    individual_bd:      dict,
    debate_bd:          dict,
    is_winner:          Optional[bool],
    opponent_final:     Optional[float],
    ollama_host:        str,
    model:              str,
) -> dict:
    """
    Ask an LLM to explain WHY the speaker won or lost, grounded in the
    component scores. The LLM is given the numbers and asked to interpret
    them — it does NOT compute or change any score.

    IMPORTANT: Use a DIFFERENT model here than the one used to GENERATE the
    debate arguments. Using the same model to generate and judge introduces
    self-preference bias. See module notes / your pipeline config.

    Returns
    -------
    {
        "explanation": str,
        "key_strengths": list[str],
        "key_weaknesses": list[str],
        "source": "llm" | "fallback",
    }
    """
    def _fmt(bd: dict) -> str:
        comps = bd.get("components_0_1", {})
        return ", ".join(
            f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}=N/A"
            for k, v in comps.items()
        )

    verdict_line = ""
    if is_winner is not None and opponent_final is not None:
        verdict_line = (
            f"This speaker {'WON' if is_winner else 'LOST'} "
            f"(their score {final_score:.1f} vs opponent {opponent_final:.1f})."
        )

    prompt = f"""You are an expert debate adjudicator writing a verdict.

Speaker: {speaker_id}
{verdict_line}

FINAL SCORE: {final_score:.1f}/100
  Individual argument quality: {individual_score if individual_score is not None else 'N/A'}/100
    ({_fmt(individual_bd)})
  Debate-level performance:    {debate_score if debate_score is not None else 'N/A'}/100
    ({_fmt(debate_bd)})
  Emotional composure:         {composure_score if composure_score is not None else 'N/A'}/100

Write a concise adjudicator's explanation (3-4 sentences) of WHY this speaker
achieved this result. Reference the SPECIFIC components that helped or hurt
them most. Be concrete: name the strongest and weakest dimensions and what
they imply about the speaker's debating.

Then list their key strengths and weaknesses.

Reply with ONLY this JSON (no markdown):
{{
  "explanation": "<3-4 sentence adjudicator verdict>",
  "key_strengths": ["<strength>", "<strength>"],
  "key_weaknesses": ["<weakness>", "<weakness>"]
}}"""

    try:
        r = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 350, "temperature": 0.2},
            },
            timeout=60,
        )
        r.raise_for_status()
        raw    = r.json().get("response", "")
        parsed = _extract_json(raw)

        if parsed.get("explanation"):
            return {
                "explanation":    str(parsed["explanation"]),
                "key_strengths":  parsed.get("key_strengths", []),
                "key_weaknesses": parsed.get("key_weaknesses", []),
                "source":         "llm",
            }
    except Exception as e:
        logger.warning(f"LLM reasoning generation failed: {e}")

    # ── Deterministic fallback explanation ────────────────────────────────────
    return _fallback_reasoning(
        speaker_id, individual_score, debate_score,
        composure_score, final_score, individual_bd, debate_bd, is_winner,
    )


def _fallback_reasoning(
    speaker_id, individual_score, debate_score,
    composure_score, final_score, individual_bd, debate_bd, is_winner,
) -> dict:
    """
    Build a rule-based explanation when the LLM is unavailable.
    Identifies the highest and lowest contributing components numerically.
    """
    parts = []
    if is_winner is True:
        parts.append(f"{speaker_id} won with a final score of {final_score:.1f}/100.")
    elif is_winner is False:
        parts.append(f"{speaker_id} lost with a final score of {final_score:.1f}/100.")
    else:
        parts.append(f"{speaker_id} scored {final_score:.1f}/100.")

    # Identify strongest / weakest debate-level component
    debate_comps = {
        k: v for k, v in debate_bd.get("components_0_1", {}).items()
        if isinstance(v, (int, float))
    }
    strengths, weaknesses = [], []
    if debate_comps:
        best  = max(debate_comps, key=debate_comps.get)
        worst = min(debate_comps, key=debate_comps.get)
        strengths.append(f"{best} ({debate_comps[best]:.2f})")
        weaknesses.append(f"{worst} ({debate_comps[worst]:.2f})")
        parts.append(
            f"Strongest debate dimension was {best} "
            f"({debate_comps[best]:.2f}); weakest was {worst} "
            f"({debate_comps[worst]:.2f})."
        )

    if individual_score is not None and debate_score is not None:
        if individual_score > debate_score:
            parts.append(
                "Individual argument quality outpaced debate-level engagement."
            )
        else:
            parts.append(
                "Debate-level engagement was stronger than standalone argument quality."
            )

    return {
        "explanation":    " ".join(parts),
        "key_strengths":  strengths,
        "key_weaknesses": weaknesses,
        "source":         "fallback",
    }


# ════════════════════════════════════════════════════════════════════════════
#  Main aggregator
# ════════════════════════════════════════════════════════════════════════════

def compute_final_speaker_score(
    inputs:          dict,
) -> dict:
    """
    Aggregate pre-computed scores into a final speaker score with breakdown.

    This is the ONLY public entry point. It does not recompute any metric.

    Parameters
    ----------
    inputs : SpeakerScoreInputs
        Pre-computed component scores for the speaker (0-1 scale).
    weights : ScoreWeights, optional
        Full weight configuration. Defaults to the built-module weights.
    opponent_final : float, optional
        The opponent's final score (0-100). If provided, winner_confidence
        and an is_winner verdict are computed.
    generate_reasoning : bool
        If True, call the LLM (or fallback) to explain the result.
    ollama_host : str
        Ollama base URL for reasoning.
    ollama_model : str
        Model for reasoning. SHOULD differ from the argument-generation model
        to avoid self-preference bias.

    Returns
    -------
    {
        "speaker_id":                str,
        "individual_argument_score": float | None,   0-100
        "debate_level_score":        float | None,   0-100
        "emotional_composure_score": float | None,   0-100
        "final_score":               float | None,   0-100
        "winner_confidence":         float | None,   0-1, if opponent given
        "is_winner":                 bool  | None,
        "reasoning":                 dict,            LLM explanation
        "breakdown": {
            "individual": dict,
            "debate":     dict,
            "composure":  dict,
            "final_weights": dict,
            "final_provenance": dict,
        },
        "warnings": list[str],
    }
    """
    ollama_host = OLLAMA_HOST
    ollama_model = OLLAMA_MODEL
    final_score = {}
    warnings: list[str] = []

    # print(f"Computing final score for speaker '{inputs}'")

    # ── Component scores (each 0-100) ─────────────────────────────────────────
    for speaker, rounds in inputs.items():
        print(f"Processing speaker: {speaker}")
        for round_name, round_data in rounds.items():
            print(f"Processing {round_name} {round_data} for individual argument score...")
            argument = round_data["argument"]
            claim = round_data["claim"]
            premises = round_data["premises"]

            individual_scores = compute_individual_arg_score(argument,claim, premises)
            print(f"Computed individual argument scores (0-1): {individual_scores}")
            # Store the result back into the same JSON
            round_data["individual_arg_scores"] = individual_scores
            print(f"Raw individual argument scores (0-1): {individual_scores}")

    
    final_score["individual_argument_score"] = individual_scores
    print(f"Final individual argument score (0-100): {final_score['individual_argument_score']}")
    debate_score, debate_bd = compute_debate_level_score(
        inputs.debate
    )
    logger.info(f"Computed debate level score: {debate_score} with breakdown {debate_bd}")
    composure_score, composure_bd = compute_composure_score(inputs.composure)
    logger.info(f"Computed composure score: {composure_score} with breakdown {composure_bd}")

    individual_score = _validate_100(individual_score, "individual_argument_score")
    debate_score     = _validate_100(debate_score,     "debate_level_score")
    composure_score  = _validate_100(composure_score,  "emotional_composure_score")

    logger.info(f"Validated scores - Individual: {individual_score}, Debate: {debate_score}, Composure: {composure_score}")
    if individual_score is None:
        warnings.append("Individual argument score missing — excluded from final.")
    if debate_score is None:
        warnings.append("Debate-level score missing — excluded from final.")
    if composure_score is None:
        warnings.append("Emotional composure score missing — excluded from final.")



    # # ── Winner determination ──────────────────────────────────────────────────
    # is_winner: Optional[bool] = None
    # winner_confidence: Optional[float] = None
    # if opponent_final is not None and final_score is not None:
    #     is_winner = final_score >= opponent_final
    #     # winner_confidence: scaled margin, mapped to 0-1 via a soft function
    #     margin = abs(final_score - opponent_final)
    #     # A 20-point margin → ~0.95 confidence; a 0-point margin → 0.5
    #     winner_confidence = round(min(0.5 + margin / 40.0, 1.0), 3)

    # # ── Reasoning ─────────────────────────────────────────────────────────────
    # reasoning = {"explanation": "", "key_strengths": [],
    #              "key_weaknesses": [], "source": "skipped"}
    # if generate_reasoning and final_score is not None:
    #     reasoning = _generate_reasoning_llm(
    #         speaker_id=inputs.speaker_id,
    #         individual_score=individual_score,
    #         debate_score=debate_score,
    #         composure_score=composure_score,
    #         final_score=final_score,
    #         individual_bd=individual_bd,
    #         debate_bd=debate_bd,
    #         is_winner=is_winner,
    #         opponent_final=opponent_final,
    #         ollama_host=ollama_host,
    #         model=ollama_model,
    #     )
    #     logger.info(f"Generated reasoning: {reasoning}")

    return {
        "speaker_id":                inputs.speaker_id,
        "individual_argument_score": individual_score,
        "debate_level_score":        debate_score,
        "emotional_composure_score": composure_score,
        "final_score":               final_score,
        # "winner_confidence":         winner_confidence,
        # "is_winner":                 is_winner,
        # "reasoning":                 reasoning,
        # "breakdown": {
        #     "individual":        individual_bd,
        #     "debate":            debate_bd,
        #     "composure":         composure_bd,
        #     "final_weights":     asdict(weights.final),
        #     "final_provenance":  final_prov,
        # },
        "warnings": warnings,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Convenience builder — maps raw pipeline outputs to typed inputs
# ════════════════════════════════════════════════════════════════════════════

def from_pipeline_outputs(
    speaker_id:               str,
    individual_score_dict:    Optional[dict] = None,
    argument_quality_dict:    Optional[dict] = None,
    engagement_dict:          Optional[dict] = None,
    rebuttal_dict:            Optional[dict] = None,
    information_density_dict: Optional[dict] = None,
    composure_dict:           Optional[dict] = None,
) -> SpeakerScoreInputs:
    """
    Map the raw per-speaker output dicts from the scorer modules into a
    typed SpeakerScoreInputs. Each argument is the slice of that module's
    output for THIS speaker.

    This isolates the mapping (which depends on each module's output schema)
    in one place, so schema changes touch only this function.

    Example
    -------
        inputs = from_pipeline_outputs(
            speaker_id="Alice",
            individual_score_dict={"nli": 0.72, "argquality": 0.81,
                                   "delta": 0.55, "votes": 0.60},
            argument_quality_dict=arg_quality_result["Alice"],
            engagement_dict=engagement_result["Alice"],
            rebuttal_dict=rebuttal_result["Alice"],
            information_density_dict=info_density_result["Alice"],
            composure_dict={"composure": 0.78},
        )
    """
    ind = individual_score_dict or {}
    individual = IndividualArgumentInputs(
        nli=ind.get("nli"),
        argquality=ind.get("argquality"),
        delta=ind.get("delta"),
        votes=ind.get("votes"),
    )

    debate = DebateLevelInputs(
        rebuttal_effectiveness=(rebuttal_dict or {}).get("rebuttal_effectiveness"),
        argument_quality=(argument_quality_dict or {}).get("argument_quality"),
        engagement=(engagement_dict or {}).get("engagement_score"),
        information_density=(information_density_dict or {}).get("information_density"),
    )

    composure = EmotionalComposureInput(
        composure=(composure_dict or {}).get("composure"),
    )

    return SpeakerScoreInputs(
        speaker_id=speaker_id,
        individual=individual,
        debate=debate,
        composure=composure,
    )