"""
debate_flow.py — Debate State Machine
======================================
DEBATE Project · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Drives one complete LLM-vs-LLM debate:
  1. Generates arguments via Ollama
  2. Scores each argument via the Emotion API (composure)
  3. Per-argument: extracts claim/premises, runs NLI + ArgQuality + ES scoring
  4. After all turns: runs debate-level DL scoring locally, then calls Judge API for winner
  5. Synthesises TTS for host + each argument via Emotion API /synthesize
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import AsyncIterator

import httpx

import host_script
from judge_api.argument_miner import extract_claim_and_premises
from judge_api.individual_arg_final_score import compute_individual_arg_score
from judge_api.argument_quality_dl import compute_argument_quality
from judge_api.information_density_dl import compute_information_density
from judge_api.rebuttual_effectiveness_dl import compute_rebuttal_effectiveness
from judge_api.engagement_parallel_dl import compute_engagement_parallel
from judge_api.judge_config import DEBATE_LEVEL_WEIGHTS, FINAL_WEIGHTS

log = logging.getLogger(__name__)

# ── System prompts for each speaker role ─────────────────────────────────────

_OPENING_PROMPT = (
    "You are {speaker}, a fierce competitive debater arguing {stance} on: '{topic}'. "
    "Deliver a sharp, confident opening statement. "
    "Make a bold claim and back it with 1-2 hard-hitting reasons. "
    "Use strong assertive language — no hedging, no apologies. "
    "Maximum 3 sentences. Respond in English only. Do not mention you are an AI."
)

_REBUTTAL_PROMPT = (
    "You are {speaker}, a fierce competitive debater arguing {stance} on: '{topic}'. "
    "Your opponent just said: \"{opponent_argument}\". "
    "First, tear apart the weakest point of their argument in one sharp sentence. "
    "Then drive home a new argument they have not addressed. "
    "Be direct, aggressive, and uncompromising. Maximum 3 sentences. Respond in English only. Do not mention you are an AI."
)

_COACH_DIRECTIVES = {
    "statistics":  "Back every claim with a specific number, percentage, or named study. Facts over rhetoric.",
    "examples":    "Ground every point in a concrete real-world example — name a country, event, or person.",
    "empathy":     "Appeal to human suffering and real-world consequences. Speak about specific people and real lives affected — make it personal, emotional, and impossible to ignore.",
    "aggressive":  "Attack the opponent's logic head-on. Use words like 'absurd', 'dangerously naive', 'completely fails'. Show no mercy.",
    "calm":        "Speak with cold, measured authority. No emotional language — let the logic demolish them.",
    "realistic":   "Use only grounded, real-world arguments. No hypotheticals — cite what is actually happening today.",
    "simple":      "Use plain language anyone can understand. Short sentences, everyday words, no jargon.",
    "technical":   "Go deep — use precise terminology, cite mechanisms, show you understand the topic at an expert level.",
    "rhetorical":  "Use rhetorical questions to challenge the opponent. Make the audience question what they believe — never state, always ask.",
}

# ── Coach steering ────────────────────────────────────────────────────────────
_steering_a: str | None = None  # Coach A → LLM-A (Pro)
_steering_b: str | None = None  # Coach B → LLM-B (Con)
_paused: bool = False


def pause_debate() -> None:
    global _paused
    _paused = True


def resume_debate() -> None:
    global _paused
    _paused = False


def is_paused() -> bool:
    return _paused


def set_steering_a(instruction: str) -> None:
    global _steering_a
    _steering_a = instruction.strip() if instruction else None


def set_steering_b(instruction: str) -> None:
    global _steering_b
    _steering_b = instruction.strip() if instruction else None


def _consume_steering(is_llm_a: bool) -> str | None:
    global _steering_a, _steering_b
    if is_llm_a:
        instruction, _steering_a = _steering_a, None
    else:
        instruction, _steering_b = _steering_b, None
    return instruction


async def _fetch_es_references(es_api: str, query: str, top_n: int = 3) -> list:
    """Fetch semantically similar arguments from the ES relay. Returns [] if unreachable."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{es_api}/semantic-search",
                json={"argument": query, "top_n": top_n, "quality_only": False, "min_score": 1.3},
            )
            resp.raise_for_status()
            return resp.json().get("similar_arguments", [])
    except Exception:
        return []


async def _generate_argument(
    ollama_url: str,
    model: str,
    prompt: str,
    references: list | None = None,
) -> str:
    """Call Ollama and return the generated text."""
    if references:
        claims = [r.get("claim", "").strip() for r in references if r.get("claim", "").strip()]
        if claims:
            prompt = (
                prompt
                + "\n\nRelevant reference points from the debate corpus (use as inspiration — do not copy verbatim):\n"
                + "".join(f"- {c}\n" for c in claims)
            )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 120},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()


async def _score_argument(emotion_api: str, text: str) -> dict:
    """Score one argument via the Emotion API delivery endpoint."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{emotion_api}/emotion/delivery",
                json={"text": text},
            )
            resp.raise_for_status()
            data = resp.json()
            intensity = data.get("delivery_vector", {}).get("intensity", 0.5)
            composure = round(1.0 - intensity, 3)
            return {
                "composure": composure,
                "intensity": intensity,
                "emotional_direction": data.get("emotional_direction", "neutral"),
                "dominant_emotion": data.get("dominant_emotion", "neutral"),
            }
    except Exception as exc:
        log.warning("Emotion API scoring failed: %s", exc)
        return {"composure": 0.5, "intensity": 0.5, "emotional_direction": "neutral", "dominant_emotion": "neutral"}


async def _synthesize(emotion_api: str, text: str, speaker_role: str, style: str = "default") -> str | None:
    """Call /synthesize and return base64-encoded WAV bytes, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{emotion_api}/synthesize",
                json={"text": text, "provider": "kokoro", "speaker_role": speaker_role, "style": style},
            )
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
    except Exception as exc:
        log.warning("TTS synthesis failed for role '%s': %s", speaker_role, exc)
        return None


async def _judge_winner(judge_api: str, turns: list[dict], topic: str = "") -> dict:
    """
    Run debate-level DL scoring locally, then call Judge API /judge for winner.

    turns: raw judge_turns from run_debate, each with keys:
        round, speaker_id, text, claim, premises, individual_arg_scores
    """
    try:
        # Reformat turns for DL functions: need turn/speaker/argument/claim
        dl_turns = []
        scored_arguments: dict[str, list] = {}
        for i, item in enumerate(turns, start=1):
            speaker = item["speaker_id"]
            dl_t = {
                "turn": i,
                "speaker": speaker,
                "argument": item["text"],
            }
            if item.get("claim"):
                dl_t["claim"] = item["claim"]
            if item.get("premises"):
                dl_t["premises"] = item["premises"]
            dl_turns.append(dl_t)

            # Build scored_arguments for compute_argument_quality
            arg_scores = item.get("individual_arg_scores") or {}
            quality = arg_scores.get("quality")
            scored_arguments.setdefault(speaker, []).append({
                "turn": i,
                "quality": quality if quality is not None else 0.5,
                "source": "full_pipeline" if quality is not None else "unavailable",
            })

        # Run all four debate-level DL scorers locally
        rebuttal_results            = compute_rebuttal_effectiveness(dl_turns)
        argument_quality_results    = compute_argument_quality(dl_turns, scored_arguments)
        engagement_results          = compute_engagement_parallel(dl_turns)
        information_density_results = compute_information_density(dl_turns)

        final_debate_scores = compute_debate_level_scores(
            rebuttal_results,
            argument_quality_results,
            engagement_results,
            information_density_results,
        )
        log.info("Local debate-level scores: %s", final_debate_scores)

        # Call Judge API /judge for final winner determination
        http_turns = [
            {"turn": t["turn"], "speaker": t["speaker"], "argument": t["argument"]}
            for t in dl_turns
        ]
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{judge_api}/judge",
                    json={"topic": topic, "turns": http_turns},
                )
                resp.raise_for_status()
                result = resp.json()
                if "verdict" in result and "explanation" not in result:
                    result["explanation"] = result["verdict"]
                return result
        except Exception as exc:
            log.warning("Judge API HTTP call failed, using local DL scores: %s", exc)

        # Fallback: determine winner from local DL scores
        if not final_debate_scores:
            return {}
        winner_id = max(final_debate_scores, key=lambda s: final_debate_scores[s]["debate_level_score"])
        loser_id  = [s for s in final_debate_scores if s != winner_id][0]
        w = final_debate_scores[winner_id]
        l = final_debate_scores[loser_id]
        margin = round(w["debate_level_score"] - l["debate_level_score"], 3)
        explanation = (
            f"{winner_id} wins on debate scoring ({w['debate_level_score']:.3f} vs "
            f"{l['debate_level_score']:.3f}). "
            f"Rebuttal: {w['rebuttal_effectiveness']:.2f} vs {l['rebuttal_effectiveness']:.2f}, "
            f"Argument Quality: {w['argument_quality']:.2f} vs {l['argument_quality']:.2f}, "
            f"Engagement: {w['engagement_score']:.2f} vs {l['engagement_score']:.2f}."
        )
        return {"winner": winner_id, "explanation": explanation, "margin": margin}

    except Exception as exc:
        log.warning("Judge API failed, falling back to composure winner: %s", exc)
        return {}


def compute_debate_level_scores(
    rebuttal_results: dict,
    argument_quality_results: dict,
    engagement_results: dict,
    information_density_results: dict,
) -> dict:
    """
    Aggregate all 4 debate-level DL metrics into one final debate score per speaker.

    Formula (from judge_config.DEBATE_LEVEL_WEIGHTS):
        0.30 * rebuttal_effectiveness
      + 0.27 * argument_quality
      + 0.25 * engagement_parallel
      + 0.18 * information_density
    """
    _non_speaker = {"summary", "error"}
    speakers = (
        set(rebuttal_results.keys())
        & set(argument_quality_results.keys())
        & set(engagement_results.keys())
        & set(information_density_results.keys())
    ) - _non_speaker

    final_results = {}

    for speaker in speakers:
        rebuttal     = float(rebuttal_results[speaker]["rebuttal_effectiveness"])
        arg_quality  = float(argument_quality_results[speaker]["argument_quality"])
        engagement   = float(engagement_results[speaker]["engagement_score"])
        info_density = float(information_density_results[speaker]["information_density"])

        debate_score = round(
            DEBATE_LEVEL_WEIGHTS["rebuttal_effectiveness"] * rebuttal
            + DEBATE_LEVEL_WEIGHTS["argument_quality"]     * arg_quality
            + DEBATE_LEVEL_WEIGHTS["engagement_parallel"]  * engagement
            + DEBATE_LEVEL_WEIGHTS["information_density"]  * info_density,
            3,
        )

        final_results[speaker] = {
            "debate_level_score":   debate_score,
            "rebuttal_effectiveness": rebuttal,
            "argument_quality":     arg_quality,
            "engagement_score":     engagement,
            "information_density":  info_density,
            "score_breakdown": {
                "rebuttal_component":          round(DEBATE_LEVEL_WEIGHTS["rebuttal_effectiveness"] * rebuttal, 3),
                "argument_quality_component":  round(DEBATE_LEVEL_WEIGHTS["argument_quality"] * arg_quality, 3),
                "engagement_component":        round(DEBATE_LEVEL_WEIGHTS["engagement_parallel"] * engagement, 3),
                "information_density_component": round(DEBATE_LEVEL_WEIGHTS["information_density"] * info_density, 3),
            },
        }

    return final_results


def _composure_winner(transcript: list[dict], llm_a: str, llm_b: str) -> tuple[str, str]:
    """Fallback winner by average composure if Judge API is unavailable."""
    scores = {llm_a: [], llm_b: []}
    for turn in transcript:
        speaker  = turn.get("speaker")
        composure = turn.get("scores", {}).get("composure")
        if speaker in scores and composure is not None:
            scores[speaker].append(composure)

    avg = {k: (sum(v) / len(v) if v else 0.5) for k, v in scores.items()}
    if abs(avg[llm_a] - avg[llm_b]) < 0.03:
        return "tie", f"Both speakers held comparable composure ({avg[llm_a]:.2f} vs {avg[llm_b]:.2f})."

    winner = max(avg, key=avg.get)
    loser  = llm_b if winner == llm_a else llm_a
    return winner, (
        f"{winner} maintained stronger composure throughout "
        f"({avg[winner]:.2f} vs {avg[loser]:.2f})."
    )


async def run_debate(
    topic: str,
    llm_a: str,
    llm_b: str,
    turn_count: int,
    ollama_url: str,
    ollama_model: str,
    emotion_api: str,
    judge_api: str,
    es_api: str = "http://141.54.159.66:8000",
) -> AsyncIterator[dict]:
    """
    Async generator that yields debate events as dicts.

    Event types:
      host_intro    — {type, text, audio_b64}
      argument      — {type, speaker, role, round, text, scores, individual_arg_scores, audio_b64, es_references}
      scores_update — {type, llm_a_avg_composure, llm_b_avg_composure}
      round_complete— {type, round}
      winner        — {type, winner, explanation, text, audio_b64}
      error         — {type, message}
    """
    global _paused
    transcript: list[dict] = []
    last_a_arg = ""
    last_b_arg = ""

    _paused = False

    # ── Host intro ────────────────────────────────────────────────────────────
    intro_text  = host_script.intro(topic, llm_a, llm_b, turn_count)
    intro_audio = await _synthesize(emotion_api, intro_text, "host", style="cheerful")
    yield {"type": "host_intro", "text": intro_text, "audio_b64": intro_audio}

    # ── Debate turns ──────────────────────────────────────────────────────────
    for turn_index in range(turn_count):
        is_llm_a   = (turn_index % 2 == 0)
        speaker    = llm_a if is_llm_a else llm_b
        stance     = "in favour" if is_llm_a else "against"
        role       = "llm_a" if is_llm_a else "llm_b"
        round_num  = (turn_index // 2) + 1
        is_opening = turn_index < 2

        if is_opening:
            prompt = _OPENING_PROMPT.format(speaker=speaker, topic=topic, stance=stance)
        else:
            opponent_arg = last_b_arg if is_llm_a else last_a_arg
            prompt = _REBUTTAL_PROMPT.format(
                speaker=speaker, topic=topic, stance=stance, opponent_argument=opponent_arg
            )

        # Wait if paused
        while _paused:
            await asyncio.sleep(0.5)

        # Coach steering
        coach_instruction = _consume_steering(is_llm_a)
        if coach_instruction:
            from difflib import get_close_matches
            words = coach_instruction.lower().split()
            directive = coach_instruction
            for word in words:
                matches = get_close_matches(word, _COACH_DIRECTIVES.keys(), n=1, cutoff=0.75)
                if matches:
                    directive = _COACH_DIRECTIVES[matches[0]]
                    break
            prompt = f"{prompt}\n\nCOACH OVERRIDE (supersedes everything above): {directive}"

        # Fetch ES references
        es_query   = (last_b_arg if is_llm_a else last_a_arg) or topic
        references = await _fetch_es_references(es_api, es_query)

        # Generate argument
        try:
            argument_text = await _generate_argument(ollama_url, ollama_model, prompt, references)
        except Exception as exc:
            yield {"type": "error", "message": f"Ollama generation failed: {exc}"}
            return

        # Score (emotion) + synthesise TTS in parallel
        scores, audio_b64 = await asyncio.gather(
            _score_argument(emotion_api, argument_text),
            _synthesize(emotion_api, argument_text, role,
                        style="cheerful" if is_opening else "empathetic"),
        )

        # Per-argument: extract claim/premises and run individual scoring
        try:
            claim, premises = extract_claim_and_premises(argument_text)
            individual_arg_scores = compute_individual_arg_score(argument_text, claim, premises)
        except Exception as exc:
            log.warning("Individual arg scoring failed: %s", exc)
            claim, premises, individual_arg_scores = "", [], {}

        if is_llm_a:
            last_a_arg = argument_text
        else:
            last_b_arg = argument_text

        turn_record = {
            "speaker":              speaker,
            "role":                 role,
            "round":                round_num,
            "text":                 argument_text,
            "claim":                claim,
            "premises":             premises,
            "scores":               scores,
            "individual_arg_scores": individual_arg_scores,
            "speaker_id":           role.upper().replace("_", "-"),
        }
        transcript.append(turn_record)

        ref_claims = [r.get("claim", "").strip() for r in references if r.get("claim", "").strip()]
        yield {
            "type":                 "argument",
            "speaker":              speaker,
            "role":                 role,
            "round":                round_num,
            "text":                 argument_text,
            "scores":               scores,
            "individual_arg_scores": individual_arg_scores,
            "audio_b64":            audio_b64,
            "es_references":        ref_claims,
        }

        # Wait if paused after argument
        while _paused:
            await asyncio.sleep(0.5)

        # Running composure averages
        a_scores = [t["scores"]["composure"] for t in transcript if t["role"] == "llm_a"]
        b_scores = [t["scores"]["composure"] for t in transcript if t["role"] == "llm_b"]
        yield {
            "type":                 "scores_update",
            "llm_a_avg_composure": round(sum(a_scores) / len(a_scores), 3) if a_scores else None,
            "llm_b_avg_composure": round(sum(b_scores) / len(b_scores), 3) if b_scores else None,
        }

        # Auto-pause for coach after each full round (Con just spoke), except last turn
        is_last_turn = (turn_index == turn_count - 1)
        if not is_llm_a and not is_last_turn:
            _paused = True
            yield {"type": "round_complete", "round": round_num}
            while _paused:
                await asyncio.sleep(0.5)

    # ── Winner determination ──────────────────────────────────────────────────
    judge_turns = [
        {
            "round":                t["round"],
            "speaker_id":           t["speaker_id"],
            "text":                 t["text"],
            "claim":                t["claim"],
            "premises":             t["premises"],
            "individual_arg_scores": t["individual_arg_scores"],
        }
        for t in transcript
    ]
    judge_result = await _judge_winner(judge_api, judge_turns, topic=topic)

    if judge_result.get("winner"):
        winner_id   = judge_result["winner"]
        winner      = llm_a if "A" in winner_id.upper() else llm_b
        explanation = judge_result.get("explanation", "")
    else:
        winner, explanation = _composure_winner(transcript, llm_a, llm_b)

    if winner == "tie":
        outro_text = host_script.tie_announcement()
    else:
        outro_text = host_script.winner_announcement(winner, explanation)

    outro_audio = await _synthesize(emotion_api, outro_text, "host", style="cheerful")

    yield {
        "type":        "winner",
        "winner":      winner,
        "explanation": explanation,
        "text":        outro_text,
        "audio_b64":   outro_audio,
    }
