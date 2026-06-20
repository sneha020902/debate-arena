"""
debate_flow.py — Debate State Machine
======================================
DEBATE Project · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Drives one complete LLM-vs-LLM debate:
  1. Generates arguments via Ollama
  2. Scores each argument via the Emotion API (composure)
  3. After all turns, calls Judge API for final winner
  4. Synthesises TTS for host + each argument via Emotion API /synthesize
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator

import httpx

import host_script

log = logging.getLogger(__name__)

# ── System prompts for each speaker role ─────────────────────────────────────

_OPENING_PROMPT = (
    "You are {speaker}, debating the topic: '{topic}'. "
    "You are arguing {stance}. "
    "Give a clear, confident opening statement in 2-3 sentences. "
    "Do not mention you are an AI."
)

_REBUTTAL_PROMPT = (
    "You are {speaker}, debating the topic: '{topic}'. "
    "You are arguing {stance}. "
    "Your opponent just said: \"{opponent_argument}\". "
    "First, directly rebut that point in one sentence. "
    "Then advance a new argument of your own in 1-2 sentences. "
    "Total: 2-3 sentences. Do not mention you are an AI."
)

# ── Coach steering ────────────────────────────────────────────────────────────
# A coach can inject a one-off instruction (e.g. "be more aggressive") that gets
# added to the next generated argument's prompt, then is cleared so it doesn't
# repeat on later turns.
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


async def _generate_argument(
    ollama_url: str,
    model: str,
    prompt: str,
) -> str:
    """Call Ollama and return the generated text."""
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
    """
    Call /synthesize and return base64-encoded WAV bytes,
    or None if synthesis fails (debate continues without audio).
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{emotion_api}/synthesize",
                json={"text": text, "provider": "kokoro", "speaker_role": speaker_role, "style": style},
            )
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
    except Exception as exc:
        log.warning("TTS synthesis failed for role '%s': %s", speaker_role, exc)
        return None


async def _judge_winner(judge_api: str, turns: list[dict]) -> dict:
    """Call the Judge API winner endpoint for final scoring."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{judge_api}/winner",
                json={"turns": turns},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        log.warning("Judge API failed, falling back to composure winner: %s", exc)
        return {}


def _composure_winner(transcript: list[dict], llm_a: str, llm_b: str) -> tuple[str, str]:
    """Fallback winner by average composure if Judge API is unavailable."""
    scores = {llm_a: [], llm_b: []}
    for turn in transcript:
        speaker = turn.get("speaker")
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
) -> AsyncIterator[dict]:
    """
    Async generator that yields debate events as dicts.

    Event types:
      host_intro   — {type, text, audio_b64}
      argument     — {type, speaker, role, round, text, scores, audio_b64}
      scores_update— {type, llm_a_avg_composure, llm_b_avg_composure}
      winner       — {type, winner, explanation, text, audio_b64}
      error        — {type, message}
    """
    global _paused
    transcript: list[dict] = []
    last_a_arg = ""
    last_b_arg = ""

    # Reset pause state for each new debate
    _paused = False

    # ── Host intro ────────────────────────────────────────────────────────────
    intro_text = host_script.intro(topic, llm_a, llm_b, turn_count)
    intro_audio = await _synthesize(emotion_api, intro_text, "host", style="cheerful")
    yield {"type": "host_intro", "text": intro_text, "audio_b64": intro_audio}

    # ── Debate turns ──────────────────────────────────────────────────────────
    # turn_count is total turns; speakers alternate A, B, A, B, ...
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

        # ── Pause check — polls every 0.5s until resumed ──────────────────
        while _paused:
            await asyncio.sleep(0.5)

        coach_instruction = _consume_steering(is_llm_a)
        if coach_instruction:
            prompt = (
                f"COACH INSTRUCTION (mandatory): {coach_instruction}. "
                f"Concretely: use short punchy sentences, strong emotionally charged words "
                f"(e.g. 'absurd', 'dangerously naive', 'completely fails'), express visible frustration "
                f"or urgency, and challenge the opponent directly by name rather than abstractly. "
                f"Do not soften your tone with hedging phrases like 'while it's true' or 'admittedly'. "
                f"\n\n{prompt}"
            )

        # Generate + score + synthesise in parallel where possible
        try:
            argument_text = await _generate_argument(ollama_url, ollama_model, prompt)
        except Exception as exc:
            yield {"type": "error", "message": f"Ollama generation failed: {exc}"}
            return

        scores, audio_b64 = await asyncio.gather(
            _score_argument(emotion_api, argument_text),
            _synthesize(emotion_api, argument_text, role,
                        style="cheerful" if is_opening else "empathetic"),
        )

        if is_llm_a:
            last_a_arg = argument_text
        else:
            last_b_arg = argument_text

        turn_record = {
            "speaker": speaker,
            "role": role,
            "round": round_num,
            "text": argument_text,
            "scores": scores,
            "speaker_id": role.upper().replace("_", "-"),
        }
        transcript.append(turn_record)

        yield {
            "type": "argument",
            "speaker": speaker,
            "role": role,
            "round": round_num,
            "text": argument_text,
            "scores": scores,
            "audio_b64": audio_b64,
        }

        # ── Pause check after each argument ──────────────────────────────
        while _paused:
            await asyncio.sleep(0.5)

        # Running composure averages for live score panel
        a_scores = [t["scores"]["composure"] for t in transcript if t["role"] == "llm_a"]
        b_scores = [t["scores"]["composure"] for t in transcript if t["role"] == "llm_b"]
        yield {
            "type": "scores_update",
            "llm_a_avg_composure": round(sum(a_scores) / len(a_scores), 3) if a_scores else None,
            "llm_b_avg_composure": round(sum(b_scores) / len(b_scores), 3) if b_scores else None,
        }

        # After each full round (Con just spoke), pause for coach
        is_last_turn = (turn_index == turn_count - 1)
        if not is_llm_a and not is_last_turn:
            _paused = True
            yield {"type": "round_complete", "round": round_num}
            while _paused:
                await asyncio.sleep(0.5)

    # ── Winner determination ──────────────────────────────────────────────────
    judge_turns = [
        {"round": t["round"], "speaker_id": t["speaker_id"], "text": t["text"]}
        for t in transcript
    ]
    judge_result = await _judge_winner(judge_api, judge_turns)

    if judge_result.get("winner"):
        winner_id  = judge_result["winner"]
        winner     = llm_a if "A" in winner_id.upper() else llm_b
        explanation = judge_result.get("explanation", "")
    else:
        winner, explanation = _composure_winner(transcript, llm_a, llm_b)

    if winner == "tie":
        outro_text = host_script.tie_announcement()
    else:
        outro_text = host_script.winner_announcement(winner, explanation)

    outro_audio = await _synthesize(emotion_api, outro_text, "host", style="cheerful")

    yield {
        "type": "winner",
        "winner": winner,
        "explanation": explanation,
        "text": outro_text,
        "audio_b64": outro_audio,
    }