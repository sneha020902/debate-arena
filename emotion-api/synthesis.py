"""
synthesis.py — Text-to-Voice (TTS)
==================================
DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Provider: kokoro, running as a SEPARATE microservice (Singularity container
on SLURM) to avoid dependency conflicts with the main emotion-api venv.

This file calls that microservice over HTTP instead of importing kokoro
directly. See kokoro_service/ for the standalone service + container.
"""

from __future__ import annotations

import os

import httpx

# ── Kokoro voice map ─────────────────────────────────────────────────────────
# af_heart is warm/neutral, good default for the host announcer.
# am_adam / bf_emma used for LLM-A / LLM-B respectively.
KOKORO_VOICES = {
    "host":   "af_heart",
    "llm_a":  "am_adam",
    "llm_b":  "bf_emma",
    "default": "af_heart",
}

# Emotion → speed multiplier for Kokoro (pitch is not exposed directly)
KOKORO_SPEED_PRESETS = {
    "default":    1.0,
    "angry":      1.2,
    "cheerful":   1.25,
    "sad":        0.8,
    "fearful":    1.3,
    "whispering": 0.85,
    "shouting":   1.35,
    "unfriendly": 0.9,
    "hopeful":    1.1,
    "excited":    1.35,
    "empathetic": 0.85,
}

# URL of the standalone Kokoro microservice (Singularity container on SLURM,
# or localhost if you run kokoro_service/main.py directly for local testing).
# Override with the KOKORO_SERVICE_URL env var if it runs elsewhere.
KOKORO_SERVICE_URL = os.environ.get("KOKORO_SERVICE_URL", "http://localhost:8003")


async def synthesize_to_bytes(
    text: str,
    voice: str = "default",
    *,
    style: str = "default",
    speaker_role: str = "default",
    # legacy edge-tts params kept for API compatibility — ignored by Kokoro
    rate: str = "+0%",
    pitch: str = "+0Hz",
    style_degree: float = 1.0,
) -> bytes:
    """
    Synthesise `text` to audio bytes by calling the Kokoro microservice.

    Args:
        text:         The text to speak.
        voice:        A Kokoro voice name, or one of the role shortcuts:
                      'host', 'llm_a', 'llm_b', 'default'.
        style:        Speaking style (angry, cheerful, sad, etc.) — mapped to speed.
        speaker_role: Alternative to voice — 'host', 'llm_a', or 'llm_b'.
        rate, pitch, style_degree: Kept for backwards-compatibility; not used.

    Returns:
        Raw WAV bytes (24 kHz, mono).

    Raises:
        ValueError:   if `text` is empty.
        RuntimeError: if the Kokoro service is unreachable or synthesis fails.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    resolved_voice = KOKORO_VOICES.get(speaker_role, KOKORO_VOICES.get(voice, KOKORO_VOICES["default"]))
    speed = KOKORO_SPEED_PRESETS.get(style, 1.0)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{KOKORO_SERVICE_URL}/synthesize",
                json={"text": text, "voice": resolved_voice, "speed": speed},
            )
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Kokoro service returned an error: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Could not reach Kokoro service at {KOKORO_SERVICE_URL}. "
            f"Is it running? (see kokoro_service/run_kokoro.sh)"
        ) from exc