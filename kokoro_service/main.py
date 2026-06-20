"""
kokoro_service/main.py — Standalone Kokoro TTS microservice
=============================================================
DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Runs INSIDE a Singularity container on SLURM, isolated from the main
emotion-api venv to avoid dependency conflicts (spacy/thinc/blis build
failures on the shared environment).

Exposes a single endpoint:
    POST /synthesize  {text, voice, speed}  →  raw WAV bytes (24kHz mono)

The main emotion-api's synthesis.py calls this over HTTP instead of
importing kokoro directly.
"""

from __future__ import annotations

import io
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Kokoro TTS Microservice", version="1.0.0")

_kokoro_pipeline = None  # lazily initialised on first request


def _get_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        log.info("Loading Kokoro pipeline (first request — downloads model if needed)...")
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
        log.info("Kokoro pipeline ready.")
    return _kokoro_pipeline


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field("af_heart", description="Kokoro voice name")
    speed: float = Field(1.0, description="Speech speed multiplier")


@app.get("/health")
def health():
    return {"status": "ok", "service": "kokoro-tts"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be non-empty")

    try:
        import numpy as np
        import soundfile as sf

        pipeline = _get_kokoro()
        audio_chunks = []
        for _, _, audio in pipeline(req.text, voice=req.voice, speed=req.speed, split_pattern=r"\n+"):
            audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Kokoro produced no audio output.")

        combined = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, combined, samplerate=24000, format="WAV")
        wav_bytes = buf.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Kokoro synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}")
