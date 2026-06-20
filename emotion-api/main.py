"""
main.py — Emotion API
=====================
FastAPI service for Team Emotion — DEBATE Project
Bauhaus-Universität Weimar · Webis Lab · SS 2026

Run:
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import os
import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import text_emotion
import emotion_utils
import transcription
import synthesis
from speech_emotion_module import SpeechEmotionRecognizer


# ── Schemas ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "emotion-api"
    version: str = "2.0.0"


class TextEmotionRequest(BaseModel):
    text: str = Field(..., description="Debate argument to analyse")
    speaker_id: Optional[str] = Field(None, description="Optional speaker identifier")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "I am absolutely furious about this policy!",
                "speaker_id": "LLM-A",
            }]
        }
    }


class TextEmotionResponse(BaseModel):
    speaker_id:       Optional[str]
    dominant_emotion: str
    confidence:       float
    emotion_vector:   dict[str, float]
    stance_label:     str
    error:            Optional[str] = None


class BatchTurn(BaseModel):
    round:      int
    speaker_id: str
    text:       str


class BatchTextEmotionRequest(BaseModel):
    turns: list[BatchTurn]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "turns": [
                    {"round": 1, "speaker_id": "LLM-A", "text": "This policy is completely wrong!"},
                    {"round": 2, "speaker_id": "LLM-B", "text": "I respectfully disagree."},
                ]
            }]
        }
    }


class BatchTurnResult(BaseModel):
    round:            int
    speaker_id:       str
    dominant_emotion: str
    confidence:       float
    stance_label:     str
    arousal_proxy:    float


class ShiftSummary(BaseModel):
    dominant_sequence: list[str]
    arousal_trend:     list[float]
    stance_trend:      list[str]
    notable_shifts:    list[str]


class BatchTextEmotionResponse(BaseModel):
    turns:         list[BatchTurnResult]
    shift_summary: ShiftSummary


class ProsodyFeatures(BaseModel):
    pitch_variation: float = Field(..., description="Intonation expressiveness, 0..1")
    speech_rate:     float = Field(..., description="Speaking speed, 0..1")
    volume_dynamics: float = Field(..., description="Loudness modulation, 0..1")
    vocal_tension:   float = Field(..., description="Strain/pressed phonation, 0..1")
    feature_vector:  list[float] = Field(..., description="[pitch, rate, volume, tension]")
    raw:             dict = Field(..., description="Raw physical measurements (Hz, dB, etc.)")


class VoiceEmotionResponse(BaseModel):
    dominant_emotion:  str
    confidence:        float
    emotion_vector:    dict[str, float]
    chunks_analyzed:   int
    canonical_emotion: str
    prosody:           ProsodyFeatures
    error:             Optional[str] = None


# ── Transcription ─────────────────────────────────────────────────────────────

class TranscriptWord(BaseModel):
    word:       str
    start:      float
    end:        float
    confidence: float


class TranscriptSegment(BaseModel):
    id:         int
    start:      float
    end:        float
    text:       str
    confidence: float
    words:      list[TranscriptWord]


class TranscriptionResponse(BaseModel):
    text:                 str
    language:             str
    language_probability: float
    duration:             float
    overall_confidence:   float
    segments:             list[TranscriptSegment]


# ── Synthesis ─────────────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text:         str = Field(...,        description="Text to speak")
    speaker_role: str = Field("default", description="Voice role: 'host', 'llm_a', 'llm_b', or 'default'")
    style:        str = Field("default", description="Speed/style preset: angry, cheerful, sad, empathetic, etc.")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "Welcome to DebateArena. Today's topic: should AI replace human judges?",
                "speaker_role": "host",
                "style": "cheerful",
            }]
        }
    }


# ── Cross-modal comparison ────────────────────────────────────────────────────

class CompareResponse(BaseModel):
    text_input:        str
    text_emotion:      str
    text_canonical:    str
    text_confidence:   float
    voice_emotion:     str
    voice_canonical:   str
    voice_confidence:  float
    agree:             bool
    arousal_text:      float
    arousal_voice:     float
    interpretation:    str
    prosody:           ProsodyFeatures


# ── Delivery Analysis ─────────────────────────────────────────────────────────

class DeliveryVector(BaseModel):
    reasoning_density: float = Field(..., description="0=pure emotion, 1=pure logic/evidence")
    intensity:         float = Field(..., description="0=very calm, 1=very aggressive")
    yielding:          float = Field(..., description="0=dominant/dismissive, 1=conciliatory/open")
    focus:             float = Field(..., description="0=breadth/many short points, 1=depth/one deep point")


class DeliveryRequest(BaseModel):
    text:       str           = Field(..., description="Debate argument to analyse")
    speaker_id: Optional[str] = Field(None, description="Optional speaker identifier")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "The evidence clearly shows this policy is outrageous and must be stopped immediately.",
                "speaker_id": "LLM-A",
            }]
        }
    }


class DeliveryResponse(BaseModel):
    speaker_id:          Optional[str]
    dominant_emotion:    str
    confidence:          float
    emotion_vector:      dict[str, float]
    stance_label:        str
    emotional_direction: str
    intensity_level:     float          # ← CHANGED: was str ("high"/"medium"/"low"), now float 0.0–1.0
    delivery_vector:     DeliveryVector
    error:               Optional[str] = None


# ── Arousal map ───────────────────────────────────────────────────────────────

AROUSAL = {
    "anger": 0.9, "fear": 0.8, "surprise": 0.7,
    "joy": 0.6, "disgust": 0.55, "sadness": 0.35, "neutral": 0.1,
}


# ── Startup ───────────────────────────────────────────────────────────────────

_speech_recognizer: Optional[SpeechEmotionRecognizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _speech_recognizer
    print("Loading text emotion model...")
    text_emotion.get_classifier()
    print("Initializing speech emotion recognizer...")
    _speech_recognizer = SpeechEmotionRecognizer(use_gpu=True)
    print("Warming up Whisper transcription model...")
    try:
        transcription.get_model("base")
    except Exception as exc:
        print(f"  (Whisper not preloaded: {exc})")
    print("All models ready. (/synthesize uses Kokoro locally — no API key needed.)")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DEBATE — Emotion API",
    description="Affective layer for the DEBATE pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Liveness check."""
    return HealthResponse()


# ── /emotion/text ─────────────────────────────────────────────────────────────

@app.post("/emotion/text", response_model=TextEmotionResponse, tags=["Text Emotion"])
def emotion_text(request: TextEmotionRequest):
    """
    Detect emotion in a single debate argument.

    Returns dominant emotion, confidence, full emotion vector, and stance label.
    Minimum 5 words required for reliable results.
    """
    text = request.text.strip()

    if len(text.split()) < 5:
        raise HTTPException(
            status_code=422,
            detail="Text too short — provide at least 5 words for reliable results.",
        )

    try:
        result = text_emotion.analyse_text(text, text_emotion.get_classifier())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

    return TextEmotionResponse(
        speaker_id=request.speaker_id,
        dominant_emotion=result["dominant_emotion"],
        confidence=result["confidence"],
        emotion_vector=result["emotion_vector"],
        stance_label=result["stance_label"],
    )


# ── /emotion/text/batch ───────────────────────────────────────────────────────

@app.post("/emotion/text/batch", response_model=BatchTextEmotionResponse, tags=["Text Emotion"])
def emotion_text_batch(request: BatchTextEmotionRequest):
    """
    Analyse a full multi-turn debate transcript.

    Returns per-turn emotion results and a shift summary showing how
    emotion, stance, and arousal change across rounds.
    """
    if not request.turns:
        raise HTTPException(status_code=422, detail="Provide at least 1 turn.")

    classifier = text_emotion.get_classifier()
    turn_results = []

    for turn in request.turns:
        try:
            r = text_emotion.analyse_text(turn.text, classifier)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error on round {turn.round}: {exc}")

        turn_results.append(BatchTurnResult(
            round=turn.round,
            speaker_id=turn.speaker_id,
            dominant_emotion=r["dominant_emotion"],
            confidence=r["confidence"],
            stance_label=r["stance_label"],
            arousal_proxy=AROUSAL.get(r["dominant_emotion"], 0.5),
        ))

    notable_shifts = []
    for i in range(1, len(turn_results)):
        prev, curr = turn_results[i - 1], turn_results[i]
        parts = []
        if prev.dominant_emotion != curr.dominant_emotion:
            parts.append(
                f"Round {prev.round}→{curr.round}: {curr.speaker_id} "
                f"shifted from {prev.dominant_emotion} to {curr.dominant_emotion}"
            )
        if prev.stance_label != curr.stance_label:
            parts.append(f"stance changed from {prev.stance_label} to {curr.stance_label}")
        if parts:
            notable_shifts.append(". ".join(parts))

    return BatchTextEmotionResponse(
        turns=turn_results,
        shift_summary=ShiftSummary(
            dominant_sequence=[r.dominant_emotion for r in turn_results],
            arousal_trend=[r.arousal_proxy for r in turn_results],
            stance_trend=[r.stance_label for r in turn_results],
            notable_shifts=notable_shifts,
        ),
    )


# ── /emotion/voice ────────────────────────────────────────────────────────────

@app.post("/emotion/voice", response_model=VoiceEmotionResponse, tags=["Voice Emotion"])
async def emotion_voice(file: UploadFile = File(...)):
    """
    Detect emotion from a raw audio file, with a prosodic feature vector.

    Returns speech-model emotion labels, canonical emotion, and the
    prosodic feature vector: pitch variation, speech rate, volume dynamics,
    and vocal tension.
    """
    if _speech_recognizer is None:
        raise HTTPException(status_code=503, detail="Speech model not loaded yet.")

    suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = _speech_recognizer.predict_emotion_with_prosody(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice analysis error: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    canonical = emotion_utils.normalize_speech_emotion(result["primary_emotion"])
    return VoiceEmotionResponse(
        dominant_emotion=result["primary_emotion"],
        confidence=result["confidence"],
        emotion_vector=result["all_emotions"],
        chunks_analyzed=result["chunks_analyzed"],
        canonical_emotion=canonical,
        prosody=ProsodyFeatures(**result["prosody"]),
    )


# ── /transcribe ───────────────────────────────────────────────────────────────

@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Voice Emotion"],
          summary="Audio → transcript with word timestamps + confidence")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    """
    Transcribe an audio file to text with word-level timestamps and confidence.
    Accepts WAV, MP3, or WebM.
    """
    suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = transcription.transcribe_file(tmp_path, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return TranscriptionResponse(**result)


# ── /synthesize ───────────────────────────────────────────────────────────────

@app.post("/synthesize", tags=["Voice Emotion"], summary="Text → audio (WAV)")
async def synthesize(request: SynthesizeRequest):
    """Convert text to spoken audio using Kokoro (local, no API key needed)."""
    try:
        audio_bytes = await synthesis.synthesize_to_bytes(
            request.text,
            speaker_role=request.speaker_role,
            style=request.style,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Missing API key, missing package, or the provider call itself failed
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


# ── /emotion/compare ──────────────────────────────────────────────────────────

@app.post("/emotion/compare", response_model=CompareResponse, tags=["Voice Emotion"],
          summary="Compare text-emotion vs voice-emotion on the same segment")
async def emotion_compare(text: str, file: UploadFile = File(...)):
    """
    Run BOTH the text model and the voice model on the same debate segment,
    normalise both to canonical labels, and report whether they agree.
    """
    if _speech_recognizer is None:
        raise HTTPException(status_code=503, detail="Speech model not loaded yet.")

    text = (text or "").strip()
    if len(text.split()) < 5:
        raise HTTPException(status_code=422, detail="Provide at least 5 words of text.")

    try:
        text_result = text_emotion.analyse_text(text, text_emotion.get_classifier())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text model error: {exc}") from exc

    suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        voice_result = _speech_recognizer.predict_emotion_with_prosody(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice analysis error: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not voice_result.get("success"):
        raise HTTPException(status_code=500, detail=voice_result.get("error", "Voice error"))

    canon_text  = emotion_utils.normalize_text_emotion(text_result["dominant_emotion"])
    canon_voice = emotion_utils.normalize_speech_emotion(voice_result["primary_emotion"])
    ar_text  = AROUSAL.get(canon_text,  0.5)
    ar_voice = AROUSAL.get(canon_voice, 0.5)
    agree    = canon_text == canon_voice

    if agree:
        interp = (f"Text and voice agree on '{canon_text}'. "
                  "Semantic and prosodic signals are consistent.")
    elif ar_voice > ar_text + 0.1:
        interp = (f"Mismatch: text reads '{canon_text}', voice reads '{canon_voice}'. "
                  "Delivery carries higher arousal than the words alone suggest.")
    elif ar_voice < ar_text - 0.1:
        interp = (f"Mismatch: text reads '{canon_text}', voice reads '{canon_voice}'. "
                  "Delivery is calmer than the word choice implies.")
    else:
        interp = (f"Labels differ ('{canon_text}' vs '{canon_voice}') but arousal "
                  "is comparable — similar intensity, different colour.")

    return CompareResponse(
        text_input=text,
        text_emotion=text_result["dominant_emotion"],
        text_canonical=canon_text,
        text_confidence=text_result["confidence"],
        voice_emotion=voice_result["primary_emotion"],
        voice_canonical=canon_voice,
        voice_confidence=voice_result["confidence"],
        agree=agree,
        arousal_text=ar_text,
        arousal_voice=ar_voice,
        interpretation=interp,
        prosody=ProsodyFeatures(**voice_result["prosody"]),
    )


# ── /emotion/delivery ─────────────────────────────────────────────────────────

@app.post("/emotion/delivery", response_model=DeliveryResponse, tags=["Delivery Analysis"],
          summary="Full delivery analysis — emotion + 4D delivery vector + emotional direction")
def emotion_delivery(request: DeliveryRequest):
    """
    Analyse the delivery style of a debate argument.

    Returns:
    - dominant_emotion and emotion_vector (from DistilRoBERTa)
    - stance_label       (aggressive / defensive / open / neutral)
    - emotional_direction (anger / fear / hope / compassion / neutral)
    - intensity_level    (float 0.0–1.0 — continuous scale, replaces high/medium/low)
    - delivery_vector with 4 dimensions:
        - reasoning_density : 0 = pure emotion,  1 = pure logic/evidence
        - intensity         : 0 = very calm,     1 = very aggressive
        - yielding          : 0 = dominant,       1 = conciliatory
        - focus             : 0 = breadth,        1 = depth

    Minimum 5 words required for reliable results.
    """
    text = request.text.strip()

    if len(text.split()) < 5:
        raise HTTPException(
            status_code=422,
            detail="Text too short — provide at least 5 words for reliable results.",
        )

    try:
        result = text_emotion.analyse_delivery(text, text_emotion.get_classifier())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

    return DeliveryResponse(
        speaker_id=request.speaker_id,
        dominant_emotion=result["dominant_emotion"],
        confidence=result["confidence"],
        emotion_vector=result["emotion_vector"],
        stance_label=result["stance_label"],
        emotional_direction=result["emotional_direction"],
        intensity_level=result["intensity_level"],
        delivery_vector=DeliveryVector(**result["delivery_vector"]),
    )