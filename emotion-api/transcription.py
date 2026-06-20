"""
transcription.py — Voice-to-Text (Whisper)
===========================================
DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Implements the Task 4 "Voice-to-Text Transcription" requirement:
accept an audio file and return transcript text, *word-level timestamps*,
and *confidence scores*.

Model choice — faster-whisper
-----------------------------
The brief allows any approach. We use `faster-whisper` (a CTranslate2
re-implementation of OpenAI Whisper) rather than the reference `openai-whisper`
package because:

  * It returns per-word timestamps AND a per-word probability out of the box
    (word.probability), which maps directly onto the "confidence scores" the
    brief asks for. Vanilla openai-whisper only exposes segment-level
    avg_logprob unless word_timestamps=True, and even then word probabilities
    are less convenient.
  * It is 4-5x faster and uses less memory for the same accuracy, which matters
    for a live demo on a shared student machine.
  * It does not require a separate ffmpeg invocation for WAV/MP3 (it decodes
    via PyAV), reducing system dependencies.

If the team prefers openai-whisper, the public function signature here is the
same shape they would build, so swapping the backend is a localised change.

Confidence
----------
Whisper is generative, so "confidence" is derived from token log-probabilities.
We expose:
  * per word: word.probability (already 0..1)
  * per segment: exp(avg_logprob) as a 0..1 segment confidence
  * overall: duration-weighted mean of segment confidence
"""

from __future__ import annotations

from typing import Dict, Optional
import math

_model = None
_model_size = "base"


def get_model(model_size: str = "base"):
    """
    Lazy-load the Whisper model once and cache it.

    Args:
        model_size: Whisper size ("tiny", "base", "small", "medium", "large-v3").
            "base" is a good speed/accuracy trade-off for a demo.
    """
    global _model, _model_size
    if _model is None or model_size != _model_size:
        from faster_whisper import WhisperModel

        # int8 keeps it light on CPU; use "float16" + device="cuda" on GPU.
        _model = WhisperModel(model_size, device="auto", compute_type="int8")
        _model_size = model_size
    return _model


def _logprob_to_prob(logprob: float) -> float:
    """Convert a natural-log probability into a 0..1 linear confidence."""
    try:
        return max(0.0, min(1.0, math.exp(logprob)))
    except (OverflowError, ValueError):
        return 0.0


def transcribe_file(
    audio_path: str,
    *,
    language: Optional[str] = None,
    model_size: str = "base",
) -> Dict:
    """
    Transcribe an audio file to text with word-level timestamps and confidence.

    Args:
        audio_path: Path to a WAV / MP3 / WebM file on disk.
        language: Optional ISO language code (e.g. "en"). None = auto-detect.
        model_size: Whisper model size to use.

    Returns:
        {
            "text": str,                       # full transcript
            "language": str,                   # detected or supplied language
            "language_probability": float,     # detector confidence (0..1)
            "duration": float,                 # audio duration in seconds
            "overall_confidence": float,       # duration-weighted segment conf.
            "segments": [
                {
                    "id": int,
                    "start": float, "end": float,
                    "text": str,
                    "confidence": float,       # exp(avg_logprob)
                    "words": [
                        {"word": str, "start": float,
                         "end": float, "confidence": float},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    model = get_model(model_size)

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,  # drop long silences -> cleaner debate timestamps
    )

    segments = []
    full_text_parts = []
    weighted_conf_sum = 0.0
    total_dur = 0.0

    for seg in segments_iter:
        seg_conf = _logprob_to_prob(seg.avg_logprob)
        seg_dur = max(seg.end - seg.start, 1e-6)
        weighted_conf_sum += seg_conf * seg_dur
        total_dur += seg_dur

        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(float(w.probability), 4),
                })

        segments.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "confidence": round(seg_conf, 4),
            "words": words,
        })
        full_text_parts.append(seg.text.strip())

    overall_conf = round(weighted_conf_sum / total_dur, 4) if total_dur > 0 else 0.0

    return {
        "text": " ".join(full_text_parts).strip(),
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "duration": round(float(info.duration), 3),
        "overall_confidence": overall_conf,
        "segments": segments,
    }
