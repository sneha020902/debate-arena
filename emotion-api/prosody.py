"""
prosody.py — Prosodic feature extraction for voice emotion
============================================================
DEBATE Project · Team Emotion · Bauhaus-Universität Weimar · Webis Lab · SS 2026

The Task 4 brief requires the voice-emotion endpoint to return, alongside the
emotion labels, a *prosodic feature vector*:

    pitch variation · speech rate · volume dynamics · vocal tension

The speech model (wav2vec2) only outputs categorical emotion probabilities.
Prosody is the *acoustic*, model-independent description of *how* something was
said — it is what lets Team Logic reason about delivery (e.g. "emotional
composure" in the composite winner formula) separately from word content.

This module computes those four feature groups from a raw waveform using
librosa, plus a small set of supporting low-level descriptors. Everything is
returned as plain floats so the vector is trivially serialisable to JSON and
directly consumable by downstream scoring.

Design notes
------------
- We operate on a mono waveform at a known sample rate (the caller is expected
  to pass audio already loaded by SpeechEmotionRecognizer's loaders, so we do
  not re-implement file decoding here — single responsibility).
- All four headline features are normalised to a roughly 0..1 range so they can
  be compared across clips and folded into a weighted score. Raw physical units
  (Hz, syllables/sec, dB) are *also* returned under `raw` for interpretability.
- "Vocal tension" has no single canonical definition. We approximate it with a
  blend of jitter (cycle-to-cycle f0 instability), high-frequency spectral
  energy (spectral centroid / rolloff), and zero-crossing rate — all of which
  rise with strained, pressed phonation. This is documented as an approximation,
  per the brief's instruction to justify choices.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


# Reference ranges used to normalise raw physical measurements into 0..1.
# These are deliberately conservative, speech-oriented bounds; they are tuning
# constants, not ground truth, and are documented so they can be adjusted.
_PITCH_MIN_HZ = 75.0      # low end of typical adult speaking f0
_PITCH_MAX_HZ = 400.0     # high end of typical adult speaking f0
_PITCH_STD_REF_HZ = 60.0  # an f0 std-dev of ~60 Hz is treated as "very expressive"
_RATE_REF_SYL_PER_SEC = 6.0   # ~6 syllables/sec is fast conversational speech
_RMS_DYNAMIC_REF = 0.15       # std-dev of frame RMS treated as "high dynamics"


def _safe_float(x, default: float = 0.0) -> float:
    """Convert to a finite python float, falling back to default on NaN/inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _normalise(value: float, ref: float) -> float:
    """Map a non-negative measurement to 0..1 using a soft reference scale."""
    if ref <= 0:
        return 0.0
    return float(np.clip(value / ref, 0.0, 1.0))


def extract_prosody(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    duration_seconds: Optional[float] = None,
) -> Dict:
    """
    Extract the prosodic feature vector from a mono waveform.

    Args:
        waveform: 1-D float array of audio samples (mono), values roughly in
            [-1, 1]. Accepts a torch tensor too (converted via np.asarray).
        sample_rate: Sample rate of the waveform in Hz.
        duration_seconds: Optional precomputed duration; derived from the
            waveform length if omitted.

    Returns:
        Dict with structure:
        {
            "pitch_variation":  float (0..1),
            "speech_rate":      float (0..1),
            "volume_dynamics":  float (0..1),
            "vocal_tension":    float (0..1),
            "feature_vector":   [pitch_variation, speech_rate,
                                 volume_dynamics, vocal_tension],
            "raw": {
                "pitch_mean_hz": float,
                "pitch_std_hz": float,
                "voiced_fraction": float,
                "rate_syllables_per_sec": float,
                "rms_mean": float,
                "rms_std": float,
                "jitter": float,
                "spectral_centroid_hz": float,
                "zero_crossing_rate": float,
                "duration_seconds": float
            }
        }

    The function never raises on degenerate audio; it returns zeros for any
    feature it cannot compute and reports the issue in raw["note"].
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "prosody.py requires librosa. Install with: pip install librosa"
        ) from exc

    y = np.asarray(waveform, dtype=np.float32).squeeze()
    if y.ndim > 1:
        y = y.mean(axis=0)

    note = None
    if y.size == 0:
        return _empty_result(note="empty waveform")

    if duration_seconds is None:
        duration_seconds = y.size / float(sample_rate)
    duration_seconds = max(duration_seconds, 1e-6)

    # ---- 1. PITCH (f0) variation --------------------------------------------
    # librosa.pyin gives a per-frame f0 track with NaN on unvoiced frames.
    pitch_mean_hz = 0.0
    pitch_std_hz = 0.0
    voiced_fraction = 0.0
    jitter = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=_PITCH_MIN_HZ,
            fmax=_PITCH_MAX_HZ,
            sr=sample_rate,
        )
        voiced_f0 = f0[np.isfinite(f0)]
        if voiced_f0.size > 0:
            pitch_mean_hz = _safe_float(np.mean(voiced_f0))
            pitch_std_hz = _safe_float(np.std(voiced_f0))
            voiced_fraction = _safe_float(voiced_f0.size / f0.size)
            # Jitter: mean absolute relative difference between consecutive f0.
            if voiced_f0.size > 1:
                diffs = np.abs(np.diff(voiced_f0))
                denom = voiced_f0[:-1]
                rel = diffs[denom > 0] / denom[denom > 0]
                jitter = _safe_float(np.mean(rel)) if rel.size else 0.0
    except Exception as exc:  # pyin can fail on very short / silent clips
        note = f"pitch extraction failed: {exc}"

    # Pitch variation feature: how expressive the intonation is.
    pitch_variation = _normalise(pitch_std_hz, _PITCH_STD_REF_HZ)

    # ---- 2. SPEECH RATE ------------------------------------------------------
    # Approximate syllable nuclei via onset detection on the energy envelope.
    rate_syllables_per_sec = 0.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sample_rate, units="frames"
        )
        n_events = int(len(onsets))
        rate_syllables_per_sec = _safe_float(n_events / duration_seconds)
    except Exception as exc:
        note = (note + " | " if note else "") + f"rate extraction failed: {exc}"

    speech_rate = _normalise(rate_syllables_per_sec, _RATE_REF_SYL_PER_SEC)

    # ---- 3. VOLUME DYNAMICS --------------------------------------------------
    # Frame-wise RMS energy; its spread captures loud/soft modulation.
    rms_mean = 0.0
    rms_std = 0.0
    try:
        rms = librosa.feature.rms(y=y)[0]
        if rms.size:
            rms_mean = _safe_float(np.mean(rms))
            rms_std = _safe_float(np.std(rms))
    except Exception as exc:
        note = (note + " | " if note else "") + f"rms extraction failed: {exc}"

    volume_dynamics = _normalise(rms_std, _RMS_DYNAMIC_REF)

    # ---- 4. VOCAL TENSION ----------------------------------------------------
    # Blend of jitter, spectral centroid (brightness/strain) and ZCR.
    spectral_centroid_hz = 0.0
    zero_crossing_rate = 0.0
    try:
        spectral_centroid_hz = _safe_float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate))
        )
        zero_crossing_rate = _safe_float(
            np.mean(librosa.feature.zero_crossing_rate(y))
        )
    except Exception as exc:
        note = (note + " | " if note else "") + f"tension extraction failed: {exc}"

    # Normalise the three tension contributors and average them.
    jitter_n = _normalise(jitter, 0.05)            # 5% f0 jitter ~ very tense
    centroid_n = _normalise(spectral_centroid_hz, 3000.0)  # bright/pressed voice
    zcr_n = _normalise(zero_crossing_rate, 0.15)
    vocal_tension = float(np.clip((jitter_n + centroid_n + zcr_n) / 3.0, 0.0, 1.0))

    feature_vector = [
        round(pitch_variation, 4),
        round(speech_rate, 4),
        round(volume_dynamics, 4),
        round(vocal_tension, 4),
    ]

    raw = {
        "pitch_mean_hz": round(pitch_mean_hz, 2),
        "pitch_std_hz": round(pitch_std_hz, 2),
        "voiced_fraction": round(voiced_fraction, 4),
        "rate_syllables_per_sec": round(rate_syllables_per_sec, 2),
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "jitter": round(jitter, 5),
        "spectral_centroid_hz": round(spectral_centroid_hz, 1),
        "zero_crossing_rate": round(zero_crossing_rate, 5),
        "duration_seconds": round(duration_seconds, 3),
    }
    if note:
        raw["note"] = note

    return {
        "pitch_variation": round(pitch_variation, 4),
        "speech_rate": round(speech_rate, 4),
        "volume_dynamics": round(volume_dynamics, 4),
        "vocal_tension": round(vocal_tension, 4),
        "feature_vector": feature_vector,
        "raw": raw,
    }


def _empty_result(note: str) -> Dict:
    """Return a zeroed prosody result with an explanatory note."""
    return {
        "pitch_variation": 0.0,
        "speech_rate": 0.0,
        "volume_dynamics": 0.0,
        "vocal_tension": 0.0,
        "feature_vector": [0.0, 0.0, 0.0, 0.0],
        "raw": {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "voiced_fraction": 0.0,
            "rate_syllables_per_sec": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "jitter": 0.0,
            "spectral_centroid_hz": 0.0,
            "zero_crossing_rate": 0.0,
            "duration_seconds": 0.0,
            "note": note,
        },
    }
