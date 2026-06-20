"""
Canonical emotion normalization.

The text model (j-hartmann/emotion-english-distilroberta-base) and the speech
model (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) were trained
on different datasets with different annotation schemas:

  Text  → Ekman-based NLP corpora (ISEAR, MELD, SemEval …)
  Speech → RAVDESS (acted speech, 2018 Ryerson taxonomy)

Only "disgust" and "neutral" share the same name. Everything else differs in
spelling or concept. This module maps both to a single canonical set so that
cross-modal comparisons are label-agnostic.

Canonical set (7, follows the text model's schema):
  anger, disgust, fear, joy, neutral, sadness, surprise

RAVDESS "calm" has no text-model equivalent — it maps to "neutral" (closest
acoustic and semantic match).
"""

# Maps each model's raw label → canonical label
_TEXT_TO_CANONICAL = {
    "anger":   "anger",
    "disgust": "disgust",
    "fear":    "fear",
    "joy":     "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}

_SPEECH_TO_CANONICAL = {
    "angry":    "anger",
    "calm":     "neutral",   # no text equivalent; nearest is neutral
    "disgust":  "disgust",
    "fearful":  "fear",
    "happy":    "joy",
    "neutral":  "neutral",
    "sad":      "sadness",
    "surprised": "surprise",
}

CANONICAL_EMOTIONS = list(_TEXT_TO_CANONICAL.keys())


def normalize_text_emotion(label: str) -> str:
    return _TEXT_TO_CANONICAL.get(label.lower(), label.lower())


def normalize_speech_emotion(label: str) -> str:
    return _SPEECH_TO_CANONICAL.get(label.lower(), label.lower())


def normalize(label: str, modality: str) -> str:
    """modality: 'text' or 'speech'"""
    if modality == "speech":
        return normalize_speech_emotion(label)
    return normalize_text_emotion(label)
