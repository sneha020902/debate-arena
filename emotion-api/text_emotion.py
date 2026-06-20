from transformers import pipeline

# ── Stance mapping ─────────────────────────────────────────────────────────────
_STANCE_MAP = {
    "anger":    "aggressive",
    "disgust":  "aggressive",
    "fear":     "defensive",
    "sadness":  "defensive",
    "joy":      "open",
    "surprise": "open",
    "neutral":  "neutral",
}

# ── Emotional direction mapping ────────────────────────────────────────────────
_DIRECTION_MAP = {
    "anger":    "anger",
    "disgust":  "anger",
    "fear":     "fear",
    "sadness":  "compassion",
    "joy":      "hope",
    "surprise": "neutral",
    "neutral":  "neutral",
}

# ── Keyword lists for delivery vector ─────────────────────────────────────────

# Reasoning density — high = logical/evidence heavy
_LOGIC_WORDS = [
    "evidence", "data", "research", "study", "studies", "proven", "shows",
    "statistics", "according", "findings", "analysis", "results", "fact",
    "measured", "documented", "reported", "published", "survey", "experiment",
    "concluded", "demonstrates", "indicates", "percentage", "significantly",
]

# Reasoning density — low = emotion heavy
_EMOTION_WORDS = [
    "feel", "believe", "imagine", "fear", "hope", "heart", "soul", "dream",
    "suffer", "pain", "joy", "love", "hate", "scared", "worried", "proud",
    "shame", "disgrace", "terrible", "wonderful", "awful", "beautiful",
]

# Intensity — high = aggressive/forceful
_AGGRESSIVE_WORDS = [
    "outrageous", "furious", "demand", "refuse", "unacceptable", "absolutely",
    "must", "cannot", "never", "always", "completely", "totally", "disgrace",
    "wrong", "failure", "pathetic", "ridiculous", "absurd", "nonsense",
    "immediately", "urgent", "critical", "dangerous", "threat", "crisis",
]

# Intensity — low = calm/measured
_CALM_WORDS = [
    "perhaps", "consider", "understand", "balanced", "carefully", "suggest",
    "possible", "might", "could", "reasonable", "thoughtful", "measured",
    "nuanced", "perspective", "acknowledge", "appreciate", "respectfully",
]

# Yielding — high = conciliatory/open
_YIELDING_WORDS = [
    "however", "although", "while", "granted", "admittedly", "i understand",
    "that said", "fair point", "you raise", "i acknowledge", "valid concern",
    "nonetheless", "despite", "even so", "i agree that", "to be fair",
]

# Yielding — low = dominant/dismissive
_DOMINANT_WORDS = [
    "completely wrong", "absolutely not", "impossible", "reject", "dismiss",
    "ignore", "false", "incorrect", "misleading", "nonsense", "baseless",
    "unfounded", "simply wrong", "not true", "clearly false",
]

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
    return _classifier


def get_stance(emotion: str) -> str:
    return _STANCE_MAP.get(emotion.lower(), "neutral")


def get_emotional_direction(emotion: str) -> str:
    return _DIRECTION_MAP.get(emotion.lower(), "neutral")


def _keyword_score(text: str, positive_words: list, negative_words: list) -> float:
    """Score 0-1 based on keyword presence. Positive words push toward 1, negative toward 0."""
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    pos_hits = sum(1 for w in positive_words if w in text_lower)
    neg_hits = sum(1 for w in negative_words if w in text_lower)

    # Normalize by text length — more hits relative to length = stronger signal
    pos_score = min(pos_hits / (total_words * 0.05 + 1), 1.0)
    neg_score = min(neg_hits / (total_words * 0.05 + 1), 1.0)

    # Blend: start at 0.5, push up for positive hits, down for negative
    raw = 0.5 + (pos_score * 0.5) - (neg_score * 0.5)
    return round(max(0.0, min(1.0, raw)), 3)


def compute_delivery_vector(text: str, emotion_vector: dict) -> dict:
    """
    Compute a 4-dimensional delivery vector for a debate argument.

    Dimensions:
    - reasoning_density : 0 = pure emotion, 1 = pure logic/evidence
    - intensity         : 0 = very calm, 1 = very aggressive
    - yielding          : 0 = dominant/dismissive, 1 = conciliatory/open
    - focus             : 0 = many short points (breadth), 1 = one deep point (depth)
    """

    # ── Reasoning Density ──────────────────────────────────────────────────────
    kw_logic = _keyword_score(text, _LOGIC_WORDS, _EMOTION_WORDS)
    emotional_score = (
        emotion_vector.get("anger",   0) +
        emotion_vector.get("fear",    0) +
        emotion_vector.get("disgust", 0) +
        emotion_vector.get("sadness", 0)
    )
    model_logic = 1.0 - emotional_score
    reasoning_density = round(0.6 * kw_logic + 0.4 * model_logic, 3)

    # ── Intensity ──────────────────────────────────────────────────────────────
    kw_intensity = _keyword_score(text, _AGGRESSIVE_WORDS, _CALM_WORDS)
    model_intensity = min(
        emotion_vector.get("anger",   0) +
        emotion_vector.get("disgust", 0) +
        emotion_vector.get("fear",    0) * 0.5,
        1.0
    )
    intensity = round(0.55 * kw_intensity + 0.45 * model_intensity, 3)

    # ── Yielding ───────────────────────────────────────────────────────────────
    kw_yielding = _keyword_score(text, _YIELDING_WORDS, _DOMINANT_WORDS)
    model_yielding = (
        emotion_vector.get("neutral",  0) * 0.5 +
        emotion_vector.get("joy",      0) * 0.3 +
        (1.0 - emotion_vector.get("anger", 0)) * 0.2
    )
    yielding = round(0.5 * kw_yielding + 0.5 * model_yielding, 3)

    # ── Focus (Breadth vs Depth) ───────────────────────────────────────────────
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    words = text.split()
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    avg_sentence_length = num_words / num_sentences
    focus = round(min(avg_sentence_length / 30.0, 1.0), 3)

    return {
        "reasoning_density": reasoning_density,
        "intensity":         intensity,
        "yielding":          yielding,
        "focus":             focus,
    }


def get_intensity_level(intensity: float) -> float:
    """
    Return intensity as a continuous float 0.0–1.0.

    Previously returned "high" / "medium" / "low" (categorical).
    Now returns the raw float so the API and demo can display
    exact values and render a continuous progress bar.

    The intensity value comes directly from compute_delivery_vector()
    and is already on a 0–1 scale:
      0.0–0.39  → previously "low"   (calm)
      0.40–0.69 → previously "medium"
      0.70–1.0  → previously "high"  (aggressive)
    """
    return round(float(intensity), 3)


def analyse_text(text: str, classifier=None) -> dict:
    clf = classifier if classifier is not None else get_classifier()
    results = clf(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    dominant = results[0]["label"]
    emotion_vector = {r["label"]: round(r["score"], 4) for r in results}

    return {
        "dominant_emotion": dominant,
        "confidence":       round(results[0]["score"], 4),
        "emotion_vector":   emotion_vector,
        "stance_label":     get_stance(dominant),
    }


def analyse_delivery(text: str, classifier=None) -> dict:
    """
    Full delivery analysis — returns emotion + stance + delivery vector
    + emotional direction + continuous intensity score.

    Change from Task 4/5:
      intensity_level is now a float 0.0–1.0 (continuous scale)
      instead of "high" / "medium" / "low".
    """
    # Step 1: run emotion model
    base = analyse_text(text, classifier)

    # Step 2: compute delivery vector
    delivery_vector = compute_delivery_vector(text, base["emotion_vector"])

    # Step 3: emotional direction
    emotional_direction = get_emotional_direction(base["dominant_emotion"])

    # Step 4: continuous intensity score (UPDATED — was categorical string)
    intensity_score = get_intensity_level(delivery_vector["intensity"])

    return {
        "dominant_emotion":    base["dominant_emotion"],
        "confidence":          base["confidence"],
        "emotion_vector":      base["emotion_vector"],
        "stance_label":        base["stance_label"],
        "emotional_direction": emotional_direction,
        "intensity_level":     intensity_score,   # now float, not string
        "delivery_vector": {
            "reasoning_density": delivery_vector["reasoning_density"],
            "intensity":         delivery_vector["intensity"],
            "yielding":          delivery_vector["yielding"],
            "focus":             delivery_vector["focus"],
        },
    }