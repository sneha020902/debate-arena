from sentence_transformers import CrossEncoder
from .argument_miner import extract_claim_and_premises

_NLI_MODEL = None

def _get_nli_model():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        _NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _NLI_MODEL

def score_nli(argument: str, claim: str = None,
              premises: list = None) -> dict:

    if not claim:
        claim, premises = extract_claim_and_premises(argument)
    print("premises",premises)
    premise_text = " ".join(premises) if premises else argument[:400]

    # STEP 1: Get accurate score from dedicated NLI model
    nli_model = _get_nli_model()
    raw_scores = nli_model.predict(
        [(premise_text, claim)],
        apply_softmax=True
    )[0]

    entailment    = float(raw_scores[2])
    neutral       = float(raw_scores[1])
    contradiction = float(raw_scores[0])

    nli_score = round((entailment - contradiction + 1) / 2, 3)

    return {
        "nli_score":     nli_score,
        "claim":         claim,
        "premise_count": len(premises),
        "raw_scores": {
            "entailment":    round(entailment, 3),
            "neutral":       round(neutral, 3),
            "contradiction": round(contradiction, 3),
        }
    }