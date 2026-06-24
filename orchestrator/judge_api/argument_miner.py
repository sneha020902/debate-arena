"""
Claim & premise extraction for individual arguments.

Belongs in: services/judge-api/app/judging/argument_miner.py

Uses a local Ollama model with structured (JSON-schema-constrained) output, so the
response is guaranteed to parse into ArgumentStructure -- no manual JSON cleanup.

Prerequisites:
    pip install ollama pydantic
    ollama pull llama3.1          # or any chat model you have
    # Ollama >= 0.5 is required for schema-constrained `format`.
"""
from __future__ import annotations

import logging
import requests
import json
from pydantic import BaseModel, Field, ValidationError

from .judge_config import OLLAMA_MODEL
from .judge_config import OLLAMA_HOST

logger = logging.getLogger(__name__)



class ArgumentStructure(BaseModel):
    """The mined structure of a single argument."""

    claim: str = Field(
        description="The single main conclusion the argument is trying to establish."
    )
    premises: list[str] = Field(
        default_factory=list,
        description="The reasons or evidence offered in support of the claim.",
    )


_SYSTEM_PROMPT = """You are an argument-mining assistant. Given a short argument \
(a few sentences), identify its structure:

- claim: the single main conclusion the author wants the reader to accept.
- premises: each distinct reason, piece of evidence, or supporting statement \
offered for that claim, listed as a separate item.

Rules:
- Use only what is stated or directly implied in the text; do not invent content.
- Each premise must be one self-contained statement.
- Preserve the author's meaning; light rephrasing for clarity is acceptable.
- If the text states no real claim, return an empty string for claim."""

def extract_claim_and_premises(
    argument: str,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.0,
) -> ArgumentStructure:

    if not argument or not argument.strip():
        raise ValueError("argument must be a non-empty string")

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": argument.strip(),
                    },
                ],
                "stream": False,
                "format": ArgumentStructure.model_json_schema(),
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=60,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}"
        ) from exc

    try:
        raw = response.json()["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            "Invalid response received from Ollama."
        ) from exc

    try:
        parsed = json.loads(raw)

        return parsed["claim"], parsed["premises"]

    except ValidationError as exc:
        logger.error("Model output did not match schema: %s", raw)
        raise RuntimeError(
            "Model returned output that did not match the schema"
        ) from exc
