import json
import re
import logging
import ollama
import httpx

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # greedy match — gets the full outermost JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # field-level fallback — extract each key individually via regex
    result = {}
    for key in ("relation", "type", "engagement", "verdict", "supported", "confidence", "quality_score", "new_information_score", "reasoning"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if m:
            result[key] = m.group(1)
            continue
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+|true|false)', text)
        if m:
            val = m.group(1)
            if val in ("true", "false"):
                result[key] = val == "true"
            else:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


class LogicDetector:
    def __init__(self, model: str = None, ollama_url: str = None):
        # Single source of truth: the OLLAMA_HOST / OLLAMA_MODEL defaults live in
        # judge_config (which already resolves them from the environment), so
        # Pavan's Part-2 detector and the Part-1/3 LLM judge can never drift to
        # different servers. Explicit constructor args still win if passed; an
        # OLLAMA_HOST / OLLAMA_MODEL env var overrides the default for both.
        from services.judge_config import OLLAMA_HOST, OLLAMA_MODEL
        self.model = model or OLLAMA_MODEL
        ollama_url = ollama_url or OLLAMA_HOST
        # Fail fast on an unreachable host (10s connect) but allow long
        # generations (120s read) so a wrong URL errors in seconds, not minutes.
        self.client = ollama.Client(
            host=ollama_url, timeout=httpx.Timeout(120.0, connect=10.0))

    def classify_relation(self, argument_1: dict, argument_2: dict) -> dict:
        """Classify whether argument_2 rebuts or undercuts argument_1."""
        def fmt(arg):
            prems = "\n".join(f"  - {p}" for p in arg.get("premises", []))
            claim = arg.get("claim", "")
            return f"Premises:\n{prems}\nClaim: {claim}" if prems else f"Argument: {claim}"

        prompt = f"""You are an argumentation theory expert.

Argument 1:
{fmt(argument_1)}

Argument 2:
{fmt(argument_2)}

Classify how Argument 2 relates to Argument 1:
- "rebuttal": Argument 2 directly contradicts the CONCLUSION (claim) of Argument 1.
- "undercut": Argument 2 attacks the EVIDENCE or REASONING of Argument 1 without denying its conclusion.
- "unrelated": Argument 2 does not respond to Argument 1 at all.

For "confidence", give YOUR OWN honest estimate of how certain you are, as a
number between 0.0 (pure guess) and 1.0 (certain). Do not copy any number from
this prompt — judge this specific pair and report your real confidence in it.

Reply with ONLY this JSON object, nothing else, with the placeholders below
replaced by your actual answer:
{{
  "relation": "<rebuttal | undercut | unrelated>",
  "confidence": <your own confidence between 0.0 and 1.0>,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        relation = str(result.get("relation", "unrelated")).lower().strip()
        if relation not in ("rebuttal", "undercut", "unrelated"):
            relation = "unrelated"
        return {
            "relation": relation,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    def classify_engagement(self, argument_1: dict, argument_2: dict) -> dict:
        """Classify whether argument_2 actually engages with argument_1, or argues in parallel without addressing it."""
        def fmt(arg):
            prems = "\n".join(f"  - {p}" for p in arg.get("premises", []))
            claim = arg.get("claim", "")
            return f"Premises:\n{prems}\nClaim: {claim}" if prems else f"Argument: {claim}"

        prompt = f"""You are an argumentation theory expert.

Argument 1:
{fmt(argument_1)}

Argument 2:
{fmt(argument_2)}

Classify whether Argument 2 engages with Argument 1:
- "engaged": Argument 2 directly addresses, responds to, or builds on the actual content of Argument 1.
- "parallel": Argument 2 largely ignores Argument 1 and pursues its own separate point, without meaningfully addressing what Argument 1 said.

For "confidence", give YOUR OWN honest estimate of how certain you are, as a
number between 0.0 (pure guess) and 1.0 (certain). Do not copy any number from
this prompt — judge this specific pair and report your real confidence in it.

Reply with ONLY this JSON object, nothing else, with the placeholders below
replaced by your actual answer:
{{
  "engagement": "<engaged | parallel>",
  "confidence": <your own confidence between 0.0 and 1.0>,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        engagement = str(result.get("engagement", "parallel")).lower().strip()
        if engagement not in ("engaged", "parallel"):
            engagement = "parallel"
        return {
            "engagement": engagement,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    def score_response_quality(self, original_argument: dict, response_argument: dict) -> dict:
        """Score how effectively a response addresses the original argument it's replying to."""
        def fmt(arg):
            prems = "\n".join(f"  - {p}" for p in arg.get("premises", []))
            claim = arg.get("claim", "")
            return f"Premises:\n{prems}\nClaim: {claim}" if prems else f"Argument: {claim}"

        prompt = f"""You are an argumentation theory expert.

Original argument:
{fmt(original_argument)}

Response:
{fmt(response_argument)}

Judge whether the response engages substantively with the original claim, or merely deflects:
- "substantive": The response directly grapples with the actual substance of the original claim — addressing its reasoning, evidence, or conclusion head-on.
- "deflecting": The response sidesteps the original claim — changing the subject, attacking something irrelevant, or offering only a surface-level reply that doesn't truly address what was said.

For "quality_score", give YOUR OWN honest rating of how EFFECTIVELY the response
addresses the original argument, as a number between 0.0 (does not address it at
all) and 1.0 (addresses it thoroughly and convincingly). Do not copy any number
from this prompt — judge this specific pair and report your real rating of it.

Reply with ONLY this JSON object, nothing else, with the placeholders below
replaced by your actual answer:
{{
  "verdict": "<substantive | deflecting>",
  "quality_score": <your own rating between 0.0 and 1.0>,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        verdict = str(result.get("verdict", "deflecting")).lower().strip()
        if verdict not in ("substantive", "deflecting"):
            verdict = "deflecting"
        return {
            "verdict": verdict,
            "quality_score": max(0.0, min(1.0, float(result.get("quality_score", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    def classify_new_point(self, argument: str, previous_opponent_arguments: list[str]) -> dict:
        """Classify whether an argument is original, reactive, or mixed (responds + adds new reasoning)."""
        if not previous_opponent_arguments:
            return {
                "type": "original",
                "confidence": 1.0,
                "reasoning": "No opponent arguments yet — this is an opening statement."
            }

        opponent_text = "\n".join(f"- {a}" for a in previous_opponent_arguments)

        prompt = f"""You are a debate analyst.

Opponent's previous arguments:
{opponent_text}

Current argument:
{argument}

Classify the current argument into exactly one of these three types:
- "original": Introduces a completely new line of reasoning with no connection to what the opponent said.
- "reactive": Purely responds to the opponent's points without adding any new evidence, reasoning, or examples of its own.
- "mixed": Responds to the opponent's points AND introduces new evidence, examples, or reasoning that was not in the opponent's arguments.

A good debater often produces "mixed" arguments — they engage with the opponent but also advance their own case with fresh material.

For "confidence", give YOUR OWN honest estimate of how certain you are, as a
number between 0.0 (pure guess) and 1.0 (certain). Do not copy any number from
this prompt — judge this specific argument and report your real confidence in it.

Reply with ONLY this JSON object, nothing else, with the placeholders below
replaced by your actual answer:
{{
  "type": "<original | reactive | mixed>",
  "confidence": <your own confidence between 0.0 and 1.0>,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        arg_type = str(result.get("type", "original")).lower().strip()
        if arg_type not in ("reactive", "original", "mixed"):
            arg_type = "original"
        return {
            "type": arg_type,
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }

    def score_information_density(self, argument: str, prior_team_arguments: list[str]) -> dict:
        """Score how much NEW information an argument adds compared to this same team's own earlier arguments."""
        if not prior_team_arguments:
            return {
                "new_information_score": 1.0,
                "reasoning": "No prior arguments from this team yet — everything here is new."
            }

        prior_text = "\n".join(f"- {a}" for a in prior_team_arguments)

        prompt = f"""You are a debate analyst.

This team's previous arguments:
{prior_text}

This team's current argument:
{argument}

Rate how much NEW information the current argument introduces, compared to
what this same team has ALREADY said above — new claims, new evidence, or new
reasoning not already present in their earlier arguments.
- A score near 0.0 means the argument mostly rephrases or repeats points the team already made.
- A score near 1.0 means the argument opens a genuinely new line of reasoning, not covered before.

For "new_information_score", give YOUR OWN honest rating, as a number between
0.0 (pure repetition) and 1.0 (entirely new ground). Do not copy any number
from this prompt — judge this specific argument and report your real rating.

Reply with ONLY this JSON object, nothing else, with the placeholder below
replaced by your actual answer:
{{
  "new_information_score": <your own rating between 0.0 and 1.0>,
  "reasoning": "..."
}}"""
        response = self.client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        result = _extract_json(raw)
        return {
            "new_information_score": max(0.0, min(1.0, float(result.get("new_information_score", 0.5)))),
            "reasoning": str(result.get("reasoning", "No reasoning provided.")),
        }
