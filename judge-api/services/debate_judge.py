from services.logic_modules import LogicDetector

# qwen2.5 was the consistent winner across all five model comparisons —
# it gave varied, genuine judgments where llama3.2 repeatedly defaulted
# to the same answer regardless of the actual argument content.
detector = LogicDetector()


def compute_rebuttal_coverage(turns: list) -> dict:
    """
    For each team, compute what fraction of the opponent's arguments were
    rebutted or undercut. Returns per-argument status + aggregate score.

    turns: list of {"turn": int, "speaker": str, "argument": str}
    """
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}

    speaker_a, speaker_b = speakers[0], speakers[1]

    args_a = [t for t in turns if t["speaker"] == speaker_a]
    args_b = [t for t in turns if t["speaker"] == speaker_b]

    def check_coverage(attacker_args, defender_args, attacker_name, defender_name):
        per_argument = []
        rebutted_count = 0

        # Only score defender arguments the attacker actually had a chance to rebut
        # (i.e. appeared before the attacker's last turn). This prevents the
        # last-speaker from getting a free rebuttal-coverage advantage.
        last_attacker_turn = max((t["turn"] for t in attacker_args), default=0)
        rebuttable_defender_args = [t for t in defender_args if t["turn"] < last_attacker_turn]

        for defender_turn in rebuttable_defender_args:
            status = "unanswered"
            rebutted_by_turn = None
            relation = None
            confidence = None
            reasoning = None

            # Only check attacker arguments that come AFTER the defender's turn
            later_attacker_args = [
                t for t in attacker_args if t["turn"] > defender_turn["turn"]
            ]

            for attacker_turn in later_attacker_args:
                result = detector.classify_relation(
                    {"claim": defender_turn["argument"], "premises": []},
                    {"claim": attacker_turn["argument"], "premises": []}
                )
                if result["relation"] in ("rebuttal", "undercut") and result["confidence"] >= 0.55:
                    status = result["relation"]
                    rebutted_by_turn = attacker_turn["turn"]
                    relation = result["relation"]
                    confidence = result["confidence"]
                    reasoning = result["reasoning"]
                    rebutted_count += 1
                    break

            per_argument.append({
                "turn": defender_turn["turn"],
                "argument_snippet": defender_turn["argument"][:120] + "..." if len(defender_turn["argument"]) > 120 else defender_turn["argument"],
                "status": status,
                "rebutted_by_turn": rebutted_by_turn,
                "relation": relation,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        # Also append unreachable arguments as "unreachable" (informational only)
        unreachable = [t for t in defender_args if t["turn"] >= last_attacker_turn]
        for defender_turn in unreachable:
            per_argument.append({
                "turn": defender_turn["turn"],
                "argument_snippet": defender_turn["argument"][:120] + "..." if len(defender_turn["argument"]) > 120 else defender_turn["argument"],
                "status": "unreachable",
                "rebutted_by_turn": None,
                "relation": None,
                "confidence": None,
                "reasoning": "This argument came after the opponent's last turn — no rebuttal opportunity.",
            })

        total = len(rebuttable_defender_args)  # denominator = only rebuttable args
        coverage_score = round(rebutted_count / total, 3) if total > 0 else 0.0

        return {
            "arguments_made": total,
            "arguments_addressed_by_opponent": rebutted_count,
            "unanswered": total - rebutted_count,
            "coverage_score": coverage_score,
            "per_argument": per_argument,
        }

    return {
        speaker_a: check_coverage(args_b, args_a, speaker_b, speaker_a),
        speaker_b: check_coverage(args_a, args_b, speaker_a, speaker_b),
        "summary": {
            "topic": "rebuttal_coverage",
            "total_turns": len(turns),
        }
    }


def compute_new_point_detection(turns: list) -> dict:
    """
    For each argument, classify whether it is reactive (responding to opponent)
    or original (brand new point). Report ratio per team.

    turns: list of {"turn": int, "speaker": str, "argument": str}
    """
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}

    speaker_a, speaker_b = speakers[0], speakers[1]

    results = {speaker_a: {"per_argument": [], "original": 0, "reactive": 0, "mixed": 0},
               speaker_b: {"per_argument": [], "original": 0, "reactive": 0, "mixed": 0}}

    for turn in turns:
        speaker = turn["speaker"]
        opponent = speaker_b if speaker == speaker_a else speaker_a

        # Collect all opponent arguments that came BEFORE this turn
        previous_opponent_args = [
            t["argument"] for t in turns
            if t["speaker"] == opponent and t["turn"] < turn["turn"]
        ]

        classification = detector.classify_new_point(
            turn["argument"],
            previous_opponent_args
        )

        results[speaker]["per_argument"].append({
            "turn": turn["turn"],
            "argument_snippet": turn["argument"][:120] + "..." if len(turn["argument"]) > 120 else turn["argument"],
            "type": classification["type"],
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"],
        })

        results[speaker][classification["type"]] += 1

    # Compute ratios
    for speaker in speakers:
        total = results[speaker]["original"] + results[speaker]["reactive"] + results[speaker]["mixed"]
        results[speaker]["total_arguments"] = total
        results[speaker]["original_ratio"] = round(results[speaker]["original"] / total, 3) if total > 0 else 0.0
        results[speaker]["reactive_ratio"] = round(results[speaker]["reactive"] / total, 3) if total > 0 else 0.0
        results[speaker]["mixed_ratio"] = round(results[speaker]["mixed"] / total, 3) if total > 0 else 0.0

    results["summary"] = {
        "topic": "new_point_detection",
        "total_turns": len(turns),
    }

    return results


def _find_prior_opponent_turn(turns: list, index: int) -> dict | None:
    """Most recent prior turn from the OTHER speaker — the 'response pairs' shape used across engagement, rebuttal, and quality scoring."""
    speaker = turns[index]["speaker"]
    for prior in reversed(turns[:index]):
        if prior["speaker"] != speaker:
            return prior
    return None


def compute_engagement(turns: list) -> dict:
    """
    For each response (paired with the most recent prior turn from the
    opposing speaker), classify whether it actually engages with that
    argument or argues in parallel without addressing it.

    turns: list of {"turn": int, "speaker": str, "argument": str}
    """
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}

    speaker_a, speaker_b = speakers[0], speakers[1]
    results = {speaker_a: {"per_pair": [], "engaged": 0, "parallel": 0},
               speaker_b: {"per_pair": [], "engaged": 0, "parallel": 0}}

    for i, turn in enumerate(turns):
        prior = _find_prior_opponent_turn(turns, i)
        if prior is None:
            continue

        classification = detector.classify_engagement(
            {"claim": prior["argument"], "premises": []},
            {"claim": turn["argument"], "premises": []}
        )

        results[turn["speaker"]]["per_pair"].append({
            "turn": turn["turn"],
            "responding_to_turn": prior["turn"],
            "argument_snippet": turn["argument"][:120] + "..." if len(turn["argument"]) > 120 else turn["argument"],
            "engagement": classification["engagement"],
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"],
        })
        results[turn["speaker"]][classification["engagement"]] += 1

    for speaker in speakers:
        total = results[speaker]["engaged"] + results[speaker]["parallel"]
        results[speaker]["total_responses"] = total
        results[speaker]["engagement_ratio"] = round(results[speaker]["engaged"] / total, 3) if total > 0 else 0.0

    results["summary"] = {
        "topic": "engagement",
        "total_turns": len(turns),
    }

    return results


def compute_response_quality(turns: list) -> dict:
    """
    For each rebuttal pair (original argument -> response), score how
    effectively the response addresses the original — substantively or
    by deflecting. Reports per-pair verdict + score, and per-team averages.

    turns: list of {"turn": int, "speaker": str, "argument": str}
    """
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}

    speaker_a, speaker_b = speakers[0], speakers[1]
    results = {speaker_a: {"per_pair": [], "substantive": 0, "deflecting": 0, "_scores": []},
               speaker_b: {"per_pair": [], "substantive": 0, "deflecting": 0, "_scores": []}}

    for i, turn in enumerate(turns):
        original = _find_prior_opponent_turn(turns, i)
        if original is None:
            continue

        scoring = detector.score_response_quality(
            {"claim": original["argument"], "premises": []},
            {"claim": turn["argument"], "premises": []}
        )

        results[turn["speaker"]]["per_pair"].append({
            "turn": turn["turn"],
            "responding_to_turn": original["turn"],
            "argument_snippet": turn["argument"][:120] + "..." if len(turn["argument"]) > 120 else turn["argument"],
            "verdict": scoring["verdict"],
            "quality_score": scoring["quality_score"],
            "reasoning": scoring["reasoning"],
        })
        results[turn["speaker"]][scoring["verdict"]] += 1
        results[turn["speaker"]]["_scores"].append(scoring["quality_score"])

    for speaker in speakers:
        scores = results[speaker].pop("_scores")
        total = len(scores)
        results[speaker]["total_responses"] = total
        results[speaker]["average_quality"] = round(sum(scores) / total, 3) if total > 0 else 0.0

    results["summary"] = {
        "topic": "response_quality",
        "total_turns": len(turns),
    }

    return results


def compute_information_density(turns: list) -> dict:
    """
    For each argument, measure how much NEW information it adds compared
    to this same team's own prior arguments — claims, evidence, or
    reasoning not already covered. Reports per-argument score + per-team
    average.

    turns: list of {"turn": int, "speaker": str, "argument": str}
    """
    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) < 2:
        return {"error": "Need at least 2 speakers"}

    speaker_a, speaker_b = speakers[0], speakers[1]
    results = {speaker_a: {"per_argument": [], "_scores": []},
               speaker_b: {"per_argument": [], "_scores": []}}

    for i, turn in enumerate(turns):
        own_prior_arguments = [t["argument"] for t in turns[:i] if t["speaker"] == turn["speaker"]]

        scoring = detector.score_information_density(turn["argument"], own_prior_arguments)

        results[turn["speaker"]]["per_argument"].append({
            "turn": turn["turn"],
            "argument_snippet": turn["argument"][:120] + "..." if len(turn["argument"]) > 120 else turn["argument"],
            "new_information_score": scoring["new_information_score"],
            "reasoning": scoring["reasoning"],
        })
        results[turn["speaker"]]["_scores"].append(scoring["new_information_score"])

    for speaker in speakers:
        scores = results[speaker].pop("_scores")
        total = len(scores)
        results[speaker]["total_arguments"] = total
        results[speaker]["average_new_information"] = round(sum(scores) / total, 3) if total > 0 else 0.0

    results["summary"] = {
        "topic": "information_density",
        "total_turns": len(turns),
    }

    return results
