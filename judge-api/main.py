from fastapi import FastAPI
from models import DebateTranscriptRequest
from services.debate_judge import (
    compute_engagement,
    compute_rebuttal_coverage,
    compute_new_point_detection,
    compute_response_quality,
    compute_information_density,
)

app = FastAPI(title="Debate Judge API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "debate-judge"}


@app.post("/engagement")
def engagement(request: DebateTranscriptRequest):
    """
    For each response (paired with the most recent prior turn from the
    opposing speaker): does it actually engage with that argument, or
    argue in parallel without addressing it?
    Returns per-pair classification + engagement ratio per team.
    """
    turns = [t.model_dump() for t in request.turns]
    return compute_engagement(turns)


@app.post("/rebuttal-coverage")
def rebuttal_coverage(request: DebateTranscriptRequest):
    """
    For each team: what fraction of the opponent's arguments were rebutted or undercut?
    Returns per-argument status (rebutted / undercut / unanswered) + aggregate coverage score.
    """
    turns = [t.model_dump() for t in request.turns]
    return compute_rebuttal_coverage(turns)


@app.post("/new-point-detection")
def new_point_detection(request: DebateTranscriptRequest):
    """
    For each argument: is it reactive (responding to opponent) or original (new point)?
    Returns per-argument classification + ratio per team.
    """
    turns = [t.model_dump() for t in request.turns]
    return compute_new_point_detection(turns)


@app.post("/response-quality")
def response_quality(request: DebateTranscriptRequest):
    """
    For each rebuttal pair (original argument -> response): how effectively
    does the response address the original — substantively, or by deflecting?
    Returns per-pair verdict + quality score, plus per-team averages.
    """
    turns = [t.model_dump() for t in request.turns]
    return compute_response_quality(turns)


@app.post("/information-density")
def information_density(request: DebateTranscriptRequest):
    """
    For each argument: how much NEW information does it add compared to
    this same team's own prior arguments — claims, evidence, or reasoning
    not already covered?
    Returns per-argument score plus per-team average.
    """
    turns = [t.model_dump() for t in request.turns]
    return compute_information_density(turns)
