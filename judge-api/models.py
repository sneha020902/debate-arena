from pydantic import BaseModel


class DebateTurn(BaseModel):
    turn: int
    speaker: str
    argument: str


class DebateTranscriptRequest(BaseModel):
    topic: str
    turns: list[DebateTurn]
