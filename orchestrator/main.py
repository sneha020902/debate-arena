"""
main.py — Debate Orchestrator API
===================================
DEBATE Project · Bauhaus-Universität Weimar · Webis Lab · SS 2026

Run:
    uvicorn main:app --reload --port 8002

Docs:
    http://localhost:8002/docs

WebSocket endpoint:
    ws://localhost:8002/ws/debate
    → send DebateConfig JSON, receive a stream of debate events
"""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from debate_flow import run_debate, set_steering_a, set_steering_b, pause_debate, resume_debate, is_paused

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="DEBATE — Orchestrator API",
    description="Manages debate state, calls Ollama + Emotion API + Judge API.",
    version="1.0.0",
)

# Allow the frontend (any local origin) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Config schema ─────────────────────────────────────────────────────────────

class DebateConfig(BaseModel):
    topic:        str   = Field(...,                          description="Debate topic (free text)")
    llm_a:        str   = Field("LLM-A",                     description="Name for the Pro speaker")
    llm_b:        str   = Field("LLM-B",                     description="Name for the Con speaker")
    turn_count:   int   = Field(6,  ge=2, le=12,             description="Total turns (must be even)")
    ollama_url:   str   = Field("http://localhost:11434",     description="Ollama server URL")
    ollama_model: str   = Field("qwen2.5:7b",                description="Ollama model name")
    emotion_api:  str   = Field("http://localhost:8000",      description="Emotion API base URL")
    judge_api:    str   = Field("http://localhost:8001",      description="Judge API base URL")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "topic": "Artificial intelligence should replace human judges in debates",
                "llm_a": "Qwen-Pro",
                "llm_b": "Qwen-Con",
                "turn_count": 6,
                "ollama_url": "http://localhost:11434",
                "ollama_model": "qwen2.5:7b",
                "emotion_api": "http://localhost:8000",
                "judge_api": "http://localhost:8001",
            }]
        }
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "service": "orchestrator", "version": "1.0.0"}


# ── Coach steering ────────────────────────────────────────────────────────────

class SteerRequest(BaseModel):
    instruction: str = Field(..., description="Coach instruction to inject into the next argument")


@app.post("/steer/a", tags=["General"])
def steer_a(req: SteerRequest):
    """Coach A sends instruction for LLM-A (Pro)."""
    set_steering_a(req.instruction)
    log.info("Coach A instruction: %s", req.instruction)
    return {"status": "ok"}


@app.post("/steer/b", tags=["General"])
def steer_b(req: SteerRequest):
    """Coach B sends instruction for LLM-B (Con)."""
    set_steering_b(req.instruction)
    log.info("Coach B instruction: %s", req.instruction)
    return {"status": "ok"}


@app.post("/pause", tags=["General"])
def pause():
    pause_debate()
    log.info("Debate paused by coach")
    return {"status": "paused"}


@app.post("/resume", tags=["General"])
def resume():
    resume_debate()
    log.info("Debate resumed by coach")
    return {"status": "resumed"}


@app.get("/paused", tags=["General"])
def paused():
    return {"paused": is_paused()}

# ── WebSocket debate stream ───────────────────────────────────────────────────

@app.websocket("/ws/debate")
async def ws_debate(websocket: WebSocket):
    """
    WebSocket endpoint for a live debate session.

    1. Client connects and sends a DebateConfig JSON object.
    2. Server streams debate events as JSON text frames:
       - {type: "host_intro",    text, audio_b64}
       - {type: "argument",      speaker, role, round, text, scores, audio_b64}
       - {type: "scores_update", llm_a_avg_composure, llm_b_avg_composure}
       - {type: "winner",        winner, explanation, text, audio_b64}
       - {type: "error",         message}
    3. Connection closes after the winner event.

    audio_b64 is base64-encoded WAV (24 kHz, mono). May be null if TTS fails.
    """
    await websocket.accept()
    log.info("WebSocket connection accepted")

    try:
        raw = await websocket.receive_text()
        config = DebateConfig(**json.loads(raw))
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": f"Invalid config: {exc}"}))
        await websocket.close()
        return

    log.info("Starting debate: topic='%s', turns=%d", config.topic, config.turn_count)

    try:
        async for event in run_debate(
            topic=config.topic,
            llm_a=config.llm_a,
            llm_b=config.llm_b,
            turn_count=config.turn_count,
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
            emotion_api=config.emotion_api,
            judge_api=config.judge_api,
        ):
            await websocket.send_text(json.dumps(event))

            if event.get("type") == "error":
                break

    except WebSocketDisconnect:
        log.info("Client disconnected mid-debate")
    except Exception as exc:
        log.exception("Unexpected orchestrator error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("WebSocket closed")
