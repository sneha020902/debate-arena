# setup_venvs.ps1 — Create all three virtual environments
# Run once from d:\debate-arena-integrated\
# Usage: .\setup_venvs.ps1

$ErrorActionPreference = "Stop"
$python = "python"

Write-Host "`n=== 1/3  venv-emotion  (Emotion API + Kokoro TTS) ===" -ForegroundColor Cyan
& $python -m venv venv-emotion
.\venv-emotion\Scripts\pip install --upgrade pip
.\venv-emotion\Scripts\pip install `
    fastapi uvicorn[standard] python-multipart pydantic `
    transformers torch torchaudio `
    faster-whisper `
    librosa soundfile scipy `
    streamlit plotly pandas `
    audio-recorder-streamlit `
    huggingface_hub safetensors requests `
    kokoro soundfile

Write-Host "`n=== 2/3  venv-judge  (Judge API) ===" -ForegroundColor Cyan
& $python -m venv venv-judge
.\venv-judge\Scripts\pip install --upgrade pip
.\venv-judge\Scripts\pip install `
    fastapi uvicorn[standard] pydantic httpx requests

Write-Host "`n=== 3/3  venv-orchestrator  (Orchestrator) ===" -ForegroundColor Cyan
& $python -m venv venv-orchestrator
.\venv-orchestrator\Scripts\pip install --upgrade pip
.\venv-orchestrator\Scripts\pip install `
    fastapi uvicorn[standard] httpx pydantic websockets

Write-Host "`n=== Done! Run .\run.ps1 to start everything. ===" -ForegroundColor Green
