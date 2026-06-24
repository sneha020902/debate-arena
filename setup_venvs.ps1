# setup_venvs.ps1 — Create virtual environments for local services
# Judge API and Emotion API run on the SLURM cluster — no local venvs needed for them.
# Run once from d:\debate-arena-1\
# Usage: .\setup_venvs.ps1

$ErrorActionPreference = "Stop"
$python = "python"

Write-Host "`n=== venv-orchestrator  (Orchestrator — lightweight, no ML) ===" -ForegroundColor Cyan
& $python -m venv venv-orchestrator
.\venv-orchestrator\Scripts\pip install --upgrade pip
.\venv-orchestrator\Scripts\pip install `
    fastapi uvicorn[standard] httpx pydantic websockets requests

Write-Host "`n=== Done! Run .\run.ps1 to start everything. ===" -ForegroundColor Green
