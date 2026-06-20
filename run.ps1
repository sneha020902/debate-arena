# run.ps1 — Start all services in separate windows
# Run from d:\debate-arena-integrated\
# Usage: .\run.ps1

$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$root\emotion-api'; ..\venv-emotion\Scripts\uvicorn main:app --port 8000 --reload"

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$root\judge-api'; ..\venv-judge\Scripts\uvicorn main:app --port 8001 --reload"

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$root\orchestrator'; ..\venv-orchestrator\Scripts\uvicorn main:app --port 8002 --reload"

Start-Sleep 3
Start-Process "$root\frontend\index.html"

Write-Host "All services started. Frontend opening in browser." -ForegroundColor Green
Write-Host "Emotion API  → http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Judge API    → http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "Orchestrator → http://localhost:8002/docs" -ForegroundColor Cyan
