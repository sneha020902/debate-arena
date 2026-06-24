# run.ps1 — Start local services
# Emotion API, Kokoro, and Judge API run on the SLURM cluster via SSH tunnels.
# Run from d:\debate-arena-1\
# Usage: .\run.ps1

$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$root\orchestrator'; ..\venv-orchestrator\Scripts\uvicorn main:app --port 8002 --reload"

Start-Sleep 2
Start-Process "$root\frontend\index.html"

Write-Host "Orchestrator started. Frontend opening in browser." -ForegroundColor Green
Write-Host "Orchestrator → http://localhost:8002/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure SSH tunnels are open:" -ForegroundColor Yellow
Write-Host "  Kokoro  : ssh -N -L 18003:<kokoro-node>.medien.uni-weimar.de:18003 joso1563@ssh.webis.de" -ForegroundColor Yellow
Write-Host "  Emotion : ssh -N -L 8000:<emotion-node>.medien.uni-weimar.de:8000 joso1563@ssh.webis.de" -ForegroundColor Yellow
Write-Host "  Ollama  : ssh -N -L 11437:gammaweb07.medien.uni-weimar.de:11437 joso1563@ssh.webis.de" -ForegroundColor Yellow
Write-Host "  Judge   : ssh -N -L 8001:<judge-node>.medien.uni-weimar.de:8001 joso1563@ssh.webis.de" -ForegroundColor Yellow
