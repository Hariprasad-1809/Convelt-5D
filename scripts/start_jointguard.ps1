# JointGuard Full System Startup Script (PowerShell)
# Launches FastAPI Backend (managing Arduino Serial COM5 + USB Webcam Vision) & React Frontend

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         JointGuard Full System Startup                 " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI Backend & React SCADA Frontend..." -ForegroundColor Yellow

$ROOT_DIR = Get-Item $PSScriptRoot\..

# 1. Start FastAPI Backend Server
Write-Host "[1/2] Launching FastAPI Backend Server (COM5 + YOLO Camera)..." -ForegroundColor Green
$backendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT_DIR'; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru

# Wait 3 seconds for FastAPI startup
Start-Sleep -Seconds 3

# 2. Start Vite React Frontend
Write-Host "[2/2] Launching React Vite SCADA Dashboard..." -ForegroundColor Green
$frontendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT_DIR\frontend'; npm run dev" -PassThru

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "System Active!" -ForegroundColor Green
Write-Host "FastAPI Interactive Docs: http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "React SCADA Dashboard:   http://localhost:5173" -ForegroundColor Gray
Write-Host "========================================================" -ForegroundColor Cyan
