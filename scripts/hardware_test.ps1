# JointGuard Hardware Test Mode (PowerShell)
# Launches PlatformIO Serial Monitor on COM4 WITHOUT starting backend serial reader to prevent port conflict.

Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "         JointGuard Standalone Hardware Test Mode       " -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "NOTE: Ensure FastAPI backend is NOT running serial_service on COM4!" -ForegroundColor Red

$ROOT_DIR = Get-Item $PSScriptRoot\..

Write-Host "Launching PlatformIO Serial Monitor on COM4 (9600 baud)..." -ForegroundColor Green
Set-Location -Path "$ROOT_DIR\esp32\sensor_node"
pio device monitor --port COM4 --baud 9600
