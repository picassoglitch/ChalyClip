# Start the PC worker with worker.env loaded.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_pc_worker.ps1 [-Port 8100]
param([int]$Port = 8100)

$ErrorActionPreference = "Stop"

# Anchor to the repo root regardless of who launched us (Startup folder,
# scheduled task, manual shell) - worker.env and .venv are repo-relative.
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path "worker.env")) {
    Write-Error "worker.env not found - run scripts\make_worker_env.ps1 first."
}

foreach ($line in Get-Content "worker.env") {
    if ($line -match "^\s*#" -or $line -notmatch "=") { continue }
    $name, $value = $line -split "=", 2
    Set-Item -Path "env:$name" -Value $value
}

# Launch detached with file logs. NOT a PowerShell stream redirect: in 5.1
# a native command's stderr through *>> becomes ErrorRecords, and with
# ErrorActionPreference=Stop the first uvicorn INFO line (stderr) kills the
# launcher AND the worker with it.
$logDir = Join-Path $env:LOCALAPPDATA "chalybclip-worker"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"[$(Get-Date -Format o)] launcher starting worker on port $Port" |
    Add-Content (Join-Path $logDir "launcher.log")

# Ollama is the upstream of the worker's /v1 LLM proxy. If it isn't up
# (crashed / quit from the tray), every cloud run's hooks silently degrade
# to the deterministic fallback - so any worker launch revives it too.
$ollamaUp = $false
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    $ollamaUp = $true
} catch {}
if (-not $ollamaUp) {
    $ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $ollamaExe) {
        "[$(Get-Date -Format o)] launcher reviving Ollama (was down)" |
            Add-Content (Join-Path $logDir "launcher.log")
        Start-Process -FilePath $ollamaExe -WindowStyle Hidden
    }
}

Start-Process -FilePath (Resolve-Path ".venv\Scripts\python.exe") `
    -ArgumentList "-m", "chalybclip.cli", "worker", "--port", $Port `
    -WorkingDirectory (Get-Location) `
    -RedirectStandardOutput (Join-Path $logDir "worker.out.log") `
    -RedirectStandardError (Join-Path $logDir "worker.err.log") `
    -WindowStyle Hidden
