$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not $env:DOCMIND_APP_MODE) {
    $env:DOCMIND_APP_MODE = "lite"
}

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
