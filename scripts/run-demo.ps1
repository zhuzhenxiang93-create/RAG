$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Run: py -3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements-lite.txt"
}

$env:DOCMIND_APP_MODE = "lite"
$server = $null
try {
    $server = Start-Process `
        -FilePath $python `
        -ArgumentList "-B", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    $ready = $false
    foreach ($attempt in 1..30) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 300
        }
    }
    if (-not $ready) {
        throw "DocMind-RAG did not become ready on port 8000."
    }

    Start-Process "http://127.0.0.1:8000/ui/"
    Write-Host "DocMind-RAG demo is running at http://127.0.0.1:8000/ui/"
    Write-Host "Close this window or press Ctrl+C to stop the server."
    Wait-Process -Id $server.Id
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
