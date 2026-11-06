param(
    [ValidateSet('test','train','report','predict')][string]$Action = 'test',
    [string]$Text = 'The movie was thoughtful and beautifully acted.',
    [string]$Model = 'mlp'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$candidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $projectRoot '..\..\work\venv\Scripts\python.exe')
)
$pythonExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Install Python 3.12 and the requirements as explained in README.md.' }
    $pythonExe = $pythonCommand.Source
}
Push-Location -LiteralPath $projectRoot
try {
    $localCache = Join-Path $projectRoot '..\..\work\hf-cache'
    if (Test-Path -LiteralPath $localCache) { $env:HF_HOME = (Resolve-Path -LiteralPath $localCache).Path }
    if ($Action -eq 'test') { & $pythonExe -m pytest -q }
    if ($Action -eq 'train') { & $pythonExe -u run_all.py }
    if ($Action -eq 'report') {
        & $pythonExe docs/build_report.py
        if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
        & $pythonExe docs/build_slides.py
        if ($LASTEXITCODE -ne 0) { throw 'Slide generation failed.' }
        & $pythonExe docs/export_pdf.py
    }
    if ($Action -eq 'predict') {
        & $pythonExe -m src.predict --model-dir "results/$Model" --text $Text
    }
    if ($LASTEXITCODE -ne 0) { throw "Project command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
