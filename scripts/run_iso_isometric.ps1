$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found."
    Write-Host ""
    Write-Host "Create it with:"
    Write-Host "python -m venv .venv"
    Write-Host "pip install -r requirements.txt"
    exit 1
}

if ($args.Count -gt 0) {
    $Model = $args[0]
}
else {
    $Model = Join-Path `
        $ProjectRoot `
        "examples\models\part.obj"
}

Write-Host "Running ISO Isometric baseline"
Write-Host "Model: $Model"

& $Python `
    "$ProjectRoot\run.py" `
    --baseline iso_isometric `
    --input "$Model"