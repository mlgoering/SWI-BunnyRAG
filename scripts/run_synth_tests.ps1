param(
    [string]$BaseTempRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "pytest_temp_ps")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Missing .venv\Scripts\python.exe. Create the project venv first."
}

$runId = "run_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), ([Guid]::NewGuid().ToString("N").Substring(0, 8))
$BaseTemp = Join-Path $BaseTempRoot $runId
New-Item -ItemType Directory -Path $BaseTemp -Force | Out-Null

& .venv\Scripts\python.exe -m pytest -q `
    --basetemp="$BaseTemp" `
    tests/test_synthetic_bunnyrag.py `
    tests/test_synthetic_graphrag.py `
    tests/test_synthetic_lambda_sweep.py `
    tests/test_visualize_lambda_sweep.py `
    tests/test_generate_synthetic_data.py

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    # Cleanup run temp folder only on success; keep it on failures for debugging.
    Remove-Item -LiteralPath $BaseTemp -Recurse -Force -ErrorAction SilentlyContinue
}

exit $exitCode
