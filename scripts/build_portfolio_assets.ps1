param(
    [string]$RunName = "behavior_runner_n500_d6_cosine_beta_same_30x30_20260304",
    [switch]$SkipScatter
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "[1/2] Building interactive portfolio page bundle..."
& $pythonCmd "presentation/build_interactive_projection_google_sites.py" `
    --output-html "docs/portfolio/interactive_projection.html"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipScatter) {
    Write-Host "[2/2] Building portfolio scatter figure for run: $RunName"
    & $pythonCmd "presentation/build_behavior_single_scatter_lambda_gradient.py" `
        --run $RunName `
        --formats "png"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $scatterSource = Join-Path $repoRoot "presentation/testing/$RunName/single_scatter_lambda_gradient/single_scatter_lambda_gradient.png"
    if (Test-Path $scatterSource) {
        $figureDir = Join-Path $repoRoot "docs/portfolio/figures"
        New-Item -ItemType Directory -Path $figureDir -Force | Out-Null
        $scatterTarget = Join-Path $figureDir "relevance_vs_coverage.png"
        Copy-Item -Path $scatterSource -Destination $scatterTarget -Force
        Write-Host "Copied figure: $scatterTarget"
    } else {
        Write-Warning "Scatter PNG not found at: $scatterSource"
    }
} else {
    Write-Host "[2/2] Skipped scatter build (--SkipScatter)."
}

Write-Host "Done."
