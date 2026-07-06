$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$DataDir = Resolve-Path (Join-Path $Root "..\prediction-market-analysis\data")

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing virtualenv Python at $Python. Run: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e ."
}

$env:DATA_DIR = $DataDir.Path
$env:OUTPUT_DIR = Join-Path $Root "output"

$Steps = @(
    "scripts\run_kalshi.py",
    "scripts\run_bayesian.py",
    "scripts\run_cross_platform.py",
    "scripts\run_robustness.py",
    "scripts\generate_figures.py"
)

foreach ($Step in $Steps) {
    Write-Host ""
    Write-Host "==> $Step"
    & $Python (Join-Path $Root $Step)
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
