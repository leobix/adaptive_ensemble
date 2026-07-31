param(
    [string]$Julia = "julia",
    [switch]$SkipGurobiCheck
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
Write-Host "Repository: $Repo"
Write-Host "Active scientific scope: publication-scale synthetic experiments + Safi only."
Write-Host "Energy and hurricane applications are excluded from the active configurations."

& $Julia --version
if ($LASTEXITCODE -ne 0) {
    throw "Julia is not available on PATH."
}

& $Julia `
  --startup-file=no `
  .\scripts\publication\verify_original_python_integrity.jl
if ($LASTEXITCODE -ne 0) {
    throw "Original Python/notebook integrity check failed."
}

& $Julia `
  --startup-file=no `
  --project=. `
  -e 'using Pkg; Pkg.instantiate(; julia_version_strict=true); Pkg.precompile(); Pkg.status()'
if ($LASTEXITCODE -ne 0) {
    throw "Pkg.instantiate/precompile failed."
}

if (-not $SkipGurobiCheck) {
    & $Julia `
      --startup-file=no `
      --project=. `
      .\scripts\publication\gurobi_preflight.jl
    if ($LASTEXITCODE -ne 0) {
        throw "Gurobi license/conic-solver check failed. Install Gurobi, activate a valid license, and rerun setup. Use -SkipGurobiCheck only for non-Gurobi stages."
    }
    $env:RUN_GUROBI_TESTS = "1"
}

try {
    & $Julia `
      --startup-file=no `
      --project=. `
      --threads=4 `
      .\test\runtests.jl
    if ($LASTEXITCODE -ne 0) {
        throw "Julia test suite failed with exit code $LASTEXITCODE."
    }
}
finally {
    Remove-Item Env:RUN_GUROBI_TESTS -ErrorAction SilentlyContinue
}

Write-Host "Julia publication environment is ready."
