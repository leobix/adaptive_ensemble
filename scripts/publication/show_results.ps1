param(
    [string]$Julia = "julia",
    [string]$Output = "publication_artifacts"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
& $Julia --startup-file=no --project=. scripts/publication/show_results.jl $Output
if ($LASTEXITCODE -ne 0) { throw "Result summary command failed" }

$FigureDir = Join-Path $Repo "$Output\figures"
if (Test-Path $FigureDir) {
    Write-Host "`nOpen the figure directory with:"
    Write-Host "  Invoke-Item `"$FigureDir`""
}
