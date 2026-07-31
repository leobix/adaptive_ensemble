param(
    [string]$Julia = "julia",
    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Pointer = Join-Path $Repo "latest_publication_output.txt"
if (-not (Test-Path -LiteralPath $Pointer -PathType Leaf)) {
    throw "No latest publication-output pointer exists at $Pointer. Launch a detached run first or call show_codex_run.ps1 with -Output explicitly."
}
$Output = (Get-Content -LiteralPath $Pointer -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($Output)) {
    throw "The latest publication-output pointer is empty: $Pointer"
}
& (Join-Path $PSScriptRoot "show_codex_run.ps1") -Output $Output -Julia $Julia -Tail $Tail
exit $LASTEXITCODE
