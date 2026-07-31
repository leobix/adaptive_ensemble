param(
    [string]$Julia = "julia",
    [string]$Output = "publication_artifacts",
    [string]$TaskName = "",
    [int]$Tail = 40
)

$Monitor = Join-Path $PSScriptRoot "show_codex_run.ps1"
& $Monitor -Julia $Julia -Output $Output -TaskName $TaskName -Tail $Tail
