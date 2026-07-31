param(
    [string]$Julia = "julia",
    [int]$Threads = 4,
    [string]$Output = "publication_artifacts",
    [string]$Config = "config\publication_full.toml",
    [string]$TaskName = "",
    [switch]$NoResume,
    [switch]$SkipTests
)

$Launcher = Join-Path $PSScriptRoot "start_codex_detached.ps1"
& $Launcher -Julia $Julia -Threads $Threads -Output $Output -Config $Config -TaskName $TaskName -NoResume:$NoResume -SkipTests:$SkipTests
