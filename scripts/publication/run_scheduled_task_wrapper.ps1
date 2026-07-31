param(
    [Parameter(Mandatory = $true)][string]$Julia,
    [Parameter(Mandatory = $true)][int]$Threads,
    [Parameter(Mandatory = $true)][string]$Repo,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$Stdout,
    [Parameter(Mandatory = $true)][string]$Stderr,
    [Parameter(Mandatory = $true)][string]$ExitCodePath,
    [Parameter(Mandatory = $true)][string]$PidPath,
    [Parameter(Mandatory = $true)][string]$StartedPath,
    [Parameter(Mandatory = $true)][string]$FinishedPath,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$ExitCode = 1
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Stdout) | Out-Null
Set-Content -LiteralPath $PidPath -Value $PID -Encoding Ascii
Set-Content -LiteralPath $StartedPath -Value (Get-Date).ToString("o") -Encoding Ascii

$env:JULIA_NUM_THREADS = [string]$Threads
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

try {
    Set-Location -LiteralPath $Repo
    $JuliaArguments = @(
        "--startup-file=no",
        "--project=$Repo",
        "--threads=$Threads",
        (Join-Path $Repo "scripts\publication\run_publication_suite.jl"),
        "suite",
        "--config", $Config,
        "--output", $Output,
        "--root", $Repo
    )
    if ($NoResume) {
        $JuliaArguments += "--no-resume"
    }

    "[$((Get-Date).ToString('o'))] Task Scheduler wrapper PID=$PID" | Out-File -LiteralPath $Stdout -Append -Encoding utf8
    "[$((Get-Date).ToString('o'))] Julia command: $Julia $($JuliaArguments -join ' ')" | Out-File -LiteralPath $Stdout -Append -Encoding utf8
    & $Julia @JuliaArguments 1>> $Stdout 2>> $Stderr
    $ExitCode = if ($null -eq $LASTEXITCODE) { 1 } else { [int]$LASTEXITCODE }
}
catch {
    $_ | Out-String | Out-File -LiteralPath $Stderr -Append -Encoding utf8
    $ExitCode = 1
}
finally {
    Set-Content -LiteralPath $ExitCodePath -Value $ExitCode -Encoding Ascii
    Set-Content -LiteralPath $FinishedPath -Value (Get-Date).ToString("o") -Encoding Ascii
}

exit $ExitCode
