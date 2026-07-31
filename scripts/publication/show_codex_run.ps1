param(
    [string]$Output = "publication_artifacts",
    [string]$TaskName = "",
    [string]$Julia = "julia",
    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
}
$LogDir = Join-Path $OutputPath "logs"
$Receipts = Get-ChildItem -LiteralPath $LogDir -Filter "codex_*.receipt.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
if ($Receipts.Count -eq 0) { throw "No Codex detached-run receipt found under $LogDir" }
$Receipt = Get-Content -LiteralPath $Receipts[0].FullName -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($TaskName)) { $TaskName = [string]$Receipt.task_name }
$JuliaForStatus = if ($Julia -eq "julia" -and $null -ne $Receipt.julia -and
    -not [string]::IsNullOrWhiteSpace([string]$Receipt.julia)) {
    [string]$Receipt.julia
} else {
    $Julia
}

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$Info = if ($null -ne $Task) { Get-ScheduledTaskInfo -TaskName $TaskName } else { $null }
Write-Host "Task: $TaskName"
Write-Host "Task Scheduler state: $(if ($null -eq $Task) { 'Missing' } else { $Task.State })"
if ($null -ne $Info) {
    Write-Host "Last run: $($Info.LastRunTime)"
    Write-Host "Last task result: $($Info.LastTaskResult)"
}
Write-Host "Output: $($Receipt.output)"
Write-Host "Receipt: $($Receipts[0].FullName)"

if (Test-Path -LiteralPath $Receipt.wrapper_pid) {
    $WrapperPid = [int](Get-Content -LiteralPath $Receipt.wrapper_pid -Raw).Trim()
    $Wrapper = Get-Process -Id $WrapperPid -ErrorAction SilentlyContinue
    if ($null -ne $Wrapper) {
        Write-Host "Wrapper process: PID=$WrapperPid CPU=$([math]::Round($Wrapper.CPU, 2))s RAM=$([math]::Round($Wrapper.WorkingSet64 / 1MB, 1)) MiB"
        $Children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $WrapperPid" -ErrorAction SilentlyContinue
        foreach ($Child in $Children) {
            $ChildProcess = Get-Process -Id $Child.ProcessId -ErrorAction SilentlyContinue
            if ($null -ne $ChildProcess) {
                Write-Host "Child process: $($Child.Name) PID=$($Child.ProcessId) CPU=$([math]::Round($ChildProcess.CPU, 2))s RAM=$([math]::Round($ChildProcess.WorkingSet64 / 1MB, 1)) MiB"
            }
        }
    } else {
        Write-Host "Wrapper process PID $WrapperPid is no longer running."
    }
}
$RecordedExitCode = $null
if (Test-Path -LiteralPath $Receipt.exitcode) {
    $RecordedExitCode = [int](Get-Content -LiteralPath $Receipt.exitcode -Raw).Trim()
    Write-Host "Exit code: $RecordedExitCode"
}
$ScientificState = if ($null -ne $RecordedExitCode) {
    if ($RecordedExitCode -eq 0) { "completed_successfully" } else { "failed" }
} elseif ($null -ne $Task -and [string]$Task.State -eq "Running") {
    "running"
} else {
    "launch_or_owner_requires_inspection"
}
Write-Host "Scientific state: $ScientificState"
if (Test-Path -LiteralPath $Receipt.started) {
    Write-Host "Started: $((Get-Content -LiteralPath $Receipt.started -Raw).Trim())"
}
if (Test-Path -LiteralPath $Receipt.finished) {
    Write-Host "Finished: $((Get-Content -LiteralPath $Receipt.finished -Raw).Trim())"
}

$Heartbeat = Join-Path $OutputPath "manifests\heartbeat.toml"
if (Test-Path -LiteralPath $Heartbeat) {
    Write-Host "`nLatest heartbeat:"
    Get-Content -LiteralPath $Heartbeat
    try {
        $HeartbeatData = Get-Content -LiteralPath $Heartbeat -Raw | ConvertFrom-Toml
        $HeartbeatTime = [DateTimeOffset]::Parse([string]$HeartbeatData.updated_at_utc)
        Write-Host "Heartbeat age: $([math]::Round(((Get-Date).ToUniversalTime() - $HeartbeatTime.UtcDateTime).TotalMinutes, 1)) minutes"
    }
    catch {
        # ConvertFrom-Toml is unavailable in Windows PowerShell 5.1; raw TOML above is authoritative.
    }
}

Write-Host "`nPublication stage status:"
& $JuliaForStatus --startup-file=no --project=$Repo (Join-Path $Repo "scripts\publication\run_publication_suite.jl") status --output $OutputPath --root $Repo

if (Test-Path -LiteralPath $Receipt.stdout) {
    Write-Host "`nLatest stdout ($($Receipt.stdout)):"
    Get-Content -LiteralPath $Receipt.stdout -Tail $Tail
}
if (Test-Path -LiteralPath $Receipt.stderr) {
    $StderrItem = Get-Item -LiteralPath $Receipt.stderr
    if ($StderrItem.Length -gt 0) {
        Write-Host "`nLatest stderr ($($Receipt.stderr)):"
        Get-Content -LiteralPath $Receipt.stderr -Tail $Tail
    }
}
