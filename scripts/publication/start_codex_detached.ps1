param(
    [string]$Julia = "julia",
    [int]$Threads = 4,
    [string]$Output = "publication_artifacts",
    [string]$Config = "config\publication_full.toml",
    [string]$TaskName = "",
    [switch]$NoResume,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FromRepo([string]$Repo, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Repo $Path))
}

function Quote-TaskArgument([string]$Value) {
    if ($Value.Contains('"')) { throw "Task arguments may not contain a double quote: $Value" }
    return '"' + $Value + '"'
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$JuliaCommand = Get-Command $Julia -ErrorAction Stop
$JuliaExe = $JuliaCommand.Source
$ConfigPath = Resolve-FromRepo $Repo $Config
$OutputPath = Resolve-FromRepo $Repo $Output
$Runner = Join-Path $Repo "scripts\publication\run_publication_suite.jl"
$Wrapper = Join-Path $Repo "scripts\publication\run_scheduled_task_wrapper.ps1"

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { throw "Configuration not found: $ConfigPath" }
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) { throw "Runner not found: $Runner" }
if (-not (Test-Path -LiteralPath $Wrapper -PathType Leaf)) { throw "Wrapper not found: $Wrapper" }
if ($Threads -lt 1) { throw "Threads must be at least 1" }

$LogDir = Join-Path $OutputPath "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
if ([string]::IsNullOrWhiteSpace($TaskName)) {
    $TaskName = "AdaptiveEnsemblePublication_$Stamp"
}
if ($TaskName -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "TaskName may contain only letters, numbers, underscore, period, and hyphen"
}

# Refuse concurrent writers to the same output directory. A second publication process can
# corrupt aggregate CSVs even though individual task outputs are atomic.
$PriorReceipts = Get-ChildItem -LiteralPath $LogDir -Filter "codex_*.receipt.json" -ErrorAction SilentlyContinue
foreach ($PriorReceiptFile in $PriorReceipts) {
    try {
        $PriorReceipt = Get-Content -LiteralPath $PriorReceiptFile.FullName -Raw | ConvertFrom-Json
        $PriorTaskName = [string]$PriorReceipt.task_name
        $PriorTask = Get-ScheduledTask -TaskName $PriorTaskName -ErrorAction SilentlyContinue
        if ($null -ne $PriorTask -and [string]$PriorTask.State -eq "Running") {
            throw "A detached publication run already owns this output directory: task '$PriorTaskName'. Monitor it or stop it before launching another writer."
        }
        if (Test-Path -LiteralPath ([string]$PriorReceipt.wrapper_pid)) {
            $PriorPid = [int](Get-Content -LiteralPath ([string]$PriorReceipt.wrapper_pid) -Raw).Trim()
            if ($null -ne (Get-Process -Id $PriorPid -ErrorAction SilentlyContinue)) {
                throw "A detached publication wrapper (PID $PriorPid) already owns this output directory."
            }
        }
    }
    catch {
        if ($_.Exception.Message -like "A detached publication*") { throw }
        # Malformed historical receipts are ignored here; run-identity and output hashes
        # still protect scientific checkpoints during preflight.
    }
}

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $Existing) {
    throw "Scheduled task '$TaskName' already exists. Choose another -TaskName or unregister the old task."
}

$Stdout = Join-Path $LogDir "codex_$Stamp.stdout.log"
$Stderr = Join-Path $LogDir "codex_$Stamp.stderr.log"
$ExitCodePath = Join-Path $LogDir "codex_$Stamp.exitcode.txt"
$PidPath = Join-Path $LogDir "codex_$Stamp.wrapper_pid.txt"
$StartedPath = Join-Path $LogDir "codex_$Stamp.started.txt"
$FinishedPath = Join-Path $LogDir "codex_$Stamp.finished.txt"
$ReceiptPath = Join-Path $LogDir "codex_$Stamp.receipt.json"

Write-Host "Running immutable Python/notebook integrity check..."
& $JuliaExe --startup-file=no (Join-Path $Repo "scripts\publication\verify_original_python_integrity.jl")
if ($LASTEXITCODE -ne 0) { throw "Original Python/notebook integrity check failed" }

if (-not $SkipTests) {
    Write-Host "Running Julia release tests before detaching..."
    & $JuliaExe --startup-file=no --project=$Repo --threads=$Threads (Join-Path $Repo "test\runtests.jl")
    if ($LASTEXITCODE -ne 0) { throw "Julia release tests failed; no detached task was registered" }
}

Write-Host "Running publication preflight before detaching..."
$PreflightArgs = @(
    "--startup-file=no",
    "--project=$Repo",
    "--threads=$Threads",
    $Runner,
    "preflight",
    "--config", $ConfigPath,
    "--output", $OutputPath,
    "--root", $Repo
)
& $JuliaExe @PreflightArgs
if ($LASTEXITCODE -ne 0) { throw "Publication preflight failed; no detached task was registered" }

$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$ActionArguments = @(
    "-NoLogo",
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $Wrapper),
    "-Julia", (Quote-TaskArgument $JuliaExe),
    "-Threads", [string]$Threads,
    "-Repo", (Quote-TaskArgument $Repo),
    "-Config", (Quote-TaskArgument $ConfigPath),
    "-Output", (Quote-TaskArgument $OutputPath),
    "-Stdout", (Quote-TaskArgument $Stdout),
    "-Stderr", (Quote-TaskArgument $Stderr),
    "-ExitCodePath", (Quote-TaskArgument $ExitCodePath),
    "-PidPath", (Quote-TaskArgument $PidPath),
    "-StartedPath", (Quote-TaskArgument $StartedPath),
    "-FinishedPath", (Quote-TaskArgument $FinishedPath)
)
if ($NoResume) { $ActionArguments += "-NoResume" }
$ActionArgumentString = $ActionArguments -join " "

$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $ActionArgumentString -WorkingDirectory $Repo
# The trigger is deliberately far in the future. Start-ScheduledTask launches the task now;
# the trigger exists only because some Windows editions require one at registration.
$Trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddYears(10))
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Julia publication experiment; registered by Codex-safe launcher" `
        -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
}
catch {
    $Message = $_.Exception.Message
    if ($Message -match "Access is denied|0x80070005") {
        throw "Task Scheduler denied registration (0x80070005). The scientific experiment was not started. Open Windows PowerShell as Administrator once, return to this repository, and rerun the same launcher; alternatively ask your administrator to permit creation of a limited, interactive scheduled task."
    }
    throw "Task Scheduler launch failed. The scientific experiment was not detached. Error: $Message"
}

$Receipt = [ordered]@{
    schema_version = 1
    task_name = $TaskName
    task_owner = "Windows Task Scheduler"
    registered_user = $CurrentUser
    created_at = (Get-Date).ToString("o")
    repository = $Repo
    julia = $JuliaExe
    threads = $Threads
    config = $ConfigPath
    output = $OutputPath
    stdout = $Stdout
    stderr = $Stderr
    exitcode = $ExitCodePath
    wrapper_pid = $PidPath
    started = $StartedPath
    finished = $FinishedPath
    resume = (-not $NoResume.IsPresent)
    release_tests_run = (-not $SkipTests.IsPresent)
}
$Receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8

$State = "Unknown"
$LaunchObserved = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    Start-Sleep -Seconds 1
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $Task) {
        $State = [string]$Task.State
    }
    $LaunchObserved = ($State -eq "Running") -or
        (Test-Path -LiteralPath $StartedPath) -or
        (Test-Path -LiteralPath $PidPath) -or
        (Test-Path -LiteralPath $ExitCodePath)
    if ($LaunchObserved) { break }
}
if (-not $LaunchObserved) {
    throw "The scheduled task was registered but did not reach Running or create a launch marker within 60 seconds. Inspect Task Scheduler task '$TaskName'."
}
if (Test-Path -LiteralPath $ExitCodePath) {
    $ImmediateExit = [int](Get-Content -LiteralPath $ExitCodePath -Raw).Trim()
    if ($ImmediateExit -ne 0) {
        $Tail = if (Test-Path -LiteralPath $Stderr) {
            (Get-Content -LiteralPath $Stderr -Tail 40) -join [Environment]::NewLine
        } else {
            "No stderr log was created."
        }
        throw "The detached wrapper started but Julia exited immediately with code $ImmediateExit. The experiment is not running.`n$Tail"
    }
}

$LatestOutputPointer = Join-Path $Repo "latest_publication_output.txt"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($LatestOutputPointer, $OutputPath, $Utf8NoBom)

Write-Host "DETACHED JULIA PUBLICATION RUN STARTED"
Write-Host "Owner: Windows Task Scheduler"
Write-Host "Task: $TaskName"
Write-Host "State: $State"
Write-Host "Output: $OutputPath"
Write-Host "Receipt: $ReceiptPath"
Write-Host "Latest-output pointer: $LatestOutputPointer"
Write-Host "stdout: $Stdout"
Write-Host "stderr: $Stderr"
Write-Host "Monitor: .\scripts\publication\show_codex_run.ps1 -Output $(Quote-TaskArgument $OutputPath) -TaskName $TaskName"
Write-Host "Safe to close Codex/terminal: yes, after this confirmation. Keep Windows logged in and prevent sleep/hibernation."
