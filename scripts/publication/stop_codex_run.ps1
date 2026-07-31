param(
    [string]$Output = "publication_artifacts",
    [string]$TaskName = "",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
}
if ([string]::IsNullOrWhiteSpace($TaskName)) {
    $Receipt = Get-ChildItem (Join-Path $OutputPath "logs") -Filter "codex_*.receipt.json" -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $TaskName = [string]((Get-Content -LiteralPath $Receipt.FullName -Raw | ConvertFrom-Json).task_name)
}
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Write-Host "Stop requested for scheduled task $TaskName"
if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    Write-Host "Scheduled task unregistered: $TaskName"
}
