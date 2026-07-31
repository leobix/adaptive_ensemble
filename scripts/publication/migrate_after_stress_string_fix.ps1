param(
    [Parameter(Mandatory = $true)]
    [string]$SourceOutput,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [string]$Config = "config\publication_full.toml",
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $Root).Path
$Julia = (Get-Command julia -ErrorAction Stop).Source
$Script = Join-Path $RepoRoot "scripts\publication\migrate_after_stress_string_fix.jl"
$SourcePath = (Resolve-Path $SourceOutput).Path
$ConfigPath = (Resolve-Path $Config).Path
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Output))
}

if (Test-Path $OutputPath) {
    $Entries = @(Get-ChildItem $OutputPath -Force -ErrorAction Stop)
    if ($Entries.Count -gt 0) {
        throw "Migration target must be absent or empty: $OutputPath"
    }
}

Write-Host "Auditing and migrating completed stages plus 420 stress task checkpoints..."
Write-Host "Source: $SourcePath"
Write-Host "Target: $OutputPath"
Write-Host "The incomplete stress stage marker and all old failure/launcher logs will not be copied."

& $Julia `
  --startup-file=no `
  --project=$RepoRoot `
  --threads=1 `
  $Script `
  --source-output $SourcePath `
  --output $OutputPath `
  --config $ConfigPath `
  --root $RepoRoot

if ($LASTEXITCODE -ne 0) {
    throw "Stress-string-fix result migration failed with Julia exit code $LASTEXITCODE."
}

$LatestOutputPointer = Join-Path $RepoRoot "latest_publication_output.txt"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($LatestOutputPointer, $OutputPath, $Utf8NoBom)

Write-Host "Migration completed successfully."
Write-Host "Resume target: $OutputPath"
Write-Host "Latest-output pointer: $LatestOutputPointer"
