$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $Python) {
    & $Python.Source (Join-Path $Root "scripts\ci\repository_audit.py") $Root
}
else {
    $Py = Get-Command py -ErrorAction Stop
    & $Py.Source -3 (Join-Path $Root "scripts\ci\repository_audit.py") $Root
}
if ($LASTEXITCODE -ne 0) { throw "Repository audit failed." }
$Git = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $Git -and (& $Git.Source -C $Root rev-parse --is-inside-work-tree 2>$null)) {
    & $Git.Source -C $Root diff --check
    if ($LASTEXITCODE -ne 0) { throw "git diff --check failed." }
}
