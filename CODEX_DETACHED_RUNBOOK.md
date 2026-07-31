# Codex detached-run guide (Windows)

Use Codex to invoke the repository launcher, not to own the scientific Julia process directly.

## Setup

```powershell
Set-Location <path-to-repository>
Set-ExecutionPolicy -Scope Process Bypass -Force
Get-ChildItem .\scripts -Recurse -File |
  Where-Object { $_.Extension -in '.ps1', '.psm1', '.psd1' } |
  Unblock-File
.\scripts\publication\setup_windows.ps1
```

## Launch

```powershell
.\scripts\publication\start_codex_detached.ps1 `
  -Threads 4 `
  -Output publication_artifacts `
  -Config .\config\publication_full.toml
```

Do not close Codex until the launcher states all three facts:

```text
DETACHED JULIA PUBLICATION RUN STARTED
Owner: Windows Task Scheduler
Safe to close Codex/terminal: yes
```

If Task Scheduler registration is denied, no experiment is running. Re-run the launcher from an Administrator PowerShell window.

## Monitor

```powershell
.\scripts\publication\show_latest_codex_run.ps1
```

A running task normally reports `LastTaskResult = 267009`; this is the Windows status code meaning that the task is currently running, not a failure.

## Resume

Reissue the identical launcher command against the same output directory. Only fingerprint- and hash-valid checkpoints are reused. After source, configuration, environment, or data changes, use a new output directory.

## Completion

Task Scheduler returning to `Ready` only means that no task instance is active. Require:

```text
Exit code: 0
Scientific state: complete
remaining = []
failure_files = []
```

Then run `scripts/publication/verify_publication_artifacts.jl` and require `PUBLICATION ARTIFACT VERIFICATION PASSED`.
