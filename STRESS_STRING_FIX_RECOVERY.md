# Recovery after the completed stress tasks and heterogeneous setting-label failure

## Failure diagnosed

The failed run reached the end of all 420 stress tasks. The task heartbeat showed
`completed = 420`, `fraction = 1.0`, and no task-failure CSV. It then failed while
aggregating the task rows because `stress_tests_per_seed.csv` contains a heterogeneous
`setting` column:

- limited-history settings are numeric values such as `100.0`;
- correlation settings are numeric values such as `0.95`;
- boundary settings are textual values such as `reset` and `carry`.

The previous aggregation code called:

```julia
String(subset.setting[1])
```

Julia's `String(x)` is a conversion constructor and has no method for `Float64`; generic
text formatting must use `string(x)`. The corrected code normalizes every setting with
`string(value)` and parses numeric plot settings explicitly with `parse(Float64, string(value))`.

This was an aggregation and figure-label bug. The 420 scientific stress tasks completed and
are reusable. No model fit, seed, correlation level, history size, pruning grid, or boundary
policy must be recomputed.

## Results already complete in the failed output

The source output directory is expected to be:

```text
publication_artifacts_synthetic_safi_20260730_203420
```

It contains current-fingerprint completion markers for:

```text
legacy_audit
synthetic_full
manuscript_sweeps
synthetic_meta
real_world_safi
real_world_all
sensitivity_safi
sensitivity_all
```

It also contains 420 completed stress-task markers and their hash-bound output files:

```text
180 limited-history tasks
210 correlated-member/pruning tasks
30 boundary-policy tasks
```

The migration utility verifies every source marker and re-hashes every referenced output
before copying it. It does not copy the incomplete stress-stage marker, old Task Scheduler
receipts, old wrapper logs, or serialized model objects.

## 1. Preserve the old repository and output

From `C:\Users\henry\Documents\Research`:

```powershell
Set-Location C:\Users\henry\Documents\Research

$ArchiveName = "adaptive-ensemble_before_stress_string_fix_$(
  Get-Date -Format yyyyMMdd_HHmmss
)"

Rename-Item `
  -Path .\adaptive-ensemble `
  -NewName $ArchiveName

$SourceOutput = Join-Path `
  (Join-Path (Get-Location) $ArchiveName) `
  "publication_artifacts_synthetic_safi_20260730_203420"

if (-not (Test-Path $SourceOutput)) {
    throw "Source output not found: $SourceOutput"
}
```

Do not delete `$SourceOutput` until the new run passes final artifact verification.

## 2. Extract the corrected repository

Extract the ZIP beside the archived repository. The resulting directory must be:

```text
C:\Users\henry\Documents\Research\adaptive-ensemble
```

Then:

```powershell
Set-Location C:\Users\henry\Documents\Research\adaptive-ensemble

Set-ExecutionPolicy `
  -Scope Process `
  -ExecutionPolicy Bypass `
  -Force

Get-ChildItem .\scripts -Recurse -File |
  Where-Object { $_.Extension -in @(".ps1", ".psm1", ".psd1") } |
  Unblock-File
```

## 3. Instantiate and run code tests

```powershell
.\scripts\publication\setup_windows.ps1
```

Required ending:

```text
Julia publication environment is ready.
```

The regression suite now contains an explicit test in which stress settings mix
`Float64` and `String` values.

## 4. Migrate the successful stages and all 420 stress tasks

```powershell
$Output = "publication_artifacts_resumed_after_stress_fix_$(
  Get-Date -Format yyyyMMdd_HHmmss
)"

.\scripts\publication\migrate_after_stress_string_fix.ps1 `
  -SourceOutput $SourceOutput `
  -Output $Output `
  -Config .\config\publication_full.toml
```

Required ending:

```text
STRESS-STRING-FIX RESULT MIGRATION PASSED
Migrated stress tasks: 420 / 420
Next work: aggregate stress tables/figures, run solver verification, runtime, and final verifier
```

Do not bypass any fingerprint, stage-marker, task-marker, grid, or SHA-256 refusal.

## 5. Launch the remaining work detached

```powershell
.\scripts\publication\start_codex_detached.ps1 `
  -Threads 4 `
  -Output $Output `
  -Config .\config\publication_full.toml
```

The launcher now automatically writes the absolute output path to:

```text
latest_publication_output.txt
```

Wait for:

```text
DETACHED JULIA PUBLICATION RUN STARTED
Owner: Windows Task Scheduler
Safe to close Codex/terminal: yes
```

The detached suite will reuse all migrated artifacts. It will not rerun the 270 expanded
synthetic tasks, 1,020 manuscript-sweep tasks, 120 meta-learning tasks, Safi, sensitivity,
or the 420 stress tasks. It will perform only:

```text
stress summary/figure aggregation
controlled solver verification
same-machine runtime benchmark
final publication-suite aggregation and artifact verification
```

## 6. Monitor without managing the output variable manually

```powershell
.\scripts\publication\show_latest_codex_run.ps1
```

The explicit equivalent is:

```powershell
$Output = (Get-Content .\latest_publication_output.txt -Raw).Trim()
.\scripts\publication\show_codex_run.ps1 -Output $Output
```

Healthy status has:

```text
failure_files = []
```

After stress aggregation completes, `stress_full` must appear in the completed-stage list.

## 7. Final scientific verification

After the scheduled task stops with exit code 0:

```powershell
$Output = (Get-Content .\latest_publication_output.txt -Raw).Trim()

julia `
  --startup-file=no `
  --project=. `
  --threads=4 `
  .\scripts\publication\verify_publication_artifacts.jl `
  --config .\config\publication_full.toml `
  --output $Output `
  --root .
```

Required ending:

```text
PUBLICATION ARTIFACT VERIFICATION PASSED
```

Only after that message should the generated runtime, solver-verification, stress-summary,
and final reviewer-response tables be used in the manuscript.

## 8. Results and figures

```powershell
.\scripts\publication\show_results.ps1 -Output $Output
Invoke-Item ".\$Output\tables"
Invoke-Item ".\$Output\figures"
```

The regenerated stress artifacts are:

```text
tables\stress_tests_per_seed.csv
tables\stress_tests_summary.csv
tables\stress_correlation_pruning_validation.csv
figures\stress_limited_history_rmse.svg
figures\stress_correlation_rmse.svg
figures\stress_correlation_condition.svg
figures\stress_boundary_cold_start.svg
```
