# Adaptive Ensemble Forecasting with Adaptive Robust Optimization

This repository contains the historical research code and a corrected Julia publication pipeline for adaptive time-series forecast ensembles. The active pipeline learns coefficients of the form

\[
\beta_t = \beta_0 + V_0 Z_t,
\]

where `Z_t` contains only forecast errors that are causally available before prediction time `t`.

## Active, verified experiment scope

The current publication profiles are deliberately limited to:

- publication-scale synthetic experiments (nine nonstationarity regimes, 30 seeds);
- corrected synthetic sweeps over ensemble size, drift, history length, and window size;
- modern online and meta-learning comparisons;
- the Safi one-hour-ahead wind-speed application;
- Safi sensitivity, limited-history, correlation, boundary, runtime, and controlled solver-verification studies.

Energy and tropical-cyclone files remain as historical research inputs, but they are not part of the active `publication_full*.toml` profiles. The final verifier reports this scope explicitly and does not claim that every historical paper application has been regenerated.

## Requirements

- Julia **1.12.6** (the version recorded by `Manifest.toml`)
- Windows, Linux, or macOS for ordinary runs
- Windows Task Scheduler for the provided detached Codex launcher
- A valid Gurobi license only for `config/publication_full.toml`; use `config/publication_full_no_gurobi.toml` otherwise

The root `Project.toml` is an application environment, not a registered Julia package. Run `test/runtests.jl` directly rather than `Pkg.test()`.

## Setup and tests

### Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
Get-ChildItem .\scripts -Recurse -File |
  Where-Object { $_.Extension -in '.ps1', '.psm1', '.psd1' } |
  Unblock-File

.\scripts\publication\setup_windows.ps1
```

Use `-SkipGurobiCheck` only with the no-Gurobi profile.

### Linux or macOS

```bash
bash scripts/publication/setup_unix.sh
julia --startup-file=no --project=. --threads=4 test/runtests.jl
```

## Run the publication suite

Foreground:

```bash
julia --startup-file=no --project=. --threads=4 \
  scripts/publication/run_publication_suite.jl suite \
  --config config/publication_full_no_gurobi.toml \
  --output publication_artifacts \
  --root .
```

Windows detached run:

```powershell
.\scripts\publication\start_codex_detached.ps1 `
  -Threads 4 `
  -Output publication_artifacts `
  -Config .\config\publication_full.toml
```

Close Codex or the launching terminal only after the launcher prints `DETACHED JULIA PUBLICATION RUN STARTED`, identifies Windows Task Scheduler as the owner, and confirms that it is safe to close the terminal.

Monitor the latest detached run:

```powershell
.\scripts\publication\show_latest_codex_run.ps1
```

## Verify results

A process exit code of zero is necessary but not sufficient. Run the independent artifact verifier:

```bash
julia --startup-file=no --project=. --threads=4 \
  scripts/publication/verify_publication_artifacts.jl \
  --config config/publication_full_no_gurobi.toml \
  --output publication_artifacts \
  --root .
```

Only the final message

```text
PUBLICATION ARTIFACT VERIFICATION PASSED
```

authorizes use of the configured output in a paper.

## Repository layout

```text
src/AdaptiveEnsemblePublication.jl   active Julia module
src/publication/                     corrected publication implementation
scripts/publication/                 launch, monitoring, verification, and setup tools
config/                              full and no-Gurobi experiment profiles
test/runtests.jl                     executable Julia test suite
reports/preserved_current_results/   immutable evidence snapshot from a prior verified run
docs/                                architecture, runbook, data/license, and response materials
src/main*.jl, src/algos/             historical Julia pipeline
python_code/, notebooks/             historical Python/notebook materials (hash protected)
```

## Data and large-file note

Two archival energy CSV files exceed 50 MiB but remain below GitHub's 100 MiB hard limit. Command-line pushes are required; browser uploads will not work for them. For a public long-lived repository, Git LFS or release assets are preferable. The MIT license applies to repository code, not automatically to every included dataset. Review [`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md) before making the repository public.

## Historical code

Historical entry points are preserved for labeled reproduction. Revised-paper results must use `scripts/publication/run_publication_suite.jl`; the historical scripts contain implementation choices documented in `docs/ARCHITECTURE.md` and should not be mixed with the corrected artifact tree.

## Citation

See [`CITATION.cff`](CITATION.cff).
