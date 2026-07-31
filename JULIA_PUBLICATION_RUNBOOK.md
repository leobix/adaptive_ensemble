# Julia publication runbook

## 1. Environment

Use Julia 1.12.6, the version recorded by `Manifest.toml`:

```bash
julia --version
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The environment is application-style. Run tests directly:

```bash
julia --startup-file=no --project=. --threads=4 test/runtests.jl
```

On Windows, `scripts/publication/setup_windows.ps1` performs the integrity, dependency, Gurobi, and test preflights.

## 2. Profiles

- `config/publication_full.toml`: synthetic + Safi plus controlled exact Gurobi verification.
- `config/publication_full_no_gurobi.toml`: the same statistical scope without commercial-solver checks.

Neither profile runs the archival energy or missing hurricane applications.

## 3. Foreground commands

Full suite:

```bash
julia --startup-file=no --project=. --threads=4 \
  scripts/publication/run_publication_suite.jl suite \
  --config config/publication_full_no_gurobi.toml \
  --output publication_artifacts \
  --root .
```

Individual stages replace `suite` with one of:

```text
audit
synthetic
manuscript-sweeps
synthetic-meta
real-world
sensitivity
stress
verification
runtime
```

The runner binds every checkpoint to a fingerprint of the effective configuration, active source files, environment files, and inputs. Do not bypass a fingerprint mismatch.

## 4. Detached Windows execution

```powershell
.\scripts\publication\start_codex_detached.ps1 `
  -Threads 4 `
  -Output publication_artifacts `
  -Config .\config\publication_full.toml
```

The launcher runs preflight and tests before registering a Windows Task Scheduler task. It sets one BLAS thread while Julia uses the requested task threads to avoid nested oversubscription.

Monitor:

```powershell
.\scripts\publication\show_latest_codex_run.ps1
```

Stop or unregister:

```powershell
.\scripts\publication\stop_codex_run.ps1 `
  -Output publication_artifacts `
  -Unregister
```

## 5. Output tree

```text
publication_artifacts/
  tables/
  figures/
  predictions/
  models/
  manifests/
  checkpoints/
  logs/
```

Writes are atomic. Completed task and stage markers include output hashes. A task is resumable only when its fingerprint and output hashes remain valid.

## 6. Final verification

```bash
julia --startup-file=no --project=. --threads=4 \
  scripts/publication/verify_publication_artifacts.jl \
  --config config/publication_full_no_gurobi.toml \
  --output publication_artifacts \
  --root .
```

Do not use results until the verifier prints `PUBLICATION ARTIFACT VERIFICATION PASSED`.

## 7. Reproducibility boundary

The verifier distinguishes completion of the configured synthetic + Safi scope from completion of every application in the historical manuscript. The former can pass while `paper_application_scope_complete=false`; this is intentional and must be disclosed in any paper or release.
