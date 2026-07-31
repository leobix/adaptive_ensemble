# Instructions for coding agents

## Active boundary

- The corrected Julia pipeline lives in `src/AdaptiveEnsemblePublication.jl`, `src/publication/`, `scripts/publication/`, `config/`, and `test/`.
- Do not modify historical `.py` or `.ipynb` files. Their immutable hashes are checked against `reports/ORIGINAL_PYTHON_INTEGRITY.csv`.
- Do not silently substitute datasets or fabricate missing hurricane inputs.
- The active publication profiles intentionally include synthetic experiments and Safi only. Energy and hurricane materials are archival.
- Revised-paper results must use the publication runner; historical `src/main*.jl` entry points are reproduction-only.

## Required validation

Run:

```bash
julia --startup-file=no --project=. --threads=4 test/runtests.jl
python scripts/ci/repository_audit.py .
```

Use the Gurobi-enabled tests only when a valid license is available. Never claim Gurobi verification from the no-Gurobi profile.

## Long runs

On Windows, launch through `scripts/publication/start_codex_detached.ps1`. Never describe a run as detached until the launcher confirms Windows Task Scheduler ownership. Do not bypass preflight, test, fingerprint, concurrent-writer, or output-hash gates.

## Outputs

Generated publication artifacts belong in a dedicated ignored `publication_artifacts*` directory. Do not overwrite historical `results*` folders or manually create completion markers.

## Pull requests

- Keep scientific changes separate from generated artifacts.
- Add regression tests for every bug fix.
- Document changes to experiment scope, grids, metrics, or claims.
- Do not commit credentials, Gurobi license files, local paths, or machine-specific run receipts.
