# GitHub commit-readiness audit — 2026-07-31

## Scope

This audit covers the corrected Julia publication pipeline, repository hygiene, reproducibility
contracts, GitHub staging behavior, security-sensitive files, preserved historical Python/notebooks,
and the active synthetic + Safi experiment scope.

## Material bugs corrected

### 1. Repository data were unintentionally untrackable

The prior `.gitignore` contained a global `*.csv` rule. A fresh `git add --all` therefore omitted
input data, result snapshots, integrity manifests, and many review artifacts. The rule was replaced by
targeted ignores for generated publication-output roots. Representative input and preserved-result
CSVs are now confirmed staged in a clean Git simulation.

### 2. Best-member comparators ignored the configured tuning metric

`validation_best_member` and `oracle_best_member` silently used RMSE even when an experiment
selected models by MAE or CVaR. This undermined the stated uniform tuning protocol. Both comparators
now use the configured `selection_metric`, record the selected member name and score, and have a
regression test in which the MAE-optimal and RMSE-optimal members differ.

### 3. Paired intervals could silently multiply duplicate rows

A duplicated `(group, method, seed)` key could create a many-to-many join and invalid uncertainty
intervals. The interval routine now rejects duplicate reference or method seed rows before joining.

### 4. Generic conversion paths were brittle

Generic textual and grid-conversion helpers were hardened to accept numbers, strings, and vectors
without reproducing the prior `String(::Float64)` failure class. Empty and nonfinite grids are rejected.

### 5. CSV index detection was overbroad

Any feature whose name started with `column` was previously dropped as an inferred CSV index.
Only generated names matching `Column<digits>` are now removed; legitimate names such as
`column_temperature` are preserved.

### 6. Target loaders accepted ambiguous multi-column targets

Safi and archival energy target loaders now require exactly one target column after documented index
handling rather than silently selecting the first column.

## GitHub release engineering

Added:

- targeted `.gitignore` and normalization-aware `.gitattributes`;
- Julia GitHub Actions CI using the direct application test entry point;
- a standard-library-only repository audit;
- citation, contribution, security, data/license, architecture, and release documentation;
- pull-request and bug-report templates;
- a complete data SHA-256 inventory.

Removed:

- one-machine recovery instructions tied to a single failed run;
- obsolete stress-fix migration utilities;
- superseded one-run audit/change manifests.

## Preservation and security checks

- 28 historical Python/notebook files match their immutable byte counts and SHA-256 values.
- No private keys, GitHub tokens, AWS access keys, or Gurobi WLS credential assignments were found
  in active code/configuration/documentation.
- No generated publication-output root, run receipt, Gurobi license file, or local user path is tracked.
- Every TOML file and notebook JSON document parses.
- All Python files compile in an isolated temporary directory.
- Shell scripts pass `bash -n`.
- Active Julia and PowerShell files pass lexical string/comment/delimiter checks.

## Git commit simulation

A temporary Git repository was initialized, all files were staged, representative input/result CSVs
were confirmed tracked, generated publication outputs were confirmed ignored, and
`git diff --cached --check` passed. The simulated repository contained 209 tracked files and a clean
initial commit. The two largest tracked files are 83.0 MB and 57.9 MB, below GitHub's 100 MiB hard
limit but above its warning threshold.

## Scientific scope

The active profiles remain synthetic + Safi. Archival energy files are retained but not loaded by the
active profiles, and hurricane matrices are absent. The final artifact verifier continues to distinguish
completion of the configured scope from complete reproduction of all applications in the historical
manuscript.

## Executable-validation boundary

The artifact-building container did not contain Julia, PowerShell, Windows Task Scheduler, or a
Gurobi license. Julia compilation and the release test suite therefore remain mandatory on the target
machine and in GitHub Actions. Static validation is not presented as a substitute for those tests.
