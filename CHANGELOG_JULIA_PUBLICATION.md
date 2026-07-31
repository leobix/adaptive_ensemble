# Julia publication-pipeline changelog

## 2026-07-31 GitHub release audit

- Corrected validation-selected and test-oracle member selection to honor the configured chronological selection metric (`rmse`, `mae`, or `cvar_05`) rather than silently defaulting to RMSE.
- Added duplicate scientific-key checks before paired seed intervals are formed.
- Hardened generic string/grid conversion and CSV index-column detection.
- Added regression tests for mixed numeric configuration values, metric-consistent member selection, duplicate paired rows, and legitimate `column_*` feature names.
- Replaced the global `*.csv` ignore rule with targeted generated-output rules so input data and preserved evidence are actually commit-able.
- Added GitHub Actions CI, repository-audit tooling, `.gitattributes`, contribution/security/data-license documentation, citation metadata, and generic runbooks.
- Removed machine-specific recovery scripts and stale one-run audit files from the default release tree.
- Preserved all historical Python and notebook files byte-for-byte.

## 2026-07-30 stress aggregation and recovery fix

- Fixed `String(::Float64)` in stress aggregation by using generic textual conversion for heterogeneous numeric/text setting labels.
- Added explicit numeric parsing for stress and runtime plots after CSV/DataFrame promotion.
- Added a mixed numeric/text stress aggregation regression test.
- Added a guarded migration for eight completed stages and all 420 hash-verified stress tasks.
- Added automatic creation of `latest_publication_output.txt` after detached Task Scheduler ownership is observed.
- Added `show_latest_codex_run.ps1`.
- Removed bulky failed-run artifacts and superseded diagnostic/recovery reports from the release while preserving a compact current-results evidence snapshot.
- Added a 24-page LaTeX/PDF reviewer-response evidence audit.

## 2026-07-30 synthetic + Safi scope stabilization

- Removed the two energy datasets from the active real-world and sensitivity configurations.
- Restricted the reviewer-facing Safi sensitivity surface to the primary smooth Adaptive Ridge estimator; the theory-aligned norm-plus-norm method remains in the completed Safi benchmark and controlled solver verification.
- Disabled large data-set IRLS-to-Gurobi fallback in the active profiles; exact conic solves remain only in the controlled synthetic verification stage.
- Added an explicit `[scope]` block that records included and excluded applications.
- Updated the final verifier to distinguish configured-scope completion from complete reproduction of the historical paper application scope.
- Added a hash- and grid-validated migration utility for completed synthetic, manuscript-sweep, meta-learning, and Safi artifacts.
- Removed obsolete energy diagnostics, diagnostic configurations, superseded migration scripts, and intermediate recovery reports.
- Preserved all historical Python and notebook assets byte-for-byte.
- Preserved historical data and historical result folders; they are not used by the active publication runner.

## Prior scientific corrections retained

- Causal error histories with verification delays and event boundaries.
- Reduced direct, matrix-free PCG, and observation-space dual quadratic solvers.
- Corrected manuscript high-dimensional solver routing.
- Separate quadratic Adaptive Ridge and paper-aligned norm-plus-norm implementations.
- Correct distinction between Hedge and bandit-feedback Exp3.
- Chronological tuning, training-only standardization, exact finite-sample CVaR, modern baselines, moving-block bootstrap, sensitivity, stress, runtime, and numerical-equivalence studies.
- Immutable run fingerprints, atomic artifacts, hash-bound task/stage markers, and Task Scheduler detachment.

## Scope statement

A verified run of `config/publication_full.toml` supports the expanded synthetic evidence and the Safi application. It does not regenerate or validate the appliance-energy or hurricane application claims. The manuscript must be edited accordingly.
