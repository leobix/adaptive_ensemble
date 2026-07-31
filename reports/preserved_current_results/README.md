# Preserved current-results snapshot

These compact tables were copied or derived from the uploaded failed run with fingerprint `0e9320deaf6da755c152474c1e96dc22bc90719f90b27213cb523f4aade21c9a`. They preserve the completed scientific summaries while the bulky failed output directory is intentionally excluded from the source release.

`stress_tests_summary_provisional.csv` was deterministically aggregated from all 3,180 completed Julia stress-task rows after the original stage failed during label conversion. It is evidence for audit and reviewer-response drafting, but the corrected Julia pipeline must regenerate `tables/stress_tests_summary.csv`, figures, and the `stress_full` stage marker before final publication verification.

The guarded migration utility copies the original fingerprint-bound stage and task outputs from the user's retained output directory into a fresh corrected-run directory after re-hashing every file.
