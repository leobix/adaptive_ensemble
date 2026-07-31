# Static validation report: stress heterogeneous-setting fix

The artifact-building environment did not provide Julia, Windows PowerShell, Task
Scheduler, or a Gurobi license. Executable Julia and Windows validation must therefore run
on the target machine. Static and provenance checks completed successfully before
packaging.

## Repository checks

- 7 TOML files parsed.
- 18 notebooks parsed as JSON.
- 10 Python files compiled into a temporary directory.
- 9 shell scripts passed `bash -n`.
- 61 Julia files passed delimiter/string/comment lexical checking.
- 11 PowerShell files passed delimiter/string/comment lexical checking.
- 195 packaged files were scanned for text-file NUL bytes and conflict markers.
- 28 historical Python/notebook files matched the uploaded repository byte-for-byte.
- No `publication_artifacts*` run directory is packaged.
- The active profile label is `synthetic_safi`.
- The active real-world and sensitivity dataset lists are exactly `["safi"]`.

## Source-run salvage checks

The migration source fingerprint was verified as:

```text
0e9320deaf6da755c152474c1e96dc22bc90719f90b27213cb523f4aade21c9a
```

The following were re-hashed successfully against their recorded marker hashes:

- 8 completed stage markers and all required stage outputs;
- 420 completed stress task markers;
- 3,180 finite stress method rows;
- 1,050 finite correlation-pruning validation rows;
- the 3,180-row aggregate stress task table;
- the 1,050-row aggregate pruning table.

The 420 tasks decompose into 180 limited-history tasks, 210 correlation/pruning tasks, and
30 boundary-policy tasks. The failed source run had no stress-task failure file; it failed only
while converting a numeric setting label during summary construction.

## PDF validation

`reports/Interim_Reviewer_Response_Current_Results.pdf` is a 24-page, 456,633-byte PDF.
It was rendered at 150 DPI. The rendered contact sheet and the corrected correlation page
were inspected for clipping, broken glyphs, table overflow, and unreadable figures.

## Mandatory executable sequence on Windows

```text
scripts/publication/setup_windows.ps1
scripts/publication/migrate_after_stress_string_fix.ps1
scripts/publication/start_codex_detached.ps1
scripts/publication/verify_publication_artifacts.jl
```

Static validation is not a substitute for Julia compilation, Gurobi verification, runtime
execution, or the final scientific artifact verifier.
