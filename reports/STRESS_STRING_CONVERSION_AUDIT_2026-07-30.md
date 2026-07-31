# Audit of the stress-stage `String(::Float64)` failure

## Executive finding

All 420 configured stress tasks finished successfully. The run failed afterward in the
summary/plotting layer, not in any scientific fit. The immediate exception was:

```text
MethodError: no method matching String(::Float64)
```

The source expression was the summary-row construction in
`src/publication/experiments.jl`:

```julia
setting = [String(subset.setting[1])]
```

## Why the type became `Float64`

The task outputs are written separately and concatenated through CSV/DataFrames. The
`setting` field intentionally represents different experiment axes:

- `100`, `200`, ..., `3000` for limited history;
- `0.0`, `0.25`, ..., `0.99` for correlation;
- text such as `reset`, `carry`, and `borrow_across_events` for boundary policies.

CSV type inference and DataFrame column promotion therefore produce a heterogeneous
column whose numeric cells can be `Float64`. Julia has no `String(::Float64)` constructor.
The general textual representation function is `string(x)`.

## Scientific integrity of the completed task outputs

The source run contains:

```text
3,180 stress method rows
1,050 correlation-pruning validation rows
420 completed stress-task markers
0 stress task failure files
```

The 420 tasks comprise:

```text
6 history sizes x 30 seeds       = 180
7 correlation levels x 30 seeds  = 210
1 boundary bundle x 30 seeds     = 30
```

The stage marker is absent because aggregation failed. The task-level outputs are valid
and are migrated only after marker-fingerprint and output-hash verification.

## Code correction

The corrected implementation defines:

```julia
_stress_setting_text(value) = string(value)
_stress_numeric_settings(values) = parse.(Float64, string.(values))
```

and centralizes stress aggregation in `_summarize_stress_rows`. The same generic text
conversion is used in runtime plotting and the final artifact verifier where values may have
passed through heterogeneous CSV columns.

## Regression coverage

`test/runtests.jl` now constructs a DataFrame containing both:

```text
100.0
"reset"
```

in the same `setting` column and verifies summary aggregation, cold-start metrics, and
numeric plot parsing.

## Recovery design

`scripts/publication/migrate_after_stress_string_fix.jl` requires the exact source
fingerprint:

```text
0e9320deaf6da755c152474c1e96dc22bc90719f90b27213cb523f4aade21c9a
```

It re-hashes every referenced stage and task output, copies only verified artifacts into a
new run-fingerprint directory, and intentionally excludes the incomplete stress marker and
old launcher/failure logs. The resumed run performs stress aggregation, solver verification,
runtime measurement, and final artifact verification only.

## Scope qualification

The active configuration is synthetic + Safi. Energy was removed after an uncertified
robust-norm real-data path and an interrupted exact fallback; hurricane matrices are absent.
A successful verifier therefore certifies the configured synthetic + Safi scope, not all
historical applications.
