# Architecture

## Active pipeline

`src/AdaptiveEnsemblePublication.jl` assembles the active module from `src/publication/`:

- `datasets.jl`: strict loaders, provenance, and chronological splits;
- `features.jl`: causal error histories, verification delays, and event boundaries;
- `adaptive_ridge.jl`: reduced quadratic, matrix-free, robust-norm, and controlled JuMP solvers;
- `baselines.jl`: static, recursive, expert-weighting, and online comparators;
- `validation.jl`: uniform chronological selection and final refitting;
- `statistics.jl`: replication summaries and block-bootstrap inference;
- `experiments.jl`: resumable stage/task orchestration;
- `io_utils.jl`: atomic writes, run fingerprints, and output hashes;
- `runtime.jl`: controlled computational benchmarks;
- `svg_plots.jl`: dependency-light publication graphics.

## Model

For `m` members and history length `tau`, the adaptive coefficient rule is

```text
beta_t = beta_0 + V_0 z_t
```

with `m + m^2*tau` reduced parameters. `z_t` contains only member errors whose outcomes are
available before prediction time. Incomplete histories use the configured static fallback; event-aware
runs do not borrow error histories across event boundaries.

## Selection protocol

All tunable methods use the same chronological training and validation segments. The configured
selection metric is passed to Adaptive Ridge, baselines, neural/meta learners, the validation-selected
member, and the diagnostic test oracle. The test segment is evaluated once after selection.

## Reproducibility

Every task marker records the run fingerprint and SHA-256 hashes of required outputs. The fingerprint
covers active source, configuration, environment, and input data. Atomic writes prevent partially
written files from being accepted. The final verifier checks the complete configured grid independently
of process exit status.

## Historical boundary

`src/main*.jl`, `src/algos/`, `python_code/`, and notebooks are historical research materials. They
are intentionally retained, but their semantics differ from the corrected publication pipeline in areas
such as history construction, objective definitions, baseline labels, and test-oracle usage.
