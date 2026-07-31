# Contributing

## Development setup

1. Install Julia 1.12.6.
2. Run `julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`.
3. Run `julia --startup-file=no --project=. --threads=4 test/runtests.jl`.
4. Run `python scripts/ci/repository_audit.py .`.

The project is an application environment; use the direct test entry point instead of `Pkg.test()`.

## Scientific changes

A pull request that changes data processing, causal availability, a model objective, a metric,
hyperparameter selection, experiment scope, or result interpretation must include:

- a regression test;
- the old and new mathematical or statistical behavior;
- an explanation of look-ahead and boundary handling;
- any configuration changes;
- an explicit statement about whether existing checkpoints remain valid.

Never weaken convergence, fingerprint, task-hash, or final-verification gates merely to make a run
finish. Use a new output directory after source, configuration, environment, or data changes.

## Historical files

Do not edit `.py` or `.ipynb` files unless the preservation policy is intentionally being retired and
reviewed. Their hashes are enforced by `reports/ORIGINAL_PYTHON_INTEGRITY.csv`.

## Pull-request hygiene

- Do not commit generated `publication_artifacts*` directories, local receipts, logs, licenses, or credentials.
- Keep one-off migration scripts and machine-specific recovery documents out of the default branch.
- Use descriptive commits and keep refactors separate from scientific changes when practical.
- Confirm data-redistribution permissions before adding or moving datasets.
