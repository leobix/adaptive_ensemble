# Historical pipeline

The repository preserves original research scripts under root-level `src/`, `python_code/`, notebooks,
and historical result directories. They are useful for provenance but are not the source of corrected
reviewer-response evidence.

Material differences identified by the executable audit include historical data mutation, expanded
history designs, a different squared objective, ambiguous expert-algorithm naming, test-set oracle
selection, and incomplete delayed-feedback/event handling. Run the active audit with:

```bash
julia --startup-file=no --project=. scripts/publication/run_audit.jl \
  --config config/publication_full_no_gurobi.toml \
  --output publication_artifacts_audit \
  --root .
```

Do not combine historical and active outputs unless mathematical compatibility and provenance have
been established explicitly.
