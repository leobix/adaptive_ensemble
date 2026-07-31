#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
JULIA_BIN="${JULIA_BIN:-julia}"
"$JULIA_BIN" --version
"$JULIA_BIN" --startup-file=no --project=. -e 'VERSION == v"1.12.6" || error("Julia 1.12.6 is required for the locked publication environment"); println("Julia version check passed: ", VERSION)'
"$JULIA_BIN" --startup-file=no scripts/publication/verify_original_python_integrity.jl
"$JULIA_BIN" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(; julia_version_strict=true); Pkg.precompile(); Pkg.status()'
if [[ "${SKIP_GUROBI_CHECK:-0}" != "1" ]]; then
  "$JULIA_BIN" --startup-file=no --project=. -e 'include("src/AdaptiveEnsemblePublication.jl"); using .AdaptiveEnsemblePublication; println(gurobi_preflight())'
  RUN_GUROBI_TESTS=1 "$JULIA_BIN" --startup-file=no --project=. test/runtests.jl
else
  "$JULIA_BIN" --startup-file=no --project=. test/runtests.jl
fi
printf '%s\n' 'Julia publication environment is ready.'
