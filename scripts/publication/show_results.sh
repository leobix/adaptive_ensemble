#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
JULIA_BIN="${JULIA_BIN:-julia}"
OUTPUT="${1:-publication_artifacts}"
"$JULIA_BIN" --startup-file=no --project=. scripts/publication/show_results.jl "$OUTPUT"
printf '\nFigure directory: %s\n' "$(pwd)/$OUTPUT/figures"
