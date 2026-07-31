#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
JULIA_BIN="${JULIA_BIN:-julia}"
OUTPUT="${1:-publication_artifacts}"
"$JULIA_BIN" --startup-file=no --project=. scripts/publication/run_publication_suite.jl status --output "$OUTPUT" --root .
latest="$(find "$OUTPUT/logs" -maxdepth 1 -type f -name 'codex_*.stdout.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -n "$latest" ]]; then
  echo "Latest stdout: $latest"
  tail -n 30 "$latest"
fi
