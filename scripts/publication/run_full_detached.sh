#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

JULIA_BIN="${JULIA_BIN:-julia}"
THREADS="${JULIA_THREADS:-4}"
OUTPUT="${1:-publication_artifacts}"
CONFIG="${2:-config/publication_full.toml}"

"$JULIA_BIN" --startup-file=no scripts/publication/verify_original_python_integrity.jl
"$JULIA_BIN" --startup-file=no --project=. --threads="$THREADS" \
  scripts/publication/run_publication_suite.jl preflight \
  --config "$CONFIG" --output "$OUTPUT" --root .

mkdir -p "$OUTPUT/logs"
stamp="$(date -u +%Y%m%d_%H%M%S)"
stdout="$OUTPUT/logs/codex_${stamp}.stdout.log"
stderr="$OUTPUT/logs/codex_${stamp}.stderr.log"
exitcode="$OUTPUT/logs/codex_${stamp}.exitcode.txt"
finished="$OUTPUT/logs/codex_${stamp}.finished.txt"
receipt="$OUTPUT/logs/codex_${stamp}.receipt.json"
wrapper="$OUTPUT/logs/codex_${stamp}.wrapper.sh"
repo_root="$(pwd)"

for value in "$repo_root" "$JULIA_BIN" "$THREADS" "$CONFIG" "$OUTPUT" "$stdout" "$stderr"; do
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
    echo "Paths and arguments may not contain newline characters" >&2
    exit 2
  }
done

cat > "$wrapper" <<EOF_WRAPPER
#!/usr/bin/env bash
set +e
cd $(printf '%q' "$repo_root")
export JULIA_NUM_THREADS=$(printf '%q' "$THREADS")
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
$(printf '%q' "$JULIA_BIN") --startup-file=no --project=. --threads=$(printf '%q' "$THREADS") \\
  scripts/publication/run_publication_suite.jl suite \\
  --config $(printf '%q' "$CONFIG") --output $(printf '%q' "$OUTPUT") --root . \\
  >$(printf '%q' "$stdout") 2>$(printf '%q' "$stderr")
rc=\$?
printf '%s\\n' "\$rc" > $(printf '%q' "$exitcode")
date -u +%Y-%m-%dT%H:%M:%SZ > $(printf '%q' "$finished")
exit "\$rc"
EOF_WRAPPER
chmod 700 "$wrapper"

if command -v setsid >/dev/null 2>&1; then
  nohup setsid bash "$wrapper" </dev/null >/dev/null 2>&1 &
  owner="nohup+setsid"
else
  nohup bash "$wrapper" </dev/null >/dev/null 2>&1 &
  owner="nohup"
fi
pid=$!

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g'
}
printf '{\n  "pid": %s,\n  "owner": "%s",\n  "started_utc": "%s",\n  "stdout": "%s",\n  "stderr": "%s",\n  "exitcode": "%s",\n  "finished": "%s",\n  "output": "%s",\n  "config": "%s"\n}\n' \
  "$pid" "$(json_escape "$owner")" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(json_escape "$stdout")" "$(json_escape "$stderr")" \
  "$(json_escape "$exitcode")" "$(json_escape "$finished")" \
  "$(json_escape "$OUTPUT")" "$(json_escape "$CONFIG")" > "$receipt"

echo "DETACHED JULIA PUBLICATION RUN STARTED"
echo "PID=$pid"
echo "owner=$owner"
echo "receipt: $receipt"
echo "stdout: $stdout"
echo "stderr: $stderr"
echo "Safe to close the launching terminal: yes. Prevent system sleep/hibernation."
