#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
JULIA_BIN="${JULIA_BIN:-julia}"
DATASET="${1:-synthetic}"
case "$DATASET" in
  energy)
    "$JULIA_BIN" --startup-file=no --project=. src/main.jl --data energy --end-id 8 --val 2000 --ridge --past 10 --num-past 500 --rho 0.1 --train_test_split 0.5 ;;
  safi)
    "$JULIA_BIN" --startup-file=no --project=. src/main_hypertune.jl --data safi_speed --begin-id 1 --end-id 8 --val 1699 --train_test_split 0.5 --num-past 5000 --param_combo 1 ;;
  hurricane_na)
    "$JULIA_BIN" --startup-file=no --project=. src/main.jl --data hurricane_NA --end-id 17 --val 500 --train_test_split 0.5 --past 3 --num-past 350 --rho 0.01 --rho_V 0.1 --rho_beta 0.1 --begin-id 1 ;;
  synthetic)
    "$JULIA_BIN" --startup-file=no --project=. src/main_synthetic_parallel.jl --past 5 --num-past 10 --train_test_split 0.75 --period 4 --val 1000 --total_drift_additive --bias_range 0.5 --std_range 0.5 --T 2000 --seed 1 --N_models 10 --bias_drift 0.5 --std_drift 0.5 --CVAR --rho_beta 0.001 --rho 0.001 --rho_V 0.001 ;;
  *) echo "usage: $0 synthetic|safi|energy|hurricane_na" >&2; exit 2 ;;
esac
