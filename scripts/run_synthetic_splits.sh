#!/usr/bin/env bash
set -euo pipefail

# Synthetic experiments launcher with fixed chronological splits
# Total length: 4000 time steps (T=2000 -> 2T=4000 generated)
# Train: 1..2000 (train_test_split=0.5)
# Val:   2001..3000 (val=1000)
# Test:  3001..4000 (train_test_split=0.75, val=1000)
# Repetitions: 30 independent seeds (resample everything: seed and seed_y)

# Concurrency
MAX_JOBS=${MAX_JOBS:-4}
JULIA_BIN=${JULIA_BIN:-julia}
JULIA_PROJECT_FLAG=${JULIA_PROJECT_FLAG:---project=.}

# One-time instantiate (set INSTANTIATE=1 to enable)
if [ "${INSTANTIATE:-0}" -eq 1 ]; then
  echo "Instantiating Julia project dependencies..."
  "$JULIA_BIN" "$JULIA_PROJECT_FLAG" -e 'using Pkg; Pkg.instantiate()'
else
  echo "Skipping instantiate (set INSTANTIATE=1 to enable)."
fi

# Seeds
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-30}

# Base synthetic args (adjust as needed)
BASE_ARGS=(
  --T 2000               # 2T = 4000 total samples
  --period 500
  --bias 1.0
  --std_pert 0.1
  --N_models 10
  --past 5
  --num-past 10
  --rho_beta 0.001
  --rho 0.001
  --rho_V 0.001
  --bias_range 0.5
  --std_range 0.5
  --bias_drift 0.5
  --std_drift 0.5
  --total_drift_additive
)

# Baseline hyperparameters (single values; script evaluates ALL baselines per run)
HEDGE_ETA=${HEDGE_ETA:-0.05}
RLS_LAMBDA=${RLS_LAMBDA:-0.99}
TREE_MAX_DEPTH=${TREE_MAX_DEPTH:-2}
TREE_MIN_LEAF=${TREE_MIN_LEAF:-5}
TREE_NUM_THRS=${TREE_NUM_THRS:-25}
GBRT_ESTIMATORS=${GBRT_ESTIMATORS:-30}
GBRT_LR=${GBRT_LR:-0.1}
GBRT_MAX_DEPTH=${GBRT_MAX_DEPTH:-2}
GBRT_MIN_LEAF=${GBRT_MIN_LEAF:-5}
GBRT_NUM_THRS=${GBRT_NUM_THRS:-25}

run_job() {
  local stage="$1"; shift
  local seed="$1"; shift
  local results_dir="$1"; shift
  mkdir -p "$results_dir"
  echo "[RUN][$stage][seed=$seed] $results_dir"
  "$JULIA_BIN" "$JULIA_PROJECT_FLAG" src/main_synthetic_parallel.jl \
    "${BASE_ARGS[@]}" \
    --seed "$seed" --seed_y "$seed" \
    --hedge_eta "$HEDGE_ETA" \
    --rls_lambda "$RLS_LAMBDA" \
    --tree_max_depth "$TREE_MAX_DEPTH" \
    --tree_min_leaf "$TREE_MIN_LEAF" \
    --tree_num_thresholds "$TREE_NUM_THRS" \
    --gbrt_estimators "$GBRT_ESTIMATORS" \
    --gbrt_lr "$GBRT_LR" \
    --gbrt_max_depth "$GBRT_MAX_DEPTH" \
    --gbrt_min_leaf "$GBRT_MIN_LEAF" \
    --gbrt_num_thresholds "$GBRT_NUM_THRS" \
    "$@" \
    --results_dir "$results_dir" > >(tee "$results_dir/run.log") 2>&1
}

# Simple job queue (portable, no wait -n)
PIDS=()
enqueue() {
  if [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; then
    local pid_to_wait=${PIDS[0]}
    wait "$pid_to_wait" || true
    PIDS=("${PIDS[@]:1}")
  fi
  run_job "$@" &
  PIDS+=("$!")
}

for seed in $(seq "$SEED_START" "$SEED_END"); do
  # Validation stage (2001..3000): split at 0.5, val=1000
  VAL_DIR="results/splits/val/seed_${seed}_eta${HEDGE_ETA}_lam${RLS_LAMBDA}_td${TREE_MAX_DEPTH}_tl${TREE_MIN_LEAF}_tt${TREE_NUM_THRS}_ge${GBRT_ESTIMATORS}_glr${GBRT_LR}_gmd${GBRT_MAX_DEPTH}_gml${GBRT_MIN_LEAF}_gnt${GBRT_NUM_THRS}"
  enqueue val "$seed" "$VAL_DIR" \
    --train_test_split 0.5 \
    --val 1000

  # Test stage (3001..4000): split at 0.75, val=1000
  TEST_DIR="results/splits/test/seed_${seed}_eta${HEDGE_ETA}_lam${RLS_LAMBDA}_td${TREE_MAX_DEPTH}_tl${TREE_MIN_LEAF}_tt${TREE_NUM_THRS}_ge${GBRT_ESTIMATORS}_glr${GBRT_LR}_gmd${GBRT_MAX_DEPTH}_gml${GBRT_MIN_LEAF}_gnt${GBRT_NUM_THRS}"
  enqueue test "$seed" "$TEST_DIR" \
    --train_test_split 0.75 \
    --val 1000
done

# Wait remaining jobs
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "All jobs completed."

