#!/usr/bin/env bash
set -euo pipefail

# Synthetic benchmarks sweep launcher
# - Runs src/main_synthetic_parallel.jl once per hyperparameter combo.
# - Each run evaluates ALL baselines (mean, best, bandits, PA, ridge,
#   adaptive variants, hedge, hedge-ewa, rls, tree, gbrt) on the same split.
# - Results are written to a unique --results_dir encoding the hyperparams.

# Tweak concurrency to your machine
MAX_JOBS=${MAX_JOBS:-4}

JULIA_BIN=${JULIA_BIN:-julia}
JULIA_PROJECT_FLAG=${JULIA_PROJECT_FLAG:---project=.}

BASE_ARGS=(
  --T 2000
  --period 500
  --bias 1.0
  --std_pert 0.1
  --N_models 10
  --train_test_split 0.75
  --val 1000
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
  --seed 1
  --seed_y 1
)

# Sweeps (adjust to budget)
HEDGE_ETAS=(0 0.01 0.05 0.1)          # 0 => auto
RLS_LAMBDAS=(0.95 0.98 0.99 1.0)

TREE_MAX_DEPTHS=(2 3)
TREE_MIN_LEAFS=(5 10)
TREE_NUM_THRS=(25)

GBRT_ESTIMATORS=(30 60)
GBRT_LRS=(0.05 0.1)
GBRT_MAX_DEPTHS=(1 2)
GBRT_MIN_LEAFS=(5)
GBRT_NUM_THRS=(25)

run_job() {
  local results_dir="$1"; shift
  mkdir -p "$results_dir"
  echo "[RUN] $results_dir"
  "$JULIA_BIN" "$JULIA_PROJECT_FLAG" src/main_synthetic_parallel.jl \
    "${BASE_ARGS[@]}" \
    "$@" \
    --results_dir "$results_dir" >/dev/stdout 2>&1 | tee "$results_dir/run.log"
}

# Simple job queue compatible with older bash (no wait -n)
PIDS=()
enqueue() {
  local results_dir="$1"; shift
  # If too many jobs, wait for the oldest to finish
  if [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; then
    local pid_to_wait=${PIDS[0]}
    wait "$pid_to_wait" || true
    PIDS=("${PIDS[@]:1}")
  fi
  run_job "$results_dir" "$@" &
  PIDS+=("$!")
}

main() {
  if [ "${INSTANTIATE:-0}" -eq 1 ]; then
    echo "Instantiating Julia project dependencies..."
    "$JULIA_BIN" "$JULIA_PROJECT_FLAG" -e 'using Pkg; Pkg.instantiate()'
  else
    echo "Skipping instantiate (set INSTANTIATE=1 to enable)."
  fi
  for eta in "${HEDGE_ETAS[@]}"; do
    for lam in "${RLS_LAMBDAS[@]}"; do
      for tmd in "${TREE_MAX_DEPTHS[@]}"; do
        for tml in "${TREE_MIN_LEAFS[@]}"; do
          for tnt in "${TREE_NUM_THRS[@]}"; do
            for ge in "${GBRT_ESTIMATORS[@]}"; do
              for glr in "${GBRT_LRS[@]}"; do
                for gmd in "${GBRT_MAX_DEPTHS[@]}"; do
                  for gml in "${GBRT_MIN_LEAFS[@]}"; do
                    for gnt in "${GBRT_NUM_THRS[@]}"; do
                      dir="results/synth_eta_${eta}_lam_${lam}_td${tmd}_tl${tml}_tt${tnt}_ge${ge}_glr${glr}_gmd${gmd}_gml${gml}_gnt${gnt}"
                      enqueue "$dir" \
                        --hedge_eta "$eta" \
                        --rls_lambda "$lam" \
                        --tree_max_depth "$tmd" \
                        --tree_min_leaf "$tml" \
                        --tree_num_thresholds "$tnt" \
                        --gbrt_estimators "$ge" \
                        --gbrt_lr "$glr" \
                        --gbrt_max_depth "$gmd" \
                        --gbrt_min_leaf "$gml" \
                        --gbrt_num_thresholds "$gnt"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done

  # Wait for remaining jobs
  for pid in "${PIDS[@]}"; do
    wait "$pid" || true
  done
  echo "All jobs completed."
}

main "$@"
