#!/usr/bin/env bash
set -euo pipefail

# Concurrency / Julia
MAX_JOBS=${MAX_JOBS:-4}
JULIA_BIN=${JULIA_BIN:-julia}
JULIA_PROJECT_FLAG=${JULIA_PROJECT_FLAG:---project=.}
INSTANTIATE=${INSTANTIATE:-0}

# Experiment axes
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-30}
M_LIST=${M_LIST:-"3 6 10 15 25 50"}
OUT_ROOT=${OUT_ROOT:-results/exp1_vary_m}

# Grids
RIDGE_LAM_GRID=${RIDGE_LAM_GRID:-"1e-4 1e-3 1e-2 1e-1 1"}
AR_LAM_GRID=${AR_LAM_GRID:-"1e-4 1e-3 1e-2 1e-1 1"}
PA_EPS_GRID=${PA_EPS_GRID:-"1e-4 1e-3 1e-2 1e-1 1"}
HEDGE_ETA_GRID=${HEDGE_ETA_GRID:-"0.01 0.05 0.1"}
FS_ALPHA_GRID=${FS_ALPHA_GRID:-"0.001 0.01 0.05"}
RLS_LAMBDA_GRID=${RLS_LAMBDA_GRID:-"0.95 0.98 0.99 1.0"}

TREE_MAXD_GRID=${TREE_MAXD_GRID:-"3 4 5"}
TREE_MINLEAF_GRID=${TREE_MINLEAF_GRID:-"5 10"}
TREE_THR_GRID=${TREE_THR_GRID:-"25"}

GBRT_EST_GRID=${GBRT_EST_GRID:-"300 600"}
GBRT_LR_GRID=${GBRT_LR_GRID:-"0.05 0.1"}
GBRT_MAXD_GRID=${GBRT_MAXD_GRID:-"3 4"}
GBRT_MINLEAF_GRID=${GBRT_MINLEAF_GRID:-"5 10"}
GBRT_THR_GRID=${GBRT_THR_GRID:-"25"}

# Optional: instantiate deps
if [ "$INSTANTIATE" -eq 1 ]; then
  "$JULIA_BIN" "$JULIA_PROJECT_FLAG" -e 'using Pkg; Pkg.instantiate()'
fi

# Fixed base args for EXP1
BASE_ARGS=(
  --T 4000
  --period 500
  --bias 1.0
  --std_pert 0.1
  --past 5
  --num-past 10
  --rho_beta 0.001
  --rho 0.001
  --rho_V 0.001
  --bias_range 0.5
  --std_range 0.5
  --bias_drift 0.5
  --std_drift 0.5
  --p_ber 1.0
  --total_drift_additive
  --lead_time 1
  --seed_y 1
)

# Encode floats for folder names
enc(){ printf "%s" "$1" | sed 's/\./p/g'; }

run_job() {
  local stage="$1"; shift   # val | test
  local m="$1"; shift
  local seed="$1"; shift

  # Hyperparams (one config per run)
  local ridge_lam="$1"; shift
  local ar_lam="$1"; shift
  local pa_eps="$1"; shift
  local hedge_eta="$1"; shift
  local fs_alpha="$1"; shift
  local rls_lambda="$1"; shift
  local tree_d="$1"; shift
  local tree_leaf="$1"; shift
  local tree_thr="$1"; shift
  local gbrt_est="$1"; shift
  local gbrt_lr="$1"; shift
  local gbrt_d="$1"; shift
  local gbrt_leaf="$1"; shift
  local gbrt_thr="$1"; shift

  # Encode EVERYTHING in directory name (for Python pairing)
  local tag="m_${m}_seed_${seed}__rid_$(enc "$ridge_lam")__ar_$(enc "$ar_lam")__pa_$(enc "$pa_eps")__ewa_$(enc "$hedge_eta")__fs_$(enc "$fs_alpha")__rls_$(enc "$rls_lambda")__td${tree_d}_tl${tree_leaf}_tt${tree_thr}__ge${gbrt_est}_glr_$(enc "$gbrt_lr")_gmd${gbrt_d}_gml${gbrt_leaf}_gnt${gbrt_thr}"
  local results_dir="${OUT_ROOT}/${stage}/${tag}"
  mkdir -p "$results_dir"

  echo "[RUN][$stage] m=${m} seed=${seed} ridgeλ=${ridge_lam} arλ=${ar_lam} PAε=${pa_eps} η=${hedge_eta} α=${fs_alpha} RLSλ=${rls_lambda} tree(d=${tree_d},leaf=${tree_leaf}) gbrt(est=${gbrt_est},lr=${gbrt_lr})"

  "$JULIA_BIN" "$JULIA_PROJECT_FLAG" src/main_synthetic_parallel.jl \
    "${BASE_ARGS[@]}" \
    --N_models "${m}" \
    --seed "${seed}" \
    --rho_stat "${ridge_lam}" \
    --rho "${ridge_lam}" \
    --rho_beta "${ar_lam}" \
    --hedge_eta "${hedge_eta}" \
    --rls_lambda "${rls_lambda}" \
    --epsilon-inf "${pa_eps}" \
    --tree_max_depth "${tree_d}" \
    --tree_min_leaf "${tree_leaf}" \
    --tree_num_thresholds "${tree_thr}" \
    --gbrt_estimators "${gbrt_est}" \
    --gbrt_lr "${gbrt_lr}" \
    --gbrt_max_depth "${gbrt_d}" \
    --gbrt_min_leaf "${gbrt_leaf}" \
    --gbrt_num_thresholds "${gbrt_thr}" \
    --results_dir "${results_dir}" \
    --filename-results "results_seed${seed}.csv" \
    "$@" \
    > >(tee "${results_dir}/run.log") 2>&1
}

# Simple job queue
PIDS=()
enqueue(){ if [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; then wait "${PIDS[0]}" || true; PIDS=("${PIDS[@]:1}"); fi; run_job "$@" & PIDS+=("$!"); }

# Sweep
for m in $M_LIST; do
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for ridge_lam in $RIDGE_LAM_GRID; do
      for ar_lam in $AR_LAM_GRID; do
        for pa_eps in $PA_EPS_GRID; do
          for hedge_eta in $HEDGE_ETA_GRID; do
            for fs_alpha in $FS_ALPHA_GRID; do
              for rls_lambda in $RLS_LAMBDA_GRID; do
                for tree_d in $TREE_MAXD_GRID; do
                  for tree_leaf in $TREE_MINLEAF_GRID; do
                    for tree_thr in $TREE_THR_GRID; do
                      for gbrt_est in $GBRT_EST_GRID; do
                        for gbrt_lr in $GBRT_LR_GRID; do
                          for gbrt_d in $GBRT_MAXD_GRID; do
                            for gbrt_leaf in $GBRT_MINLEAF_GRID; do
                              for gbrt_thr in $GBRT_THR_GRID; do
                                # Validation stage
                                enqueue "val" "$m" "$seed" \
                                  "$ridge_lam" "$ar_lam" "$pa_eps" "$hedge_eta" "$fs_alpha" "$rls_lambda" \
                                  "$tree_d" "$tree_leaf" "$tree_thr" \
                                  "$gbrt_est" "$gbrt_lr" "$gbrt_d" "$gbrt_leaf" "$gbrt_thr" \
                                  --train_test_split 0.5 --val 1000
                                # Test stage (same config)
                                enqueue "test" "$m" "$seed" \
                                  "$ridge_lam" "$ar_lam" "$pa_eps" "$hedge_eta" "$fs_alpha" "$rls_lambda" \
                                  "$tree_d" "$tree_leaf" "$tree_thr" \
                                  "$gbrt_est" "$gbrt_lr" "$gbrt_d" "$gbrt_leaf" "$gbrt_thr" \
                                  --train_test_split 0.75 --val 1000
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
          done
        done
      done
    done
  done
done

for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "[done] EXP1 seed-wise validation sweeps finished → ${OUT_ROOT}/{val,test}/m_*/"