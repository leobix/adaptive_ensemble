module AdaptiveEnsemblePublication

using ArgParse
using CSV
using DataFrames
using Dates
using Distributions
using Gurobi
using JuMP
using LinearAlgebra
using Printf
using Random
using Serialization
using SHA
using Sockets
using SparseArrays
using Statistics
using StatsBase
using TOML
using Base.Threads

# MOI is exported by `using JuMP` above.

include("publication/types.jl")
include("publication/io_utils.jl")
include("publication/metrics.jl")
include("publication/features.jl")
include("publication/adaptive_ridge.jl")
include("publication/baselines.jl")
include("publication/synthetic.jl")
include("publication/datasets.jl")
include("publication/validation.jl")
include("publication/statistics.jl")
include("publication/svg_plots.jl")
include("publication/experiments.jl")
include("publication/runtime.jl")
include("publication/legacy_audit.jl")
include("publication/cli.jl")

export DatasetBundle, SplitSpec, RunPaths, Standardizer,
       AdaptiveQuadraticWorkspace, AdaptiveRidgeModel, StaticRidgeModel,
       NeuralGateModel, BoostedMetaModel, DEFAULT_METHODS, SYNTHETIC_REGIMES,
       ensure_run_paths, load_config, save_run_manifest, publication_main,
       initialize_run_identity, current_run_identity, current_run_fingerprint,
       stage_complete, task_complete, expected_publication_stages,
       run_publication_suite, run_synthetic_publication,
       run_manuscript_sweeps_publication, run_synthetic_meta_publication,
       run_real_world_publication, run_sensitivity_publication,
       run_stress_publication, run_solver_verification_publication,
       run_runtime_publication, run_legacy_audit_publication,
       run_original_legacy_command, metric_table, metrics_dataframe,
       empirical_cvar, mae, rmse, mape_safe, smape, wape,
       fit_adaptive_quadratic, fit_adaptive_quadratic_path,
       fit_adaptive_robust_norm, fit_adaptive_norm_gurobi,
       predict_adaptive, fit_static_ridge, predict_static,
       prepare_adaptive_quadratic_workspace,
       build_error_histories, reduced_design, unpack_theta,
       explicit_beta_map, stacked_beta_gram, adaptive_coefficients,
       hedge_predictions, exp3_predictions, fixed_share_predictions,
       passive_aggressive_predictions, rls_predictions,
       rolling_ridge_predictions, dma_predictions,
       fit_neural_gate, predict_neural_gate,
       fit_boosted_meta, predict_boosted_meta,
       load_safi, load_energy, load_hurricane, load_dataset,
       standardize_bundle, inverse_transform, generate_synthetic_regime,
       generate_correlation_stress, gurobi_preflight,
       verify_reduced_against_jump

end # module
