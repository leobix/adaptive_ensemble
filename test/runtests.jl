using Test
using LinearAlgebra
using Random
using CSV
using DataFrames
using TOML
using Statistics
using SHA

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "AdaptiveEnsemblePublication.jl"))
using .AdaptiveEnsemblePublication
const AEP = AdaptiveEnsemblePublication

@testset "exact metrics" begin
    y = [1.0, 2.0, 3.0, 4.0]
    prediction = [0.0, 2.0, 5.0, 4.0]
    @test AEP.mae(y, prediction) ≈ 0.75
    @test AEP.rmse(y, prediction) ≈ sqrt(1.25)
    @test AEP.empirical_cvar([4.0, 3.0, 2.0, 1.0], 0.25) ≈ 4.0
    @test AEP.empirical_cvar([4.0, 3.0, 2.0, 1.0], 0.375) ≈ (4.0 + 0.5 * 3.0) / 1.5
    table = AEP.metric_table(y, prediction)
    @test haskey(table, "cvar_05")
    @test haskey(table, "smape")
    @test all(isfinite, [table["mae"], table["rmse"], table["cvar_05"]])
end

@testset "causal histories, delays, and boundaries" begin
    X = reshape(collect(1.0:18.0), 6, 3)
    y = collect(1.0:6.0)
    Z, counts = AEP.build_error_histories(X, y, 2)
    @test size(Z) == (6, 6)
    @test all(Z[1, :] .== 0.0)
    @test counts[1] == 0
    @test Z[3, 1:3] ≈ X[1, :] .- y[1]
    @test Z[3, 4:6] ≈ X[2, :] .- y[2]

    events = [1, 1, 1, 2, 2, 2]
    event_Z, event_counts = AEP.build_error_histories(X, y, 2; event_ids=events)
    @test all(event_Z[4, :] .== 0.0)
    @test event_counts[4] == 0

    delayed_Z, delayed_counts = AEP.build_error_histories(X, y, 2; availability_lag=2)
    @test all(delayed_Z[2, :] .== 0.0)
    @test delayed_counts[2] == 0
    @test delayed_Z[4, 4:6] ≈ X[2, :] .- y[2]

    # Altering an outcome that is not yet available cannot alter an earlier history row.
    y_modified = copy(y)
    y_modified[5:6] .+= 10_000.0
    Z_modified, _ = AEP.build_error_histories(X, y_modified, 2)
    @test Z_modified[1:5, :] == Z[1:5, :]
end

@testset "reduced affine algebra and stacked-beta Gram" begin
    rng = MersenneTwister(1)
    n, m, tau = 30, 3, 2
    X = randn(rng, n, m)
    y = randn(rng, n)
    Z, _ = AEP.build_error_histories(X, y, tau)
    indices = collect(1:n)
    A = AEP.reduced_design(X, Z, indices)
    theta = randn(rng, m + m * m * tau)
    beta0, V = AEP.unpack_theta(theta, m, tau)
    direct = [dot(X[t, :], beta0 + V * Z[t, :]) for t in indices]
    @test maximum(abs.(A * theta - direct)) < 1e-10

    B = AEP.explicit_beta_map(Z, indices, m)
    H = Matrix(AEP.stacked_beta_gram(Z, indices, m))
    @test norm(H - transpose(B) * B) / max(norm(H), eps()) < 1e-11
end

@testset "quadratic solver, matrix-free solver, and regularization path" begin
    rng = MersenneTwister(2)
    n, m = 260, 3
    X = randn(rng, n, m)
    true_weights = [0.2, -0.3, 1.1]
    y = X * true_weights + 0.01 .* randn(rng, n)
    train = collect(1:200)
    test = collect(201:n)

    direct = AEP.fit_adaptive_quadratic(X, y, train, 2, 1e-2;
        direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int))
    matrix_free = AEP.fit_adaptive_quadratic(X, y, train, 2, 1e-2;
        direct_max_parameters=0, max_dense_bytes=0,
        cg_rtol=1e-11, cg_atol=1e-13, cg_maxiter=5000,
        allow_dual=false)
    p_direct = AEP.predict_adaptive(direct, X, y, test)
    p_matrix_free = AEP.predict_adaptive(matrix_free, X, y, test)
    @test all(isfinite, p_direct)
    @test AEP.rmse(y[test], p_direct) < 0.15
    @test maximum(abs.(p_direct .- p_matrix_free)) < 1e-6
    @test abs(direct.objective - matrix_free.objective) / max(abs(direct.objective), eps()) < 1e-7

    lambdas = [0.0, 1e-3, 1e-2, 1e-1]
    workspace = AEP.prepare_adaptive_quadratic_workspace(X, y, train, 2;
        direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int))
    path = AEP.fit_adaptive_quadratic_path(workspace, lambdas)
    @test length(path) == length(lambdas)
    for (lambda, model) in zip(lambdas, path)
        independent = AEP.fit_adaptive_quadratic(X, y, train, 2, lambda;
            direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int))
        @test maximum(abs.(model.theta .- independent.theta)) < 1e-7
    end
end



@testset "PCG stopping rule is relative to the right-hand side" begin
    operator = x -> [x[1], 2.0 * x[2]]
    rhs = [1.0, 1.0]
    initial = [1.0 + 5e-10, 0.5]
    solution, iterations, converged, residual = AEP._pcg(
        operator, rhs; diagonal=[1.0, 2.0], x0=initial,
        rtol=1e-9, atol=0.0, maxiter=10)
    @test converged
    @test iterations == 0
    @test residual <= 1e-9 * norm(rhs)
    @test norm(operator(solution) - rhs) <= 1e-9 * norm(rhs)
end

@testset "dual stacked-beta solver matches direct and covers manuscript high dimensions" begin
    rng = MersenneTwister(20260729)
    n, m, tau = 320, 5, 3
    X = randn(rng, n, m)
    y = X * randn(rng, m) + 0.05 .* randn(rng, n)
    train = collect(1:240)
    validation = collect(241:n)

    direct = AEP.fit_adaptive_quadratic(X, y, train, tau, 1e-2;
        direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int))
    dual = AEP.fit_adaptive_quadratic(X, y, train, tau, 1e-2;
        direct_max_parameters=0, max_dense_bytes=0, dual_max_rows=1000)
    p_direct = AEP.predict_adaptive(direct, X, y, validation)
    p_dual = AEP.predict_adaptive(dual, X, y, validation)
    @test startswith(dual.linear_solver, "dual_")
    @test maximum(abs.(p_direct .- p_dual)) < 1e-7
    @test abs(direct.objective - dual.objective) / max(abs(direct.objective), eps()) < 1e-8

    # Regression for the failed m=50, tau=5 manuscript setting. The dual system
    # is only n×n even though the reduced parameter vector has 12,550 entries.
    bundle = AEP.generate_synthetic_regime("gradual", 1;
        n=500, m=50, train_end=300, val_end=400)
    workspace = AEP.prepare_adaptive_quadratic_workspace(
        bundle.X, bundle.y, collect(bundle.split.train), 5;
        direct_max_parameters=0, max_dense_bytes=0)
    models = AEP.fit_adaptive_quadratic_path(workspace, [1e-4, 1e-2];
        dual_max_rows=1000)
    @test workspace.d == 12_550
    @test all(model -> model.linear_solver == "dual_eigen_stacked_beta", models)
    @test all(model -> model.converged && isfinite(model.objective), models)
    @test all(model -> model.residual_norm / max(norm(workspace.rhs), eps()) < 1e-8, models)
end

@testset "paper-aligned robust-norm IRLS" begin
    rng = MersenneTwister(22)
    n, m = 240, 3
    X = randn(rng, n, m)
    y = X * [0.4, -0.2, 0.9] + 0.03 .* randn(rng, n)
    train = collect(1:180)
    validation = collect(181:n)
    model = AEP.fit_adaptive_robust_norm(X, y, train, 2, 1e-2;
        irls_rtol=1e-6, irls_maxiter=500,
        direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int))
    prediction = AEP.predict_adaptive(model, X, y, validation)
    @test model.objective_variant == :norm_irls
    @test model.converged
    @test model.iterations >= 2
    @test all(isfinite, prediction)
    @test AEP.rmse(y[validation], prediction) < 0.20
end

@testset "event-aware moving-block bootstrap stays within storm boundaries" begin
    event_ids = [1, 1, 1, 2, 2, 3, 3, 3]
    rng = MersenneTwister(9)
    sampled = AEP._event_block_indices(rng, event_ids, 2)
    @test length(sampled) == length(event_ids)
    @test all((1 .<= sampled) .& (sampled .<= length(event_ids)))
    # Each output segment is sampled only from the corresponding original event run.
    @test all(event_ids[sampled[1:3]] .== 1)
    @test all(event_ids[sampled[4:5]] .== 2)
    @test all(event_ids[sampled[6:8]] .== 3)
end

@testset "original Python and notebook files are byte-identical" begin
    manifest = DataFrame(CSV.File(joinpath(REPO, "reports", "ORIGINAL_PYTHON_INTEGRITY.csv")))
    @test nrow(manifest) == 28
    for row in eachrow(manifest)
        path = joinpath(REPO, split(String(row.relative_path), '/')...)
        @test isfile(path)
        @test filesize(path) == Int(row.bytes)
        @test bytes2hex(SHA.sha256(read(path))) == String(row.sha256)
    end
end

@testset "synthetic publication regimes" begin
    for regime in AEP.SYNTHETIC_REGIMES
        bundle = AEP.generate_synthetic_regime(regime, 3)
        @test size(bundle.X) == (4000, 10)
        @test length(bundle.y) == 4000
        @test length(bundle.split.train) == 2000
        @test length(bundle.split.validation) == 1000
        @test length(bundle.split.test) == 1000
        @test all(isfinite, bundle.X)
        @test all(isfinite, bundle.y)
    end
end

@testset "real-data loaders and exact chronological splits" begin
    safi = AEP.load_safi(REPO)
    energy = AEP.load_energy(REPO)
    energy_all = AEP.load_energy(REPO; member_set=:all_available)
    @test size(safi.X) == (8494, 6)
    @test length(safi.y) == 8494
    @test length(safi.split.train) == 4247
    @test length(safi.split.validation) == 1699
    @test length(safi.split.test) == 2548
    @test !all(safi.X[:, 1] .== 1.0)  # first forecast was not overwritten by an intercept

    @test size(energy.X) == (9858, 10)
    @test length(energy.y) == 9858
    @test length(energy.split.train) == 4929
    @test length(energy.split.validation) == 1973
    @test length(energy.split.test) == 2956
    @test energy.y[1] == parse(Float64, strip(first(readlines(joinpath(REPO, "data", "energy_y_test_val.csv")))))
    @test energy.name == "energy_historical_selected_10"
    @test size(energy_all.X) == (9858, 24)
    @test energy_all.name == "energy_all_available"
    @test AEP.load_dataset("energy_historical_selected_10", REPO).name == energy.name
    @test AEP.load_dataset("energy_all_available", REPO).name == energy_all.name
end

@testset "standardization uses training targets only" begin
    bundle = AEP.generate_synthetic_regime("stationary", 11; n=400, m=4, train_end=200, val_end=300)
    standardized, transform = AEP.standardize_bundle(bundle)
    @test abs(mean(standardized.y[bundle.split.train])) < 1e-12
    @test std(standardized.y[bundle.split.train]) ≈ 1.0
    @test maximum(abs.(AEP.inverse_transform(transform, standardized.y) .- bundle.y)) < 1e-10
end

@testset "online comparison methods" begin
    bundle = AEP.generate_synthetic_regime("recurring", 4;
        n=500, m=4, train_end=250, val_end=375)
    initialization = collect(bundle.split.train)
    prediction_indices = collect(bundle.split.validation)
    calls = [
        () -> AEP.hedge_predictions(bundle.X, bundle.y, initialization, prediction_indices),
        () -> AEP.exp3_predictions(bundle.X, bundle.y, initialization, prediction_indices; seed=4),
        () -> AEP.fixed_share_predictions(bundle.X, bundle.y, initialization, prediction_indices),
        () -> AEP.passive_aggressive_predictions(bundle.X, bundle.y, initialization, prediction_indices),
        () -> AEP.rls_predictions(bundle.X, bundle.y, initialization, prediction_indices),
        () -> AEP.rolling_ridge_predictions(bundle.X, bundle.y, initialization, prediction_indices; window=50),
        () -> AEP.dma_predictions(bundle.X, bundle.y, initialization, prediction_indices),
    ]
    for call in calls
        prediction, weights = call()
        @test length(prediction) == length(prediction_indices)
        @test size(weights) == (length(prediction_indices), 4)
        @test all(isfinite, prediction)
        @test all(isfinite, weights)
    end
    hedge_prediction, hedge_weights = AEP.hedge_predictions(
        bundle.X, bundle.y, initialization, prediction_indices)
    @test all(abs.(vec(sum(hedge_weights; dims=2)) .- 1.0) .< 1e-10)
    @test all(isfinite, hedge_prediction)

    exp3_prediction, exp3_probabilities = AEP.exp3_predictions(
        bundle.X, bundle.y, initialization, prediction_indices; seed=44)
    exp3_prediction_repeat, exp3_probabilities_repeat = AEP.exp3_predictions(
        bundle.X, bundle.y, initialization, prediction_indices; seed=44)
    @test exp3_prediction == exp3_prediction_repeat
    @test exp3_probabilities == exp3_probabilities_repeat
    @test all(abs.(vec(sum(exp3_probabilities; dims=2)) .- 1.0) .< 1e-10)
    @test minimum(exp3_probabilities) > 0.0
end

@testset "Julia-native modern meta learners" begin
    bundle = AEP.generate_synthetic_regime("recurring", 5;
        n=500, m=4, train_end=250, val_end=375)
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    gate = AEP.fit_neural_gate(bundle.X, bundle.y, train, 2;
        epochs=3, batch_size=64, seed=5)
    gate_prediction, gate_weights = AEP.predict_neural_gate(
        gate, bundle.X, bundle.y, validation)
    @test all(isfinite, gate_prediction)
    @test all(abs.(vec(sum(gate_weights; dims=2)) .- 1.0) .< 1e-8)

    boosted = AEP.fit_boosted_meta(bundle.X, bundle.y, train, 2;
        n_trees=3, min_leaf=10)
    boosted_prediction = AEP.predict_boosted_meta(
        boosted, bundle.X, bundle.y, validation)
    @test all(isfinite, boosted_prediction)
end

@testset "limited-history construction cannot borrow earlier observations" begin
    bundle = AEP._limited_history_bundle(8, 100)
    @test length(bundle.split.train) + length(bundle.split.validation) == 100
    @test first(bundle.split.test) == 101
    @test first(bundle.split.train) == 1
end


@testset "complete-history cold start is explicit" begin
    X = reshape(collect(1.0:24.0), 8, 3)
    y = collect(1.0:8.0)
    complete, complete_counts = AEP.build_error_histories(X, y, 3)
    partial, partial_counts = AEP.build_error_histories(X, y, 3; require_complete=false)
    @test all(complete[1:3, :] .== 0.0)
    @test complete_counts[1:3] == [0, 0, 0]
    @test complete_counts[4] == 3
    @test any(partial[2, :] .!= 0.0)
    @test partial_counts[2] == 1
end

@testset "rolling Ridge sufficient statistics match brute-force refits" begin
    rng = MersenneTwister(31)
    n, m = 90, 4
    X = randn(rng, n, m)
    y = X * [0.4, -0.2, 0.7, 0.1] + 0.02 .* randn(rng, n)
    init = collect(1:50)
    pred = collect(51:70)
    window = 17
    lambda = 0.03
    optimized, optimized_weights = AEP.rolling_ridge_predictions(
        X, y, init, pred; window=window, lambda=lambda, availability_lag=1)
    brute = Float64[]
    brute_weights = Matrix{Float64}(undef, length(pred), m)
    for (row, t) in enumerate(pred)
        stop = t - 1
        history = collect(max(1, stop - window + 1):stop)
        model = AEP.fit_static_ridge(X, y, history, lambda)
        brute_weights[row, :] .= model.weights
        push!(brute, dot(model.weights, @view X[t, :]))
    end
    @test maximum(abs.(optimized .- brute)) < 1e-8
    @test maximum(abs.(optimized_weights .- brute_weights)) < 1e-8
end

@testset "online methods cannot use outcomes before verification" begin
    rng = MersenneTwister(41)
    X = randn(rng, 45, 4)
    y = randn(rng, 45)
    altered = copy(y)
    altered[19:end] .+= 10_000.0
    init = collect(1:20)
    pred = collect(21:25)
    lag = 3
    calls = [
        (target, seed) -> AEP.hedge_predictions(X, target, init, pred; availability_lag=lag),
        (target, seed) -> AEP.exp3_predictions(X, target, init, pred; availability_lag=lag, seed=seed),
        (target, seed) -> AEP.fixed_share_predictions(X, target, init, pred; availability_lag=lag),
        (target, seed) -> AEP.passive_aggressive_predictions(X, target, init, pred; availability_lag=lag),
        (target, seed) -> AEP.rls_predictions(X, target, init, pred; availability_lag=lag),
        (target, seed) -> AEP.rolling_ridge_predictions(X, target, init, pred; availability_lag=lag, window=12),
        (target, seed) -> AEP.dma_predictions(X, target, init, pred; availability_lag=lag),
    ]
    for call in calls
        baseline, _ = call(y, 1234)
        counterfactual, _ = call(altered, 1234)
        @test baseline[1] == counterfactual[1]
    end
end

@testset "run identities and checkpoint hashes reject stale artifacts" begin
    temporary = mktempdir()
    config_path = joinpath(temporary, "config.toml")
    write(config_path, "[run]\nprofile = \"test\"\n")
    config = TOML.parsefile(config_path)
    paths = AEP.ensure_run_paths(joinpath(temporary, "output"))
    AEP.initialize_run_identity(paths, temporary, config_path, config; command="test")
    result_path = joinpath(paths.tables, "result.csv")
    AEP.atomic_csv_write(result_path, DataFrame(value=[1.0]))
    AEP.mark_task_complete(paths, "unit_task"; outputs=[result_path])
    @test AEP.task_complete(paths, "unit_task")
    AEP.atomic_csv_write(result_path, DataFrame(value=[2.0]))
    @test !AEP.task_complete(paths, "unit_task")

    write(config_path, "[run]\nprofile = \"changed\"\n")
    changed = TOML.parsefile(config_path)
    @test_throws ErrorException AEP.initialize_run_identity(
        paths, temporary, config_path, changed; command="test")
end

@testset "SVG artifacts are atomic and reject invalid inputs" begin
    directory = mktempdir()
    line_path = joinpath(directory, "line.svg")
    bar_path = joinpath(directory, "bar.svg")
    AEP.svg_line_chart(line_path, Dict("series" => ([1.0, 2.0], [2.0, 1.0]));
        title="line", xlabel="x", ylabel="y")
    AEP.svg_bar_chart(bar_path, ["negative", "positive"], [-1.0, 2.0]; title="bar")
    @test isfile(line_path) && filesize(line_path) > 0
    @test isfile(bar_path) && filesize(bar_path) > 0
    @test_throws ArgumentError AEP.svg_line_chart(joinpath(directory, "empty.svg"),
        Dict{String,Tuple{Vector{Float64},Vector{Float64}}}())
    @test_throws ArgumentError AEP.svg_bar_chart(joinpath(directory, "nan.svg"),
        ["bad"], [NaN])
end

@testset "configuration and publication paths" begin
    config = TOML.parsefile(joinpath(REPO, "config", "publication_full.toml"))
    @test config["synthetic"]["seeds"] == 30
    @test length(config["synthetic"]["regimes"]) == 9
    @test "discrete_gaussian" in config["synthetic"]["regimes"]
    @test "exp3" in config["synthetic"]["methods"]
    @test config["real_world"]["bootstrap_reps"] == 2000
    @test config["real_world"]["datasets"] == ["safi"]
    @test config["sensitivity"]["datasets"] == ["safi"]
    @test config["sensitivity"]["objective_variants"] == ["quadratic_stacked"]
    @test config["scope"]["label"] == "synthetic_safi"
    @test "energy_historical_selected_10" in config["scope"]["excluded"]
    @test config["scope"]["paper_application_scope_complete"] == false
    @test config["manuscript_sweeps"]["selection_metric"] == "mae"
    @test config["manuscript_sweeps"]["fixed_tau"] == 5
    @test config["adaptive"]["direct_max_parameters"] == 3500
    @test config["adaptive"]["dual_max_rows"] == 3500
    @test config["adaptive"]["irls_objective_rtol"] == 1.0e-8
    @test config["adaptive"]["irls_stationarity_rtol"] == 1.0e-8
    @test config["adaptive"]["irls_practical_stationarity_rtol"] == 1.0e-7
    @test config["adaptive"]["irls_maxiter"] == 1000
    @test config["adaptive"]["irls_min_iterations"] == 2
    @test config["adaptive"]["irls_fallback_solver"] == "none"
    @test config["adaptive"]["gurobi_threads"] == 1
    @test config["adaptive"]["gurobi_time_limit"] == 1800.0
    @test length(config["manuscript_sweeps"]["discrete_probabilities"]) == 6
    no_gurobi = TOML.parsefile(joinpath(REPO, "config", "publication_full_no_gurobi.toml"))
    @test no_gurobi["synthetic"]["seeds"] == config["synthetic"]["seeds"]
    @test no_gurobi["real_world"]["bootstrap_reps"] == config["real_world"]["bootstrap_reps"]
    @test no_gurobi["sensitivity"]["objective_variants"] == ["quadratic_stacked"]
    @test no_gurobi["adaptive"]["irls_fallback_solver"] == "none"
    @test no_gurobi["verification"]["require_gurobi"] == false
    @test no_gurobi["verification"]["run_literal_jump_equivalence"] == false
    @test no_gurobi["runtime"]["run_gurobi_verification"] == false
    expected = AEP.expected_publication_stages(config)
    @test "real_world_safi" in expected
    @test !("real_world_energy_historical_selected_10" in expected)
    @test !("real_world_energy_all_available" in expected)
    @test "real_world_all" in expected
    @test "sensitivity_safi" in expected
    @test !("sensitivity_energy_historical_selected_10" in expected)
    @test !("sensitivity_energy_all_available" in expected)
    @test "sensitivity_all" in expected
    @test "publication_suite" in expected
    @test isfile(joinpath(REPO, "scripts", "publication", "start_codex_detached.ps1"))
    @test isfile(joinpath(REPO, "scripts", "publication", "verify_publication_artifacts.jl"))
    temporary = mktempdir()
    paths = AEP.ensure_run_paths(temporary)
    @test isdir(paths.tables)
    @test isdir(paths.figures)
    @test isdir(paths.checkpoints)
end



@testset "robust-norm practical certificate rejects false energy nonconvergence" begin
    reason = AEP._robust_norm_convergence_reason(
        1000,
        7.34045999474977e-6,
        3.259583246899126e-10,
        3.371132348144002e-8;
        min_iterations=2,
        iterate_rtol=1e-8,
        objective_rtol=1e-8,
        stationarity_rtol=1e-8,
        practical_stationarity_rtol=1e-7,
    )
    @test reason == "objective_and_practical_stationarity"

    # The same stationarity residual must not be accepted without objective
    # stabilization; the practical certificate is conjunctive by design.
    rejected = AEP._robust_norm_convergence_reason(
        1000,
        7.34045999474977e-6,
        1e-4,
        3.371132348144002e-8;
        min_iterations=2,
        iterate_rtol=1e-8,
        objective_rtol=1e-8,
        stationarity_rtol=1e-8,
        practical_stationarity_rtol=1e-7,
    )
    @test isempty(rejected)
end

@testset "robust-norm IRLS diagnostics satisfy the declared convergence contract" begin
    bundle = AEP.generate_synthetic_regime("recurring", 73;
        n=700, m=5, train_end=350, val_end=525)
    train = collect(bundle.split.train)
    workspace = AEP.prepare_adaptive_quadratic_workspace(
        bundle.X, bundle.y, train, 3)

    iterate_rtol = 1e-12
    objective_rtol = 1e-8
    strict_stationarity_rtol = 1e-8
    practical_stationarity_rtol = 1e-7
    models = AEP.fit_adaptive_robust_norm_path(
        workspace, [1e-6, 1e-3, 0.1, 2.0];
        irls_rtol=iterate_rtol,
        irls_objective_rtol=objective_rtol,
        irls_stationarity_rtol=strict_stationarity_rtol,
        irls_practical_stationarity_rtol=practical_stationarity_rtol,
        irls_maxiter=500,
        irls_min_iterations=2,
        require_convergence=true,
    )

    @test all(model -> model.converged, models)
    @test all(model -> !isempty(model.convergence_reason), models)
    @test all(model -> isfinite(model.objective), models)
    @test all(model -> isfinite(model.residual_norm), models)
    @test all(model -> isfinite(model.relative_objective_change), models)
    @test all(model -> isfinite(model.stationarity_residual), models)

    for model in models
        reason = model.convergence_reason
        if reason == "iterate_objective_and_strict_stationarity"
            @test model.residual_norm <= max(iterate_rtol, objective_rtol)
            @test model.stationarity_residual <= strict_stationarity_rtol
        elseif reason == "strict_smoothed_stationarity"
            @test model.stationarity_residual <= strict_stationarity_rtol
        elseif reason == "iterate_and_objective"
            # `residual_norm` stores max(relative_theta, relative_objective)
            # for robust-norm models, so this is a conservative observable
            # check of the two stabilization quantities.
            @test model.residual_norm <= max(iterate_rtol, objective_rtol)
        elseif reason == "objective_and_practical_stationarity"
            @test model.relative_objective_change <= objective_rtol
            @test model.stationarity_residual <= practical_stationarity_rtol
        else
            @test false  # An undocumented convergence reason is never accepted.
        end
    end
end

@testset "robust-norm fallback solver is explicit" begin
    bundle = AEP.generate_synthetic_regime("stationary", 109;
        n=120, m=3, train_end=70, val_end=95)
    train = collect(bundle.split.train)
    @test_throws ArgumentError AEP.fit_adaptive_robust_norm(
        bundle.X, bundle.y, train, 2, 1e-2;
        fallback_solver=:invalid,
    )
end

if get(ENV, "RUN_GUROBI_TESTS", "0") == "1"
    @testset "Gurobi exact-objective and literal-equivalence checks" begin
        bundle = AEP.generate_synthetic_regime("stationary", 7;
            n=140, m=3, train_end=80, val_end=110)
        train = collect(bundle.split.train)
        exact = AEP.fit_adaptive_norm_gurobi(bundle.X, bundle.y, train, 2, 1e-2)
        irls = AEP.fit_adaptive_robust_norm(bundle.X, bundle.y, train, 2, 1e-2;
            irls_rtol=1e-7, irls_maxiter=500)
        @test exact.objective_variant == :norm
        @test irls.objective_variant == :norm_irls
        @test abs(irls.objective - exact.objective) / max(abs(exact.objective), eps()) < 1e-4
        result = AEP.verify_reduced_against_jump(bundle.X, bundle.y, train, 2, 1e-2)
        @test result["max_prediction_difference"] < 1e-6
        @test result["relative_objective_difference"] < 1e-7

        # Force the fast IRLS path to hit its limit and exercise the exact
        # SOCP fallback on a small controlled synthetic problem. The active
        # synthetic+Safi profile never invokes this fallback on real data.
        fallback = AEP.fit_adaptive_robust_norm(
            bundle.X, bundle.y, train, 2, 1e-2;
            irls_rtol=1e-16,
            irls_objective_rtol=1e-16,
            irls_stationarity_rtol=1e-16,
            irls_practical_stationarity_rtol=1e-16,
            irls_maxiter=1,
            irls_min_iterations=1,
            fallback_solver=:gurobi,
            gurobi_output_flag=0,
            gurobi_threads=1,
            gurobi_optimality_tolerance=1e-8,
            require_convergence=true,
        )
        @test fallback.converged
        @test fallback.objective_variant == :norm
        @test fallback.linear_solver == "irls_limit_then_gurobi_socp"
        @test fallback.convergence_reason ==
              "gurobi_socp_fallback_after_irls_limit"
        @test abs(fallback.objective - exact.objective) /
              max(abs(exact.objective), eps()) < 1e-7
    end
end

@testset "one-seed paired interval output is schemaful" begin
    input = DataFrame(
        regime = [
            "stationary",
            "stationary",
        ],
        seed = [
            1,
            1,
        ],
        method = [
            "adaptive_ridge",
            "static_ridge",
        ],
        rmse = [
            0.10,
            0.12,
        ],
    )

    result =
        AdaptiveEnsemblePublication.paired_seed_intervals(
            input;
            reference = "adaptive_ridge",
            metric = :rmse,
            groupcol = :regime,
        )

    @test nrow(result) == 1
    @test result.n[1] == 1
    @test result.interval_available[1] == false
    @test ismissing(result.ci_low[1])
    @test ismissing(result.ci_high[1])
    @test result.mean_method_minus_reference[1] ≈ 0.02
end
@testset "stress aggregation handles numeric and textual settings" begin
    rows = DataFrame(
        analysis = Any["limited_history", "limited_history", "boundary"],
        setting = Any[100.0, 100.0, "reset"],
        method = Any["adaptive_ridge", "adaptive_ridge", "adaptive_ridge__reset"],
        rmse = [0.20, 0.30, 0.10],
        cvar_05 = [0.50, 0.70, 0.25],
        cold_start_rmse = Union{Missing,Float64}[missing, missing, 0.12],
        condition_number = Union{Missing,Float64}[1.0e6, 2.0e6, missing],
        max_abs_weight = Union{Missing,Float64}[2.0, 4.0, 1.5],
    )

    summary = AEP._summarize_stress_rows(rows)
    history = summary[(summary.analysis .== "limited_history") .&
                      (summary.setting .== "100.0"), :]
    boundary = summary[(summary.analysis .== "boundary") .&
                       (summary.setting .== "reset"), :]

    @test nrow(history) == 1
    @test history.replications[1] == 2
    @test history.rmse_mean[1] ≈ 0.25
    @test nrow(boundary) == 1
    @test boundary.cold_start_rmse_mean[1] ≈ 0.12
    @test AEP._stress_numeric_settings(Any[100.0, "0.95"]) == [100.0, 0.95]
end

@testset "generic configuration and label conversions" begin
    @test AEP.parse_float_grid(0.25) == [0.25]
    @test AEP.parse_float_grid("0.1, 0.2") == [0.1, 0.2]
    @test AEP.parse_int_grid(4) == [4]
    @test AEP.parse_int_grid("2, 5") == [2, 5]
    @test_throws ArgumentError AEP.parse_float_grid("")
    @test_throws ArgumentError AEP.parse_float_grid([1.0, Inf])
    @test AEP._string_vector("safi", ["fallback"]) == ["safi"]
    @test AEP._svg_escape(1.5) == "1.5"
end

@testset "member selection honors the configured validation metric" begin
    X = zeros(7, 2)
    y = zeros(7)
    # On validation rows, member 1 has lower MAE but higher RMSE than member 2.
    X[3:5, 1] .= [0.0, 0.0, 10.0]
    X[3:5, 2] .= [4.0, 4.0, 4.0]
    X[6:7, 1] .= 1.0
    X[6:7, 2] .= 9.0
    split = AEP.SplitSpec(train=1:2, validation=3:5, test=6:7)
    bundle = AEP.DatasetBundle(
        name="selection_metric_probe",
        X=X,
        y=y,
        member_names=["mae_member", "rmse_member"],
        split=split,
        availability_lag=1,
    )

    _, mae_member, _ = AEP.oracle_best_member_predictions(
        X, y, collect(split.validation); metric=:mae)
    _, rmse_member, _ = AEP.oracle_best_member_predictions(
        X, y, collect(split.validation); metric=:rmse)
    @test mae_member == 1
    @test rmse_member == 2
    @test_throws ArgumentError AEP.oracle_best_member_predictions(
        X, y, collect(split.validation); metric=:unsupported)

    config = Dict{String,Any}(
        "experiment" => Dict{String,Any}("selection_metric" => "mae"),
    )
    metrics, predictions, _, _, _ = AEP.evaluate_bundle(
        bundle;
        methods=["validation_best_member"],
        config=config,
        standardize=false,
        seed=1,
    )
    @test nrow(metrics) == 1
    @test predictions.validation_best_member == [1.0, 1.0]
    @test metrics.selected_member_name[1] == "mae_member"
    @test metrics.selection_metric[1] == "mae"
end

@testset "paired seed intervals reject duplicate scientific keys" begin
    duplicate_reference = DataFrame(
        regime=["stationary", "stationary", "stationary"],
        seed=[1, 1, 1],
        method=["adaptive_ridge", "adaptive_ridge", "static_ridge"],
        rmse=[0.10, 0.11, 0.12],
    )
    @test_throws ArgumentError AEP.paired_seed_intervals(
        duplicate_reference; reference="adaptive_ridge", metric=:rmse,
        groupcol=:regime)

    duplicate_method = DataFrame(
        regime=["stationary", "stationary", "stationary"],
        seed=[1, 1, 1],
        method=["adaptive_ridge", "static_ridge", "static_ridge"],
        rmse=[0.10, 0.12, 0.13],
    )
    @test_throws ArgumentError AEP.paired_seed_intervals(
        duplicate_method; reference="adaptive_ridge", metric=:rmse,
        groupcol=:regime)
end

@testset "CSV loader distinguishes generated from legitimate column names" begin
    frame = DataFrame(
        Column1=[1, 2],
        column_temperature=[10.0, 11.0],
        forecast=[12.0, 13.0],
    )
    cleaned = AEP._drop_index_columns(frame)
    @test names(cleaned) == ["column_temperature", "forecast"]
end
