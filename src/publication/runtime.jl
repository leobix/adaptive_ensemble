"""Benchmark one callable after explicit warm-up; compilation is excluded from summaries."""
function _benchmark_call(f::Function; reps::Int=11, warmup::Int=3)
    reps >= 3 || throw(ArgumentError("runtime benchmark requires at least three measured repetitions"))
    warmup >= 1 || throw(ArgumentError("runtime benchmark requires at least one warm-up"))
    for _ in 1:warmup
        f()
    end
    seconds = zeros(Float64, reps)
    allocated_bytes = zeros(Int, reps)
    for replication in 1:reps
        GC.gc()
        measurement = @timed f()
        seconds[replication] = measurement.time
        allocated_bytes[replication] = measurement.bytes
    end
    return (
        seconds=seconds,
        allocated_bytes=allocated_bytes,
        median_seconds=median(seconds),
        q10_seconds=quantile(seconds, 0.10),
        q90_seconds=quantile(seconds, 0.90),
        median_allocated_bytes=median(allocated_bytes),
        reps=reps,
        warmup=warmup,
    )
end

function _append_runtime!(summary::DataFrame, raw::DataFrame,
                          kind, setting,
                          numeric_setting::Real, benchmark;
                          observations::Int=0, parameters::Int=0,
                          solver="", notes="")
    append!(summary, DataFrame(
        kind=[string(kind)], setting=[string(setting)], numeric_setting=[Float64(numeric_setting)],
        observations=[observations], parameters=[parameters], solver=[string(solver)],
        median_seconds=[benchmark.median_seconds], q10_seconds=[benchmark.q10_seconds],
        q90_seconds=[benchmark.q90_seconds],
        median_allocated_bytes=[Float64(benchmark.median_allocated_bytes)],
        reps=[benchmark.reps], warmup=[benchmark.warmup], notes=[string(notes)],
    ); cols=:union)
    for replication in eachindex(benchmark.seconds)
        append!(raw, DataFrame(
            kind=[string(kind)], setting=[string(setting)], numeric_setting=[Float64(numeric_setting)],
            replication=[replication], seconds=[benchmark.seconds[replication]],
            allocated_bytes=[benchmark.allocated_bytes[replication]],
        ); cols=:union)
    end
    return nothing
end

function _parameter_count_row(n::Int, m::Int, tau::Int)
    reduced = m + m * m * tau
    legacy_beta = n * m
    return DataFrame(
        observations=[n], members=[m], tau=[tau],
        reduced_parameters=[reduced],
        legacy_time_varying_beta_variables=[legacy_beta],
        legacy_beta_link_equalities=[legacy_beta],
        reduced_dense_gram_bytes=[8.0 * reduced * reduced],
        legacy_beta_storage_bytes=[8.0 * legacy_beta],
    )
end

function _runtime_plot(paths::RunPaths, summary::DataFrame, kind::String;
                       xlabel::String, logx::Bool=false, logy::Bool=true)
    subset = summary[summary.kind .== kind, :]
    nrow(subset) == 0 && return nothing
    order = sortperm(Float64.(subset.numeric_setting))
    x = Float64.(subset.numeric_setting[order])
    y = Float64.(subset.median_seconds[order])
    svg_line_chart(joinpath(paths.figures, "runtime_$(kind).svg"),
        Dict(kind => (x, y));
        title="Runtime: $(replace(kind, '_' => ' ')) (median; compilation excluded; BLAS=1)",
        xlabel=xlabel, ylabel="seconds", logx=logx, logy=logy)
    return nothing
end

function _adaptive_runtime_kwargs(config::AbstractDict)
    settings = _adaptive_settings(config)
    return (
        direct_max_parameters=Int(settings["direct_max_parameters"]),
        max_dense_bytes=Int(settings["max_dense_bytes"]),
        cg_rtol=Float64(settings["cg_rtol"]),
        cg_atol=Float64(settings["cg_atol"]),
        cg_maxiter=Int(settings["cg_maxiter"]),
        dual_max_rows=Int(settings["dual_max_rows"]),
    )
end

function _benchmark_validation_search(bundle::DatasetBundle, config::AbstractDict)
    grids = _merge_grids(config)
    return tune_adaptive_quadratic(bundle, grids, config, _selection_metric(config))
end

function run_runtime_publication(config::AbstractDict, paths::RunPaths; resume::Bool=true)
    if resume && stage_complete(paths, "runtime_full")
        return DataFrame(CSV.File(joinpath(paths.tables, "runtime_benchmarks.csv")))
    end
    section = _cfg_section(config, "runtime")
    run_gurobi_verification = Bool(_cfg_get(section, "run_gurobi_verification", true))
    run_gurobi_verification && _preflight_gurobi(config, "runtime literal JuMP/Gurobi equivalence")
    reps = Int(_cfg_get(section, "reps", 11))
    warmup = Int(_cfg_get(section, "warmup", 3))
    adaptive_kwargs = _adaptive_runtime_kwargs(config)
    summary = DataFrame()
    raw = DataFrame()
    parameter_counts = DataFrame()
    equivalence = DataFrame()
    update_progress(paths, "runtime_full"; completed=0, total=8,
        message="starting/resuming controlled runtime benchmark")
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        # Scaling with the number of training observations at the manuscript m=10, tau=5 setting.
        for ntrain in Int.(_cfg_get(section, "sample_sizes", [100, 200, 400, 750, 1500, 3000]))
            n = max(4000, ntrain + 1000)
            val_end = min(ntrain + 500, n - 1)
            bundle = generate_synthetic_regime("recurring", 101;
                n=n, m=10, train_end=ntrain, val_end=val_end)
            train = collect(bundle.split.train)
            benchmark = _benchmark_call(
                () -> fit_adaptive_quadratic(bundle.X, bundle.y, train, 5, 1e-2;
                    adaptive_kwargs...);
                reps=reps, warmup=warmup)
            d = 10 + 10 * 10 * 5
            _append_runtime!(summary, raw, "adaptive_fit_samples", string(ntrain), ntrain,
                benchmark; observations=ntrain, parameters=d,
                notes="complete feature construction plus fit")
            append!(parameter_counts, _parameter_count_row(ntrain, 10, 5); cols=:union)
        end
        update_progress(paths, "runtime_full"; completed=1, total=8,
            message="sample-size scaling complete")

        # Window-size scaling at N=3000, m=10, exactly matching the paper's runtime table.
        window_bundle = generate_synthetic_regime("recurring", 102;
            n=4000, m=10, train_end=3000, val_end=3500)
        window_train = collect(window_bundle.split.train)
        for tau in Int.(_cfg_get(section, "window_sizes", [2, 3, 5, 10, 15, 25]))
            benchmark = _benchmark_call(
                () -> fit_adaptive_quadratic(window_bundle.X, window_bundle.y,
                    window_train, tau, 1e-2; adaptive_kwargs...);
                reps=reps, warmup=warmup)
            d = 10 + 10 * 10 * tau
            solver = d <= Int(adaptive_kwargs.direct_max_parameters) ?
                "direct_cholesky_or_qr" : "dual_eigen_or_matrix_free_pcg"
            _append_runtime!(summary, raw, "adaptive_fit_window", string(tau), tau,
                benchmark; observations=3000, parameters=d, solver=solver,
                notes="complete feature construction plus fit")
            append!(parameter_counts, _parameter_count_row(3000, 10, tau); cols=:union)
        end
        update_progress(paths, "runtime_full"; completed=2, total=8,
            message="window-size scaling complete")

        # Member-count scaling at N=3000, tau=5.
        for m in Int.(_cfg_get(section, "member_sizes", [3, 6, 10, 15, 25, 50]))
            bundle = generate_synthetic_regime("recurring", 103;
                n=4000, m=m, train_end=3000, val_end=3500)
            train = collect(bundle.split.train)
            benchmark = _benchmark_call(
                () -> fit_adaptive_quadratic(bundle.X, bundle.y, train, 5, 1e-2;
                    adaptive_kwargs...);
                reps=reps, warmup=warmup)
            d = m + m * m * 5
            solver = d <= Int(adaptive_kwargs.direct_max_parameters) ?
                "direct_cholesky_or_qr" : "dual_eigen_or_matrix_free_pcg"
            _append_runtime!(summary, raw, "adaptive_fit_members", string(m), m,
                benchmark; observations=3000, parameters=d, solver=solver,
                notes="complete feature construction plus fit")
            append!(parameter_counts, _parameter_count_row(3000, m, 5); cols=:union)
        end
        update_progress(paths, "runtime_full"; completed=3, total=8,
            message="ensemble-member scaling complete")

        # Regularization path: independent recomputation versus one reusable causal workspace.
        lambdas = Float64.(_cfg_get(section, "regularization_path",
            [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]))
        path_bundle = generate_synthetic_regime("recurring", 104)
        path_train = collect(path_bundle.split.train)
        independent = _benchmark_call(
            () -> [fit_adaptive_quadratic(path_bundle.X, path_bundle.y, path_train, 5, lambda;
                    adaptive_kwargs...) for lambda in lambdas];
            reps=reps, warmup=warmup)
        shared = _benchmark_call(
            () -> begin
                workspace = prepare_adaptive_quadratic_workspace(
                    path_bundle.X, path_bundle.y, path_train, 5;
                    direct_max_parameters=Int(adaptive_kwargs.direct_max_parameters),
                    max_dense_bytes=Int(adaptive_kwargs.max_dense_bytes))
                fit_adaptive_quadratic_path(workspace, lambdas;
                    cg_rtol=Float64(adaptive_kwargs.cg_rtol),
                    cg_atol=Float64(adaptive_kwargs.cg_atol),
                    cg_maxiter=Int(adaptive_kwargs.cg_maxiter))
            end;
            reps=reps, warmup=warmup)
        _append_runtime!(summary, raw, "regularization_path", "independent", 1,
            independent; observations=length(path_train), parameters=510,
            notes="reconstructs histories/design for every lambda")
        _append_runtime!(summary, raw, "regularization_path", "shared_workspace", 2,
            shared; observations=length(path_train), parameters=510,
            notes="one feature/design preparation, warm-started path")
        update_progress(paths, "runtime_full"; completed=4, total=8,
            message="regularization-path benchmark complete")

        # Full validation search cost, which is the reviewer-relevant online deployment cost.
        search_bundle = generate_synthetic_regime("recurring", 106;
            n=4000, m=10, train_end=2000, val_end=3000)
        validation_search = _benchmark_call(
            () -> _benchmark_validation_search(search_bundle, config);
            reps=max(3, min(reps, 5)), warmup=1)
        search_grids = _merge_grids(config)
        grid_count = length(search_grids["lambda"]) * length(search_grids["tau"])
        max_search_parameters = maximum(10 + 10 * 10 * Int(tau) for tau in search_grids["tau"])
        _append_runtime!(summary, raw, "validation_search", "adaptive_lambda_by_tau", grid_count,
            validation_search; observations=3000, parameters=max_search_parameters,
            notes="chronological selection plus train+validation refit; $(grid_count) candidates; parameter count is the largest tau candidate")
        update_progress(paths, "runtime_full"; completed=5, total=8,
            message="full validation-search benchmark complete")

        # Fit plus 1,000 sequential forecasts for all lightweight baselines on one machine.
        method_bundle = generate_synthetic_regime("recurring", 107)
        train = collect(method_bundle.split.train)
        test = collect(method_bundle.split.test)
        method_calls = [
            ("adaptive_ridge", () -> begin
                model = fit_adaptive_quadratic(method_bundle.X, method_bundle.y, train, 5, 1e-2;
                    adaptive_kwargs...)
                predict_adaptive(model, method_bundle.X, method_bundle.y, test)
            end),
            ("static_ridge", () -> begin
                model = fit_static_ridge(method_bundle.X, method_bundle.y, train, 1e-2)
                predict_static(model, method_bundle.X, test)
            end),
            ("rolling_ridge", () -> rolling_ridge_predictions(
                method_bundle.X, method_bundle.y, train, test; window=168, lambda=1e-2)),
            ("rls", () -> rls_predictions(
                method_bundle.X, method_bundle.y, train, test; forgetting=0.99, ridge=1e-3)),
            ("hedge", () -> hedge_predictions(
                method_bundle.X, method_bundle.y, train, test; eta=0.25, window=168)),
            ("exp3", () -> exp3_predictions(
                method_bundle.X, method_bundle.y, train, test; gamma=0.10, seed=2026)),
            ("fixed_share", () -> fixed_share_predictions(
                method_bundle.X, method_bundle.y, train, test; eta=0.25, share=0.02)),
            ("passive_aggressive", () -> passive_aggressive_predictions(
                method_bundle.X, method_bundle.y, train, test; C=0.1, epsilon=0.01)),
            ("dma", () -> dma_predictions(
                method_bundle.X, method_bundle.y, train, test; forgetting=0.99,
                variance_forgetting=0.98)),
        ]
        for (position, (name, call)) in enumerate(method_calls)
            benchmark = _benchmark_call(call; reps=reps, warmup=warmup)
            _append_runtime!(summary, raw, "method_fit_plus_roll", name, position,
                benchmark; observations=length(test), notes="fit plus $(length(test)) held-out forecasts")
        end
        update_progress(paths, "runtime_full"; completed=6, total=8,
            message="same-machine method comparison complete")

        # Deployment-only batch prediction and per-observation latency after a model is fitted.
        fitted = fit_adaptive_quadratic(method_bundle.X, method_bundle.y, train, 5, 1e-2;
            adaptive_kwargs...)
        prediction_benchmark = _benchmark_call(
            () -> predict_adaptive(fitted, method_bundle.X, method_bundle.y, test);
            reps=reps, warmup=warmup)
        _append_runtime!(summary, raw, "adaptive_prediction", "1000_forecasts", length(test),
            prediction_benchmark; observations=length(test), parameters=fitted.n_parameters,
            solver=fitted.linear_solver, notes="fixed fitted model; causal history construction included")
        per_forecast = prediction_benchmark.median_seconds / length(test)
        atomic_csv_write(joinpath(paths.tables, "runtime_prediction_latency.csv"), DataFrame(
            forecasts=[length(test)], median_batch_seconds=[prediction_benchmark.median_seconds],
            median_seconds_per_forecast=[per_forecast],
            median_microseconds_per_forecast=[1e6 * per_forecast],
        ))

        # Algebraic and numerical equivalence without requiring a commercial solver license.
        direct = fit_adaptive_quadratic(method_bundle.X, method_bundle.y, train, 5, 1e-2;
            direct_max_parameters=typemax(Int), max_dense_bytes=typemax(Int),
            cg_rtol=1e-11, cg_atol=1e-13)
        matrix_free = fit_adaptive_quadratic(method_bundle.X, method_bundle.y, train, 5, 1e-2;
            direct_max_parameters=0, max_dense_bytes=0,
            cg_rtol=1e-11, cg_atol=1e-13, cg_maxiter=5000,
            allow_dual=false)
        Atest, _, _ = causal_design(method_bundle.X, method_bundle.y, 5, test)
        affine_prediction = predict_adaptive(direct, method_bundle.X, method_bundle.y, test)
        design_prediction = Atest * direct.theta
        pcg_prediction = predict_adaptive(matrix_free, method_bundle.X, method_bundle.y, test)
        append!(equivalence, DataFrame(
            check=["reduced_design_vs_affine_rule"],
            max_abs_difference=[maximum(abs.(design_prediction .- affine_prediction))],
            relative_l2_difference=[norm(design_prediction - affine_prediction) / max(norm(affine_prediction), eps())],
            relative_objective_difference=[0.0],
        ); cols=:union)
        append!(equivalence, DataFrame(
            check=["direct_vs_matrix_free_pcg"],
            max_abs_difference=[maximum(abs.(pcg_prediction .- affine_prediction))],
            relative_l2_difference=[norm(pcg_prediction - affine_prediction) / max(norm(affine_prediction), eps())],
            relative_objective_difference=[abs(matrix_free.objective - direct.objective) / max(abs(direct.objective), eps())],
        ); cols=:union)
        atomic_csv_write(joinpath(paths.tables, "runtime_numerical_equivalence.csv"), equivalence)
        update_progress(paths, "runtime_full"; completed=7, total=8,
            message="prediction latency and numerical-equivalence checks complete")

        if run_gurobi_verification
            small = generate_synthetic_regime("stationary", 108;
                n=300, m=4, train_end=180, val_end=240)
            result = verify_reduced_against_jump(
                small.X, small.y, collect(small.split.train), 3, 1e-2)
            atomic_csv_write(joinpath(paths.tables, "runtime_gurobi_equivalence.csv"),
                DataFrame(Dict(Symbol(key) => [value] for (key, value) in result)))
        end
    finally
        BLAS.set_num_threads(old_blas)
    end

    atomic_csv_write(joinpath(paths.tables, "runtime_benchmarks.csv"), summary)
    atomic_csv_write(joinpath(paths.tables, "runtime_benchmark_raw.csv"), raw)
    atomic_csv_write(joinpath(paths.tables, "runtime_parameter_counts.csv"), parameter_counts)
    _runtime_plot(paths, summary, "adaptive_fit_samples"; xlabel="training observations", logx=true)
    _runtime_plot(paths, summary, "adaptive_fit_window"; xlabel="window size tau")
    _runtime_plot(paths, summary, "adaptive_fit_members"; xlabel="ensemble members", logx=true)
    _runtime_plot(paths, summary, "regularization_path"; xlabel="1=independent, 2=shared workspace")
    _runtime_plot(paths, summary, "method_fit_plus_roll"; xlabel="method index", logy=true)
    method_subset = summary[summary.kind .== "method_fit_plus_roll", :]
    if nrow(method_subset) > 0
        method_order = sortperm(Float64.(method_subset.numeric_setting))
        svg_bar_chart(joinpath(paths.figures, "runtime_method_fit_plus_roll.svg"),
            string.(method_subset.setting[method_order]),
            Float64.(method_subset.median_seconds[method_order]);
            title="Same-machine fit plus held-out forecasting runtime",
            ylabel="median seconds (compilation excluded; BLAS=1)", lower_better=true)
    end
    output_paths = [
        joinpath(paths.tables, "runtime_benchmarks.csv"),
        joinpath(paths.tables, "runtime_benchmark_raw.csv"),
        joinpath(paths.tables, "runtime_parameter_counts.csv"),
        joinpath(paths.tables, "runtime_prediction_latency.csv"),
        joinpath(paths.tables, "runtime_numerical_equivalence.csv"),
        joinpath(paths.figures, "runtime_adaptive_fit_samples.svg"),
        joinpath(paths.figures, "runtime_adaptive_fit_window.svg"),
        joinpath(paths.figures, "runtime_adaptive_fit_members.svg"),
        joinpath(paths.figures, "runtime_regularization_path.svg"),
        joinpath(paths.figures, "runtime_method_fit_plus_roll.svg"),
    ]
    run_gurobi_verification && push!(output_paths,
        joinpath(paths.tables, "runtime_gurobi_equivalence.csv"))
    update_progress(paths, "runtime_full"; completed=8, total=8,
        message="runtime benchmark and figures complete")
    mark_stage_complete(paths, "runtime_full"; extras=Dict(
        "reps" => reps, "warmup" => warmup, "blas_threads" => 1,
        "compilation_excluded" => true,
        "gurobi_verification" => run_gurobi_verification), outputs=output_paths)
    return summary
end
