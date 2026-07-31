function _default_grids()
    return Dict{String,Any}(
        "lambda" => [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0],
        "tau" => [1, 2, 3, 5, 7, 10],
        "ridge_lambda" => [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "rolling_window" => [24, 72, 168, 336],
        "forgetting" => [0.90, 0.95, 0.98, 0.99, 0.995],
        "rls_ridge" => [1e-6, 1e-4, 1e-2, 1e-1],
        "hedge_eta" => [0.01, 0.05, 0.10, 0.25, 0.50, 1.0],
        "hedge_window" => [0, 24, 72, 168, 336],
        "exp3_gamma" => [0.01, 0.05, 0.10, 0.20, 0.40],
        "fixed_share" => [0.005, 0.01, 0.02, 0.05, 0.10],
        "pa_C" => [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "pa_epsilon" => [0.0, 0.01, 0.05, 0.10],
        "dma_forgetting" => [0.90, 0.95, 0.98, 0.99, 0.995],
        "dma_variance_forgetting" => [0.90, 0.95, 0.98, 0.995],
    )
end

function _merge_grids(config::AbstractDict)
    grids = _default_grids()
    if haskey(config, "grids")
        for (key, value) in config["grids"]
            grids[string(key)] = value
        end
    end
    return grids
end

function _selection_metric(config::AbstractDict)
    experiment = haskey(config, "experiment") ? config["experiment"] : Dict{String,Any}()
    name = lowercase(string(get(experiment, "selection_metric", "rmse")))
    name in ("rmse", "mae", "cvar_05") ||
        throw(ArgumentError("selection_metric must be rmse, mae, or cvar_05"))
    return Symbol(name)
end

function _validation_score(y::AbstractVector, prediction::AbstractVector, metric::Symbol)
    metric == :rmse && return rmse(y, prediction)
    metric == :mae && return mae(y, prediction)
    metric == :cvar_05 && return empirical_cvar(abs.(Float64.(y) .- Float64.(prediction)), 0.05)
    throw(ArgumentError("unsupported validation metric $(metric)"))
end

function _best_row(df::DataFrame, metric::Symbol)
    nrow(df) > 0 || throw(ArgumentError("empty tuning table"))
    column = Symbol("validation_", metric)
    column in propertynames(df) || throw(ArgumentError("tuning table lacks $(column)"))
    values = Float64.(df[!, column])
    all(isfinite, values) || throw(ArgumentError("non-finite validation scores"))
    # DataFrame preserves deterministic grid order, so argmin provides a deterministic tie break.
    return df[argmin(values), :]
end

function _append_tuning_row!(rows::DataFrame, values::AbstractDict)
    append!(rows, one_row_dataframe(values); cols=:union)
    return rows
end

function _adaptive_settings(config::AbstractDict)
    section = haskey(config, "adaptive") ? config["adaptive"] : Dict{String,Any}()
    return Dict{String,Any}(
        "primary_objective" => lowercase(string(get(section, "primary_objective", "quadratic_stacked"))),
        "direct_max_parameters" => Int(get(section, "direct_max_parameters", 3500)),
        "max_dense_bytes" => Int(get(section, "max_dense_bytes", 750_000_000)),
        "cg_rtol" => Float64(get(section, "cg_rtol", 1e-9)),
        "cg_atol" => Float64(get(section, "cg_atol", 1e-12)),
        "cg_maxiter" => Int(get(section, "cg_maxiter", 0)),
        "dual_max_rows" => Int(get(section, "dual_max_rows", 3500)),
        "irls_smoothing" => Float64(get(section, "irls_smoothing", 1e-8)),
        "irls_rtol" => Float64(get(section, "irls_rtol", 1e-8)),
        "irls_objective_rtol" => Float64(
            get(section, "irls_objective_rtol", 1e-8)),
        "irls_stationarity_rtol" => Float64(
            get(section, "irls_stationarity_rtol", 1e-8)),
        "irls_practical_stationarity_rtol" => Float64(
            get(section, "irls_practical_stationarity_rtol", 1e-7)),
        "irls_maxiter" => Int(get(section, "irls_maxiter", 1000)),
        "irls_min_iterations" => Int(
            get(section, "irls_min_iterations", 2)),
        # The licensed publication profile uses the exact SOCP only when IRLS
        # reaches its iteration limit without satisfying a declared certificate.
        "irls_fallback_solver" => lowercase(string(
            get(section, "irls_fallback_solver", "none"))),
        "gurobi_threads" => Int(get(section, "gurobi_threads", 1)),
        "gurobi_output_flag" => Int(get(section, "gurobi_output_flag", 0)),
        "gurobi_time_limit" => Float64(get(section, "gurobi_time_limit", Inf)),
        "gurobi_optimality_tolerance" => Float64(get(section, "gurobi_optimality_tolerance", 1e-8)),
    )
end

function _method_objective(method::String, settings::AbstractDict)
    method == "adaptive_ridge_quadratic" && return :quadratic
    method == "adaptive_robust_norm" && return :norm
    method == "adaptive_ridge" || throw(ArgumentError("not an adaptive method: $(method)"))
    primary = string(settings["primary_objective"])
    primary in ("quadratic", "quadratic_stacked", "squared") && return :quadratic
    primary in ("norm", "robust_norm", "norm_stacked") && return :norm
    throw(ArgumentError("unknown adaptive.primary_objective=$(primary)"))
end

function _combined_indices(bundle::DatasetBundle)
    return collect(first(bundle.split.train):last(bundle.split.validation))
end

function tune_adaptive_quadratic(bundle::DatasetBundle, grids::AbstractDict,
                                 config::AbstractDict, selection_metric::Symbol)
    rows = DataFrame()
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    settings = _adaptive_settings(config)
    lambdas = Float64.(grids["lambda"])
    for tau in Int.(grids["tau"])
        workspace = prepare_adaptive_quadratic_workspace(
            bundle.X, bundle.y, train, tau;
            availability_lag=bundle.availability_lag,
            event_ids=bundle.event_ids,
            direct_max_parameters=Int(settings["direct_max_parameters"]),
            max_dense_bytes=Int(settings["max_dense_bytes"]),
        )
        models = fit_adaptive_quadratic_path(
            workspace, lambdas;
            penalty=:stacked_beta,
            cg_rtol=Float64(settings["cg_rtol"]),
            cg_atol=Float64(settings["cg_atol"]),
            cg_maxiter=Int(settings["cg_maxiter"]),
            dual_max_rows=Int(settings["dual_max_rows"]),
        )
        for (lambda, model) in zip(lambdas, models)
            prediction = predict_adaptive(model, bundle.X, bundle.y, validation;
                event_ids=bundle.event_ids)
            values = Dict{Symbol,Any}(
                :objective_variant => "quadratic_stacked_beta",
                :tau => tau,
                :lambda => lambda,
                :validation_rmse => rmse(bundle.y[validation], prediction),
                :validation_mae => mae(bundle.y[validation], prediction),
                :validation_cvar_05 => empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05),
                :preparation_seconds => model.preparation_seconds,
                :solve_seconds => model.solve_seconds,
                :fit_seconds => model.fit_seconds,
                :linear_solver => model.linear_solver,
                :condition_number => model.condition_number,
                :iterations => model.iterations,
                :residual_norm => model.residual_norm,
                :n_parameters => model.n_parameters,
            )
            _append_tuning_row!(rows, values)
        end
    end
    best = _best_row(rows, selection_metric)
    combined = _combined_indices(bundle)
    model = fit_adaptive_quadratic(
        bundle.X, bundle.y, combined, Int(best.tau), Float64(best.lambda);
        availability_lag=bundle.availability_lag,
        event_ids=bundle.event_ids,
        penalty=:stacked_beta,
        direct_max_parameters=Int(settings["direct_max_parameters"]),
        max_dense_bytes=Int(settings["max_dense_bytes"]),
        cg_rtol=Float64(settings["cg_rtol"]),
        cg_atol=Float64(settings["cg_atol"]),
        cg_maxiter=Int(settings["cg_maxiter"]),
        dual_max_rows=Int(settings["dual_max_rows"]),
    )
    return model, rows, best
end

function tune_adaptive_norm(bundle::DatasetBundle, grids::AbstractDict,
                            config::AbstractDict, selection_metric::Symbol)
    rows = DataFrame()
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    settings = _adaptive_settings(config)
    lambdas = Float64.(grids["lambda"])
    for tau in Int.(grids["tau"])
        workspace = prepare_adaptive_quadratic_workspace(
            bundle.X, bundle.y, train, tau;
            availability_lag=bundle.availability_lag,
            event_ids=bundle.event_ids,
            direct_max_parameters=Int(settings["direct_max_parameters"]),
            max_dense_bytes=Int(settings["max_dense_bytes"]),
        )
        models = fit_adaptive_robust_norm_path(workspace, lambdas;
            smoothing=Float64(settings["irls_smoothing"]),
            irls_rtol=Float64(settings["irls_rtol"]),
            irls_objective_rtol=Float64(
                settings["irls_objective_rtol"]),
            irls_stationarity_rtol=Float64(
                settings["irls_stationarity_rtol"]),
            irls_practical_stationarity_rtol=Float64(
                settings["irls_practical_stationarity_rtol"]),
            irls_maxiter=Int(settings["irls_maxiter"]),
            irls_min_iterations=Int(settings["irls_min_iterations"]),
            cg_rtol=Float64(settings["cg_rtol"]),
            cg_atol=Float64(settings["cg_atol"]),
            cg_maxiter=Int(settings["cg_maxiter"]),
            dual_max_rows=Int(settings["dual_max_rows"]),
            fallback_solver=Symbol(settings["irls_fallback_solver"]),
            gurobi_output_flag=Int(settings["gurobi_output_flag"]),
            gurobi_time_limit=Float64(settings["gurobi_time_limit"]),
            gurobi_threads=Int(settings["gurobi_threads"]),
            gurobi_optimality_tolerance=Float64(
                settings["gurobi_optimality_tolerance"]),
            require_convergence=true,
        )
        for (lambda, model) in zip(lambdas, models)
            prediction = predict_adaptive(model, bundle.X, bundle.y, validation;
                event_ids=bundle.event_ids)
            _append_tuning_row!(rows, Dict{Symbol,Any}(
                :objective_variant => model.objective_variant == :norm ?
                    "norm_plus_norm_exact" : "norm_plus_norm_irls",
                :tau => tau,
                :lambda => lambda,
                :validation_rmse => rmse(bundle.y[validation], prediction),
                :validation_mae => mae(bundle.y[validation], prediction),
                :validation_cvar_05 => empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05),
                :preparation_seconds => model.preparation_seconds,
                :solve_seconds => model.solve_seconds,
                :fit_seconds => model.fit_seconds,
                :linear_solver => model.linear_solver,
                :iterations => model.iterations,
                :converged => model.converged,
                :relative_change => model.residual_norm,
                :relative_objective_change =>
                    model.relative_objective_change,
                :stationarity_residual =>
                    model.stationarity_residual,
                :convergence_reason =>
                    model.convergence_reason,
                :exact_fallback_used =>
                    occursin("gurobi_socp_fallback",
                             model.convergence_reason),
                :n_parameters => model.n_parameters,
            ))
        end
    end
    best = _best_row(rows, selection_metric)
    combined = _combined_indices(bundle)
    model = fit_adaptive_robust_norm(
        bundle.X, bundle.y, combined, Int(best.tau), Float64(best.lambda);
        availability_lag=bundle.availability_lag,
        event_ids=bundle.event_ids,
        direct_max_parameters=Int(settings["direct_max_parameters"]),
        max_dense_bytes=Int(settings["max_dense_bytes"]),
        smoothing=Float64(settings["irls_smoothing"]),
        irls_rtol=Float64(settings["irls_rtol"]),
        irls_objective_rtol=Float64(
            settings["irls_objective_rtol"]),
        irls_stationarity_rtol=Float64(
            settings["irls_stationarity_rtol"]),
        irls_practical_stationarity_rtol=Float64(
            settings["irls_practical_stationarity_rtol"]),
        irls_maxiter=Int(settings["irls_maxiter"]),
        irls_min_iterations=Int(settings["irls_min_iterations"]),
        cg_rtol=Float64(settings["cg_rtol"]),
        cg_atol=Float64(settings["cg_atol"]),
        cg_maxiter=Int(settings["cg_maxiter"]),
        dual_max_rows=Int(settings["dual_max_rows"]),
        fallback_solver=Symbol(settings["irls_fallback_solver"]),
        gurobi_output_flag=Int(settings["gurobi_output_flag"]),
        gurobi_time_limit=Float64(settings["gurobi_time_limit"]),
        gurobi_threads=Int(settings["gurobi_threads"]),
        gurobi_optimality_tolerance=Float64(
            settings["gurobi_optimality_tolerance"]),
        require_convergence=true,
    )
    return model, rows, best
end

function tune_adaptive(bundle::DatasetBundle, grids::AbstractDict,
                       config::AbstractDict, method::String,
                       selection_metric::Symbol)
    settings = _adaptive_settings(config)
    objective = _method_objective(method, settings)
    objective == :quadratic && return tune_adaptive_quadratic(bundle, grids, config, selection_metric)
    return tune_adaptive_norm(bundle, grids, config, selection_metric)
end

function tune_static_ridge(bundle::DatasetBundle, grids::AbstractDict,
                           selection_metric::Symbol)
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    rows = DataFrame()
    for lambda in Float64.(grids["ridge_lambda"])
        model = fit_static_ridge(bundle.X, bundle.y, train, lambda)
        prediction = predict_static(model, bundle.X, validation)
        _append_tuning_row!(rows, Dict{Symbol,Any}(
            :lambda => lambda,
            :validation_rmse => rmse(bundle.y[validation], prediction),
            :validation_mae => mae(bundle.y[validation], prediction),
            :validation_cvar_05 => empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05),
            :fit_seconds => model.fit_seconds,
        ))
    end
    best = _best_row(rows, selection_metric)
    model = fit_static_ridge(bundle.X, bundle.y, _combined_indices(bundle), Float64(best.lambda))
    return model, rows, best
end

function _event_reset_policy(bundle::DatasetBundle, config::AbstractDict)
    section = haskey(config, "boundary") ? config["boundary"] : Dict{String,Any}()
    policy = lowercase(string(get(section, "online_event_policy", "reset")))
    policy in ("reset", "carry") || throw(ArgumentError("online_event_policy must be reset or carry"))
    return bundle.event_ids !== nothing && policy == "reset"
end

function _tune_online(bundle::DatasetBundle, method::String,
                      grids::AbstractDict, config::AbstractDict,
                      selection_metric::Symbol; seed::Int=2026)
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    rows = DataFrame()
    reset = _event_reset_policy(bundle, config)

    function evaluate_candidate(call::Function, parameters::AbstractDict{Symbol,V}) where {V}
        started = time_ns()
        prediction, _ = call()
        elapsed = (time_ns() - started) / 1e9

        # Widen the internal representation so that mixed numeric and future
        # metadata types can be added safely.
        values = Dict{Symbol,Any}(parameters)
        values[:validation_rmse] = rmse(bundle.y[validation], prediction)
        values[:validation_mae] = mae(bundle.y[validation], prediction)
        values[:validation_cvar_05] = empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05)
        values[:fit_and_validation_seconds] = elapsed
        _append_tuning_row!(rows, values)
    end

    if method == "rolling_ridge"
        for window in Int.(grids["rolling_window"]), lambda in Float64.(grids["ridge_lambda"])
            evaluate_candidate(Dict(:window => window, :lambda => lambda)) do
                rolling_ridge_predictions(bundle.X, bundle.y, train, validation;
                    window=window, lambda=lambda,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids)
            end
        end
    elseif method == "rls"
        for forgetting in Float64.(grids["forgetting"]), ridge in Float64.(grids["rls_ridge"])
            evaluate_candidate(Dict(:forgetting => forgetting, :ridge => ridge)) do
                rls_predictions(bundle.X, bundle.y, train, validation;
                    forgetting=forgetting, ridge=ridge,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    elseif method == "hedge"
        for eta in Float64.(grids["hedge_eta"]), window in Int.(grids["hedge_window"])
            evaluate_candidate(Dict(:eta => eta, :window => window)) do
                hedge_predictions(bundle.X, bundle.y, train, validation;
                    eta=eta, window=window,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    elseif method == "exp3"
        for gamma in Float64.(grids["exp3_gamma"])
            evaluate_candidate(Dict(:gamma => gamma, :seed => seed)) do
                exp3_predictions(bundle.X, bundle.y, train, validation;
                    gamma=gamma, seed=seed,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    elseif method == "fixed_share"
        for eta in Float64.(grids["hedge_eta"]), share in Float64.(grids["fixed_share"])
            evaluate_candidate(Dict(:eta => eta, :share => share)) do
                fixed_share_predictions(bundle.X, bundle.y, train, validation;
                    eta=eta, share=share,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    elseif method == "passive_aggressive"
        for C in Float64.(grids["pa_C"]), epsilon in Float64.(grids["pa_epsilon"])
            evaluate_candidate(Dict(:C => C, :epsilon => epsilon)) do
                passive_aggressive_predictions(bundle.X, bundle.y, train, validation;
                    C=C, epsilon=epsilon,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    elseif method == "dma"
        for forgetting in Float64.(grids["dma_forgetting"]),
            variance_forgetting in Float64.(grids["dma_variance_forgetting"])
            evaluate_candidate(Dict(
                :forgetting => forgetting,
                :variance_forgetting => variance_forgetting,
            )) do
                dma_predictions(bundle.X, bundle.y, train, validation;
                    forgetting=forgetting,
                    variance_forgetting=variance_forgetting,
                    availability_lag=bundle.availability_lag,
                    event_ids=bundle.event_ids, reset_on_event=reset)
            end
        end
    else
        throw(ArgumentError("unsupported online method $(method)"))
    end
    return rows, _best_row(rows, selection_metric)
end

function _predict_online_selected(bundle::DatasetBundle, method::String,
                                  best, config::AbstractDict; seed::Int=2026)
    initialization = _combined_indices(bundle)
    test = collect(bundle.split.test)
    reset = _event_reset_policy(bundle, config)
    common = (
        availability_lag=bundle.availability_lag,
        event_ids=bundle.event_ids,
    )
    if method == "rolling_ridge"
        return rolling_ridge_predictions(bundle.X, bundle.y, initialization, test;
            window=Int(best.window), lambda=Float64(best.lambda), common...)
    elseif method == "rls"
        return rls_predictions(bundle.X, bundle.y, initialization, test;
            forgetting=Float64(best.forgetting), ridge=Float64(best.ridge),
            reset_on_event=reset, common...)
    elseif method == "hedge"
        return hedge_predictions(bundle.X, bundle.y, initialization, test;
            eta=Float64(best.eta), window=Int(best.window),
            reset_on_event=reset, common...)
    elseif method == "exp3"
        return exp3_predictions(bundle.X, bundle.y, initialization, test;
            gamma=Float64(best.gamma), seed=seed,
            reset_on_event=reset, common...)
    elseif method == "fixed_share"
        return fixed_share_predictions(bundle.X, bundle.y, initialization, test;
            eta=Float64(best.eta), share=Float64(best.share),
            reset_on_event=reset, common...)
    elseif method == "passive_aggressive"
        return passive_aggressive_predictions(bundle.X, bundle.y, initialization, test;
            C=Float64(best.C), epsilon=Float64(best.epsilon),
            reset_on_event=reset, common...)
    elseif method == "dma"
        return dma_predictions(bundle.X, bundle.y, initialization, test;
            forgetting=Float64(best.forgetting),
            variance_forgetting=Float64(best.variance_forgetting),
            reset_on_event=reset, common...)
    end
    throw(ArgumentError("unsupported online method $(method)"))
end

function _meta_settings(config::AbstractDict)
    section = haskey(config, "meta") ? config["meta"] : Dict{String,Any}()
    return Dict{String,Any}(
        "neural_tau" => Int.(get(section, "neural_tau", [1, 3, 5])),
        "neural_hidden" => Int.(get(section, "neural_hidden", [8, 16])),
        "neural_learning_rate" => Float64.(get(section, "neural_learning_rate", [1e-3, 3e-3])),
        "neural_weight_decay" => Float64.(get(section, "neural_weight_decay", [0.0, 1e-4])),
        "neural_epochs" => Int(get(section, "neural_epochs", 300)),
        "neural_batch_size" => Int(get(section, "neural_batch_size", 128)),
        "boosted_tau" => Int.(get(section, "boosted_tau", [1, 3, 5])),
        "boosted_trees" => Int.(get(section, "boosted_trees", [100, 200])),
        "boosted_depth" => Int.(get(section, "boosted_depth", [1, 2])),
        "boosted_min_leaf" => Int.(get(section, "boosted_min_leaf", [20, 50])),
        "boosted_learning_rate" => Float64.(get(section, "boosted_learning_rate", [0.03, 0.08])),
    )
end

function tune_neural_gate(bundle::DatasetBundle, config::AbstractDict,
                          selection_metric::Symbol; seed::Int=2026)
    settings = _meta_settings(config)
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    rows = DataFrame()
    for tau in settings["neural_tau"], hidden in settings["neural_hidden"],
        learning_rate in settings["neural_learning_rate"],
        weight_decay in settings["neural_weight_decay"]
        model = fit_neural_gate(
            bundle.X, bundle.y, train, tau;
            availability_lag=bundle.availability_lag,
            event_ids=bundle.event_ids,
            hidden=hidden,
            epochs=Int(settings["neural_epochs"]),
            batch_size=Int(settings["neural_batch_size"]),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
        )
        prediction, _ = predict_neural_gate(model, bundle.X, bundle.y, validation;
            event_ids=bundle.event_ids)
        _append_tuning_row!(rows, Dict{Symbol,Any}(
            :tau => tau,
            :hidden => hidden,
            :learning_rate => learning_rate,
            :weight_decay => weight_decay,
            :epochs => Int(settings["neural_epochs"]),
            :validation_rmse => rmse(bundle.y[validation], prediction),
            :validation_mae => mae(bundle.y[validation], prediction),
            :validation_cvar_05 => empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05),
            :fit_seconds => model.fit_seconds,
        ))
    end
    best = _best_row(rows, selection_metric)
    model = fit_neural_gate(
        bundle.X, bundle.y, _combined_indices(bundle), Int(best.tau);
        availability_lag=bundle.availability_lag,
        event_ids=bundle.event_ids,
        hidden=Int(best.hidden),
        epochs=Int(best.epochs),
        batch_size=Int(settings["neural_batch_size"]),
        learning_rate=Float64(best.learning_rate),
        weight_decay=Float64(best.weight_decay),
        seed=seed,
    )
    return model, rows, best
end

function tune_boosted_meta(bundle::DatasetBundle, config::AbstractDict,
                           selection_metric::Symbol)
    settings = _meta_settings(config)
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    rows = DataFrame()
    for tau in settings["boosted_tau"], n_trees in settings["boosted_trees"],
        max_depth in settings["boosted_depth"], min_leaf in settings["boosted_min_leaf"],
        learning_rate in settings["boosted_learning_rate"]
        model = fit_boosted_meta(
            bundle.X, bundle.y, train, tau;
            availability_lag=bundle.availability_lag,
            event_ids=bundle.event_ids,
            n_trees=n_trees,
            max_depth=max_depth,
            min_leaf=min_leaf,
            learning_rate=learning_rate,
        )
        prediction = predict_boosted_meta(model, bundle.X, bundle.y, validation;
            event_ids=bundle.event_ids)
        _append_tuning_row!(rows, Dict{Symbol,Any}(
            :tau => tau,
            :n_trees => n_trees,
            :max_depth => max_depth,
            :min_leaf => min_leaf,
            :learning_rate => learning_rate,
            :validation_rmse => rmse(bundle.y[validation], prediction),
            :validation_mae => mae(bundle.y[validation], prediction),
            :validation_cvar_05 => empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05),
            :fit_seconds => model.fit_seconds,
        ))
    end
    best = _best_row(rows, selection_metric)
    model = fit_boosted_meta(
        bundle.X, bundle.y, _combined_indices(bundle), Int(best.tau);
        availability_lag=bundle.availability_lag,
        event_ids=bundle.event_ids,
        n_trees=Int(best.n_trees),
        max_depth=Int(best.max_depth),
        min_leaf=Int(best.min_leaf),
        learning_rate=Float64(best.learning_rate),
    )
    return model, rows, best
end

function _selected_row(method::String, best, selection_metric::Symbol)
    values = Dict{Symbol,Any}(:method => method, :selection_metric => string(selection_metric))
    for name in propertynames(best)
        values[name] = best[name]
    end
    return one_row_dataframe(values)
end

"""
Chronologically tune every method on the validation segment, refit on train+validation,
and evaluate exactly once on the test segment. The returned artifact dictionary contains
serializable models and all time-varying weight matrices.
"""
function evaluate_bundle(bundle_raw::DatasetBundle;
                         methods::Vector{String}=copy(DEFAULT_METHODS),
                         config::AbstractDict=Dict{String,Any}(),
                         standardize::Bool=true,
                         seed::Int=2026)
    bundle, standardizer = standardize ? standardize_bundle(bundle_raw) :
        (bundle_raw, Standardizer(0.0, 1.0))
    grids = _merge_grids(config)
    selection_metric = _selection_metric(config)
    test = collect(bundle.split.test)
    y_test_original = bundle_raw.y[test]
    metrics = DataFrame()
    predictions = DataFrame(time_index=test, target=y_test_original)
    tuning = DataFrame()
    selected = DataFrame()
    artifacts = Dict{String,Any}()

    function record(method::String, prediction_standardized::AbstractVector;
                    extras::AbstractDict=Dict{String,Any}(),
                    weights::Union{Nothing,AbstractMatrix}=nothing,
                    model=nothing)
        prediction_original = inverse_transform(standardizer, prediction_standardized)
        predictions[!, Symbol(method)] = prediction_original
        append!(metrics, metrics_dataframe(
            bundle_raw.name, method, y_test_original, prediction_original; extras=extras);
            cols=:union)
        weights !== nothing && (artifacts[method * "_weights"] = Matrix{Float64}(weights))
        model !== nothing && (artifacts[method * "_model"] = model)
        return nothing
    end

    for method in methods
        if method in ("adaptive_ridge", "adaptive_ridge_quadratic", "adaptive_robust_norm")
            model, table, best = tune_adaptive(bundle, grids, config, method, selection_metric)
            table[!, :method] = fill(method, nrow(table))
            append!(tuning, table; cols=:union)
            append!(selected, _selected_row(method, best, selection_metric); cols=:union)
            prediction = predict_adaptive(model, bundle.X, bundle.y, test;
                event_ids=bundle.event_ids)
            weights, history_counts = adaptive_coefficients(model, bundle.X, bundle.y, test;
                event_ids=bundle.event_ids)
            artifacts[method * "_history_counts"] = history_counts
            record(method, prediction;
                extras=Dict{String,Any}(
                    "tau" => model.tau,
                    "lambda" => model.lambda,
                    "objective_variant" => string(model.objective_variant),
                    "fit_seconds" => model.fit_seconds,
                    "solve_seconds" => model.solve_seconds,
                    "linear_solver" => model.linear_solver,
                    "n_parameters" => model.n_parameters,
                ),
                weights=weights,
                model=model)
        elseif method == "static_ridge"
            model, table, best = tune_static_ridge(bundle, grids, selection_metric)
            table[!, :method] = fill(method, nrow(table))
            append!(tuning, table; cols=:union)
            append!(selected, _selected_row(method, best, selection_metric); cols=:union)
            record(method, predict_static(model, bundle.X, test);
                extras=Dict("lambda" => model.lambda, "fit_seconds" => model.fit_seconds),
                model=model)
        elseif method in ("rolling_ridge", "rls", "hedge", "exp3", "fixed_share", "passive_aggressive", "dma")
            table, best = _tune_online(bundle, method, grids, config, selection_metric; seed=seed)
            table[!, :method] = fill(method, nrow(table))
            append!(tuning, table; cols=:union)
            append!(selected, _selected_row(method, best, selection_metric); cols=:union)
            started = time_ns()
            prediction, weights = _predict_online_selected(bundle, method, best, config; seed=seed)
            elapsed = (time_ns() - started) / 1e9
            record(method, prediction;
                extras=Dict("fit_and_test_seconds" => elapsed),
                weights=weights)
        elseif method == "neural_gate"
            model, table, best = tune_neural_gate(bundle, config, selection_metric; seed=seed)
            table[!, :method] = fill(method, nrow(table))
            append!(tuning, table; cols=:union)
            append!(selected, _selected_row(method, best, selection_metric); cols=:union)
            prediction, weights = predict_neural_gate(model, bundle.X, bundle.y, test;
                event_ids=bundle.event_ids)
            record(method, prediction;
                extras=Dict("fit_seconds" => model.fit_seconds, "tau" => model.tau),
                weights=weights, model=model)
        elseif method == "boosted_meta"
            model, table, best = tune_boosted_meta(bundle, config, selection_metric)
            table[!, :method] = fill(method, nrow(table))
            append!(tuning, table; cols=:union)
            append!(selected, _selected_row(method, best, selection_metric); cols=:union)
            record(method, predict_boosted_meta(model, bundle.X, bundle.y, test;
                    event_ids=bundle.event_ids);
                extras=Dict("fit_seconds" => model.fit_seconds, "tau" => model.tau),
                model=model)
        elseif method == "ensemble_mean"
            record(method, ensemble_mean_predictions(bundle.X, test))
        elseif method == "oracle_best_member"
            prediction, member, score = oracle_best_member_predictions(
                bundle.X, bundle.y, test; metric=selection_metric)
            record(method, prediction;
                extras=Dict(
                    "diagnostic_only" => true,
                    "oracle_member_index" => member,
                    "oracle_member_name" => bundle.member_names[member],
                    "oracle_selection_metric" => string(selection_metric),
                    "oracle_standardized_selection_score" => score,
                ))
        elseif method == "validation_best_member"
            _, member, score = oracle_best_member_predictions(
                bundle.X, bundle.y, collect(bundle.split.validation);
                metric=selection_metric)
            record(method, Vector{Float64}(@view bundle.X[test, member]);
                extras=Dict(
                    "selected_member_index" => member,
                    "selected_member_name" => bundle.member_names[member],
                    "selection_metric" => string(selection_metric),
                    "validation_standardized_selection_score" => score,
                ))
        else
            throw(ArgumentError("unknown method $(method)"))
        end
    end

    if haskey(bundle.metadata, "benchmark_predictions")
        benchmarks = bundle.metadata["benchmark_predictions"]
        for (name, full_prediction) in benchmarks
            method = "benchmark_" * string(name)
            standardized_prediction = transform(standardizer, Float64.(full_prediction[test]))
            record(method, standardized_prediction; extras=Dict("external_operational_benchmark" => true))
        end
    end

    return metrics, predictions, tuning, selected, artifacts
end
