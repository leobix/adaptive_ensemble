function _string_vector(value, default)
    value === nothing && return copy(default)
    value isa AbstractString && return [string(value)]
    value isa AbstractVector || throw(ArgumentError("expected a string or vector of strings"))
    return string.(value)
end

function _cfg_section(config::AbstractDict, name::String)
    return haskey(config, name) ? config[name] : Dict{String,Any}()
end

function _cfg_get(section::AbstractDict, key::String, default)
    return haskey(section, key) ? section[key] : default
end

function _failure_log_path(paths::RunPaths, stage::AbstractString)
    return joinpath(paths.logs, _safe_marker_name(stage) * "_failures.csv")
end

"""
Run independent publication tasks with hash-validated, run-fingerprint-bound checkpoints.
A task is resumable only when every required output still exists and matches the hashes in
its marker. This prevents a lone or partially written CSV from being mistaken for a complete
experiment after a crash or code/configuration change.
"""
function _run_checkpointed_tasks(paths::RunPaths, stage::AbstractString, tasks::AbstractVector;
                                 resume::Bool, task_name::Function,
                                 task_outputs::Function, action::Function,
                                 task_extras::Function=(task -> Dict{String,Any}()),
                                 threaded::Bool=true, blas_threads::Union{Nothing,Int}=1)
    total = length(tasks)
    completed_before = resume ? count(task -> task_complete(paths, task_name(task)), tasks) : 0
    pending = resume ? [task for task in tasks if !task_complete(paths, task_name(task))] : collect(tasks)
    update_progress(paths, stage; completed=completed_before, total=total,
        message=isempty(pending) ? "all tasks already complete" : "starting/resuming task execution")

    failures = Vector{NamedTuple{(:task, :error),Tuple{String,String}}}()
    failure_lock = ReentrantLock()
    progress_lock = ReentrantLock()
    completed_counter = Ref(completed_before)

    function run_one(task)
        identifier = string(task_name(task))
        try
            action(task)
            outputs = string.(task_outputs(task))
            mark_task_complete(paths, identifier; outputs=outputs, extras=task_extras(task))
        catch err
            diagnostic = sprint(showerror, err, catch_backtrace())
            lock(failure_lock) do
                push!(failures, (task=identifier, error=diagnostic))
            end
        finally
            lock(progress_lock) do
                completed_counter[] += 1
                update_progress(paths, stage; completed=completed_counter[], total=total,
                    message="last task: $(identifier)")
            end
        end
        return nothing
    end

    old_blas = BLAS.get_num_threads()
    blas_threads === nothing || BLAS.set_num_threads(blas_threads)
    try
        if threaded && length(pending) > 1
            Threads.@threads for index in eachindex(pending)
                run_one(pending[index])
            end
        else
            for task in pending
                run_one(task)
            end
        end
    finally
        blas_threads === nothing || BLAS.set_num_threads(old_blas)
    end

    failure_path = _failure_log_path(paths, stage)
    if !isempty(failures)
        atomic_csv_write(failure_path, DataFrame(
            task=[failure.task for failure in failures],
            error=[failure.error for failure in failures],
        ))
        error("$(length(failures)) task(s) failed in $(stage); see $(failure_path)")
    end
    isfile(failure_path) && rm(failure_path; force=true)
    update_progress(paths, stage; completed=total, total=total, message="all tasks complete")
    return nothing
end

function _preflight_gurobi(config::AbstractDict, stage::String)
    settings = _adaptive_settings(config)
    try
        return gurobi_preflight(
            output_flag=Int(settings["gurobi_output_flag"]),
            threads=Int(settings["gurobi_threads"]))
    catch err
        error("Gurobi preflight failed before $(stage): " * sprint(showerror, err))
    end
end

function _dataset_manifest(bundle::DatasetBundle)
    metadata = Dict{String,Any}()
    for (key, value) in bundle.metadata
        key == "benchmark_predictions" && continue
        metadata[string(key)] = value
    end
    return Dict{String,Any}(
        "name" => bundle.name,
        "n" => length(bundle.y),
        "m" => size(bundle.X, 2),
        "member_names" => bundle.member_names,
        "train" => [first(bundle.split.train), last(bundle.split.train)],
        "validation" => [first(bundle.split.validation), last(bundle.split.validation)],
        "test" => [first(bundle.split.test), last(bundle.split.test)],
        "availability_lag" => bundle.availability_lag,
        "event_count" => bundle.event_ids === nothing ? 0 : length(unique(bundle.event_ids)),
        "units" => bundle.units,
        "metadata" => metadata,
    )
end

function _weight_dataframe(matrix::AbstractMatrix, time_index::AbstractVector{Int},
                           member_names::Vector{String})
    size(matrix, 1) == length(time_index) || throw(DimensionMismatch("weight rows differ from time index"))
    size(matrix, 2) == length(member_names) || throw(DimensionMismatch("weight columns differ from members"))
    df = DataFrame(time_index=time_index)
    for (column, name) in enumerate(member_names)
        df[!, Symbol(name)] = Float64.(matrix[:, column])
    end
    return df
end

function _save_evaluation(paths::RunPaths, prefix::String, bundle::DatasetBundle,
                          metrics::DataFrame, predictions::DataFrame,
                          tuning::DataFrame, selected::DataFrame,
                          artifacts::Dict{String,Any})
    atomic_csv_write(joinpath(paths.tables, prefix * "_metrics.csv"), metrics)
    atomic_csv_write(joinpath(paths.predictions, prefix * "_predictions.csv"), predictions)
    atomic_csv_write(joinpath(paths.tables, prefix * "_tuning.csv"), tuning)
    atomic_csv_write(joinpath(paths.tables, prefix * "_selected_hyperparameters.csv"), selected)
    atomic_toml_write(joinpath(paths.manifests, prefix * "_dataset.toml"), _dataset_manifest(bundle))
    member_manifest = DataFrame(member_index=collect(1:length(bundle.member_names)),
                                member_name=bundle.member_names)
    atomic_csv_write(joinpath(paths.manifests, prefix * "_members.csv"), member_manifest)
    test_index = collect(bundle.split.test)
    for (name, artifact) in artifacts
        if artifact isa AbstractMatrix
            weight_df = _weight_dataframe(artifact, test_index, bundle.member_names)
            atomic_csv_write(joinpath(paths.predictions, prefix * "_" * name * ".csv"), weight_df)
        elseif artifact isa AbstractVector{<:Integer}
            atomic_csv_write(joinpath(paths.predictions, prefix * "_" * name * ".csv"),
                DataFrame(time_index=test_index, history_count=artifact))
        else
            save_model(joinpath(paths.models, prefix * "_" * name * ".jls"), artifact)
        end
    end
    return nothing
end

function _real_world_figures(paths::RunPaths, prefix::String, metrics::DataFrame)
    for (metric, label) in ((:rmse, "RMSE"), (:mae, "MAE"), (:cvar_05, "CVaR 5%"))
        order = sortperm(Float64.(metrics[!, metric]))
        labels = string.(metrics.method[order])
        values = Float64.(metrics[order, metric])
        svg_bar_chart(joinpath(paths.figures, prefix * "_" * lowercase(string(metric)) * ".svg"),
            labels, values; title="$(prefix): held-out $(label)", ylabel=label)
    end
end

function _bootstrap_block_length(section::AbstractDict, n::Int)
    requested = Int(_cfg_get(section, "bootstrap_block_length", 0))
    return requested > 0 ? clamp(requested, 2, n) : max(2, round(Int, n^(1 / 3)))
end

function run_real_world_publication(config::AbstractDict, paths::RunPaths;
                                    root::AbstractString=".", resume::Bool=true)
    section = _cfg_section(config, "real_world")
    methods = _string_vector(_cfg_get(section, "methods", nothing), DEFAULT_METHODS)
    bootstrap_reps = Int(_cfg_get(section, "bootstrap_reps", 2000))
    datasets = _string_vector(_cfg_get(section, "datasets", nothing), [
        "safi", "energy_historical_selected_10", "energy_all_available",
    ])
    all_metrics = DataFrame()
    all_bootstrap = DataFrame()
    update_progress(paths, "real_world_datasets"; completed=0, total=length(datasets),
        message="starting/resuming real-world datasets")
    for (position, dataset_name) in enumerate(datasets)
        stage = "real_world_" * dataset_name
        metrics_path = joinpath(paths.tables, dataset_name * "_metrics.csv")
        predictions_path = joinpath(paths.predictions, dataset_name * "_predictions.csv")
        tuning_path = joinpath(paths.tables, dataset_name * "_tuning.csv")
        selected_path = joinpath(paths.tables, dataset_name * "_selected_hyperparameters.csv")
        bootstrap_path = joinpath(paths.tables, dataset_name * "_moving_block_bootstrap.csv")
        manifest_path = joinpath(paths.manifests, dataset_name * "_dataset.toml")
        members_path = joinpath(paths.manifests, dataset_name * "_members.csv")
        figure_paths = [joinpath(paths.figures, dataset_name * "_$(metric).svg")
                        for metric in ("rmse", "mae", "cvar_05")]
        required_outputs = vcat([
            metrics_path, predictions_path, tuning_path, selected_path,
            bootstrap_path, manifest_path, members_path,
        ], figure_paths)

        if resume && stage_complete(paths, stage)
            write_log(paths, "suite", "Skipping verified $(stage) and loading saved aggregates")
            append!(all_metrics, DataFrame(CSV.File(metrics_path)); cols=:union)
            append!(all_bootstrap, DataFrame(CSV.File(bootstrap_path)); cols=:union)
        else
            bundle = load_dataset(dataset_name, root)
            metrics, predictions, tuning, selected, artifacts = evaluate_bundle(
                bundle; methods=methods, config=config, standardize=true, seed=2026)
            _save_evaluation(paths, dataset_name, bundle,
                metrics, predictions, tuning, selected, artifacts)
            # Bootstrap every saved forecast column, including external hurricane
            # benchmarks such as OFCL/FSSE, while keeping Adaptive Ridge as the paired
            # reference. This prevents the operational comparators from disappearing
            # from the inferential table merely because they are not tunable methods.
            available_methods = [string(name) for name in propertynames(predictions)
                                 if !(name in (:time_index, :target))]
            block_length = _bootstrap_block_length(section, length(bundle.split.test))
            bootstrap = moving_block_bootstrap(
                bundle.y[bundle.split.test], predictions;
                methods=available_methods,
                reps=bootstrap_reps,
                block_length=block_length,
                reference="adaptive_ridge",
                seed=2026 + length(dataset_name),
                event_ids=bundle.event_ids === nothing ? nothing :
                    bundle.event_ids[bundle.split.test],
            )
            bootstrap[!, :dataset] = fill(dataset_name, nrow(bootstrap))
            atomic_csv_write(bootstrap_path, bootstrap)
            _real_world_figures(paths, dataset_name, metrics)
            append!(all_metrics, metrics; cols=:union)
            append!(all_bootstrap, bootstrap; cols=:union)
            mark_stage_complete(paths, stage; extras=Dict(
                "methods" => methods,
                "bootstrap_reps" => bootstrap_reps,
                "bootstrap_block_length" => block_length,
            ), outputs=required_outputs)
        end
        update_progress(paths, "real_world_datasets"; completed=position,
            total=length(datasets), message="last dataset: $(dataset_name)")
    end

    aggregate_outputs = String[]
    if nrow(all_metrics) > 0
        path = joinpath(paths.tables, "real_world_all_metrics.csv")
        atomic_csv_write(path, all_metrics)
        push!(aggregate_outputs, path)
    end
    if nrow(all_bootstrap) > 0
        path = joinpath(paths.tables, "real_world_all_bootstrap.csv")
        atomic_csv_write(path, all_bootstrap)
        push!(aggregate_outputs, path)
    end
    !isempty(aggregate_outputs) && mark_stage_complete(paths, "real_world_all";
        outputs=aggregate_outputs, extras=Dict("datasets" => datasets))
    return all_metrics
end

function _synthetic_checkpoint(paths::RunPaths, regime::String, seed::Int, suffix::String="metrics")
    return joinpath(paths.checkpoints, @sprintf("synthetic_%s_seed_%03d_%s.csv", regime, seed, suffix))
end

function _synthetic_task_name(regime::String, seed::Int)
    return @sprintf("synthetic_%s_seed_%03d", regime, seed)
end

function _synthetic_task_outputs(paths::RunPaths, regime::String, seed::Int,
                                 save_prediction_seeds::Set{Int})
    outputs = [
        _synthetic_checkpoint(paths, regime, seed, "metrics"),
        _synthetic_checkpoint(paths, regime, seed, "selected"),
        _synthetic_checkpoint(paths, regime, seed, "tuning"),
    ]
    if seed in save_prediction_seeds
        prefix = @sprintf("synthetic_%s_seed_%03d", regime, seed)
        append!(outputs, [
            joinpath(paths.predictions, prefix * "_predictions.csv"),
            joinpath(paths.tables, prefix * "_tuning.csv"),
            joinpath(paths.tables, prefix * "_selected.csv"),
            joinpath(paths.figures, prefix * "_rolling_rmse.svg"),
        ])
    end
    return outputs
end

function _rolling_metric(y::AbstractVector, prediction::AbstractVector, window::Int)
    n = length(y)
    n >= window || return Float64[], Float64[]
    x = collect(Float64(window):Float64(n))
    values = [rmse(@view(y[(i - window + 1):i]), @view(prediction[(i - window + 1):i])) for i in window:n]
    return x, values
end

function _change_point_diagnostics(bundle::DatasetBundle, predictions::DataFrame;
                                   window::Int=25)
    points = Int.(get(bundle.metadata, "change_points", Int[]))
    test = collect(bundle.split.test)
    test_start = first(test)
    test_end = last(test)
    relevant = [point for point in points if test_start <= point <= test_end]
    rows = DataFrame()
    isempty(relevant) && return rows
    method_names = [string(name) for name in names(predictions) if !(name in ("time_index", "target"))]
    for point in relevant, method in method_names
        prediction = Float64.(predictions[!, Symbol(method)])
        local_point = point - test_start + 1
        intervals = Dict(
            "pre" => max(1, local_point - 4window):(local_point - 1),
            "immediate" => local_point:min(length(test), local_point + window - 1),
            "early" => min(length(test), local_point + window):min(length(test), local_point + 4window - 1),
            "late" => min(length(test), local_point + 4window):min(length(test), local_point + 8window - 1),
        )
        for (phase, interval) in intervals
            isempty(interval) && continue
            first(interval) > last(interval) && continue
            value = rmse(bundle.y[test][interval], prediction[interval])
            append!(rows, DataFrame(
                dataset=[bundle.name],
                regime=[string(bundle.metadata["regime"])],
                change_point=[point],
                method=[method],
                phase=[phase],
                observations=[length(interval)],
                rmse=[value],
            ); cols=:union)
        end
    end
    return rows
end

function _save_seed_weight_artifacts(paths::RunPaths, prefix::String,
                                     bundle::DatasetBundle,
                                     artifacts::Dict{String,Any})
    test = collect(bundle.split.test)
    for (name, artifact) in artifacts
        if artifact isa AbstractMatrix
            atomic_csv_write(joinpath(paths.predictions, prefix * "_" * name * ".csv"),
                _weight_dataframe(artifact, test, bundle.member_names))
        elseif artifact isa AbstractVector{<:Integer}
            atomic_csv_write(joinpath(paths.predictions, prefix * "_" * name * ".csv"),
                DataFrame(time_index=test, history_count=artifact))
        end
    end
end

function _synthetic_task(config::AbstractDict, paths::RunPaths,
                         regime::String, seed::Int, n::Int, m::Int,
                         methods::Vector{String}, save_prediction_seeds::Set{Int})
    bundle = generate_synthetic_regime(
        regime, seed; n=n, m=m, train_end=div(n, 2), val_end=div(3n, 4))
    metrics, predictions, tuning, selected, artifacts = evaluate_bundle(
        bundle; methods=methods, config=config, standardize=false, seed=seed)
    metrics[!, :regime] = fill(regime, nrow(metrics))
    metrics[!, :seed] = fill(seed, nrow(metrics))
    selected[!, :regime] = fill(regime, nrow(selected))
    selected[!, :seed] = fill(seed, nrow(selected))
    tuning[!, :regime] = fill(regime, nrow(tuning))
    tuning[!, :seed] = fill(seed, nrow(tuning))
    atomic_csv_write(_synthetic_checkpoint(paths, regime, seed, "metrics"), metrics)
    atomic_csv_write(_synthetic_checkpoint(paths, regime, seed, "selected"), selected)
    atomic_csv_write(_synthetic_checkpoint(paths, regime, seed, "tuning"), tuning)
    diagnostics = _change_point_diagnostics(bundle, predictions)
    if nrow(diagnostics) > 0
        atomic_csv_write(_synthetic_checkpoint(paths, regime, seed, "adaptation"), diagnostics)
    end
    if seed in save_prediction_seeds
        prefix = @sprintf("synthetic_%s_seed_%03d", regime, seed)
        atomic_csv_write(joinpath(paths.predictions, prefix * "_predictions.csv"), predictions)
        atomic_csv_write(joinpath(paths.tables, prefix * "_tuning.csv"), tuning)
        atomic_csv_write(joinpath(paths.tables, prefix * "_selected.csv"), selected)
        _save_seed_weight_artifacts(paths, prefix, bundle, artifacts)
        series = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
        for method in methods
            Symbol(method) in propertynames(predictions) || continue
            x, values = _rolling_metric(bundle.y[bundle.split.test],
                Float64.(predictions[!, Symbol(method)]), 25)
            !isempty(values) && (series[method] = (x, values))
        end
        !isempty(series) && svg_line_chart(
            joinpath(paths.figures, prefix * "_rolling_rmse.svg"), series;
            title="$(regime), seed $(seed): 25-step rolling test RMSE",
            xlabel="test observation", ylabel="rolling RMSE")
    end
    return nothing
end

function _aggregate_synthetic(paths::RunPaths, regimes::Vector{String}, seeds::Int)
    per_seed = DataFrame()
    selected = DataFrame()
    tuning = DataFrame()
    adaptation = DataFrame()
    for regime in regimes, seed in 1:seeds
        append!(per_seed, DataFrame(CSV.File(_synthetic_checkpoint(paths, regime, seed, "metrics"))); cols=:union)
        append!(selected, DataFrame(CSV.File(_synthetic_checkpoint(paths, regime, seed, "selected"))); cols=:union)
        tuning_path = _synthetic_checkpoint(paths, regime, seed, "tuning")
        isfile(tuning_path) && append!(tuning, DataFrame(CSV.File(tuning_path)); cols=:union)
        adaptation_path = _synthetic_checkpoint(paths, regime, seed, "adaptation")
        isfile(adaptation_path) && filesize(adaptation_path) > 0 &&
            append!(adaptation, DataFrame(CSV.File(adaptation_path)); cols=:union)
    end
    atomic_csv_write(joinpath(paths.tables, "synthetic_regimes_per_seed.csv"), per_seed)
    atomic_csv_write(joinpath(paths.tables, "synthetic_selected_hyperparameters.csv"), selected)
    nrow(tuning) > 0 && atomic_csv_write(joinpath(paths.tables, "synthetic_all_tuning.csv"), tuning)
    nrow(adaptation) > 0 && atomic_csv_write(joinpath(paths.tables, "synthetic_adaptation_diagnostics.csv"), adaptation)
    summary = summarize_replications(per_seed)
    paired = paired_seed_intervals(per_seed)
    atomic_csv_write(joinpath(paths.tables, "synthetic_regimes_summary.csv"), summary)
    atomic_csv_write(joinpath(paths.tables, "synthetic_paired_rmse_intervals.csv"), paired)
    return per_seed, summary
end

function _synthetic_summary_figure(paths::RunPaths, summary::DataFrame,
                                   regimes::Vector{String}, seeds::Int,
                                   metric::Symbol)
    value_column = Symbol(metric, "_mean")
    series = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    x = Float64.(collect(1:length(regimes)))
    for method in unique(string.(summary.method))
        subset = summary[summary.method .== method, :]
        lookup = Dict(string(row.regime) => Float64(row[value_column]) for row in eachrow(subset))
        all(haskey(lookup, regime) for regime in regimes) &&
            (series[method] = (x, [lookup[regime] for regime in regimes]))
    end
    svg_line_chart(
        joinpath(paths.figures, "synthetic_regime_$(metric).svg"), series;
        title="Synthetic regimes: mean $(uppercase(string(metric))) over $(seeds) seeds",
        xlabel="Regime index: " * join(["$(i)=$(regime)" for (i, regime) in enumerate(regimes)], ", "),
        ylabel=uppercase(string(metric)))
end

function run_synthetic_publication(config::AbstractDict, paths::RunPaths; resume::Bool=true)
    if resume && stage_complete(paths, "synthetic_full")
        return DataFrame(CSV.File(joinpath(paths.tables, "synthetic_regimes_per_seed.csv"))),
               DataFrame(CSV.File(joinpath(paths.tables, "synthetic_regimes_summary.csv")))
    end
    section = _cfg_section(config, "synthetic")
    regimes = _string_vector(_cfg_get(section, "regimes", nothing), SYNTHETIC_REGIMES)
    seeds = Int(_cfg_get(section, "seeds", 30))
    n = Int(_cfg_get(section, "n", 4000))
    m = Int(_cfg_get(section, "m", 10))
    methods = _string_vector(_cfg_get(section, "methods", nothing), [
        "adaptive_ridge", "static_ridge", "rolling_ridge", "rls", "hedge",
        "fixed_share", "passive_aggressive", "dma", "ensemble_mean",
        "oracle_best_member", "validation_best_member",
    ])
    save_prediction_seeds = Set(Int.(_cfg_get(section, "save_seed_predictions", [1])))
    tasks = [(regime=regime, seed=seed) for regime in regimes for seed in 1:seeds]
    write_log(paths, "suite",
        "Synthetic publication run: $(length(tasks)) regime-seed tasks on $(Threads.nthreads()) Julia threads")
    _run_checkpointed_tasks(paths, "synthetic_full_tasks", tasks;
        resume=resume,
        task_name=task -> _synthetic_task_name(task.regime, task.seed),
        task_outputs=task -> _synthetic_task_outputs(
            paths, task.regime, task.seed, save_prediction_seeds),
        task_extras=task -> Dict("regime" => task.regime, "seed" => task.seed),
        action=task -> _synthetic_task(
            config, paths, task.regime, task.seed, n, m, methods, save_prediction_seeds),
        threaded=true,
        blas_threads=1,
    )
    per_seed, summary = _aggregate_synthetic(paths, regimes, seeds)
    _synthetic_summary_figure(paths, summary, regimes, seeds, :rmse)
    _synthetic_summary_figure(paths, summary, regimes, seeds, :cvar_05)
    outputs = [
        joinpath(paths.tables, "synthetic_regimes_per_seed.csv"),
        joinpath(paths.tables, "synthetic_regimes_summary.csv"),
        joinpath(paths.tables, "synthetic_paired_rmse_intervals.csv"),
        joinpath(paths.tables, "synthetic_selected_hyperparameters.csv"),
        joinpath(paths.figures, "synthetic_regime_rmse.svg"),
        joinpath(paths.figures, "synthetic_regime_cvar_05.svg"),
    ]
    mark_stage_complete(paths, "synthetic_full"; extras=Dict(
        "seeds" => seeds, "regimes" => regimes, "methods" => methods), outputs=outputs)
    return per_seed, summary
end

function _manuscript_sweep_checkpoint(paths::RunPaths, sweep::String,
                                      setting::Real, seed::Int, suffix::String)
    token = replace(string(setting), "." => "p", "-" => "m")
    return joinpath(paths.checkpoints,
        @sprintf("manuscript_%s_%s_seed_%03d_%s.csv", sweep, token, seed, suffix))
end

function _manuscript_sweep_task_name(sweep::String, setting::Real, seed::Int)
    token = replace(string(setting), "." => "p", "-" => "m")
    return @sprintf("manuscript_%s_%s_seed_%03d", sweep, token, seed)
end

function _manuscript_sweep_task_outputs(paths::RunPaths, sweep::String,
                                        setting::Real, seed::Int)
    outputs = [
        _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "metrics"),
        _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "tuning"),
        _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "selected"),
    ]
    if seed == 1
        token = replace(string(setting), "." => "p", "-" => "m")
        prefix = "manuscript_$(sweep)_$(token)_seed_001"
        push!(outputs, joinpath(paths.predictions, prefix * "_predictions.csv"))
    end
    return outputs
end

function _config_with_fixed_tau(config::AbstractDict, tau::Int;
                                selection_metric::Union{Nothing,String}=nothing,
                                reproduction_grid::Union{Nothing,AbstractVector}=nothing)
    local_config = deepcopy(config)
    grids = haskey(local_config, "grids") ? local_config["grids"] : Dict{String,Any}()
    grids["tau"] = [tau]
    if reproduction_grid !== nothing
        grid = Float64.(reproduction_grid)
        grids["lambda"] = grid
        grids["ridge_lambda"] = grid
        # The paper tunes the PA margin epsilon. Use a practically unbounded PA-I cap
        # so the corrected implementation matches the stated update rather than PA-I.
        grids["pa_C"] = [1.0e12]
        grids["pa_epsilon"] = grid
    end
    local_config["grids"] = grids
    if selection_metric !== nothing
        experiment = haskey(local_config, "experiment") ? local_config["experiment"] : Dict{String,Any}()
        experiment["selection_metric"] = selection_metric
        local_config["experiment"] = experiment
    end
    return local_config
end

function _manuscript_history_bundle(seed::Int, history::Int)
    history >= 50 || throw(ArgumentError("manuscript history setting must be at least 50"))
    base = generate_synthetic_regime("gradual", seed)
    test_start = first(base.split.test)
    start = test_start - history
    start >= 1 || throw(ArgumentError("history=$(history) exceeds the 3000 pre-test observations"))
    stop = last(base.split.test)
    X = Matrix{Float64}(base.X[start:stop, :])
    y = Vector{Float64}(base.y[start:stop])
    validation_length = max(25, round(Int, history / 3))
    validation_length < history || (validation_length = history - 1)
    train_length = history - validation_length
    split = SplitSpec(
        train=1:train_length,
        validation=(train_length + 1):history,
        test=(history + 1):length(y),
    )
    return DatasetBundle(
        name="manuscript_history_$(history)", X=X, y=y,
        member_names=base.member_names, split=split,
        availability_lag=1, units=base.units,
        metadata=Dict{String,Any}(
            "sweep" => "history", "setting" => history, "seed" => seed,
            "test_source_indices" => [test_start, stop],
            "training_count" => train_length,
            "validation_count" => validation_length,
            "test_count" => length(split.test),
        ),
    )
end

function _manuscript_sweep_bundle(sweep::String, setting::Real, seed::Int)
    if sweep == "members"
        return generate_synthetic_regime("gradual", seed; m=Int(setting))
    elseif sweep == "drift"
        level = Float64(setting)
        return generate_synthetic_regime("gradual", seed;
            sigma_drift=level, s_drift=level)
    elseif sweep == "window"
        return generate_synthetic_regime("gradual", seed)
    elseif sweep == "history"
        return _manuscript_history_bundle(seed, Int(setting))
    elseif sweep == "discrete_probability"
        return generate_synthetic_regime("discrete_gaussian", seed;
            p_drift=Float64(setting), sigma_drift=0.5, s_drift=0.5)
    end
    throw(ArgumentError("unknown manuscript sweep $(sweep)"))
end

function _manuscript_sweep_figure(paths::RunPaths, summary::DataFrame,
                                  sweep::String, metric::Symbol)
    subset = summary[summary.sweep .== sweep, :]
    nrow(subset) == 0 && return nothing
    value_column = Symbol(metric, "_mean")
    series = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
    for method in unique(string.(subset.method))
        current = subset[subset.method .== method, :]
        settings = Float64.(current.setting)
        order = sortperm(settings)
        series[method] = (settings[order], Float64.(current[order, value_column]))
    end
    svg_line_chart(joinpath(paths.figures, "manuscript_$(sweep)_$(metric).svg"), series;
        title="Corrected Julia replication of manuscript $(sweep) sweep",
        xlabel=sweep == "members" ? "ensemble members" :
               sweep == "drift" ? "drift scale" :
               sweep == "window" ? "fixed adaptive window tau" :
               sweep == "history" ? "available train+validation observations" :
               "Bernoulli probability of discrete Gaussian drift",
        ylabel=uppercase(string(metric)),
        logx=sweep == "history")
    return nothing
end

function run_manuscript_sweeps_publication(config::AbstractDict, paths::RunPaths;
                                           resume::Bool=true)
    section = _cfg_section(config, "manuscript_sweeps")
    enabled = Bool(_cfg_get(section, "enabled", true))
    !enabled && return DataFrame(), DataFrame()
    if resume && stage_complete(paths, "manuscript_sweeps")
        return DataFrame(CSV.File(joinpath(paths.tables, "manuscript_sweeps_per_seed.csv"))),
               DataFrame(CSV.File(joinpath(paths.tables, "manuscript_sweeps_summary.csv")))
    end
    seeds = Int(_cfg_get(section, "seeds", 30))
    fixed_tau = Int(_cfg_get(section, "fixed_tau", 5))
    selection_metric = lowercase(string(_cfg_get(section, "selection_metric", "mae")))
    reproduction_grid = Float64.(_cfg_get(section, "reproduction_grid",
        [1e-4, 1e-3, 1e-2, 1e-1, 1.0]))
    methods = _string_vector(_cfg_get(section, "methods", nothing), [
        "adaptive_ridge", "static_ridge", "rls", "hedge", "exp3", "fixed_share",
        "passive_aggressive", "ensemble_mean", "oracle_best_member",
        "validation_best_member",
    ])
    settings = Dict{String,Vector{Float64}}(
        "members" => Float64.(_cfg_get(section, "member_sizes", [3, 6, 10, 15, 25, 50])),
        "drift" => Float64.(_cfg_get(section, "drift_levels", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        "window" => Float64.(_cfg_get(section, "window_sizes", [1, 2, 3, 4, 5, 8, 10, 15, 20, 25])),
        "history" => Float64.(_cfg_get(section, "history_sizes", [100, 200, 400, 750, 1500, 3000])),
        "discrete_probability" => Float64.(_cfg_get(section, "discrete_probabilities", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])),
    )
    sweep_names = ["members", "drift", "window", "history", "discrete_probability"]
    tasks = [(sweep=sweep, setting=setting, seed=seed) for sweep in sweep_names
             for setting in settings[sweep] for seed in 1:seeds]

    function run_task(task)
        bundle = _manuscript_sweep_bundle(task.sweep, task.setting, task.seed)
        local_config = _config_with_fixed_tau(
            config, task.sweep == "window" ? Int(task.setting) : fixed_tau;
            selection_metric=selection_metric,
            reproduction_grid=reproduction_grid)
        metrics, predictions, tuning, selected, artifacts = evaluate_bundle(
            bundle; methods=methods, config=local_config,
            standardize=false, seed=20_000 + task.seed)
        for table in (metrics, tuning, selected)
            table[!, :sweep] = fill(task.sweep, nrow(table))
            table[!, :setting] = fill(task.setting, nrow(table))
            table[!, :seed] = fill(task.seed, nrow(table))
        end
        atomic_csv_write(_manuscript_sweep_checkpoint(
            paths, task.sweep, task.setting, task.seed, "metrics"), metrics)
        atomic_csv_write(_manuscript_sweep_checkpoint(
            paths, task.sweep, task.setting, task.seed, "tuning"), tuning)
        atomic_csv_write(_manuscript_sweep_checkpoint(
            paths, task.sweep, task.setting, task.seed, "selected"), selected)
        if task.seed == 1
            token = replace(string(task.setting), "." => "p", "-" => "m")
            prefix = "manuscript_$(task.sweep)_$(token)_seed_001"
            atomic_csv_write(joinpath(paths.predictions, prefix * "_predictions.csv"), predictions)
            _save_seed_weight_artifacts(paths, prefix, bundle, artifacts)
        end
        return nothing
    end

    _run_checkpointed_tasks(paths, "manuscript_sweeps_tasks", tasks;
        resume=resume,
        task_name=task -> _manuscript_sweep_task_name(task.sweep, task.setting, task.seed),
        task_outputs=task -> _manuscript_sweep_task_outputs(
            paths, task.sweep, task.setting, task.seed),
        task_extras=task -> Dict(
            "sweep" => task.sweep, "setting" => task.setting, "seed" => task.seed),
        action=run_task,
        threaded=true,
        blas_threads=1,
    )

    per_seed = DataFrame()
    tuning = DataFrame()
    selected = DataFrame()
    for sweep in sweep_names, setting in settings[sweep], seed in 1:seeds
        append!(per_seed, DataFrame(CSV.File(
            _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "metrics"))); cols=:union)
        append!(tuning, DataFrame(CSV.File(
            _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "tuning"))); cols=:union)
        append!(selected, DataFrame(CSV.File(
            _manuscript_sweep_checkpoint(paths, sweep, setting, seed, "selected"))); cols=:union)
    end
    summary = summarize_replications(per_seed; groupcols=[:sweep, :setting, :method])
    paired_input = copy(per_seed)
    paired_input[!, :sweep_setting] = string.(paired_input.sweep, "=", paired_input.setting)
    paired = paired_seed_intervals(paired_input; groupcol=:sweep_setting)
    atomic_csv_write(joinpath(paths.tables, "manuscript_sweeps_per_seed.csv"), per_seed)
    atomic_csv_write(joinpath(paths.tables, "manuscript_sweeps_summary.csv"), summary)
    atomic_csv_write(joinpath(paths.tables, "manuscript_sweeps_paired_intervals.csv"), paired)
    atomic_csv_write(joinpath(paths.tables, "manuscript_sweeps_tuning.csv"), tuning)
    atomic_csv_write(joinpath(paths.tables, "manuscript_sweeps_selected_hyperparameters.csv"), selected)
    figure_paths = String[]
    for sweep in sweep_names, metric in (:rmse, :cvar_05)
        _manuscript_sweep_figure(paths, summary, sweep, metric)
        push!(figure_paths, joinpath(paths.figures, "manuscript_$(sweep)_$(metric).svg"))
    end
    outputs = vcat([
        joinpath(paths.tables, "manuscript_sweeps_per_seed.csv"),
        joinpath(paths.tables, "manuscript_sweeps_summary.csv"),
        joinpath(paths.tables, "manuscript_sweeps_paired_intervals.csv"),
        joinpath(paths.tables, "manuscript_sweeps_tuning.csv"),
        joinpath(paths.tables, "manuscript_sweeps_selected_hyperparameters.csv"),
    ], figure_paths)
    mark_stage_complete(paths, "manuscript_sweeps"; extras=Dict(
        "seeds" => seeds, "methods" => methods, "fixed_tau" => fixed_tau,
        "selection_metric" => selection_metric,
        "reproduction_grid" => reproduction_grid,
        "sweeps" => sweep_names,
        "total_regime_seed_tasks" => length(tasks)), outputs=outputs)
    return per_seed, summary
end

function _synthetic_meta_checkpoint(paths::RunPaths, regime::String, seed::Int, suffix::String)
    return joinpath(paths.checkpoints, @sprintf("synthetic_meta_%s_seed_%03d_%s.csv", regime, seed, suffix))
end

function _synthetic_meta_task_name(regime::String, seed::Int)
    return @sprintf("synthetic_meta_%s_seed_%03d", regime, seed)
end

function _synthetic_meta_task_outputs(paths::RunPaths, regime::String, seed::Int)
    outputs = [
        _synthetic_meta_checkpoint(paths, regime, seed, "metrics"),
        _synthetic_meta_checkpoint(paths, regime, seed, "tuning"),
        _synthetic_meta_checkpoint(paths, regime, seed, "selected"),
    ]
    if seed == 1
        push!(outputs, joinpath(paths.predictions,
            "synthetic_meta_$(regime)_seed_001_predictions.csv"))
    end
    return outputs
end

function run_synthetic_meta_publication(config::AbstractDict, paths::RunPaths; resume::Bool=true)
    section = _cfg_section(config, "synthetic_meta")
    enabled = Bool(_cfg_get(section, "enabled", true))
    !enabled && return DataFrame()
    if resume && stage_complete(paths, "synthetic_meta")
        return DataFrame(CSV.File(joinpath(paths.tables, "synthetic_meta_summary.csv")))
    end
    seeds = Int(_cfg_get(section, "seeds", 10))
    regimes = _string_vector(_cfg_get(section, "regimes", nothing),
        ["gradual", "abrupt", "member_degradation", "recurring"])
    methods = _string_vector(_cfg_get(section, "methods", nothing),
        ["adaptive_ridge", "rolling_ridge", "rls", "neural_gate", "boosted_meta"])
    n = Int(_cfg_get(section, "n", 4000))
    m = Int(_cfg_get(section, "m", 10))
    tasks = [(regime=regime, seed=seed) for regime in regimes for seed in 1:seeds]

    function run_task(task)
        bundle = generate_synthetic_regime(task.regime, task.seed; n=n, m=m,
            train_end=div(n, 2), val_end=div(3n, 4))
        metrics, predictions, tuning, selected, artifacts = evaluate_bundle(
            bundle; methods=methods, config=config, standardize=false,
            seed=10_000 + task.seed)
        metrics[!, :regime] = fill(task.regime, nrow(metrics))
        metrics[!, :seed] = fill(task.seed, nrow(metrics))
        tuning[!, :regime] = fill(task.regime, nrow(tuning))
        tuning[!, :seed] = fill(task.seed, nrow(tuning))
        selected[!, :regime] = fill(task.regime, nrow(selected))
        selected[!, :seed] = fill(task.seed, nrow(selected))
        atomic_csv_write(_synthetic_meta_checkpoint(
            paths, task.regime, task.seed, "metrics"), metrics)
        atomic_csv_write(_synthetic_meta_checkpoint(
            paths, task.regime, task.seed, "tuning"), tuning)
        atomic_csv_write(_synthetic_meta_checkpoint(
            paths, task.regime, task.seed, "selected"), selected)
        if task.seed == 1
            prefix = "synthetic_meta_$(task.regime)_seed_001"
            atomic_csv_write(joinpath(paths.predictions, prefix * "_predictions.csv"), predictions)
            _save_seed_weight_artifacts(paths, prefix, bundle, artifacts)
        end
        return nothing
    end

    _run_checkpointed_tasks(paths, "synthetic_meta_tasks", tasks;
        resume=resume,
        task_name=task -> _synthetic_meta_task_name(task.regime, task.seed),
        task_outputs=task -> _synthetic_meta_task_outputs(paths, task.regime, task.seed),
        task_extras=task -> Dict("regime" => task.regime, "seed" => task.seed),
        action=run_task,
        threaded=true,
        blas_threads=1,
    )

    per_seed = DataFrame()
    all_tuning = DataFrame()
    all_selected = DataFrame()
    for regime in regimes, seed in 1:seeds
        append!(per_seed, DataFrame(CSV.File(
            _synthetic_meta_checkpoint(paths, regime, seed, "metrics"))); cols=:union)
        append!(all_tuning, DataFrame(CSV.File(
            _synthetic_meta_checkpoint(paths, regime, seed, "tuning"))); cols=:union)
        append!(all_selected, DataFrame(CSV.File(
            _synthetic_meta_checkpoint(paths, regime, seed, "selected"))); cols=:union)
    end
    summary = summarize_replications(per_seed)
    paired = paired_seed_intervals(per_seed)
    output_paths = [
        joinpath(paths.tables, "synthetic_meta_per_seed.csv"),
        joinpath(paths.tables, "synthetic_meta_summary.csv"),
        joinpath(paths.tables, "synthetic_meta_paired_intervals.csv"),
        joinpath(paths.tables, "synthetic_meta_tuning.csv"),
        joinpath(paths.tables, "synthetic_meta_selected_hyperparameters.csv"),
    ]
    atomic_csv_write(output_paths[1], per_seed)
    atomic_csv_write(output_paths[2], summary)
    atomic_csv_write(output_paths[3], paired)
    atomic_csv_write(output_paths[4], all_tuning)
    atomic_csv_write(output_paths[5], all_selected)
    mark_stage_complete(paths, "synthetic_meta"; extras=Dict(
        "seeds" => seeds, "regimes" => regimes, "methods" => methods, "n" => n, "m" => m),
        outputs=output_paths)
    return summary
end

function _sensitivity_checkpoint(paths::RunPaths, prefix::String,
                                 variant::String, tau::Int, lambda::Real)
    token = replace(string(Float64(lambda)), "." => "p", "-" => "m", "+" => "p")
    directory = joinpath(paths.checkpoints, "sensitivity_" * prefix)
    mkpath(directory)
    return joinpath(directory, "$(variant)_tau_$(tau)_lambda_$(token).csv")
end

function _sensitivity_task_name(prefix::String, variant::String, tau::Int)
    return "sensitivity_$(prefix)_$(variant)_tau_$(tau)"
end

function _sensitivity_row(bundle::DatasetBundle, validation::Vector{Int},
                          prefix::String, variant::String,
                          tau::Int, lambda::Float64,
                          model::AdaptiveRidgeModel)
    prediction = predict_adaptive(model, bundle.X, bundle.y, validation;
        event_ids=bundle.event_ids)
    return DataFrame(
        dataset=[prefix], objective_variant=[variant], tau=[tau], lambda=[lambda],
        validation_rmse=[rmse(bundle.y[validation], prediction)],
        validation_mae=[mae(bundle.y[validation], prediction)],
        validation_cvar_05=[empirical_cvar(abs.(bundle.y[validation] .- prediction), 0.05)],
        fit_seconds=[model.fit_seconds],
        preparation_seconds=[model.preparation_seconds],
        solve_seconds=[model.solve_seconds],
        condition_number=[model.condition_number],
        linear_solver=[model.linear_solver],
        converged=[model.converged],
        iterations=[model.iterations],
        residual_norm=[model.residual_norm],
        relative_objective_change=[model.relative_objective_change],
        stationarity_residual=[model.stationarity_residual],
        convergence_reason=[model.convergence_reason],
        n_parameters=[model.n_parameters],
    )
end

function _sensitivity_one(bundle_raw::DatasetBundle, config::AbstractDict,
                          paths::RunPaths, prefix::String; resume::Bool=true)
    bundle, _ = standardize_bundle(bundle_raw)
    section = _cfg_section(config, "sensitivity")
    taus = Int.(_cfg_get(section, "tau", [1, 2, 3, 4, 5, 7, 10, 15, 25]))
    lambdas = Float64.(_cfg_get(section, "lambda",
        [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0]))
    variants = _string_vector(_cfg_get(section, "objective_variants", nothing),
        ["quadratic_stacked", "norm_plus_norm"])
    near_fraction = Float64(_cfg_get(section, "near_optimal_fraction", 0.01))
    train = collect(bundle.split.train)
    validation = collect(bundle.split.validation)
    settings = _adaptive_settings(config)
    tasks = [(variant=variant, tau=tau) for variant in variants for tau in taus]

    function task_outputs(task)
        return [_sensitivity_checkpoint(paths, prefix, task.variant, task.tau, lambda)
                for lambda in lambdas]
    end

    function run_task(task)
        workspace = prepare_adaptive_quadratic_workspace(
            bundle.X, bundle.y, train, task.tau;
            availability_lag=bundle.availability_lag,
            event_ids=bundle.event_ids,
            direct_max_parameters=Int(settings["direct_max_parameters"]),
            max_dense_bytes=Int(settings["max_dense_bytes"]),
        )
        models = if task.variant == "quadratic_stacked"
            fit_adaptive_quadratic_path(workspace, lambdas;
                cg_rtol=Float64(settings["cg_rtol"]),
                cg_atol=Float64(settings["cg_atol"]),
                cg_maxiter=Int(settings["cg_maxiter"]))
        elseif task.variant == "norm_plus_norm"
            fit_adaptive_robust_norm_path(workspace, lambdas;
                smoothing=Float64(settings["irls_smoothing"]),
                irls_rtol=Float64(settings["irls_rtol"]),
                irls_stationarity_rtol=Float64(
                    settings["irls_stationarity_rtol"]),
                irls_maxiter=Int(settings["irls_maxiter"]),
                irls_min_iterations=Int(settings["irls_min_iterations"]),
                cg_rtol=Float64(settings["cg_rtol"]),
                cg_atol=Float64(settings["cg_atol"]),
                cg_maxiter=Int(settings["cg_maxiter"]),
                require_convergence=true)
        else
            throw(ArgumentError("unknown sensitivity objective variant $(task.variant)"))
        end
        for (lambda, model, path) in zip(lambdas, models, task_outputs(task))
            row = _sensitivity_row(bundle, validation, prefix, task.variant,
                task.tau, Float64(lambda), model)
            atomic_csv_write(path, row)
        end
        return nothing
    end

    _run_checkpointed_tasks(paths, "sensitivity_$(prefix)_tasks", tasks;
        resume=resume,
        task_name=task -> _sensitivity_task_name(prefix, task.variant, task.tau),
        task_outputs=task_outputs,
        task_extras=task -> Dict(
            "dataset" => prefix, "objective_variant" => task.variant, "tau" => task.tau),
        action=run_task,
        threaded=true,
        blas_threads=1,
    )

    rows = DataFrame()
    for variant in variants, tau in taus, lambda in lambdas
        path = _sensitivity_checkpoint(paths, prefix, variant, tau, lambda)
        append!(rows, DataFrame(CSV.File(path)); cols=:union)
    end
    sort!(rows, [:objective_variant, :tau, :lambda])
    rows_path = joinpath(paths.tables, "sensitivity_$(prefix).csv")
    atomic_csv_write(rows_path, rows)

    stability = DataFrame()
    figure_paths = String[]
    for subset in groupby(rows, :objective_variant)
        best = minimum(Float64.(subset.validation_rmse))
        threshold = best == 0.0 ? near_fraction : best * (1.0 + near_fraction)
        near = subset.validation_rmse .<= threshold
        best_row = subset[argmin(Float64.(subset.validation_rmse)), :]
        append!(stability, DataFrame(
            dataset=[prefix],
            objective_variant=[string(subset.objective_variant[1])],
            best_validation_rmse=[best],
            best_tau=[Int(best_row.tau)],
            best_lambda=[Float64(best_row.lambda)],
            near_optimal_fraction=[near_fraction],
            near_optimal_cells=[count(near)],
            total_cells=[nrow(subset)],
            near_optimal_tau_min=[minimum(Int.(subset.tau[near]))],
            near_optimal_tau_max=[maximum(Int.(subset.tau[near]))],
            near_optimal_lambda_min=[minimum(Float64.(subset.lambda[near]))],
            near_optimal_lambda_max=[maximum(Float64.(subset.lambda[near]))],
        ); cols=:union)
        matrix = zeros(Float64, length(taus), length(lambdas))
        for (i, tau) in enumerate(taus), (j, lambda) in enumerate(lambdas)
            match = subset[(subset.tau .== tau) .& (subset.lambda .== lambda), :]
            nrow(match) == 1 || error(
                "sensitivity grid has $(nrow(match)) rows for $(prefix), " *
                "$(subset.objective_variant[1]), tau=$(tau), lambda=$(lambda)")
            matrix[i, j] = Float64(match.validation_rmse[1])
        end
        figure_path = joinpath(paths.figures,
            "sensitivity_$(prefix)_$(subset.objective_variant[1])_validation_rmse.svg")
        svg_heatmap(figure_path, string.(lambdas), string.(taus), matrix;
            title="$(prefix), $(subset.objective_variant[1]): validation RMSE",
            xlabel="regularization lambda", ylabel="window tau")
        push!(figure_paths, figure_path)
    end
    stability_path = joinpath(paths.tables, "sensitivity_$(prefix)_stability.csv")
    atomic_csv_write(stability_path, stability)
    return rows, stability, vcat([rows_path, stability_path], figure_paths)
end

function run_sensitivity_publication(config::AbstractDict, paths::RunPaths;
                                     root::AbstractString=".", resume::Bool=true)
    section = _cfg_section(config, "sensitivity")
    datasets = _string_vector(_cfg_get(section, "datasets", nothing), [
        "safi", "energy_historical_selected_10", "energy_all_available",
    ])
    all_rows = DataFrame()
    all_stability = DataFrame()
    update_progress(paths, "sensitivity_datasets"; completed=0, total=length(datasets),
        message="starting/resuming sensitivity datasets")
    for (position, dataset_name) in enumerate(datasets)
        stage = "sensitivity_" * dataset_name
        rows_path = joinpath(paths.tables, "sensitivity_$(dataset_name).csv")
        stability_path = joinpath(paths.tables, "sensitivity_$(dataset_name)_stability.csv")
        if resume && stage_complete(paths, stage)
            append!(all_rows, DataFrame(CSV.File(rows_path)); cols=:union)
            append!(all_stability, DataFrame(CSV.File(stability_path)); cols=:union)
        else
            rows, stability, outputs = _sensitivity_one(
                load_dataset(dataset_name, root), config, paths, dataset_name; resume=resume)
            append!(all_rows, rows; cols=:union)
            append!(all_stability, stability; cols=:union)
            mark_stage_complete(paths, stage; outputs=outputs,
                extras=Dict("dataset" => dataset_name))
        end
        update_progress(paths, "sensitivity_datasets"; completed=position,
            total=length(datasets), message="last dataset: $(dataset_name)")
    end
    aggregate_outputs = String[]
    if nrow(all_rows) > 0
        path = joinpath(paths.tables, "sensitivity_all.csv")
        atomic_csv_write(path, all_rows)
        push!(aggregate_outputs, path)
    end
    if nrow(all_stability) > 0
        path = joinpath(paths.tables, "sensitivity_stability_all.csv")
        atomic_csv_write(path, all_stability)
        push!(aggregate_outputs, path)
    end
    !isempty(aggregate_outputs) && mark_stage_complete(paths, "sensitivity_all";
        outputs=aggregate_outputs, extras=Dict("datasets" => datasets))
    return all_rows
end

function _limited_history_bundle(seed::Int, history::Int)
    base = generate_synthetic_regime("recurring", seed)
    original_test_start = first(base.split.test)
    start = max(1, original_test_start - history)
    stop = last(base.split.test)
    X = base.X[start:stop, :]
    y = base.y[start:stop]
    available_history = original_test_start - start
    validation_length = max(25, div(available_history, 3))
    validation_length < available_history || (validation_length = max(1, available_history - 1))
    train_length = available_history - validation_length
    train_length >= 1 || throw(ArgumentError("history=$(history) leaves no training data"))
    split = SplitSpec(
        train=1:train_length,
        validation=(train_length + 1):available_history,
        test=(available_history + 1):length(y),
    )
    return DatasetBundle(
        name="limited_history",
        X=Matrix{Float64}(X),
        y=Vector{Float64}(y),
        member_names=base.member_names,
        split=split,
        availability_lag=1,
        units=base.units,
        metadata=Dict("history" => available_history, "seed" => seed, "source_start" => start),
    )
end

function _safe_correlation(x::AbstractVector, y::AbstractVector)
    sx = std(x)
    sy = std(y)
    if sx <= 1e-12 || sy <= 1e-12
        return all(abs.(x .- y) .<= 1e-12) ? 1.0 : 0.0
    end
    return cor(x, y)
end

function _greedy_correlation_subset(bundle::DatasetBundle, threshold::Float64)
    0.0 < threshold <= 1.0 || throw(ArgumentError("correlation threshold must be in (0,1]"))
    train = collect(bundle.split.train)
    errors = Matrix(bundle.X[train, :]) .- bundle.y[train]
    scores = vec(mean(abs.(errors); dims=1))
    order = sortperm(scores)
    selected = Int[]
    for candidate in order
        admissible = all(abs(_safe_correlation(@view(errors[:, candidate]), @view(errors[:, kept]))) <= threshold + 1e-12
                         for kept in selected)
        admissible && push!(selected, candidate)
    end
    if length(selected) < min(2, size(bundle.X, 2))
        selected = order[1:min(2, length(order))]
    end
    return sort(selected)
end

function _coefficient_diagnostics(artifacts::Dict{String,Any}, method::String)
    model_key = method * "_model"
    weight_key = method * "_weights"
    model = get(artifacts, model_key, nothing)
    weights = get(artifacts, weight_key, nothing)
    return Dict{String,Any}(
        "condition_number" => model isa AdaptiveRidgeModel ? model.condition_number : NaN,
        "theta_l2" => model isa AdaptiveRidgeModel ? norm(model.theta) : NaN,
        "max_abs_weight" => weights isa AbstractMatrix ? maximum(abs, weights) : NaN,
        "mean_weight_l2" => weights isa AbstractMatrix ? mean(norm(@view(weights[i, :])) for i in axes(weights, 1)) : NaN,
    )
end

function _stress_checkpoint(paths::RunPaths, analysis::String, setting::String, seed::Int)
    safe_setting = replace(setting, r"[^A-Za-z0-9_.-]" => "_")
    return joinpath(paths.checkpoints,
        @sprintf("stress_%s_%s_seed_%03d.csv", analysis, safe_setting, seed))
end

function _stress_task_name(analysis::String, setting::String, seed::Int)
    safe_setting = replace(setting, r"[^A-Za-z0-9_.-]" => "_")
    return @sprintf("stress_%s_%s_seed_%03d", analysis, safe_setting, seed)
end

function _stress_task_outputs(paths::RunPaths, analysis::String, setting::String, seed::Int)
    outputs = [_stress_checkpoint(paths, analysis, setting, seed)]
    if analysis == "correlation"
        push!(outputs, _stress_checkpoint(paths, "correlation_pruning", setting, seed))
    end
    return outputs
end

function _stress_evaluate(bundle::DatasetBundle, methods::Vector{String},
                          config::AbstractDict, analysis::String,
                          setting::String, seed::Int)
    metrics, _, _, _, artifacts = evaluate_bundle(
        bundle; methods=methods, config=config, standardize=false, seed=seed)
    metrics[!, :analysis] = fill(analysis, nrow(metrics))
    metrics[!, :setting] = fill(setting, nrow(metrics))
    metrics[!, :seed] = fill(seed, nrow(metrics))
    diagnostic_names = ("condition_number", "theta_l2", "max_abs_weight", "mean_weight_l2")
    for key in diagnostic_names
        metrics[!, Symbol(key)] = fill(NaN, nrow(metrics))
    end
    for method in unique(string.(metrics.method))
        diagnostics = _coefficient_diagnostics(artifacts, method)
        mask = metrics.method .== method
        for (key, value) in diagnostics
            metrics[mask, Symbol(key)] .= Float64(value)
        end
    end
    return metrics
end

function _correlation_pruning_result(bundle::DatasetBundle, config::AbstractDict,
                                     thresholds::Vector{Float64}, seed::Int)
    selection_metric = _selection_metric(config)
    grids = _merge_grids(config)
    candidates = DataFrame()
    candidate_models = Dict{Float64,Tuple{DatasetBundle,AdaptiveRidgeModel,Vector{Int}}}()
    for threshold in thresholds
        selected_members = _greedy_correlation_subset(bundle, threshold)
        reduced = DatasetBundle(
            name=bundle.name * "_pruned",
            X=Matrix{Float64}(bundle.X[:, selected_members]),
            y=bundle.y,
            member_names=bundle.member_names[selected_members],
            split=bundle.split,
            event_ids=bundle.event_ids,
            availability_lag=bundle.availability_lag,
            units=bundle.units,
            metadata=copy(bundle.metadata),
        )
        model, _, best = tune_adaptive(reduced, grids, config, "adaptive_ridge", selection_metric)
        score = Float64(best[Symbol("validation_", selection_metric)])
        append!(candidates, DataFrame(
            threshold=[threshold],
            selected_members=[join(selected_members, ";")],
            selected_count=[length(selected_members)],
            validation_score=[score],
            tau=[Int(best.tau)],
            lambda=[Float64(best.lambda)],
        ); cols=:union)
        candidate_models[threshold] = (reduced, model, selected_members)
    end
    best_row = candidates[argmin(Float64.(candidates.validation_score)), :]
    reduced, model, selected_members = candidate_models[Float64(best_row.threshold)]
    test = collect(reduced.split.test)
    prediction = predict_adaptive(model, reduced.X, reduced.y, test; event_ids=reduced.event_ids)
    result = metrics_dataframe(reduced.name, "adaptive_ridge_pruned", reduced.y[test], prediction;
        extras=Dict(
            "selected_threshold" => Float64(best_row.threshold),
            "selected_count" => length(selected_members),
            "selected_members" => join(selected_members, ";"),
            "condition_number" => model.condition_number,
            "theta_l2" => norm(model.theta),
        ))
    result[!, :analysis] = ["correlation"]
    result[!, :setting] = [string(get(bundle.metadata, "rho", NaN))]
    result[!, :seed] = [seed]
    return result, candidates
end

function _boundary_cases(seed::Int, config::AbstractDict,
                         event_length::Int, cold_points::Int)
    base = generate_synthetic_regime("recurring", seed)
    event_ids = [cld(t, event_length) for t in 1:length(base.y)]
    reset_bundle = DatasetBundle(
        name="boundary_reset", X=base.X, y=base.y, member_names=base.member_names,
        split=base.split, event_ids=event_ids, availability_lag=1, units=base.units,
        metadata=Dict("event_length" => event_length),
    )
    borrow_bundle = DatasetBundle(
        name="boundary_borrow", X=base.X, y=base.y, member_names=base.member_names,
        split=base.split, event_ids=nothing, availability_lag=1, units=base.units,
        metadata=Dict("event_length" => event_length),
    )
    reset_config = deepcopy(config)
    reset_config["boundary"] = Dict("online_event_policy" => "reset")
    carry_config = deepcopy(config)
    carry_config["boundary"] = Dict("online_event_policy" => "carry")
    rows = DataFrame()
    test = collect(base.split.test)
    cold_mask = [mod(t - 1, event_length) < cold_points for t in test]

    metrics, predictions, _, _, _ = evaluate_bundle(
        reset_bundle; methods=["adaptive_ridge", "rls", "hedge", "passive_aggressive"],
        config=reset_config, standardize=false, seed=seed)
    for row in eachrow(metrics)
        method = string(row.method)
        cold_rmse = rmse(base.y[test][cold_mask], Float64.(predictions[!, Symbol(method)])[cold_mask])
        values = Dict{Symbol,Any}(name => row[name] for name in propertynames(row))
        values[:method] = method * "__reset"
        values[:analysis] = "boundary"
        values[:setting] = "reset"
        values[:seed] = seed
        values[:cold_start_rmse] = cold_rmse
        append!(rows, one_row_dataframe(values); cols=:union)
    end

    metrics, predictions, _, _, _ = evaluate_bundle(
        borrow_bundle; methods=["adaptive_ridge"], config=reset_config,
        standardize=false, seed=seed)
    row = metrics[1, :]
    cold_rmse = rmse(base.y[test][cold_mask], Float64.(predictions[!, :adaptive_ridge])[cold_mask])
    values = Dict{Symbol,Any}(name => row[name] for name in propertynames(row))
    values[:method] = "adaptive_ridge__borrow_across_events"
    values[:analysis] = "boundary"
    values[:setting] = "borrow_across_events"
    values[:seed] = seed
    values[:cold_start_rmse] = cold_rmse
    append!(rows, one_row_dataframe(values); cols=:union)

    metrics, predictions, _, _, _ = evaluate_bundle(
        reset_bundle; methods=["rls", "hedge", "passive_aggressive"],
        config=carry_config, standardize=false, seed=seed)
    for row in eachrow(metrics)
        method = string(row.method)
        cold_rmse = rmse(base.y[test][cold_mask], Float64.(predictions[!, Symbol(method)])[cold_mask])
        values = Dict{Symbol,Any}(name => row[name] for name in propertynames(row))
        values[:method] = method * "__carry"
        values[:analysis] = "boundary"
        values[:setting] = "carry"
        values[:seed] = seed
        values[:cold_start_rmse] = cold_rmse
        append!(rows, one_row_dataframe(values); cols=:union)
    end
    return rows
end

"""Return a stable textual representation for heterogeneous stress-test settings.

CSV ingestion may infer numeric settings (for example, `100.0` or `0.95`) even though
boundary-policy settings are textual.  `string(x)` is only a conversion constructor for
string-like values; `string(x)` is the Julia display conversion that is valid for numbers
and strings alike.
"""
_stress_setting_text(value) = string(value)

function _summarize_stress_rows(rows::DataFrame)
    summary = DataFrame()
    for subset in groupby(rows, [:analysis, :setting, :method])
        rmse_values = Float64.(subset.rmse)
        cvar_values = Float64.(subset.cvar_05)
        cold_values = :cold_start_rmse in propertynames(subset) ?
            Float64.(collect(skipmissing(subset.cold_start_rmse))) : Float64[]
        condition_values = :condition_number in propertynames(subset) ?
            filter(isfinite, Float64.(collect(skipmissing(subset.condition_number)))) : Float64[]
        weight_values = :max_abs_weight in propertynames(subset) ?
            filter(isfinite, Float64.(collect(skipmissing(subset.max_abs_weight)))) : Float64[]
        append!(summary, DataFrame(
            analysis=[_stress_setting_text(subset.analysis[1])],
            setting=[_stress_setting_text(subset.setting[1])],
            method=[_stress_setting_text(subset.method[1])],
            replications=[nrow(subset)],
            rmse_mean=[mean(rmse_values)],
            rmse_sd=[length(rmse_values) > 1 ? std(rmse_values) : 0.0],
            cvar05_mean=[mean(cvar_values)],
            cold_start_rmse_mean=[isempty(cold_values) ? missing : mean(cold_values)],
            condition_number_median=[isempty(condition_values) ? missing : median(condition_values)],
            max_abs_weight_mean=[isempty(weight_values) ? missing : mean(weight_values)],
        ); cols=:union)
    end
    return summary
end

_stress_numeric_settings(values) = parse.(Float64, string.(values))

function run_stress_publication(config::AbstractDict, paths::RunPaths; resume::Bool=true)
    if resume && stage_complete(paths, "stress_full")
        return DataFrame(CSV.File(joinpath(paths.tables, "stress_tests_per_seed.csv"))),
               DataFrame(CSV.File(joinpath(paths.tables, "stress_tests_summary.csv")))
    end
    section = _cfg_section(config, "stress")
    seeds = Int(_cfg_get(section, "seeds", 30))
    methods = _string_vector(_cfg_get(section, "methods", nothing),
        ["adaptive_ridge", "static_ridge", "rolling_ridge", "rls", "passive_aggressive", "ensemble_mean"])
    history_sizes = Int.(_cfg_get(section, "history_sizes", [100, 200, 400, 750, 1500, 3000]))
    correlations = Float64.(_cfg_get(section, "correlations", [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    pruning_thresholds = Float64.(_cfg_get(section, "correlation_pruning_thresholds", [0.8, 0.9, 0.95, 0.99, 1.0]))
    event_length = Int(_cfg_get(section, "event_length", 500))
    cold_points = Int(_cfg_get(section, "cold_start_points", 10))
    tasks = vcat(
        [(analysis="limited_history", setting=string(history), seed=seed)
         for history in history_sizes for seed in 1:seeds],
        [(analysis="correlation", setting=string(rho), seed=seed)
         for rho in correlations for seed in 1:seeds],
        [(analysis="boundary", setting="all", seed=seed) for seed in 1:seeds],
    )

    function run_task(task)
        if task.analysis == "limited_history"
            history = parse(Int, task.setting)
            metrics = _stress_evaluate(_limited_history_bundle(task.seed, history), methods,
                config, "limited_history", task.setting, task.seed)
            atomic_csv_write(_stress_checkpoint(
                paths, "limited_history", task.setting, task.seed), metrics)
        elseif task.analysis == "correlation"
            rho = parse(Float64, task.setting)
            bundle = generate_correlation_stress(task.seed, rho)
            metrics = _stress_evaluate(
                bundle, methods, config, "correlation", task.setting, task.seed)
            pruned_result, candidates = _correlation_pruning_result(
                bundle, config, pruning_thresholds, task.seed)
            append!(metrics, pruned_result; cols=:union)
            candidates[!, :rho] = fill(rho, nrow(candidates))
            candidates[!, :seed] = fill(task.seed, nrow(candidates))
            atomic_csv_write(_stress_checkpoint(
                paths, "correlation", task.setting, task.seed), metrics)
            atomic_csv_write(_stress_checkpoint(
                paths, "correlation_pruning", task.setting, task.seed), candidates)
        elseif task.analysis == "boundary"
            metrics = _boundary_cases(task.seed, config, event_length, cold_points)
            atomic_csv_write(_stress_checkpoint(paths, "boundary", "all", task.seed), metrics)
        else
            throw(ArgumentError("unknown stress analysis $(task.analysis)"))
        end
        return nothing
    end

    _run_checkpointed_tasks(paths, "stress_full_tasks", tasks;
        resume=resume,
        task_name=task -> _stress_task_name(task.analysis, task.setting, task.seed),
        task_outputs=task -> _stress_task_outputs(paths, task.analysis, task.setting, task.seed),
        task_extras=task -> Dict(
            "analysis" => task.analysis, "setting" => task.setting, "seed" => task.seed),
        action=run_task,
        threaded=true,
        blas_threads=1,
    )

    rows = DataFrame()
    pruning_tables = DataFrame()
    for history in history_sizes, seed in 1:seeds
        append!(rows, DataFrame(CSV.File(
            _stress_checkpoint(paths, "limited_history", string(history), seed))); cols=:union)
    end
    for rho in correlations, seed in 1:seeds
        setting = string(rho)
        append!(rows, DataFrame(CSV.File(
            _stress_checkpoint(paths, "correlation", setting, seed))); cols=:union)
        append!(pruning_tables, DataFrame(CSV.File(
            _stress_checkpoint(paths, "correlation_pruning", setting, seed))); cols=:union)
    end
    for seed in 1:seeds
        append!(rows, DataFrame(CSV.File(
            _stress_checkpoint(paths, "boundary", "all", seed))); cols=:union)
    end

    per_seed_path = joinpath(paths.tables, "stress_tests_per_seed.csv")
    pruning_path = joinpath(paths.tables, "stress_correlation_pruning_validation.csv")
    atomic_csv_write(per_seed_path, rows)
    atomic_csv_write(pruning_path, pruning_tables)
    summary = _summarize_stress_rows(rows)
    summary_path = joinpath(paths.tables, "stress_tests_summary.csv")
    atomic_csv_write(summary_path, summary)

    figure_paths = String[]
    for analysis in ("limited_history", "correlation")
        subset = summary[summary.analysis .== analysis, :]
        series = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()
        for method in unique(string.(subset.method))
            current = subset[subset.method .== method, :]
            settings_numeric = _stress_numeric_settings(current.setting)
            order = sortperm(settings_numeric)
            series[method] = (settings_numeric[order], Float64.(current.rmse_mean[order]))
        end
        figure_path = joinpath(paths.figures, "stress_$(analysis)_rmse.svg")
        svg_line_chart(figure_path, series;
            title="$(analysis): held-out RMSE",
            xlabel=analysis == "correlation" ? "member-error correlation" : "historical observations",
            ylabel="RMSE", logx=analysis == "limited_history")
        push!(figure_paths, figure_path)
    end
    boundary = summary[summary.analysis .== "boundary", :]
    if nrow(boundary) > 0
        keep = .!ismissing.(boundary.cold_start_rmse_mean)
        boundary = boundary[keep, :]
        order = sortperm(Float64.(boundary.cold_start_rmse_mean))
        figure_path = joinpath(paths.figures, "stress_boundary_cold_start_rmse.svg")
        svg_bar_chart(figure_path, string.(boundary.method[order]),
            Float64.(boundary.cold_start_rmse_mean[order]);
            title="Event-boundary policy: cold-start RMSE",
            ylabel="mean cold-start RMSE", lower_better=true)
        push!(figure_paths, figure_path)
    end

    outputs = vcat([per_seed_path, pruning_path, summary_path], figure_paths)
    mark_stage_complete(paths, "stress_full"; extras=Dict(
        "seeds" => seeds,
        "history_sizes" => history_sizes,
        "correlations" => correlations,
        "pruning_thresholds" => pruning_thresholds,
        "task_count" => length(tasks),
    ), outputs=outputs)
    return rows, summary
end

function _solver_verification_checkpoint(paths::RunPaths, regime::String,
                                         seed::Int, tau::Int, lambda::Float64)
    token = replace(string(lambda), "." => "p", "-" => "m", "+" => "p")
    directory = joinpath(paths.checkpoints, "solver_verification_tasks")
    mkpath(directory)
    return joinpath(directory,
        @sprintf("%s_seed_%03d_tau_%02d_lambda_%s.csv", regime, seed, tau, token))
end

function _solver_verification_task_name(regime::String, seed::Int,
                                        tau::Int, lambda::Float64)
    token = replace(string(lambda), "." => "p", "-" => "m", "+" => "p")
    return @sprintf("solver_%s_seed_%03d_tau_%02d_lambda_%s", regime, seed, tau, token)
end

function run_solver_verification_publication(config::AbstractDict, paths::RunPaths;
                                             resume::Bool=true)
    objective_path = joinpath(paths.tables, "solver_objective_verification.csv")
    literal_path = joinpath(paths.tables, "solver_literal_jump_equivalence.csv")
    if resume && stage_complete(paths, "solver_verification")
        return DataFrame(CSV.File(objective_path))
    end
    section = _cfg_section(config, "verification")
    run_comparison = Bool(_cfg_get(section, "run_exact_norm_vs_quadratic", true))
    run_literal = Bool(_cfg_get(section, "run_literal_jump_equivalence", true))
    require_gurobi = Bool(_cfg_get(section, "require_gurobi", true))
    (run_comparison || run_literal) && require_gurobi &&
        _preflight_gurobi(config, "solver verification")
    if (run_comparison || run_literal) && !require_gurobi
        @warn "Exact Gurobi verification disabled by configuration; solver-independent quadratic/IRLS checks still run"
    end

    settings = _adaptive_settings(config)
    seeds = Int.(_cfg_get(section, "seeds", [1, 2, 3, 4, 5]))
    regimes = _string_vector(_cfg_get(section, "regimes", nothing),
        ["stationary", "gradual", "recurring"])
    taus = Int.(_cfg_get(section, "tau", [1, 3, 5]))
    lambdas = Float64.(_cfg_get(section, "lambda", [1e-3, 1e-2, 1e-1]))
    tasks = run_comparison ? [
        (regime=regime, seed=seed, tau=tau, lambda=lambda)
        for regime in regimes for seed in seeds for tau in taus for lambda in lambdas
    ] : NamedTuple[]

    function run_task(task)
        bundle = generate_synthetic_regime(task.regime, task.seed;
            n=1200, m=6, train_end=600, val_end=900)
        train = collect(bundle.split.train)
        validation = collect(bundle.split.validation)
        quadratic = fit_adaptive_quadratic(bundle.X, bundle.y, train, task.tau, task.lambda;
            direct_max_parameters=Int(settings["direct_max_parameters"]),
            max_dense_bytes=Int(settings["max_dense_bytes"]),
            cg_rtol=Float64(settings["cg_rtol"]),
            cg_atol=Float64(settings["cg_atol"]),
            cg_maxiter=Int(settings["cg_maxiter"]))
        irls = fit_adaptive_robust_norm(bundle.X, bundle.y, train, task.tau, task.lambda;
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
            require_convergence=true)
        exact = require_gurobi ? fit_adaptive_norm_gurobi(
            bundle.X, bundle.y, train, task.tau, task.lambda;
            output_flag=Int(settings["gurobi_output_flag"]),
            time_limit=Float64(settings["gurobi_time_limit"]),
            gurobi_threads=Int(settings["gurobi_threads"]),
            optimality_tolerance=Float64(settings["gurobi_optimality_tolerance"])) : nothing
        quadratic_prediction = predict_adaptive(quadratic, bundle.X, bundle.y, validation)
        irls_prediction = predict_adaptive(irls, bundle.X, bundle.y, validation)
        exact_prediction = exact === nothing ? fill(NaN, length(validation)) :
            predict_adaptive(exact, bundle.X, bundle.y, validation)
        row = DataFrame(
            regime=[task.regime], seed=[task.seed], tau=[task.tau], lambda=[task.lambda],
            quadratic_validation_rmse=[rmse(bundle.y[validation], quadratic_prediction)],
            irls_norm_validation_rmse=[rmse(bundle.y[validation], irls_prediction)],
            exact_norm_validation_rmse=[exact === nothing ? NaN : rmse(bundle.y[validation], exact_prediction)],
            quadratic_vs_irls_prediction_max_abs=[maximum(abs.(quadratic_prediction .- irls_prediction))],
            irls_vs_exact_prediction_max_abs=[exact === nothing ? NaN : maximum(abs.(irls_prediction .- exact_prediction))],
            irls_vs_exact_prediction_relative_l2=[exact === nothing ? NaN : norm(irls_prediction - exact_prediction) / max(norm(exact_prediction), eps())],
            quadratic_fit_seconds=[quadratic.fit_seconds],
            irls_fit_seconds=[irls.fit_seconds],
            exact_fit_seconds=[exact === nothing ? NaN : exact.fit_seconds],
            quadratic_objective=[quadratic.objective],
            irls_norm_objective=[irls.objective],
            exact_norm_objective=[exact === nothing ? NaN : exact.objective],
            irls_vs_exact_relative_objective=[exact === nothing ? NaN : abs(irls.objective - exact.objective) / max(abs(exact.objective), eps())],
            irls_iterations=[irls.iterations],
            irls_converged=[irls.converged],
            irls_stationarity_residual=[irls.stationarity_residual],
            irls_relative_objective_change=[
                irls.relative_objective_change],
            irls_convergence_reason=[irls.convergence_reason],
        )
        atomic_csv_write(_solver_verification_checkpoint(
            paths, task.regime, task.seed, task.tau, task.lambda), row)
        return nothing
    end

    if run_comparison
        _run_checkpointed_tasks(paths, "solver_verification_tasks", tasks;
            resume=resume,
            task_name=task -> _solver_verification_task_name(
                task.regime, task.seed, task.tau, task.lambda),
            task_outputs=task -> [_solver_verification_checkpoint(
                paths, task.regime, task.seed, task.tau, task.lambda)],
            task_extras=task -> Dict(
                "regime" => task.regime, "seed" => task.seed,
                "tau" => task.tau, "lambda" => task.lambda),
            action=run_task,
            threaded=!require_gurobi,
            blas_threads=1,
        )
    end

    rows = DataFrame()
    if run_comparison
        for task in tasks
            append!(rows, DataFrame(CSV.File(_solver_verification_checkpoint(
                paths, task.regime, task.seed, task.tau, task.lambda))); cols=:union)
        end
        sort!(rows, [:regime, :seed, :tau, :lambda])
    else
        rows = DataFrame(
            status=["not_run"],
            reason=["run_exact_norm_vs_quadratic=false"],
            gurobi_required=[require_gurobi],
        )
    end
    atomic_csv_write(objective_path, rows)

    output_paths = [objective_path]
    if run_literal && require_gurobi
        small = generate_synthetic_regime("stationary", 2026;
            n=180, m=3, train_end=100, val_end=140)
        result = verify_reduced_against_jump(
            small.X, small.y, collect(small.split.train), 2, 1e-2)
        atomic_csv_write(literal_path,
            DataFrame(Dict(Symbol(key) => [value] for (key, value) in result)))
        push!(output_paths, literal_path)
    end
    mark_stage_complete(paths, "solver_verification"; extras=Dict(
        "gurobi_required" => require_gurobi,
        "exact_comparison_run" => run_comparison && require_gurobi,
        "solver_independent_comparison_run" => run_comparison,
        "literal_jump_run" => run_literal && require_gurobi,
        "task_count" => length(tasks)), outputs=output_paths)
    return rows
end

function run_publication_suite(config::AbstractDict, paths::RunPaths;
                               root::AbstractString=".", resume::Bool=true)
    if resume && stage_complete(paths, "publication_suite")
        write_log(paths, "suite", "Verified publication suite is already complete")
        return paths
    end

    # Preflight validates configured datasets and any requested solver license before the
    # multi-hour experiment begins. The same output fingerprint is checked on every resume.
    _publication_preflight(config, paths; root=root)
    suite_manifest = save_run_manifest(paths, "publication_suite", config;
        extras=Dict("source_root" => abspath(root), "resume" => resume))
    stages = expected_publication_stages(config)
    total = length(stages) - 1
    completed = 0
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="preflight complete")

    run_legacy_audit_publication(config, paths; root=root, resume=resume)
    completed += 1
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="legacy audit complete")
    run_synthetic_publication(config, paths; resume=resume)
    completed += 1
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="expanded synthetic experiment complete")

    manuscript = _cfg_section(config, "manuscript_sweeps")
    if Bool(_cfg_get(manuscript, "enabled", true))
        run_manuscript_sweeps_publication(config, paths; resume=resume)
        completed += 1
        update_progress(paths, "publication_suite"; completed=completed, total=total,
            message="corrected manuscript sweeps complete")
    end
    meta = _cfg_section(config, "synthetic_meta")
    if Bool(_cfg_get(meta, "enabled", true))
        run_synthetic_meta_publication(config, paths; resume=resume)
        completed += 1
        update_progress(paths, "publication_suite"; completed=completed, total=total,
            message="modern meta-learner comparison complete")
    end

    run_real_world_publication(config, paths; root=root, resume=resume)
    real_count = length(string.(_cfg_get(_cfg_section(config, "real_world"), "datasets", String[])))
    completed += real_count + (real_count > 0 ? 1 : 0)
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="real-world datasets complete")

    run_sensitivity_publication(config, paths; root=root, resume=resume)
    sensitivity_count = length(string.(_cfg_get(_cfg_section(config, "sensitivity"), "datasets", String[])))
    completed += sensitivity_count + (sensitivity_count > 0 ? 1 : 0)
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="sensitivity analyses complete")

    run_stress_publication(config, paths; resume=resume)
    completed += 1
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="stress tests complete")
    run_solver_verification_publication(config, paths; resume=resume)
    completed += 1
    update_progress(paths, "publication_suite"; completed=completed, total=total,
        message="solver/theory verification complete")
    run_runtime_publication(config, paths; resume=resume)
    completed += 1

    required_stages = [stage for stage in stages if stage != "publication_suite"]
    invalid = [stage for stage in required_stages if !stage_complete(paths, stage)]
    isempty(invalid) || error("Cannot complete publication suite; invalid stages: $(join(invalid, ", "))")
    inventory = DataFrame(
        stage=required_stages,
        marker=[relpath(stage_marker(paths, stage), paths.root) for stage in required_stages],
        marker_sha256=[sha256_file(stage_marker(paths, stage)) for stage in required_stages],
    )
    inventory_path = joinpath(paths.manifests, "publication_stage_inventory.csv")
    atomic_csv_write(inventory_path, inventory)
    completion_path = joinpath(paths.manifests, "publication_completion.toml")
    atomic_toml_write(completion_path, Dict{String,Any}(
        "completed_at_utc" => string(Dates.now(Dates.UTC)),
        "run_fingerprint" => current_run_fingerprint(paths),
        "profile" => string(_cfg_get(_cfg_section(config, "run"), "profile", "unknown")),
        "configured_stages" => stages,
        "verified_stages" => required_stages,
        "scope" => Dict{String,Any}(string(k) => v for (k, v) in _cfg_section(config, "scope")),
        "paper_scope_note" => string(_cfg_get(_cfg_section(config, "scope"), "reason",
            "Only the experiments explicitly configured for this run are certified.")),
    ))
    update_progress(paths, "publication_suite"; completed=total, total=total,
        message="all configured publication stages verified")
    mark_stage_complete(paths, "publication_suite"; extras=Dict(
        "configured_stages" => stages, "verified_stage_count" => length(required_stages)),
        outputs=[suite_manifest, joinpath(paths.manifests, "preflight.toml"),
                 joinpath(paths.manifests, "preflight_datasets.csv"),
                 inventory_path, completion_path])
    return paths
end
