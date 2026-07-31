#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using TOML

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "src", "AdaptiveEnsemblePublication.jl"))
using .AdaptiveEnsemblePublication
const AEP = AdaptiveEnsemblePublication

function parse_options(args)
    options = Dict{String,String}(
        "root" => REPO,
        "config" => "config/publication_full.toml",
        "output" => "publication_artifacts",
    )
    index = 1
    while index <= length(args)
        flag = args[index]
        flag in ("--root", "--config", "--output") ||
            throw(ArgumentError("unknown option $(flag)"))
        index < length(args) || throw(ArgumentError("missing value after $(flag)"))
        options[replace(flag, "--" => "")] = args[index + 1]
        index += 2
    end
    return options
end

function resolve_from_root(root::String, value::String)
    return isabspath(value) ? normpath(value) : normpath(joinpath(root, value))
end

function require_file(path::String)
    isfile(path) || error("required artifact is missing: $(path)")
    filesize(path) > 0 || error("required artifact is empty: $(path)")
    return path
end

function require_columns(table::DataFrame, columns::Vector{Symbol}, label::String)
    missing_columns = [column for column in columns if !(column in propertynames(table))]
    isempty(missing_columns) || error("$(label) is missing columns: $(join(string.(missing_columns), ", "))")
end

function finite_column(table::DataFrame, column::Symbol)
    column in propertynames(table) || return false
    for value in table[!, column]
        value === missing && return false
        value isa Real || return false
        isfinite(Float64(value)) || return false
    end
    return true
end

function require_exact_grid(table::DataFrame, dimensions::Vector{Symbol}, expected_rows::Int, label::String)
    require_columns(table, dimensions, label)
    nrow(table) == expected_rows || error(
        "$(label) has $(nrow(table)) rows; expected $(expected_rows)")
    unique_rows = unique(table[:, dimensions])
    nrow(unique_rows) == expected_rows || error(
        "$(label) has duplicate or missing combinations: $(nrow(unique_rows)) unique of $(expected_rows)")
end

function configured_methods(config, section::String)
    return string.(get(get(config, section, Dict{String,Any}()), "methods", String[]))
end

function verify_synthetic(config, paths)
    section = config["synthetic"]
    regimes = string.(section["regimes"])
    seeds = Int(section["seeds"])
    methods = string.(section["methods"])
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "synthetic_regimes_per_seed.csv"))))
    require_exact_grid(table, [:regime, :seed, :method],
        length(regimes) * seeds * length(methods), "expanded synthetic results")
    Set(string.(table.regime)) == Set(regimes) || error("synthetic regime set differs from configuration")
    Set(Int.(table.seed)) == Set(1:seeds) || error("synthetic seed set differs from configuration")
    Set(string.(table.method)) == Set(methods) || error("synthetic method set differs from configuration")
    finite_column(table, :rmse) || error("synthetic RMSE contains missing/non-finite values")
    finite_column(table, :cvar_05) || error("synthetic CVaR contains missing/non-finite values")
    return nrow(table)
end

function verify_manuscript_sweeps(config, paths)
    section = get(config, "manuscript_sweeps", Dict{String,Any}())
    Bool(get(section, "enabled", true)) || return 0
    seeds = Int(section["seeds"])
    methods = string.(section["methods"])
    settings = Dict(
        "members" => section["member_sizes"],
        "drift" => section["drift_levels"],
        "window" => section["window_sizes"],
        "history" => section["history_sizes"],
        "discrete_probability" => section["discrete_probabilities"],
    )
    task_count = sum(length(value) for value in values(settings)) * seeds
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "manuscript_sweeps_per_seed.csv"))))
    require_exact_grid(table, [:sweep, :setting, :seed, :method],
        task_count * length(methods), "manuscript sweeps")
    finite_column(table, :rmse) || error("manuscript-sweep RMSE contains missing/non-finite values")
    return nrow(table)
end

function verify_synthetic_meta(config, paths)
    section = get(config, "synthetic_meta", Dict{String,Any}())
    Bool(get(section, "enabled", true)) || return 0
    regimes = string.(section["regimes"])
    seeds = Int(section["seeds"])
    methods = string.(section["methods"])
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "synthetic_meta_per_seed.csv"))))
    require_exact_grid(table, [:regime, :seed, :method],
        length(regimes) * seeds * length(methods), "synthetic meta-learning comparison")
    finite_column(table, :rmse) || error("synthetic-meta RMSE contains missing/non-finite values")
    return nrow(table)
end

function verify_real_world(config, paths)
    section = config["real_world"]
    datasets = string.(section["datasets"])
    methods = string.(section["methods"])
    required_method_set = Set(methods)
    total = 0
    for dataset in datasets
        table = DataFrame(CSV.File(require_file(joinpath(paths.tables, dataset * "_metrics.csv"))))
        require_columns(table, [:dataset, :method, :rmse, :mae, :cvar_05], dataset * " metrics")
        nrow(table) > 0 || error("$(dataset) metrics table is empty")
        dataset_values = Set(string.(table.dataset))
        dataset_values == Set([dataset]) || error(
            "$(dataset) metrics table contains unexpected dataset labels: $(collect(dataset_values))")

        observed_methods = string.(table.method)
        length(unique(observed_methods)) == length(observed_methods) || error(
            "$(dataset) metrics table contains duplicate method rows")
        observed_method_set = Set(observed_methods)
        missing_methods = setdiff(required_method_set, observed_method_set)
        isempty(missing_methods) || error(
            "$(dataset) is missing configured methods: $(collect(missing_methods))")

        # Hurricane files may contain operational benchmark forecasts (for example,
        # FSSE and OFCL) in addition to the configured ensemble algorithms. Those rows
        # are deliberately prefixed with `benchmark_` and are valid diagnostics, but no
        # other undeclared method rows are accepted.
        extras = setdiff(observed_method_set, required_method_set)
        invalid_extras = [name for name in extras if !startswith(name, "benchmark_")]
        isempty(invalid_extras) || error(
            "$(dataset) has undeclared non-benchmark methods: $(sort(invalid_extras))")

        finite_column(table, :rmse) || error("$(dataset) RMSE contains missing/non-finite values")

        predictions = DataFrame(CSV.File(require_file(joinpath(paths.predictions,
            dataset * "_predictions.csv"))))
        require_columns(predictions, [:time_index, :target], dataset * " predictions")
        prediction_methods = Set(string(name) for name in propertynames(predictions)
                                 if !(name in (:time_index, :target)))
        missing_prediction_methods = setdiff(required_method_set, prediction_methods)
        isempty(missing_prediction_methods) || error(
            "$(dataset) predictions are missing configured methods: $(collect(missing_prediction_methods))")

        bootstrap = DataFrame(CSV.File(require_file(joinpath(paths.tables,
            dataset * "_moving_block_bootstrap.csv"))))
        require_columns(bootstrap,
            [:reference, :method, :metric, :reps, :block_length, :resampling_scheme,
             :mean_method_minus_reference, :ci_low, :ci_high],
            dataset * " bootstrap")
        expected_bootstrap_methods = setdiff(prediction_methods, Set(["adaptive_ridge"]))
        expected_metrics = Set(["rmse", "mae", "cvar_05"])
        require_exact_grid(bootstrap, [:method, :metric],
            length(expected_bootstrap_methods) * length(expected_metrics),
            dataset * " bootstrap")
        Set(string.(bootstrap.method)) == expected_bootstrap_methods || error(
            "$(dataset) bootstrap method set differs from saved prediction columns")
        Set(string.(bootstrap.metric)) == expected_metrics || error(
            "$(dataset) bootstrap metric set is incomplete")
        all(Int.(bootstrap.reps) .== Int(section["bootstrap_reps"])) || error(
            "$(dataset) bootstrap replication count differs from configuration")
        finite_column(bootstrap, :mean_method_minus_reference) || error(
            "$(dataset) bootstrap mean contrasts contain missing/non-finite values")
        finite_column(bootstrap, :ci_low) || error(
            "$(dataset) bootstrap lower intervals contain missing/non-finite values")
        finite_column(bootstrap, :ci_high) || error(
            "$(dataset) bootstrap upper intervals contain missing/non-finite values")
        if startswith(dataset, "hurricane_")
            all(string.(bootstrap.resampling_scheme) .== "within_event_moving_blocks") ||
                error("$(dataset) bootstrap crossed event boundaries or is mislabeled")
        end
        total += nrow(table)
    end
    aggregate = DataFrame(CSV.File(require_file(joinpath(paths.tables, "real_world_all_metrics.csv"))))
    nrow(aggregate) == total || error("real-world aggregate row count mismatch")
    return total
end

function verify_sensitivity(config, paths)
    section = config["sensitivity"]
    datasets = string.(section["datasets"])
    variants = string.(section["objective_variants"])
    taus = Int.(section["tau"])
    lambdas = Float64.(section["lambda"])
    expected_per_dataset = length(variants) * length(taus) * length(lambdas)
    total = 0
    for dataset in datasets
        table = DataFrame(CSV.File(require_file(joinpath(paths.tables,
            "sensitivity_$(dataset).csv"))))
        require_exact_grid(table, [:dataset, :objective_variant, :tau, :lambda],
            expected_per_dataset, "$(dataset) sensitivity")
        finite_column(table, :validation_rmse) || error("$(dataset) sensitivity RMSE is non-finite")
        total += nrow(table)
    end
    aggregate = DataFrame(CSV.File(require_file(joinpath(paths.tables, "sensitivity_all.csv"))))
    nrow(aggregate) == total || error("sensitivity aggregate row count mismatch")
    return total
end

function verify_stress(config, paths)
    section = config["stress"]
    seeds = Int(section["seeds"])
    histories = string.(section["history_sizes"])
    correlations = string.(Float64.(section["correlations"]))
    methods = string.(section["methods"])
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "stress_tests_per_seed.csv"))))
    require_columns(table, [:analysis, :setting, :seed, :method, :rmse], "stress tests")
    for (analysis, settings) in (("limited_history", histories), ("correlation", correlations))
        subset = table[table.analysis .== analysis, :]
        for setting in settings, seed in 1:seeds
            cell = subset[(string.(subset.setting) .== setting) .& (Int.(subset.seed) .== seed), :]
            missing_methods = setdiff(Set(methods), Set(string.(cell.method)))
            isempty(missing_methods) || error(
                "stress $(analysis), setting=$(setting), seed=$(seed) missing methods $(collect(missing_methods))")
        end
    end
    boundary = table[table.analysis .== "boundary", :]
    Set(Int.(boundary.seed)) == Set(1:seeds) || error("boundary stress test is missing seeds")
    finite_column(table, :rmse) || error("stress-test RMSE contains missing/non-finite values")
    return nrow(table)
end

function verify_solver(config, paths)
    section = config["verification"]
    run_comparison = Bool(get(section, "run_exact_norm_vs_quadratic", true))
    run_literal = Bool(get(section, "run_literal_jump_equivalence", true))
    require_gurobi = Bool(get(section, "require_gurobi", true))
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "solver_objective_verification.csv"))))
    if run_comparison
        expected = length(section["regimes"]) * length(section["seeds"]) *
            length(section["tau"]) * length(section["lambda"])
        require_exact_grid(table, [:regime, :seed, :tau, :lambda], expected, "solver verification")
        finite_column(table, :quadratic_validation_rmse) || error("quadratic verification RMSE is non-finite")
        finite_column(table, :irls_norm_validation_rmse) || error("IRLS verification RMSE is non-finite")
        all(Bool.(table.irls_converged)) || error("one or more IRLS verification fits did not converge")
        if require_gurobi
            finite_column(table, :exact_norm_validation_rmse) || error("exact Gurobi verification RMSE is non-finite")
            finite_column(table, :irls_vs_exact_relative_objective) || error("IRLS/Gurobi objective comparison is non-finite")
        end
    end
    if run_literal && require_gurobi
        literal = DataFrame(CSV.File(require_file(joinpath(paths.tables,
            "solver_literal_jump_equivalence.csv"))))
        nrow(literal) == 1 || error("literal JuMP equivalence table must contain one row")
    end
    return nrow(table)
end

function verify_runtime(paths)
    table = DataFrame(CSV.File(require_file(joinpath(paths.tables, "runtime_benchmarks.csv"))))
    require_columns(table, [:kind, :setting, :median_seconds], "runtime benchmark")
    required_kinds = Set(["adaptive_fit_samples", "adaptive_fit_window", "adaptive_fit_members",
        "regularization_path", "validation_search", "method_fit_plus_roll", "adaptive_prediction"])
    missing_kinds = setdiff(required_kinds, Set(string.(table.kind)))
    isempty(missing_kinds) || error("runtime benchmark is missing kinds: $(collect(missing_kinds))")
    finite_column(table, :median_seconds) || error("runtime table contains non-finite medians")
    equivalence = DataFrame(CSV.File(require_file(joinpath(paths.tables,
        "runtime_numerical_equivalence.csv"))))
    nrow(equivalence) >= 2 || error("runtime numerical-equivalence table is incomplete")
    return nrow(table)
end

function verify_no_failures(paths)
    failures = String[]
    for directory in (paths.logs, paths.checkpoints)
        isdir(directory) || continue
        for (root, _, files) in walkdir(directory)
            for filename in files
                lower = lowercase(filename)
                failed_name = endswith(lower, "_failures.csv") ||
                              endswith(lower, ".failure.txt") ||
                              endswith(lower, ".failure.log") ||
                              endswith(lower, ".error.txt") ||
                              endswith(lower, ".error.log")
                if failed_name
                    path = joinpath(root, filename)
                    filesize(path) > 0 && push!(failures, path)
                end
            end
        end
    end
    isempty(failures) || error("failure/error artifacts remain: $(join(failures, ", "))")
end

function main(args=ARGS)
    options = parse_options(args)
    root = abspath(options["root"])
    config_path = resolve_from_root(root, options["config"])
    output = resolve_from_root(root, options["output"])
    config = AEP.load_config(config_path)
    AEP._validate_publication_runtime(config, "verify")
    paths = AEP.ensure_run_paths(output)
    AEP.initialize_run_identity(paths, root, config_path, config; command="verify")

    stages = AEP.expected_publication_stages(config)
    invalid = [stage for stage in stages if !AEP.stage_complete(paths, stage)]
    isempty(invalid) || error("missing/invalid stage markers: $(join(invalid, ", "))")
    verify_no_failures(paths)

    counts = Dict{String,Any}(
        "synthetic_rows" => verify_synthetic(config, paths),
        "manuscript_sweep_rows" => verify_manuscript_sweeps(config, paths),
        "synthetic_meta_rows" => verify_synthetic_meta(config, paths),
        "real_world_rows" => verify_real_world(config, paths),
        "sensitivity_rows" => verify_sensitivity(config, paths),
        "stress_rows" => verify_stress(config, paths),
        "solver_rows" => verify_solver(config, paths),
        "runtime_rows" => verify_runtime(paths),
    )
    real_datasets = string.(config["real_world"]["datasets"])
    sensitivity_datasets = string.(config["sensitivity"]["datasets"])
    scope = get(config, "scope", Dict{String,Any}())
    scope_label = string(get(scope, "label", "unspecified"))
    declared_excluded = string.(get(scope, "excluded", String[]))

    safi_included = "safi" in real_datasets
    energy_included = any(dataset -> startswith(dataset, "energy_"), real_datasets)
    hurricane_included = all(dataset in real_datasets for dataset in
        ("hurricane_north_atlantic", "hurricane_eastern_pacific"))
    paper_scope_complete = safi_included && energy_included && hurricane_included

    if scope_label == "synthetic_safi"
        real_datasets == ["safi"] || error(
            "synthetic_safi scope must configure only the Safi real-world dataset")
        sensitivity_datasets == ["safi"] || error(
            "synthetic_safi scope must configure only Safi sensitivity")
        required_exclusions = Set([
            "energy_historical_selected_10", "energy_all_available",
            "hurricane_north_atlantic", "hurricane_eastern_pacific",
        ])
        required_exclusions ⊆ Set(declared_excluded) || error(
            "synthetic_safi scope does not explicitly declare every omitted paper application")
    end

    manifest = Dict{String,Any}(
        "verified_at_utc" => string(Dates.now(Dates.UTC)),
        "run_fingerprint" => AEP.current_run_fingerprint(paths),
        "configured_stages" => stages,
        "configured_scope_complete" => true,
        "scope_label" => scope_label,
        "configured_real_world_datasets" => real_datasets,
        "configured_sensitivity_datasets" => sensitivity_datasets,
        "declared_excluded_datasets" => declared_excluded,
        "safi_application_included" => safi_included,
        "energy_application_included" => energy_included,
        "hurricane_applications_included" => hurricane_included,
        "paper_application_scope_complete" => paper_scope_complete,
        "scope_reason" => string(get(scope, "reason", "")),
        "counts" => counts,
    )
    output_path = joinpath(paths.manifests, "artifact_verification.toml")
    AEP.atomic_toml_write(output_path, manifest)
    println("PUBLICATION ARTIFACT VERIFICATION PASSED")
    println("Run fingerprint: ", AEP.current_run_fingerprint(paths))
    println("Verification manifest: ", output_path)
    if !paper_scope_complete
        println("Scope note: all configured synthetic + Safi experiments passed. " *
                "Energy and/or hurricane applications are intentionally outside this run; " *
                "do not retain paper claims that depend on those omitted applications.")
    end
    return nothing
end

main()
