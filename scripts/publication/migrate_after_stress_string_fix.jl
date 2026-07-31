#!/usr/bin/env julia

using ArgParse
using CSV
using DataFrames
using Dates
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO_ROOT, "src", "AdaptiveEnsemblePublication.jl"))
const AEP = AdaptiveEnsemblePublication

const EXPECTED_SOURCE_FINGERPRINT =
    "0e9320deaf6da755c152474c1e96dc22bc90719f90b27213cb523f4aade21c9a"

const COMPLETED_STAGES = [
    "legacy_audit",
    "synthetic_full",
    "manuscript_sweeps",
    "synthetic_meta",
    "real_world_safi",
    "real_world_all",
    "sensitivity_safi",
    "sensitivity_all",
]

function parse_arguments()
    settings = ArgParseSettings(
        description="Migrate fingerprint- and hash-verified completed stages plus all 420 completed stress tasks after the heterogeneous-setting String fix",
    )
    @add_arg_table! settings begin
        "--source-output"
            help = "failed source output containing completed stages and 420 stress task markers"
            arg_type = String
            required = true
        "--output"
            help = "new, absent or empty output directory"
            arg_type = String
            required = true
        "--config"
            help = "corrected synthetic+Safi publication configuration"
            arg_type = String
            default = joinpath("config", "publication_full.toml")
        "--root"
            help = "corrected repository root"
            arg_type = String
            default = REPO_ROOT
    end
    return parse_args(settings)
end

function normalized_relative_path(value::AbstractString)
    return split(replace(String(value), '\\' => '/'), '/')
end

function rooted_path(root::AbstractString, relative::AbstractString)
    isabspath(relative) && error("marker output must be relative: $(relative)")
    parts = normalized_relative_path(relative)
    isempty(parts) && error("empty marker output path")
    any(part -> part == "..", parts) && error("marker output escapes root: $(relative)")
    candidate = normpath(joinpath(root, parts...))
    relationship = relpath(candidate, root)
    (relationship == ".." || startswith(relationship, "../") ||
     startswith(relationship, "..\\")) && error("marker output escapes root: $(relative)")
    return candidate
end

function marker_extras(marker::AbstractDict)
    reserved = Set(["stage", "task", "completed_at_utc", "run_fingerprint",
                    "outputs", "output_sha256"])
    return Dict{String,Any}(String(k) => v for (k, v) in marker
                            if !(String(k) in reserved))
end

function validate_marker_file(marker_path::String, source_root::String,
                              source_fingerprint::String; expected_stage=nothing,
                              expected_task=nothing)
    isfile(marker_path) || error("missing source marker: $(marker_path)")
    marker = TOML.parsefile(marker_path)
    expected_stage === nothing || String(get(marker, "stage", "")) == expected_stage ||
        error("stage mismatch in $(marker_path)")
    expected_task === nothing || String(get(marker, "task", "")) == expected_task ||
        error("task mismatch in $(marker_path)")
    String(get(marker, "run_fingerprint", "")) == source_fingerprint ||
        error("source marker belongs to another fingerprint: $(marker_path)")
    outputs = String.(get(marker, "outputs", String[]))
    hashes = String.(get(marker, "output_sha256", String[]))
    !isempty(outputs) || error("source marker has no outputs: $(marker_path)")
    length(outputs) == length(hashes) || error("output/hash length mismatch: $(marker_path)")
    for (relative, expected_hash) in zip(outputs, hashes)
        output = rooted_path(source_root, relative)
        isfile(output) || error("source output is missing: $(output)")
        filesize(output) > 0 || error("source output is empty: $(output)")
        AEP.sha256_file(output) == expected_hash || error("source output hash mismatch: $(output)")
    end
    return marker
end

function copy_marker_outputs(source_root::String, target_root::String,
                             marker::AbstractDict)
    copied = String[]
    for relative in String.(marker["outputs"])
        source = rooted_path(source_root, relative)
        target = rooted_path(target_root, relative)
        mkpath(dirname(target))
        cp(source, target; force=true)
        push!(copied, target)
    end
    return copied
end

function require_exact_grid(table::DataFrame, columns::Vector{Symbol},
                            expected_rows::Int, label::String)
    all(column -> column in propertynames(table), columns) ||
        error("$(label) lacks one or more key columns")
    nrow(table) == expected_rows ||
        error("$(label) has $(nrow(table)) rows; expected $(expected_rows)")
    nrow(unique(table[:, columns])) == expected_rows ||
        error("$(label) contains duplicate or missing key combinations")
end

function validate_completed_stage(source_paths::AEP.RunPaths,
                                  stage::String, config::AbstractDict,
                                  source_fingerprint::String)
    marker = validate_marker_file(AEP.stage_marker(source_paths, stage),
        source_paths.root, source_fingerprint; expected_stage=stage)
    if stage == "synthetic_full"
        section = config["synthetic"]
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "synthetic_regimes_per_seed.csv")))
        require_exact_grid(table, [:regime, :seed, :method],
            Int(section["seeds"]) * length(section["regimes"]) * length(section["methods"]),
            "expanded synthetic results")
    elseif stage == "manuscript_sweeps"
        section = config["manuscript_sweeps"]
        task_count = Int(section["seeds"]) * sum(length(section[key]) for key in
            ("member_sizes", "drift_levels", "window_sizes", "history_sizes",
             "discrete_probabilities"))
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "manuscript_sweeps_per_seed.csv")))
        require_exact_grid(table, [:sweep, :setting, :seed, :method],
            task_count * length(section["methods"]), "manuscript sweeps")
    elseif stage == "synthetic_meta"
        section = config["synthetic_meta"]
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "synthetic_meta_per_seed.csv")))
        require_exact_grid(table, [:regime, :seed, :method],
            Int(section["seeds"]) * length(section["regimes"]) * length(section["methods"]),
            "synthetic meta comparison")
    elseif stage == "real_world_safi"
        table = DataFrame(CSV.File(joinpath(source_paths.tables, "safi_metrics.csv")))
        methods = String.(config["real_world"]["methods"])
        Set(String.(table.method)) == Set(methods) ||
            error("Safi method set differs from active configuration")
        nrow(table) == length(methods) || error("Safi metrics contain duplicate rows")
    elseif stage == "real_world_all"
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "real_world_all_metrics.csv")))
        all(String.(table.dataset) .== "safi") ||
            error("real-world aggregate contains a dataset outside the Safi-only scope")
    elseif stage == "sensitivity_safi"
        section = config["sensitivity"]
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "sensitivity_safi.csv")))
        require_exact_grid(table, [:dataset, :objective_variant, :tau, :lambda],
            length(section["datasets"]) * length(section["objective_variants"]) *
            length(section["tau"]) * length(section["lambda"]),
            "Safi sensitivity")
    elseif stage == "sensitivity_all"
        table = DataFrame(CSV.File(joinpath(source_paths.tables,
            "sensitivity_all.csv")))
        all(String.(table.dataset) .== "safi") ||
            error("sensitivity aggregate contains a dataset outside the Safi-only scope")
    end
    return marker
end

function stress_tasks(config::AbstractDict)
    section = config["stress"]
    seeds = Int(section["seeds"])
    tasks = NamedTuple[]
    for history in Int.(section["history_sizes"]), seed in 1:seeds
        push!(tasks, (analysis="limited_history", setting=string(history), seed=seed))
    end
    for rho in Float64.(section["correlations"]), seed in 1:seeds
        push!(tasks, (analysis="correlation", setting=string(rho), seed=seed))
    end
    for seed in 1:seeds
        push!(tasks, (analysis="boundary", setting="all", seed=seed))
    end
    return tasks
end

function validate_stress_output(task, paths::AEP.RunPaths, config::AbstractDict)
    metrics_path = AEP._stress_checkpoint(paths, task.analysis, task.setting, task.seed)
    metrics = DataFrame(CSV.File(metrics_path))
    all(isfinite, Float64.(metrics.rmse)) || error("non-finite stress RMSE: $(metrics_path)")
    methods = Set(String.(config["stress"]["methods"]))
    observed = Set(String.(metrics.method))
    if task.analysis == "limited_history"
        observed == methods || error("limited-history method set mismatch: $(metrics_path)")
        nrow(metrics) == length(methods) || error("limited-history row count mismatch: $(metrics_path)")
    elseif task.analysis == "correlation"
        observed == union(methods, Set(["adaptive_ridge_pruned"])) ||
            error("correlation method set mismatch: $(metrics_path)")
        pruning_path = AEP._stress_checkpoint(
            paths, "correlation_pruning", task.setting, task.seed)
        pruning = DataFrame(CSV.File(pruning_path))
        nrow(pruning) == length(config["stress"]["correlation_pruning_thresholds"]) ||
            error("correlation-pruning grid mismatch: $(pruning_path)")
        all(isfinite, Float64.(pruning.validation_score)) ||
            error("non-finite correlation-pruning score: $(pruning_path)")
    elseif task.analysis == "boundary"
        expected = Set([
            "adaptive_ridge__reset", "rls__reset", "hedge__reset",
            "passive_aggressive__reset", "adaptive_ridge__borrow_across_events",
            "rls__carry", "hedge__carry", "passive_aggressive__carry",
        ])
        observed == expected || error("boundary method set mismatch: $(metrics_path)")
        nrow(metrics) == length(expected) || error("boundary row count mismatch: $(metrics_path)")
    end
end

function main()
    args = parse_arguments()
    root = abspath(args["root"])
    source_root = abspath(args["source-output"])
    target_root = abspath(args["output"])
    config_path = abspath(args["config"])

    isdir(source_root) || error("source output directory not found: $(source_root)")
    source_root == target_root && error("source and target output directories must differ")
    if ispath(target_root)
        isdir(target_root) || error("migration target exists but is not a directory: $(target_root)")
        isempty(readdir(target_root)) || error("migration target must be absent or empty: $(target_root)")
    end

    config = AEP.load_config(config_path)
    String(get(get(config, "scope", Dict{String,Any}()), "label", "")) == "synthetic_safi" ||
        error("migration requires the synthetic_safi profile")
    String.(config["real_world"]["datasets"]) == ["safi"] ||
        error("migration target must configure Safi as its only real-world dataset")

    source_paths = AEP.ensure_run_paths(source_root)
    source_identity = AEP.current_run_identity(source_paths)
    source_fingerprint = String(source_identity["run_fingerprint"])
    source_fingerprint == EXPECTED_SOURCE_FINGERPRINT || error(
        "unexpected source fingerprint $(source_fingerprint); expected $(EXPECTED_SOURCE_FINGERPRINT)")

    stage_markers = Dict{String,Any}()
    for stage in COMPLETED_STAGES
        stage_markers[stage] = validate_completed_stage(
            source_paths, stage, config, source_fingerprint)
    end

    tasks = stress_tasks(config)
    length(tasks) == 420 || error("expected 420 stress tasks, found $(length(tasks))")
    task_markers = Dict{String,Any}()
    for task in tasks
        task_name = AEP._stress_task_name(task.analysis, task.setting, task.seed)
        marker = validate_marker_file(AEP.task_marker(source_paths, task_name),
            source_paths.root, source_fingerprint; expected_task=task_name)
        validate_stress_output(task, source_paths, config)
        task_markers[task_name] = marker
    end

    target_paths = AEP.ensure_run_paths(target_root)
    AEP.initialize_run_identity(target_paths, root, config_path, config;
        command="migrate_after_stress_string_fix")
    target_fingerprint = AEP.current_run_fingerprint(target_paths)
    target_fingerprint != source_fingerprint ||
        error("corrected code unexpectedly has the same fingerprint as the source run")

    copied_stage_outputs = 0
    for stage in COMPLETED_STAGES
        marker = stage_markers[stage]
        outputs = copy_marker_outputs(source_paths.root, target_paths.root, marker)
        copied_stage_outputs += length(outputs)
        AEP.mark_stage_complete(target_paths, stage;
            extras=marker_extras(marker), outputs=outputs)
    end

    copied_task_outputs = 0
    for task in tasks
        task_name = AEP._stress_task_name(task.analysis, task.setting, task.seed)
        marker = task_markers[task_name]
        outputs = copy_marker_outputs(source_paths.root, target_paths.root, marker)
        copied_task_outputs += length(outputs)
        AEP.mark_task_complete(target_paths, task_name;
            outputs=outputs, extras=marker_extras(marker))
    end

    AEP.update_progress(target_paths, "stress_full_tasks";
        completed=length(tasks), total=length(tasks),
        message="migrated 420 hash-verified stress tasks; aggregate summary and figures pending")
    AEP.update_progress(target_paths, "publication_suite";
        completed=length(COMPLETED_STAGES),
        total=length(AEP.expected_publication_stages(config)) - 1,
        message="migrated completed stages and stress tasks; stress aggregation, solver verification, runtime pending")

    report = Dict{String,Any}(
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "source_output" => source_root,
        "target_output" => target_root,
        "source_run_fingerprint" => source_fingerprint,
        "target_run_fingerprint" => target_fingerprint,
        "migrated_stages" => COMPLETED_STAGES,
        "migrated_stage_outputs" => copied_stage_outputs,
        "migrated_stress_tasks" => length(tasks),
        "migrated_stress_task_outputs" => copied_task_outputs,
        "stress_stage_marker_migrated" => false,
        "next_stage" => "stress_full aggregation (no stress task recomputation)",
        "excluded_artifacts" => [
            "old failure logs", "old receipts", "old wrapper markers",
            "incomplete stress stage marker", "serialized model objects",
        ],
        "scientific_justification" => "Every migrated stage and stress task was bound to the exact source fingerprint and every referenced output was re-hashed before copying. The failed heterogeneous-setting aggregation was not migrated.",
    )
    AEP.atomic_toml_write(joinpath(target_paths.manifests,
        "stress_string_fix_migration.toml"), report)

    println("STRESS-STRING-FIX RESULT MIGRATION PASSED")
    println("Source fingerprint: ", source_fingerprint)
    println("Target fingerprint: ", target_fingerprint)
    println("Migrated completed stages: ", join(COMPLETED_STAGES, ", "))
    println("Migrated stress tasks: ", length(tasks), " / 420")
    println("Next work: aggregate stress tables/figures, run solver verification, runtime, and final verifier")
end

main()
