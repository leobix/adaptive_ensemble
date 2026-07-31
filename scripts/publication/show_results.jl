#!/usr/bin/env julia

using CSV
using DataFrames
using TOML

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUTPUT = abspath(length(ARGS) >= 1 ? ARGS[1] : joinpath(REPO, "publication_artifacts"))

function show_dataframe(path::AbstractString, title::AbstractString;
                        columns::Vector{Symbol}=Symbol[], rows::Int=80,
                        sort_columns::Vector{Symbol}=Symbol[])
    if !isfile(path)
        println("\n", title, "\n  not available yet: ", path)
        return
    end
    df = DataFrame(CSV.File(path))
    available = isempty(columns) ? Symbol.(names(df)) : [column for column in columns if column in propertynames(df)]
    view = isempty(available) ? df : select(df, available)
    if !isempty(sort_columns) && all(column -> column in propertynames(view), sort_columns)
        sort!(view, sort_columns)
    end
    println("\n", title, "  [", nrow(df), " rows]\n", repeat("-", 80))
    show(stdout, MIME("text/plain"), first(view, min(rows, nrow(view))); allrows=true, allcols=true, truncate=120)
    println()
end

println("Publication output: ", OUTPUT)
if !isdir(OUTPUT)
    error("output directory does not exist. Start the publication suite first.")
end

checkpoint_dir = joinpath(OUTPUT, "checkpoints")
completed = String[]
if isdir(checkpoint_dir)
    for path in filter(path -> endswith(path, ".done.toml"), readdir(checkpoint_dir; join=true))
        marker = TOML.parsefile(path)
        push!(completed, string(get(marker, "stage", basename(path))))
    end
end
println("Completed stages: ", isempty(completed) ? "none" : join(sort!(unique(completed)), ", "))

verification_path = joinpath(OUTPUT, "manifests", "artifact_verification.toml")
completion_path = joinpath(OUTPUT, "manifests", "publication_completion.toml")
if isfile(verification_path)
    verification = TOML.parsefile(verification_path)
    println("Configured scope: ", get(verification, "scope_label", "unknown"))
    println("Configured scope complete: ", get(verification, "configured_scope_complete", false))
    println("Historical paper application scope complete: ",
        get(verification, "paper_application_scope_complete", false))
elseif isfile(completion_path)
    completion = TOML.parsefile(completion_path)
    scope = get(completion, "scope", Dict{String,Any}())
    println("Configured scope: ", get(scope, "label", "unknown"))
    println("Scope note: ", get(completion, "paper_scope_note", ""))
end

show_dataframe(joinpath(OUTPUT, "tables", "real_world_all_metrics.csv"),
    "Real-world held-out metrics";
    columns=[:dataset, :method, :n, :mae, :rmse, :cvar_05, :cvar_15, :smape, :wape],
    sort_columns=[:dataset, :rmse])

show_dataframe(joinpath(OUTPUT, "tables", "synthetic_regimes_summary.csv"),
    "Synthetic regime summary across seeds";
    columns=[:regime, :method, :replications, :rmse_mean, :rmse_ci_low, :rmse_ci_high,
             :cvar_05_mean, :cvar_05_ci_low, :cvar_05_ci_high],
    sort_columns=[:regime, :rmse_mean])

show_dataframe(joinpath(OUTPUT, "tables", "synthetic_paired_rmse_intervals.csv"),
    "Paired synthetic RMSE contrasts (method minus Adaptive Ridge)";
    columns=[:regime, :reference, :method, :n, :mean_method_minus_reference,
             :ci_low, :ci_high, :method_win_rate],
    sort_columns=[:regime, :mean_method_minus_reference])

show_dataframe(joinpath(OUTPUT, "tables", "runtime_benchmarks.csv"),
    "Runtime benchmarks";
    rows=120)

show_dataframe(joinpath(OUTPUT, "tables", "solver_objective_verification.csv"),
    "Quadratic, IRLS robust-norm, and exact Gurobi verification";
    rows=120)

figure_dir = joinpath(OUTPUT, "figures")
if isdir(figure_dir)
    figures = sort(filter(path -> endswith(lowercase(path), ".svg"), readdir(figure_dir; join=true)))
    println("\nFigures\n", repeat("-", 80))
    isempty(figures) ? println("No figures available yet.") : foreach(path -> println(path), figures)
end
