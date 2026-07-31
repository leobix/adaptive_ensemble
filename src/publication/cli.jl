const PUBLICATION_STAGES = [
    "legacy_audit", "synthetic_full", "manuscript_sweeps", "synthetic_meta",
    "real_world_safi", "real_world_all", "sensitivity_safi", "sensitivity_all",
    "stress_full", "solver_verification", "runtime_full", "publication_suite",
]
function expected_publication_stages(config::AbstractDict)
    stages = String["legacy_audit", "synthetic_full"]
    manuscript = haskey(config, "manuscript_sweeps") ? config["manuscript_sweeps"] : Dict{String,Any}()
    Bool(get(manuscript, "enabled", true)) && push!(stages, "manuscript_sweeps")
    meta = haskey(config, "synthetic_meta") ? config["synthetic_meta"] : Dict{String,Any}()
    Bool(get(meta, "enabled", true)) && push!(stages, "synthetic_meta")

    real = haskey(config, "real_world") ? config["real_world"] : Dict{String,Any}()
    real_datasets = string.(get(real, "datasets", String[]))
    append!(stages, ["real_world_" * dataset for dataset in real_datasets])
    !isempty(real_datasets) && push!(stages, "real_world_all")

    sensitivity = haskey(config, "sensitivity") ? config["sensitivity"] : Dict{String,Any}()
    sensitivity_datasets = string.(get(sensitivity, "datasets", String[]))
    append!(stages, ["sensitivity_" * dataset for dataset in sensitivity_datasets])
    !isempty(sensitivity_datasets) && push!(stages, "sensitivity_all")

    append!(stages, ["stress_full", "solver_verification", "runtime_full", "publication_suite"])
    return unique(stages)
end

function _status_expected_stages(paths::RunPaths)
    manifest_path = joinpath(paths.manifests, "publication_suite.toml")
    if isfile(manifest_path)
        try
            manifest = TOML.parsefile(manifest_path)
            config = get(manifest, "config", Dict{String,Any}())
            !isempty(config) && return expected_publication_stages(config)
        catch
        end
    end
    return PUBLICATION_STAGES
end

function _parse_cli(args::Vector{String})
    isempty(args) && return Dict{String,Any}("command" => "help")
    options = Dict{String,Any}(
        "command" => args[1],
        "config" => "config/publication_full.toml",
        "output" => "publication_artifacts",
        "root" => ".",
        "resume" => true,
    )
    i = 2
    while i <= length(args)
        argument = args[i]
        if argument == "--no-resume"
            options["resume"] = false
            i += 1
        elseif argument in ("--config", "--output", "--root", "--dataset", "--member-set")
            i == length(args) && throw(ArgumentError("missing value after $(argument)"))
            options[replace(argument, "--" => "")] = args[i + 1]
            i += 2
        else
            throw(ArgumentError("unknown option $(argument)"))
        end
    end
    return options
end

function _print_help()
    println("""
Julia synthetic + Safi publication experiment runner

Main publication command:
  julia --project=. --threads=4 scripts/publication/run_publication_suite.jl suite \\
    --config config/publication_full.toml --output publication_artifacts --root .

Codex-safe Windows launch (Task Scheduler owns the process):
  powershell -ExecutionPolicy Bypass -File scripts/publication/start_codex_detached.ps1 \\
    -Threads 4 -Output publication_artifacts -Config config/publication_full.toml

Individual stages:
  ... run_publication_suite.jl preflight
  ... run_publication_suite.jl audit
  ... run_publication_suite.jl synthetic
  ... run_publication_suite.jl manuscript-sweeps
  ... run_publication_suite.jl synthetic-meta
  ... run_publication_suite.jl real-world
  ... run_publication_suite.jl sensitivity
  ... run_publication_suite.jl stress
  ... run_publication_suite.jl verification
  ... run_publication_suite.jl runtime
  ... run_publication_suite.jl status
  ... run_publication_suite.jl original-command --dataset synthetic|safi|energy|hurricane_na

Options:
  --config PATH       TOML configuration (default: config/publication_full.toml).
                      The active publication_full profiles intentionally exclude energy and hurricane data.
                      Use config/publication_full_no_gurobi.toml without a Gurobi license.
  --output PATH       output root (default: publication_artifacts)
  --root PATH         repository root (default: current directory)
  --dataset NAME      restrict real-world/sensitivity stage to one configured dataset
  --no-resume         recompute even when valid task/stage markers exist

An output directory is bound to a SHA-256 fingerprint of the effective configuration,
publication Julia sources, launch scripts, environment files, and supplied data. Use a new
--output directory after changing any of those inputs.
""")
end

function _status(paths::RunPaths)
    completed = String[]
    stale = String[]
    completion_times = Dict{String,String}()
    if isdir(paths.checkpoints)
        for marker in filter(file -> endswith(file, ".done.toml"), readdir(paths.checkpoints; join=true))
            data = try
                TOML.parsefile(marker)
            catch
                push!(stale, basename(marker))
                continue
            end
            stage = string(get(data, "stage", basename(marker)))
            if stage_complete(paths, stage)
                push!(completed, stage)
                completion_times[stage] = string(get(data, "completed_at_utc", ""))
            else
                push!(stale, stage)
            end
        end
    end
    completed = sort(unique(completed))
    expected = _status_expected_stages(paths)
    remaining = [stage for stage in expected if !(stage in completed)]
    identity = current_run_identity(paths; required=false)
    progress = Dict{String,Any}()
    heartbeat = joinpath(paths.manifests, "heartbeat.toml")
    isfile(heartbeat) && (progress = TOML.parsefile(heartbeat))
    failure_files = isdir(paths.logs) ? sort([
        relpath(path, paths.root) for path in readdir(paths.logs; join=true)
        if occursin("fail", lowercase(basename(path))) && filesize(path) > 0
    ]) : String[]
    status = Dict{String,Any}(
        "output_root" => paths.root,
        "run_fingerprint" => string(get(identity, "run_fingerprint", "")),
        "expected" => expected,
        "completed" => completed,
        "remaining" => remaining,
        "complete" => "publication_suite" in completed,
        "stale_or_invalid_markers" => sort(unique(stale)),
        "completion_times" => completion_times,
        "latest_progress" => progress,
        "failure_files" => failure_files,
    )
    println(sprint(io -> TOML.print(io, status)))
    return status
end

function _validate_publication_runtime(config::AbstractDict, command::String)
    command in ("status", "original-command") && return nothing
    VERSION == v"1.12.6" || error(
        "The locked publication environment requires Julia 1.12.6; found $(VERSION). " *
        "Install/select 1.12.6 with Juliaup before running experiments.")
    run_section = haskey(config, "run") ? config["run"] : Dict{String,Any}()
    required_threads = Int(get(run_section, "required_julia_threads", 0))
    if required_threads > 0 && Threads.nthreads() != required_threads
        error("This publication profile requires exactly $(required_threads) Julia threads; " *
              "found $(Threads.nthreads()). Relaunch with --threads=$(required_threads).")
    end
    return nothing
end

function _publication_preflight(config::AbstractDict, paths::RunPaths;
                                root::AbstractString=".")
    datasets = Set{String}()
    for section_name in ("real_world", "sensitivity")
        section = haskey(config, section_name) ? config[section_name] : Dict{String,Any}()
        for dataset in string.(get(section, "datasets", String[]))
            push!(datasets, dataset)
        end
    end
    scope = haskey(config, "scope") ? config["scope"] : Dict{String,Any}()
    if string(get(scope, "label", "")) == "synthetic_safi"
        forbidden = sort([dataset for dataset in datasets
            if startswith(dataset, "energy_") || startswith(dataset, "hurricane_")])
        isempty(forbidden) || error(
            "The synthetic_safi profile forbids energy/hurricane datasets: " *
            join(forbidden, ", "))
        datasets == Set(["safi"]) || error(
            "The synthetic_safi profile must configure Safi as its only data-backed application")
    end
    dataset_rows = DataFrame()
    for dataset in sort(collect(datasets))
        bundle = load_dataset(dataset, root)
        append!(dataset_rows, DataFrame(
            dataset=[bundle.name], observations=[length(bundle.y)], members=[size(bundle.X, 2)],
            train=[length(bundle.split.train)], validation=[length(bundle.split.validation)],
            test=[length(bundle.split.test)], availability_lag=[bundle.availability_lag],
        ); cols=:union)
    end
    atomic_csv_write(joinpath(paths.manifests, "preflight_datasets.csv"), dataset_rows)

    verification = _cfg_section(config, "verification")
    runtime = _cfg_section(config, "runtime")
    needs_gurobi = (Bool(_cfg_get(verification, "require_gurobi", true)) &&
        (Bool(_cfg_get(verification, "run_exact_norm_vs_quadratic", true)) ||
         Bool(_cfg_get(verification, "run_literal_jump_equivalence", true)))) ||
        Bool(_cfg_get(runtime, "run_gurobi_verification", true))
    gurobi_result = needs_gurobi ? _preflight_gurobi(config, "publication preflight") :
        Dict{String,Any}("status" => "not_required")
    manifest = Dict{String,Any}(
        "completed_at_utc" => string(Dates.now(Dates.UTC)),
        "run_fingerprint" => current_run_fingerprint(paths),
        "datasets" => sort(collect(datasets)),
        "gurobi" => gurobi_result,
        "system" => system_manifest(),
    )
    atomic_toml_write(joinpath(paths.manifests, "preflight.toml"), manifest)
    println("PUBLICATION PREFLIGHT PASSED")
    println("Run fingerprint: ", current_run_fingerprint(paths))
    return manifest
end

function publication_main(args::Vector{String}=ARGS)
    options = _parse_cli(args)
    command = string(options["command"])
    command in ("help", "-h", "--help") && return _print_help()

    root = abspath(string(options["root"]))
    output_option = string(options["output"])
    output = isabspath(output_option) ? output_option : joinpath(root, output_option)
    paths = ensure_run_paths(output)

    if command == "status"
        return _status(paths)
    elseif command == "original-command"
        haskey(options, "dataset") || throw(ArgumentError("--dataset is required"))
        println(run_original_legacy_command(string(options["dataset"])))
        return nothing
    end

    config_option = string(options["config"])
    config_path = isabspath(config_option) ? config_option : joinpath(root, config_option)
    isfile(config_path) || throw(ArgumentError("configuration file not found: $(config_path)"))
    config = load_config(config_path)
    _validate_publication_runtime(config, command)
    if haskey(options, "dataset") && command in ("real-world", "sensitivity")
        section_name = command == "real-world" ? "real_world" : "sensitivity"
        haskey(config, section_name) || (config[section_name] = Dict{String,Any}())
        config[section_name]["datasets"] = [string(options["dataset"])]
    end
    initialize_run_identity(paths, root, config_path, config; command=command)
    resume = Bool(options["resume"])

    if command == "preflight"
        return _publication_preflight(config, paths; root=root)
    elseif command == "suite"
        return run_publication_suite(config, paths; root=root, resume=resume)
    elseif command == "synthetic"
        return run_synthetic_publication(config, paths; resume=resume)
    elseif command == "manuscript-sweeps"
        return run_manuscript_sweeps_publication(config, paths; resume=resume)
    elseif command == "synthetic-meta"
        return run_synthetic_meta_publication(config, paths; resume=resume)
    elseif command == "real-world"
        return run_real_world_publication(config, paths; root=root, resume=resume)
    elseif command == "sensitivity"
        return run_sensitivity_publication(config, paths; root=root, resume=resume)
    elseif command == "stress"
        return run_stress_publication(config, paths; resume=resume)
    elseif command == "verification"
        return run_solver_verification_publication(config, paths; resume=resume)
    elseif command == "runtime"
        return run_runtime_publication(config, paths; resume=resume)
    elseif command == "audit"
        return run_legacy_audit_publication(config, paths; root=root, resume=resume)
    end
    throw(ArgumentError("unknown command $(command); use help"))
end
