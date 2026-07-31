const RUN_IDENTITY_FILENAME = "run_identity.toml"
const RUN_IDENTITY_FORMAT_VERSION = 2
const _PROGRESS_LOCK = ReentrantLock()

function ensure_run_paths(root::AbstractString)
    root_abs = abspath(root)
    paths = RunPaths(
        root=root_abs,
        tables=joinpath(root_abs, "tables"),
        figures=joinpath(root_abs, "figures"),
        predictions=joinpath(root_abs, "predictions"),
        models=joinpath(root_abs, "models"),
        logs=joinpath(root_abs, "logs"),
        checkpoints=joinpath(root_abs, "checkpoints"),
        manifests=joinpath(root_abs, "manifests"),
    )
    for p in (paths.root, paths.tables, paths.figures, paths.predictions,
              paths.models, paths.logs, paths.checkpoints, paths.manifests)
        mkpath(p)
    end
    return paths
end

load_config(path::AbstractString) = TOML.parsefile(path)

function _temporary_path(path::AbstractString)
    mkpath(dirname(path))
    return path * ".tmp-" * string(getpid()) * "-" * string(Threads.threadid()) * "-" * randstring(10)
end

function atomic_csv_write(path::AbstractString, table; append::Bool=false)
    mkpath(dirname(path))
    tmp = _temporary_path(path)
    try
        if append && isfile(path)
            # Preserve the existing file in a sibling temporary path so a crash
            # cannot leave a partially appended CSV behind.
            cp(path, tmp; force=true)
            CSV.write(tmp, table; append=true, writeheader=false)
        else
            CSV.write(tmp, table)
        end
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return path
end

function atomic_toml_write(path::AbstractString, data::AbstractDict)
    mkpath(dirname(path))
    tmp = _temporary_path(path)
    try
        open(tmp, "w") do io
            TOML.print(io, data)
            flush(io)
        end
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return path
end

function sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

_sha256_text(text::AbstractString) = bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(string(text)))))

function _blas_string()
    try
        return sprint(show, LinearAlgebra.BLAS.get_config())
    catch
        return "unavailable"
    end
end

function system_manifest()
    return Dict{String,Any}(
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "julia_version" => string(VERSION),
        "kernel" => string(Sys.KERNEL),
        "architecture" => string(Sys.ARCH),
        "cpu_threads" => Sys.CPU_THREADS,
        "julia_threads" => Threads.nthreads(),
        "word_size" => Sys.WORD_SIZE,
        "blas" => _blas_string(),
        "hostname" => gethostname(),
    )
end

function _fingerprint_files(root::AbstractString, relative_roots::Vector{String};
                            extensions::Set{String}=Set([".jl", ".toml", ".ps1", ".sh"]))
    records = Vector{Dict{String,Any}}()
    for relative_root in relative_roots
        path = joinpath(root, relative_root)
        if isfile(path)
            push!(records, Dict{String,Any}(
                "path" => replace(relpath(path, root), '\\' => '/'),
                "bytes" => filesize(path),
                "sha256" => sha256_file(path),
            ))
        elseif isdir(path)
            for (directory, _, files) in walkdir(path)
                for filename in sort(files)
                    extension = lowercase(splitext(filename)[2])
                    extension in extensions || continue
                    file_path = joinpath(directory, filename)
                    push!(records, Dict{String,Any}(
                        "path" => replace(relpath(file_path, root), '\\' => '/'),
                        "bytes" => filesize(file_path),
                        "sha256" => sha256_file(file_path),
                    ))
                end
            end
        end
    end
    sort!(records; by=record -> string(record["path"]))
    return records
end

function _records_digest(records::Vector{Dict{String,Any}})
    payload = join([
        string(record["path"], "\t", record["bytes"], "\t", record["sha256"])
        for record in records
    ], "\n")
    return _sha256_text(payload)
end

function _canonical_config_value(value)
    if value isa AbstractDict
        entries = String[]
        for (key, item) in sort(collect(pairs(value)); by=pair -> string(first(pair)))
            push!(entries, repr(string(key)) * ":" * _canonical_config_value(item))
        end
        return "{" * join(entries, ",") * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join((_canonical_config_value(item) for item in value), ",") * "]"
    elseif value isa AbstractString
        return repr(string(value))
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value === nothing
        return "nothing"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        if isnan(value)
            return "NaN"
        elseif isinf(value)
            return signbit(value) ? "-Inf" : "Inf"
        end
        return @sprintf("%.17g", Float64(value))
    else
        return repr(value)
    end
end

function _effective_config_digest(config::AbstractDict)
    canonical = _canonical_config_value(config)
    return _sha256_text(canonical), canonical
end

function run_identity_path(paths::RunPaths)
    return joinpath(paths.manifests, RUN_IDENTITY_FILENAME)
end

function current_run_identity(paths::RunPaths; required::Bool=true)
    path = run_identity_path(paths)
    if !isfile(path)
        required && error(
            "The output directory has no run identity. Launch experiments through " *
            "scripts/publication/run_publication_suite.jl so configuration and source " *
            "fingerprints are recorded before checkpoints are created.")
        return Dict{String,Any}()
    end
    return TOML.parsefile(path)
end

function current_run_fingerprint(paths::RunPaths; required::Bool=true)
    identity = current_run_identity(paths; required=required)
    isempty(identity) && return ""
    return string(get(identity, "run_fingerprint", ""))
end

function _output_has_prior_artifacts(paths::RunPaths)
    for directory in (paths.tables, paths.figures, paths.predictions, paths.models,
                      paths.checkpoints, paths.logs)
        isdir(directory) || continue
        !isempty(readdir(directory)) && return true
    end
    return false
end

"""
Create or verify the immutable identity of one output directory. The identity covers the
*effective* configuration (including command-line dataset restrictions), publication Julia
sources, launch scripts, locked environment, and supplied data inputs. Reusing an output
directory after any of those inputs change is rejected instead of silently mixing old and new
checkpoints.
"""
function initialize_run_identity(paths::RunPaths, root::AbstractString,
                                 config_path::AbstractString, config::AbstractDict;
                                 command::AbstractString="suite")
    root_abs = abspath(root)
    config_abs = abspath(config_path)
    isfile(config_abs) || throw(ArgumentError("configuration file not found: $(config_abs)"))

    code_records = _fingerprint_files(root_abs, [
        "Project.toml", "Manifest.toml", "src", "scripts/publication",
        "scripts/run_publication_suite.jl", "test/runtests.jl",
        "reports/ORIGINAL_PYTHON_INTEGRITY.csv",
    ]; extensions=Set([".jl", ".toml", ".ps1", ".sh", ".csv"]))
    data_records = _fingerprint_files(root_abs, ["data", "config/hurricane_schema.toml"];
        extensions=Set([".csv", ".toml"]))
    effective_config_sha, _ = _effective_config_digest(config)
    config_sha = sha256_file(config_abs)
    code_sha = _records_digest(code_records)
    data_sha = _records_digest(data_records)
    fingerprint_payload = join([
        "format=$(RUN_IDENTITY_FORMAT_VERSION)",
        "effective_config=$(effective_config_sha)",
        "config_file=$(config_sha)",
        "code=$(code_sha)",
        "data=$(data_sha)",
    ], "\n")
    fingerprint = _sha256_text(fingerprint_payload)
    proposed = Dict{String,Any}(
        "format_version" => RUN_IDENTITY_FORMAT_VERSION,
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "command" => string(command),
        "source_root" => root_abs,
        "output_root" => paths.root,
        "config_path" => config_abs,
        "config_file_sha256" => config_sha,
        "effective_config_sha256" => effective_config_sha,
        "code_sha256" => code_sha,
        "data_sha256" => data_sha,
        "run_fingerprint" => fingerprint,
        "code_files" => code_records,
        "data_files" => data_records,
    )

    identity_path = run_identity_path(paths)
    if isfile(identity_path)
        existing = TOML.parsefile(identity_path)
        old = string(get(existing, "run_fingerprint", ""))
        old == fingerprint || error(
            "Refusing to reuse output directory $(paths.root): its run fingerprint is $(old), " *
            "but the current code/configuration/data fingerprint is $(fingerprint). Use a new " *
            "--output directory (recommended) or archive and remove the old output first.")
        return existing
    end
    _output_has_prior_artifacts(paths) && error(
        "Refusing to adopt a nonempty legacy output directory without a run identity: $(paths.root). " *
        "Use a fresh --output directory so stale checkpoints cannot contaminate the publication run.")
    atomic_toml_write(identity_path, proposed)
    return proposed
end

function save_run_manifest(paths::RunPaths, name::AbstractString, config::AbstractDict;
                           extras::AbstractDict=Dict{String,Any}())
    data = Dict{String,Any}(
        "run" => Dict(
            "name" => string(name),
            "root" => paths.root,
            "run_fingerprint" => current_run_fingerprint(paths),
        ),
        "system" => system_manifest(),
        "config" => Dict{String,Any}(string(k) => v for (k, v) in config),
        "extras" => Dict{String,Any}(string(k) => v for (k, v) in extras),
    )
    return atomic_toml_write(joinpath(paths.manifests, string(name) * ".toml"), data)
end

function _safe_marker_name(name::AbstractString)
    return replace(string(name), r"[^A-Za-z0-9_.-]" => "_")
end

function stage_marker(paths::RunPaths, stage::AbstractString)
    return joinpath(paths.checkpoints, _safe_marker_name(stage) * ".done.toml")
end


function _resolve_output_under_root(paths::RunPaths, raw_path::AbstractString)
    root = realpath(paths.root)
    candidate = isabspath(raw_path) ? normpath(abspath(raw_path)) :
        normpath(abspath(joinpath(root, raw_path)))
    isfile(candidate) || throw(ArgumentError("output file does not exist: $(candidate)"))
    resolved = realpath(candidate)
    relative = try
        relpath(resolved, root)
    catch err
        throw(ArgumentError("output path is not on the output root filesystem: $(candidate)"))
    end
    parts = splitpath(relative)
    (isabspath(relative) || relative == ".." || any(==(".."), parts)) &&
        throw(ArgumentError("output path escapes the configured output root: $(candidate)"))
    return resolved, replace(relative, '\\' => '/')
end

function _marker_outputs_valid(paths::RunPaths, marker::AbstractDict)
    outputs = string.(get(marker, "outputs", String[]))
    hashes = string.(get(marker, "output_sha256", String[]))
    length(outputs) == length(hashes) || return false
    for (relative_path, expected_hash) in zip(outputs, hashes)
        path = try
            first(_resolve_output_under_root(paths, relative_path))
        catch
            return false
        end
        sha256_file(path) == expected_hash || return false
    end
    return true
end

function _marker_matches_run(paths::RunPaths, marker::AbstractDict)
    fingerprint = current_run_fingerprint(paths; required=false)
    isempty(fingerprint) && return false
    return string(get(marker, "run_fingerprint", "")) == fingerprint &&
        _marker_outputs_valid(paths, marker)
end

function stage_complete(paths::RunPaths, stage::AbstractString)
    path = stage_marker(paths, stage)
    isfile(path) || return false
    try
        return _marker_matches_run(paths, TOML.parsefile(path))
    catch
        return false
    end
end

function _output_metadata(paths::RunPaths, output_paths::Vector{String})
    relative = String[]
    hashes = String[]
    for raw_path in output_paths
        path, relative_path = try
            _resolve_output_under_root(paths, raw_path)
        catch err
            error("Cannot mark completion: " * sprint(showerror, err))
        end
        filesize(path) > 0 || error("Cannot mark completion: required output is empty: $(path)")
        push!(relative, relative_path)
        push!(hashes, sha256_file(path))
    end
    return relative, hashes
end

function mark_stage_complete(paths::RunPaths, stage::AbstractString;
                             extras=Dict{String,Any}(), outputs::Vector{String}=String[])
    relative, hashes = _output_metadata(paths, outputs)
    d = Dict{String,Any}(
        "stage" => string(stage),
        "completed_at_utc" => string(Dates.now(Dates.UTC)),
        "run_fingerprint" => current_run_fingerprint(paths),
        "outputs" => relative,
        "output_sha256" => hashes,
    )
    for (k, v) in extras
        d[string(k)] = v
    end
    atomic_toml_write(stage_marker(paths, stage), d)
end

function task_marker(paths::RunPaths, task::AbstractString)
    directory = joinpath(paths.checkpoints, "task_markers")
    mkpath(directory)
    return joinpath(directory, _safe_marker_name(task) * ".done.toml")
end

function task_complete(paths::RunPaths, task::AbstractString)
    path = task_marker(paths, task)
    isfile(path) || return false
    try
        return _marker_matches_run(paths, TOML.parsefile(path))
    catch
        return false
    end
end

function mark_task_complete(paths::RunPaths, task::AbstractString;
                            outputs::Vector{String}, extras=Dict{String,Any}())
    relative, hashes = _output_metadata(paths, outputs)
    data = Dict{String,Any}(
        "task" => string(task),
        "completed_at_utc" => string(Dates.now(Dates.UTC)),
        "run_fingerprint" => current_run_fingerprint(paths),
        "outputs" => relative,
        "output_sha256" => hashes,
    )
    for (key, value) in extras
        data[string(key)] = value
    end
    atomic_toml_write(task_marker(paths, task), data)
    return nothing
end

function update_progress(paths::RunPaths, stage::AbstractString;
                         completed::Int, total::Int, message::AbstractString="")
    total >= 0 || throw(ArgumentError("progress total must be nonnegative"))
    completed >= 0 || throw(ArgumentError("progress completed must be nonnegative"))
    lock(_PROGRESS_LOCK) do
        data = Dict{String,Any}(
            "stage" => string(stage),
            "completed" => completed,
            "total" => total,
            "fraction" => total == 0 ? 1.0 : completed / total,
            "message" => string(message),
            "updated_at_utc" => string(Dates.now(Dates.UTC)),
            "run_fingerprint" => current_run_fingerprint(paths),
        )
        atomic_toml_write(joinpath(paths.manifests, "progress_" * _safe_marker_name(stage) * ".toml"), data)
        atomic_toml_write(joinpath(paths.manifests, "heartbeat.toml"), data)
    end
    return nothing
end

function save_model(path::AbstractString, model)
    mkpath(dirname(path))
    tmp = _temporary_path(path)
    try
        open(tmp, "w") do io
            serialize(io, model)
            flush(io)
        end
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return path
end

function load_model(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

function parse_float_grid(x)
    values = if x isa AbstractVector
        Float64.(x)
    elseif x isa Real
        [Float64(x)]
    elseif x isa AbstractString
        stripped = strip(x)
        isempty(stripped) && throw(ArgumentError("float grid must not be empty"))
        parse.(Float64, strip.(split(stripped, ',')))
    else
        throw(ArgumentError("float grid must be a number, string, or vector"))
    end
    isempty(values) && throw(ArgumentError("float grid must not be empty"))
    all(isfinite, values) || throw(ArgumentError("float grid contains NaN or Inf"))
    return values
end

function parse_int_grid(x)
    values = if x isa AbstractVector
        Int.(x)
    elseif x isa Integer
        [Int(x)]
    elseif x isa AbstractString
        stripped = strip(x)
        isempty(stripped) && throw(ArgumentError("integer grid must not be empty"))
        parse.(Int, strip.(split(stripped, ',')))
    else
        throw(ArgumentError("integer grid must be an integer, string, or vector"))
    end
    isempty(values) && throw(ArgumentError("integer grid must not be empty"))
    return values
end

function write_log(paths::RunPaths, name::AbstractString, text::AbstractString; append::Bool=true)
    path = joinpath(paths.logs, string(name) * ".log")
    lock(_PROGRESS_LOCK) do
        open(path, append ? "a" : "w") do io
            println(io, "[", Dates.now(), "] ", text)
            flush(io)
        end
    end
    return path
end

"""Create a one-row DataFrame from a dictionary or named tuple of scalar values."""
function one_row_dataframe(row)
    pairs_iter = row isa NamedTuple ? pairs(row) : pairs(row)
    columns = Dict{Symbol,Any}()
    for (key, value) in pairs_iter
        columns[Symbol(key)] = [value]
    end
    return DataFrame(columns)
end
