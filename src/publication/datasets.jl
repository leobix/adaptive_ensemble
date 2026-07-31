function _numeric_dataframe(path::AbstractString; header::Bool=true)
    isfile(path) || throw(ArgumentError("missing data file: $(path)"))
    return header ? DataFrame(CSV.File(path)) : DataFrame(CSV.File(path; header=false))
end

function _is_explicit_index_column(name::AbstractString)
    normalized = lowercase(strip(string(name)))
    generated_column = occursin(r"^column\d+$", normalized)
    return startswith(normalized, "unnamed") || generated_column ||
           normalized in ("index", "row", "row_id", "rowid", "id", "")
end

function _drop_index_columns(df::DataFrame)
    keep = Symbol[]
    for name in names(df)
        !_is_explicit_index_column(name) && push!(keep, Symbol(name))
    end
    isempty(keep) && throw(ArgumentError("no non-index columns found"))
    return df[:, keep]
end

function _matrix_float(df::DataFrame)
    try
        matrix = Float64.(Matrix(df))
        all(isfinite, matrix) || throw(ArgumentError("data contain non-finite values"))
        return matrix
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("dataframe contains nonnumeric values after index removal"))
    end
end

function _source_metadata(paths::Vector{String})
    return Dict{String,Any}(
        basename(path) => Dict(
            "absolute_path" => abspath(path),
            "sha256" => sha256_file(path),
            "bytes" => filesize(path),
        ) for path in paths
    )
end

function load_safi(root::AbstractString="."; member_set::Symbol=:paper)
    data_dir = joinpath(root, "data")
    xfile = member_set == :paper ? "X_test_speed_adaptive_out1.csv" :
            member_set == :augmented ? "X_test_adaptive.csv" :
            throw(ArgumentError("member_set must be :paper or :augmented"))
    yfile = member_set == :paper ? "y_test_speed_adaptive_out1.csv" : "y_test_speed.csv"
    xpath = joinpath(data_dir, xfile)
    ypath = joinpath(data_dir, yfile)
    xdf = _drop_index_columns(_numeric_dataframe(xpath))
    ydf = _drop_index_columns(_numeric_dataframe(ypath))
    size(ydf, 2) == 1 || throw(ArgumentError(
        "Safi target file must contain exactly one non-index column; found $(size(ydf, 2))"))
    X = _matrix_float(xdf)
    y = vec(_matrix_float(ydf)[:, 1])
    size(X, 1) == length(y) || throw(DimensionMismatch("Safi forecast and target rows differ"))
    n = length(y)
    n == 8494 || @warn "Safi input has $(n) rows rather than the manuscript's 8,494"

    # The historical Julia command and paper use 4,247 / 1,699 / 2,548 rows.
    ntrain = n == 8494 ? 4247 : floor(Int, 0.50n)
    nval = n == 8494 ? 1699 : round(Int, 0.20n)
    ntrain + nval < n || throw(ArgumentError("Safi split leaves no test observations"))
    split = SplitSpec(
        train=1:ntrain,
        validation=(ntrain + 1):(ntrain + nval),
        test=(ntrain + nval + 1):n,
    )
    metadata = Dict{String,Any}(
        "source_X" => xfile,
        "source_y" => yfile,
        "source_files" => _source_metadata([xpath, ypath]),
        "split_counts" => [length(split.train), length(split.validation), length(split.test)],
        "split_rule" => "chronological 4247/1699/2548 for the supplied 8494-row matrix",
        "member_set" => string(member_set),
    )
    bundle = DatasetBundle(
        name=member_set == :paper ? "safi" : "safi_augmented",
        X=X,
        y=y,
        member_names=string.(names(xdf)),
        split=split,
        availability_lag=1,
        units="km/h",
        metadata=metadata,
    )
    validate_split(bundle.split, n)
    return bundle
end

function load_energy(root::AbstractString="."; member_set::Symbol=:historical_selected_10)
    data_dir = joinpath(root, "data")
    xfile = member_set == :historical_selected_10 ? "energy_predictions_test_val2.csv" :
            member_set == :all_available ? "energy_predictions_test_val.csv" :
            throw(ArgumentError("member_set must be :historical_selected_10 or :all_available"))
    xpath = joinpath(data_dir, xfile)
    ypath = joinpath(data_dir, "energy_y_test_val.csv")
    xdf = _drop_index_columns(_numeric_dataframe(xpath))
    # The target file is headerless. Treating the first value as a header drops one observation.
    ydf = _numeric_dataframe(ypath; header=false)
    size(ydf, 2) == 1 || throw(ArgumentError(
        "energy target file must contain exactly one column; found $(size(ydf, 2))"))
    Xall = _matrix_float(xdf)
    yall = vec(_matrix_float(ydf)[:, 1])
    required = 4929 + 1973 + 2956
    available = min(size(Xall, 1), length(yall))
    available >= required || throw(ArgumentError("energy files are shorter than the manuscript split"))
    X = Xall[1:required, :]
    y = yall[1:required]
    split = SplitSpec(
        train=1:4929,
        validation=4930:(4929 + 1973),
        test=(4929 + 1973 + 1):required,
    )
    manuscript_names = [
        "Ordinary Least Squares", "Ridge", "Bayesian Ridge", "ElasticNet",
        "Huber Regressor", "Least Angle Regression", "Lasso", "LassoLARS",
        "Linear Support Vector Regression", "Orthogonal Matching Pursuit",
    ]
    file_names = string.(names(xdf))
    metadata = Dict{String,Any}(
        "source_X" => xfile,
        "source_y" => basename(ypath),
        "source_files" => _source_metadata([xpath, ypath]),
        "split_counts" => [4929, 1973, 2956],
        "split_rule" => "chronological 4929/1973/2956 as stated in the manuscript",
        "rows_after_manuscript_test" => available - required,
        "manuscript_member_labels" => manuscript_names,
        "file_member_labels" => file_names,
        "member_identity_warning" => "the stored columns use validation/CV variants and do not exactly match every manuscript label",
        "member_set" => string(member_set),
    )
    bundle = DatasetBundle(
        name=member_set == :historical_selected_10 ? "energy_historical_selected_10" : "energy_all_available",
        X=X,
        y=y,
        member_names=file_names,
        split=split,
        availability_lag=1,
        units="Wh",
        metadata=metadata,
    )
    validate_split(bundle.split, required)
    return bundle
end

function _hurricane_event_ids(df::DataFrame; storm_id_column::Union{Nothing,String}=nothing)
    if storm_id_column !== nothing && storm_id_column in names(df)
        values = df[!, storm_id_column]
        levels = Dict{Any,Int}()
        ids = Vector{Int}(undef, length(values))
        next_id = 0
        for i in eachindex(values)
            if !haskey(levels, values[i])
                next_id += 1
                levels[values[i]] = next_id
            end
            ids[i] = levels[values[i]]
        end
        return ids
    end
    all(name -> name in names(df), ["days", "hours"]) ||
        throw(ArgumentError("hurricane input needs a storm-id column or the legacy days/hours columns"))
    days = Int.(df[!, "days"])
    hours = Int.(df[!, "hours"])
    ids = ones(Int, nrow(df))
    current = 1
    for i in 2:nrow(df)
        continuous = (days[i] == days[i - 1] && hours[i] == hours[i - 1] + 6) ||
                     (days[i] == days[i - 1] + 1 && hours[i - 1] == 18 && hours[i] == 0)
        if !continuous
            current += 1
        end
        ids[i] = current
    end
    return ids
end

"""
Load a tropical-cyclone forecast matrix when the original ATCF/Hurricast data have been
restored. The supplied archive does not contain these matrices. A schema TOML may define
`target`, `storm_id`, `member_columns`, and `benchmark_columns`; otherwise the legacy
`TRUTH`, `days`, `hours`, and columns 14:end conventions are used.
"""
function load_hurricane(root::AbstractString="."; basin::Symbol=:NA,
                        schema_path::AbstractString=joinpath(root, "config", "hurricane_schema.toml"))
    filename = basin == :NA ? "NA_ARO_Intensity_2014_clean_v2.csv" :
               basin == :EP ? "EP_ARO_Intensity_2014_clean_v2.csv" :
               throw(ArgumentError("basin must be :NA or :EP"))
    path = joinpath(root, "data", filename)
    isfile(path) || throw(ArgumentError(
        "The historical hurricane matrix $(filename) is absent. Restore the exact ATCF/Hurricast matrix; the publication pipeline will not substitute fabricated data."))
    df = DataFrame(CSV.File(path))
    schema = isfile(schema_path) ? TOML.parsefile(schema_path) : Dict{String,Any}()
    target_name = string(get(schema, "target", "TRUTH"))
    target_name in names(df) || throw(ArgumentError("target column $(target_name) is absent"))
    storm_name = haskey(schema, "storm_id") ? string(schema["storm_id"]) : nothing
    member_names = haskey(schema, "member_columns") ? string.(schema["member_columns"]) : string.(names(df)[14:end])
    benchmark_names = haskey(schema, "benchmark_columns") ? string.(schema["benchmark_columns"]) : String[]
    member_names = [name for name in member_names if !(name in benchmark_names)]
    all(name -> name in names(df), member_names) || throw(ArgumentError("one or more configured hurricane member columns are absent"))
    event_ids = _hurricane_event_ids(df; storm_id_column=storm_name)
    X = Float64.(Matrix(df[:, Symbol.(member_names)]))
    y = Float64.(df[!, target_name])
    all(isfinite, X) && all(isfinite, y) || throw(ArgumentError("hurricane matrix contains non-finite values"))
    n = length(y)
    ntrain = floor(Int, 0.50n)
    nval = floor(Int, 0.20n)
    split = SplitSpec(train=1:ntrain, validation=(ntrain + 1):(ntrain + nval), test=(ntrain + nval + 1):n)
    benchmark_data = Dict{String,Vector{Float64}}()
    for name in benchmark_names
        name in names(df) && (benchmark_data[name] = Float64.(df[!, name]))
    end
    bundle = DatasetBundle(
        name=basin == :NA ? "hurricane_north_atlantic" : "hurricane_eastern_pacific",
        X=X,
        y=y,
        member_names=member_names,
        split=split,
        event_ids=event_ids,
        availability_lag=4,
        units="knots",
        metadata=Dict{String,Any}(
            "source_file" => filename,
            "source_sha256" => sha256_file(path),
            "schema_file" => isfile(schema_path) ? abspath(schema_path) : "legacy positional fallback",
            "lead_time_hours" => 24,
            "row_spacing_hours" => 6,
            "benchmark_columns" => benchmark_names,
            "benchmark_predictions" => benchmark_data,
            "event_count" => maximum(event_ids),
        ),
    )
    validate_split(bundle.split, n)
    return bundle
end

function standardize_bundle(bundle::DatasetBundle)
    standardizer = fit_standardizer(bundle.y, collect(bundle.split.train))
    X = transform(standardizer, bundle.X)
    y = transform(standardizer, bundle.y)
    metadata = copy(bundle.metadata)
    metadata["standardization_center"] = standardizer.center
    metadata["standardization_scale"] = standardizer.scale
    standardized = DatasetBundle(
        name=bundle.name,
        X=Matrix{Float64}(X),
        y=Vector{Float64}(y),
        member_names=bundle.member_names,
        split=bundle.split,
        event_ids=bundle.event_ids,
        availability_lag=bundle.availability_lag,
        units="training-target standard deviations",
        metadata=metadata,
    )
    return standardized, standardizer
end

function load_dataset(name::AbstractString, root::AbstractString=".";
                      member_set::AbstractString="")
    normalized = lowercase(string(name))
    normalized == "safi" && return load_safi(root; member_set=isempty(member_set) ? :paper : Symbol(member_set))
    normalized in ("safi_augmented", "safi_transformer") && return load_safi(root; member_set=:augmented)
    normalized in ("energy", "energy_historical", "energy_selected_10", "energy_historical_selected_10", "energy_legacy") &&
        return load_energy(root; member_set=isempty(member_set) ? :historical_selected_10 : Symbol(member_set))
    normalized in ("energy_all", "energy_all_members", "energy_all_available") &&
        return load_energy(root; member_set=:all_available)
    normalized in ("hurricane_na", "north_atlantic") && return load_hurricane(root; basin=:NA)
    normalized in ("hurricane_ep", "eastern_pacific") && return load_hurricane(root; basin=:EP)
    throw(ArgumentError("unknown dataset $(name)"))
end
