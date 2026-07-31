function _critical_t(n::Int, alpha::Float64)
    n >= 2 || return NaN
    return quantile(TDist(n - 1), 1.0 - alpha / 2.0)
end

function summarize_replications(df::DataFrame;
                                groupcols=[:regime, :method],
                                metrics=[:mae, :rmse, :cvar_05, :cvar_15],
                                alpha::Float64=0.05)
    out = DataFrame()
    for sub in groupby(df, groupcols)
        row = Dict{Symbol,Any}()
        for column in groupcols
            row[column] = sub[1, column]
        end
        row[:replications] = nrow(sub)
        for metric in metrics
            values = Float64.(sub[!, metric])
            n = length(values)
            mean_value = mean(values)
            sd_value = n >= 2 ? std(values) : 0.0
            se_value = n >= 2 ? sd_value / sqrt(n) : 0.0
            critical = n >= 2 ? _critical_t(n, alpha) : 0.0
            row[Symbol(metric, "_mean")] = mean_value
            row[Symbol(metric, "_sd")] = sd_value
            row[Symbol(metric, "_se")] = se_value
            row[Symbol(metric, "_ci_low")] = mean_value - critical * se_value
            row[Symbol(metric, "_ci_high")] = mean_value + critical * se_value
        end
        append!(out, one_row_dataframe(row); cols=:union)
    end
    return out
end

function paired_seed_intervals(
    df::DataFrame;
    reference::String = "adaptive_ridge",
    metric::Symbol = :rmse,
    groupcol::Symbol = :regime,
    alpha::Float64 = 0.05,
)
    out = DataFrame(
        group = String[],
        reference = String[],
        method = String[],
        metric = String[],
        n = Int[],
        interval_available = Bool[],
        mean_method_minus_reference = Float64[],
        ci_low = Union{Missing,Float64}[],
        ci_high = Union{Missing,Float64}[],
        method_win_rate = Float64[],
        reference_win_rate = Float64[],
        tie_rate = Float64[],
    )

    for sub in groupby(df, groupcol)
        reference_rows =
            sub[sub.method .== reference, [:seed, metric]]

        rename!(
            reference_rows,
            metric => :reference_value,
        )
        length(unique(reference_rows.seed)) == nrow(reference_rows) ||
            throw(ArgumentError("duplicate reference rows for one or more seeds in group $(sub[1, groupcol])"))

        for method in unique(string.(sub.method))
            method == reference && continue

            current =
                sub[sub.method .== method, [:seed, metric]]

            rename!(
                current,
                metric => :method_value,
            )
            length(unique(current.seed)) == nrow(current) ||
                throw(ArgumentError("duplicate rows for method $(method) in group $(sub[1, groupcol])"))

            paired = innerjoin(
                current,
                reference_rows;
                on = :seed,
            )

            n = nrow(paired)
            n == 0 && continue

            differences =
                paired.method_value .-
                paired.reference_value

            mean_difference = mean(differences)

            interval_available = n >= 2

            if interval_available
                standard_error =
                    std(differences) / sqrt(n)

                critical =
                    _critical_t(n, alpha)

                ci_low =
                    mean_difference -
                    critical * standard_error

                ci_high =
                    mean_difference +
                    critical * standard_error
            else
                ci_low = missing
                ci_high = missing
            end

            push!(
                out,
                (
                    group = string(sub[1, groupcol]),
                    reference = reference,
                    method = method,
                    metric = string(metric),
                    n = n,
                    interval_available = interval_available,
                    mean_method_minus_reference =
                        mean_difference,
                    ci_low = ci_low,
                    ci_high = ci_high,
                    method_win_rate =
                        mean(differences .< 0.0),
                    reference_win_rate =
                        mean(differences .> 0.0),
                    tie_rate =
                        mean(differences .== 0.0),
                ),
            )
        end
    end

    nrow(out) > 0 &&
        rename!(out, :group => groupcol)

    return out
end

function _block_indices(rng::AbstractRNG, n::Int, block_length::Int)
    # Ordinary (non-circular) moving-block bootstrap. Every sampled block is a
    # contiguous subsequence of the observed test trajectory; no block wraps from
    # the final observation back to the first observation.
    block_length = clamp(block_length, 1, n)
    last_start = n - block_length + 1
    indices = Int[]
    sizehint!(indices, n + block_length)
    while length(indices) < n
        start = rand(rng, 1:last_start)
        append!(indices, start:(start + block_length - 1))
    end
    resize!(indices, n)
    return indices
end

function _event_block_indices(rng::AbstractRNG, event_ids::AbstractVector{Int},
                              block_length::Int)
    n = length(event_ids)
    n > 0 || throw(ArgumentError("event_ids cannot be empty"))
    indices = Int[]
    start = 1
    while start <= n
        stop = start
        while stop < n && event_ids[stop + 1] == event_ids[start]
            stop += 1
        end
        run_length = stop - start + 1
        loc = _block_indices(rng, run_length, min(block_length, run_length))
        append!(indices, (start - 1) .+ loc)
        start = stop + 1
    end
    length(indices) == n || error("event-aware bootstrap did not preserve sample size")
    return indices
end

function moving_block_bootstrap(y::AbstractVector, predictions::DataFrame;
                                methods::Vector{String},
                                reps::Int=2000,
                                block_length::Int=max(2, round(Int, length(y)^(1 / 3))),
                                seed::Int=2026,
                                reference::String="adaptive_ridge",
                                event_ids::Union{Nothing,AbstractVector{Int}}=nothing)
    reps >= 200 || @warn "Fewer than 200 bootstrap replications are not publication grade"
    n = length(y)
    n > 1 || throw(ArgumentError("bootstrap requires at least two observations"))
    event_ids !== nothing && length(event_ids) != n &&
        throw(DimensionMismatch("event_ids length differs from bootstrap sample"))
    block_length = clamp(block_length, 2, n)
    resampling_scheme = event_ids === nothing ? "moving_blocks" : "within_event_moving_blocks"
    Symbol(reference) in propertynames(predictions) ||
        throw(ArgumentError("reference prediction column $(reference) is absent"))
    out = DataFrame()
    reference_prediction = Float64.(predictions[!, Symbol(reference)])
    for method in methods
        method == reference && continue
        Symbol(method) in propertynames(predictions) || continue
        prediction = Float64.(predictions[!, Symbol(method)])
        # Reinitialize the RNG for every method so all method-reference contrasts
        # use exactly the same block resamples. This is a paired bootstrap.
        rng = MersenneTwister(seed)
        differences = Dict(
            "rmse" => zeros(Float64, reps),
            "mae" => zeros(Float64, reps),
            "cvar_05" => zeros(Float64, reps),
        )
        for replication in 1:reps
            indices = event_ids === nothing ?
                _block_indices(rng, n, block_length) :
                _event_block_indices(rng, event_ids, block_length)
            yb = y[indices]
            pb = prediction[indices]
            rb = reference_prediction[indices]
            differences["rmse"][replication] = rmse(yb, pb) - rmse(yb, rb)
            differences["mae"][replication] = mae(yb, pb) - mae(yb, rb)
            differences["cvar_05"][replication] =
                empirical_cvar(abs.(yb .- pb), 0.05) - empirical_cvar(abs.(yb .- rb), 0.05)
        end
        for metric in ("rmse", "mae", "cvar_05")
            values = differences[metric]
            append!(out, DataFrame(
                reference=[reference],
                method=[method],
                metric=[metric],
                reps=[reps],
                block_length=[block_length],
                resampling_scheme=[resampling_scheme],
                event_count=[event_ids === nothing ? 0 : length(unique(event_ids))],
                mean_method_minus_reference=[mean(values)],
                median_method_minus_reference=[median(values)],
                ci_low=[quantile(values, 0.025)],
                ci_high=[quantile(values, 0.975)],
                method_win_probability=[mean(values .< 0.0)],
                reference_win_probability=[mean(values .> 0.0)],
            ); cols=:union)
        end
    end
    return out
end
