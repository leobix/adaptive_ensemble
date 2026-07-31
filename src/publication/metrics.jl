mae(y, yhat) = mean(abs.(y .- yhat))
rmse(y, yhat) = sqrt(mean(abs2, y .- yhat))

function mape_safe(y, yhat; atol::Float64=1e-8)
    mask = abs.(y) .> atol
    if !any(mask)
        return NaN, 0.0
    end
    return 100 * mean(abs.((y[mask] .- yhat[mask]) ./ y[mask])), mean(mask)
end

function smape(y, yhat; atol::Float64=1e-8)
    denom = abs.(y) .+ abs.(yhat)
    vals = 200 .* abs.(y .- yhat) ./ max.(denom, atol)
    return mean(vals)
end

function wape(y, yhat; atol::Float64=1e-8)
    return 100 * sum(abs.(y .- yhat)) / max(sum(abs.(y)), atol)
end

"""Exact finite-sample empirical CVaR of nonnegative losses in the worst alpha fraction."""
function empirical_cvar(losses::AbstractVector{<:Real}, alpha::Real)
    0 < alpha <= 1 || throw(ArgumentError("alpha must be in (0,1]"))
    n = length(losses)
    n > 0 || throw(ArgumentError("losses cannot be empty"))
    sorted = sort(Float64.(losses); rev=true)
    mass = Float64(alpha) * n
    whole = floor(Int, mass)
    frac = mass - whole
    total = whole > 0 ? sum(@view sorted[1:whole]) : 0.0
    if frac > 0 && whole < n
        total += frac * sorted[whole + 1]
    end
    return total / mass
end

function metric_table(y::AbstractVector, yhat::AbstractVector)
    length(y) == length(yhat) || throw(DimensionMismatch("y and yhat lengths differ"))
    err = abs.(Float64.(y) .- Float64.(yhat))
    mape_value, mape_fraction = mape_safe(y, yhat)
    return Dict{String,Float64}(
        "mae" => mae(y, yhat),
        "rmse" => rmse(y, yhat),
        "mean_error" => mean(Float64.(yhat) .- Float64.(y)),
        "median_abs_error" => median(err),
        "q95_abs_error" => quantile(err, 0.95),
        "mape" => mape_value,
        "mape_valid_fraction" => mape_fraction,
        "smape" => smape(y, yhat),
        "wape" => wape(y, yhat),
        "cvar_05" => empirical_cvar(err, 0.05),
        "cvar_15" => empirical_cvar(err, 0.15),
        "max_abs_error" => maximum(err),
    )
end

function metrics_dataframe(dataset::AbstractString, method::AbstractString,
                           y::AbstractVector, yhat::AbstractVector;
                           seed::Union{Nothing,Int}=nothing,
                           regime::Union{Nothing,String}=nothing,
                           extras::AbstractDict=Dict{String,Any}())
    m = metric_table(y, yhat)
    row = Dict{Symbol,Any}(:dataset=>string(dataset), :method=>string(method), :n=>length(y))
    seed !== nothing && (row[:seed] = seed)
    regime !== nothing && (row[:regime] = regime)
    for (k,v) in m
        row[Symbol(k)] = v
    end
    for (k,v) in extras
        row[Symbol(k)] = v
    end
    return one_row_dataframe(row)
end
