function validate_xy(X::AbstractMatrix, y::AbstractVector)
    size(X, 1) == length(y) || throw(DimensionMismatch("X rows must equal y length"))
    size(X, 2) >= 1 || throw(ArgumentError("X must contain at least one ensemble member"))
    all(isfinite, X) || throw(ArgumentError("X contains non-finite values"))
    all(isfinite, y) || throw(ArgumentError("y contains non-finite values"))
    return nothing
end

function validate_split(split::SplitSpec, n::Int)
    first(split.train) == 1 || throw(ArgumentError("training split must begin at one"))
    last(split.train) + 1 == first(split.validation) || throw(ArgumentError("train and validation must be contiguous"))
    last(split.validation) + 1 == first(split.test) || throw(ArgumentError("validation and test must be contiguous"))
    last(split.test) == n || throw(ArgumentError("test split must end at the final observation"))
    return nothing
end

function fit_standardizer(y::AbstractVector, idx::AbstractVector{Int})
    isempty(idx) && throw(ArgumentError("standardizer indices cannot be empty"))
    c = mean(@view y[idx])
    # Use the same sample-standard-deviation convention as the historical Julia code.
    s = std(@view y[idx])
    if !isfinite(s) || s <= eps(Float64)
        s = 1.0
    end
    return Standardizer(Float64(c), Float64(s))
end

transform(s::Standardizer, x) = (x .- s.center) ./ s.scale
inverse_transform(s::Standardizer, x) = x .* s.scale .+ s.center

function _same_event(event_ids, t::Int, j::Int)
    event_ids === nothing && return true
    return event_ids[t] == event_ids[j]
end

"""
Construct the causal error-history matrix `Z` with exactly `m*τ` columns. Row `t`
uses only outcomes available before the forecast at `t`. `availability_lag=1` is the
ordinary one-step verification rule; `availability_lag=4` enforces a 24-hour delay
when rows are spaced six hours apart. Error histories never cross event boundaries.
By default, a row is activated only after a complete `τ`-slot history from the same
event is available. Earlier/cold-start rows are all zero, so the deployed affine rule is
exactly the static component `β₀`, rather than an undocumented partial-window rule.
Set `require_complete=false` only for an explicitly labeled partial-history ablation.
"""
function build_error_histories(X::AbstractMatrix, y::AbstractVector, tau::Int;
                               availability_lag::Int=1,
                               event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                               normalize::Bool=false,
                               require_complete::Bool=true)
    validate_xy(X, y)
    tau >= 1 || throw(ArgumentError("tau must be positive"))
    availability_lag >= 1 || throw(ArgumentError("availability_lag must be positive"))
    n, m = size(X)
    event_ids !== nothing && length(event_ids) != n && throw(DimensionMismatch("event_ids length differs"))
    Z = zeros(Float64, n, m * tau)
    counts = zeros(Int, n)
    for t in 1:n
        latest = t - availability_lag
        latest < 1 && continue
        used = 0
        for lag_slot in 1:tau
            j = latest - (tau - lag_slot)
            if j >= 1 && _same_event(event_ids, t, j)
                cols = ((lag_slot - 1) * m + 1):(lag_slot * m)
                @views Z[t, cols] .= X[j, :] .- y[j]
                used += 1
            end
        end
        if require_complete && used < tau
            # Report zero usable history whenever the affine adjustment is deliberately
            # disabled. This keeps exported cold-start diagnostics consistent with the
            # actual zero history vector used for prediction.
            counts[t] = 0
            fill!(@view(Z[t, :]), 0.0)
        else
            counts[t] = used
        end
        if !(require_complete && used < tau) && normalize && used > 0
            nr = norm(@view Z[t, :])
            nr > 0 && (@views Z[t, :] ./= nr)
        end
    end
    return Z, counts
end

"""Materialize the reduced design `A_t=[x_t; kron(z_t,x_t)]'`."""
function reduced_design(X::AbstractMatrix, Z::AbstractMatrix, idx::AbstractVector{Int})
    n, m = size(X)
    size(Z, 1) == n || throw(DimensionMismatch("Z and X rows differ"))
    q = size(Z, 2)
    A = Matrix{Float64}(undef, length(idx), m + m * q)
    for (r, t) in enumerate(idx)
        x = @view X[t, :]
        z = @view Z[t, :]
        @views A[r, 1:m] .= x
        for k in 1:q
            cols = (m + (k - 1) * m + 1):(m + k * m)
            @views A[r, cols] .= z[k] .* x
        end
    end
    return A
end

function beta_map(z::AbstractVector{<:Real}, m::Int)
    q = length(z)
    C = zeros(Float64, m, m + m * q)
    @views C[:, 1:m] .= Matrix{Float64}(I, m, m)
    for k in 1:q
        cols = (m + (k - 1) * m + 1):(m + k * m)
        @views C[:, cols] .= Float64(z[k]) .* Matrix{Float64}(I, m, m)
    end
    return C
end

"""
Return `Σ_t C_t'C_t`, where `β_t=C_t θ`, without constructing one dense `C_t`
per observation. This is the exact Gram matrix of the stacked time-varying weights.
"""
function stacked_beta_gram(Z::AbstractMatrix, idx::AbstractVector{Int}, m::Int)
    Zt = Matrix{Float64}(Z[idx, :])
    n = size(Zt, 1)
    q = size(Zt, 2)
    d = m + m * q
    H = zeros(Float64, d, d)
    I_m = Matrix{Float64}(I, m, m)
    @views H[1:m, 1:m] .= n .* I_m
    sum_z = vec(sum(Zt; dims=1))
    cross = kron(reshape(sum_z, 1, q), I_m)
    @views H[1:m, (m + 1):d] .= cross
    @views H[(m + 1):d, 1:m] .= transpose(cross)
    second = transpose(Zt) * Zt
    @views H[(m + 1):d, (m + 1):d] .= kron(second, I_m)
    return Symmetric(H)
end

function stacked_beta_diagonal(Ztrain::AbstractMatrix, m::Int)
    n = size(Ztrain, 1)
    z2 = vec(sum(abs2, Ztrain; dims=1))
    return vcat(fill(Float64(n), m), repeat(z2; inner=m))
end

function reduced_design_diagonal(Xtrain::AbstractMatrix, Ztrain::AbstractMatrix)
    n, m = size(Xtrain)
    q = size(Ztrain, 2)
    out = Vector{Float64}(undef, m + m * q)
    x2 = Xtrain .^ 2
    @views out[1:m] .= vec(sum(x2; dims=1))
    pos = m + 1
    for k in 1:q
        zk2 = reshape(Float64.(@view(Ztrain[:, k])) .^ 2, :, 1)
        @views out[pos:(pos + m - 1)] .= vec(sum(x2 .* zk2; dims=1))
        pos += m
    end
    return out
end

function explicit_beta_map(Z::AbstractMatrix, idx::AbstractVector{Int}, m::Int)
    q = size(Z, 2)
    d = m + m * q
    B = zeros(Float64, length(idx) * m, d)
    for (r, t) in enumerate(idx)
        rows = ((r - 1) * m + 1):(r * m)
        view(B, rows, :) .= beta_map(view(Z, t, :), m)
    end
    return B
end

function causal_design(X::AbstractMatrix, y::AbstractVector, tau::Int, idx::AbstractVector{Int};
                       availability_lag::Int=1,
                       event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                       normalize_history::Bool=false,
                       require_complete_history::Bool=true)
    Z, counts = build_error_histories(X, y, tau;
        availability_lag=availability_lag,
        event_ids=event_ids,
        normalize=normalize_history,
        require_complete=require_complete_history)
    A = reduced_design(X, Z, idx)
    return A, Z, counts
end

function unpack_theta(theta::AbstractVector, m::Int, tau::Int)
    q = m * tau
    length(theta) == m + m * q || throw(DimensionMismatch("theta has wrong length"))
    beta0 = Vector{Float64}(theta[1:m])
    V = reshape(Vector{Float64}(theta[(m + 1):end]), m, q)
    return beta0, V
end

function prediction_from_history(beta0::AbstractVector, V::AbstractMatrix,
                                 x::AbstractVector, z::AbstractVector)
    return dot(x, beta0 + V * z)
end

function adaptive_coefficients(model::AdaptiveRidgeModel, X::AbstractMatrix,
                               y_history::AbstractVector, idx::AbstractVector{Int};
                               event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                               normalize_history::Bool=false)
    Z, counts = build_error_histories(X, y_history, model.tau;
        availability_lag=model.availability_lag,
        event_ids=event_ids,
        normalize=normalize_history)
    W = Matrix{Float64}(undef, length(idx), size(X, 2))
    for (r, t) in enumerate(idx)
        @views W[r, :] .= model.beta0 .+ model.V * Z[t, :]
    end
    return W, counts[idx]
end
