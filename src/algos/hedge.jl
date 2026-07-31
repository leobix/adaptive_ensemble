function compute_hedge_weights(X, y; η=nothing)
    # Hedge (Exponentially Weighted Average) algorithm, full-information
    # Sequential update: w_{t+1,i} ∝ w_{t,i} * exp(-η * ℓ_{t,i})
    # ℓ_{t,i} = (X[t,i] - y[t])^2 (squared loss)
    # Inputs: X::Matrix (n x p) expert predictions, y::Vector (n)
    # Output: β::Vector (p), nonnegative and sums to 1 (final weights)

    n, p = size(X)
    η === nothing && (η = sqrt(8log(p)/max(n,1)))

    # initialize uniform weights
    w = ones(p) / p

    for t in 1:n
        losses_t = abs2.(X[t, :] .- y[t])
        # multiplicative update
        w .*= exp.(-η .* losses_t)
        Z = sum(w)
        if Z == 0 || !isfinite(Z)
            w .= 1/p
        else
            w ./= Z
        end
    end

    return w
end
