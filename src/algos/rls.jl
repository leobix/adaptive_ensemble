function rls_init(m; δ=1000.0)
    # Initialize RLS weights and covariance
    return zeros(m), δ .* I(m)
end

function rls_update(w, P, x, y, λ)
    # One step RLS with forgetting factor λ ∈ (0,1]
    # Inputs: w::Vector(m), P::Matrix(m,m), x::Vector(m), y::Float64
    # Returns: new_w, new_P
    denom = λ + dot(x, P * x)
    K = (P * x) / denom
    yhat = dot(w, x)
    e = y - yhat
    w_new = w + K * e
    P_new = (P - K * (x' * P)) / λ
    return w_new, P_new
end

function rls_fit_sequence(X, y; λ=0.99, δ=1000.0)
    # Fit RLS sequentially over all rows of X
    n, m = size(X)
    w, P = rls_init(m; δ=δ)
    for t in 1:n
        w, P = rls_update(w, P, vec(X[t, :]), y[t], λ)
    end
    return w, P
end

