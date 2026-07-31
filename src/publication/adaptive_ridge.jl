function _stable_linear_solve(G::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
                              jitter::Float64=1e-10)
    d = size(G, 1)
    d == size(G, 2) || throw(DimensionMismatch("G must be square"))
    base = Symmetric(Matrix{Float64}(G))
    base_factor = cholesky(base; check=false)
    if issuccess(base_factor)
        return base_factor \ Float64.(b), "cholesky"
    end
    scale = max(maximum(abs, diag(G)), 1.0)
    for k in 0:7
        j = jitter * scale * 10.0^k
        M = Symmetric(Matrix{Float64}(G) + j * Matrix{Float64}(I, d, d))
        factor = cholesky(M; check=false)
        if issuccess(factor)
            return factor \ Float64.(b), "cholesky+jitter=$(j)"
        end
    end
    M = Matrix{Float64}(G) + jitter * scale * Matrix{Float64}(I, d, d)
    return qr(M, ColumnNorm()) \ Float64.(b), "pivoted_qr"
end

function _condition_number(G)
    try
        return cond(Matrix(G))
    catch
        return NaN
    end
end

function _data_transpose_product(Xtrain::AbstractMatrix, Ztrain::AbstractMatrix,
                                 r::AbstractVector)
    weighted = Float64.(Xtrain) .* reshape(Float64.(r), :, 1)
    gb = vec(sum(weighted; dims=1))
    gV = transpose(weighted) * Float64.(Ztrain)
    return vcat(gb, vec(gV))
end

"""
Prepare one reusable quadratic workspace for a fixed `τ`. Dense normal equations are
materialized only when the reduced parameter dimension and estimated memory are below
the configured limits. Larger positive-lambda stacked-beta problems can use the exact
observation-space dual; matrix-free preconditioned conjugate gradients remain the fallback.
"""
function prepare_adaptive_quadratic_workspace(X::AbstractMatrix, y::AbstractVector,
                                              idx::AbstractVector{Int}, tau::Int;
                                              availability_lag::Int=1,
                                              event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                              normalize_history::Bool=false,
                                              direct_max_parameters::Int=3500,
                                              max_dense_bytes::Int=750_000_000)
    validate_xy(X, y)
    isempty(idx) && throw(ArgumentError("training indices cannot be empty"))
    issorted(idx) || throw(ArgumentError("training indices must be chronological"))
    length(unique(idx)) == length(idx) || throw(ArgumentError("training indices must be unique"))
    minimum(idx) >= 1 && maximum(idx) <= length(y) || throw(BoundsError(y, idx))
    t0 = time_ns()
    Z, _ = build_error_histories(X, y, tau;
        availability_lag=availability_lag,
        event_ids=event_ids,
        normalize=normalize_history)
    Xtrain = Matrix{Float64}(X[idx, :])
    Ztrain = Matrix{Float64}(Z[idx, :])
    ytrain = Float64.(y[idx])
    n, m = size(Xtrain)
    q = size(Ztrain, 2)
    d = m + m * q
    estimated_dense_bytes = 8 * d * d * 3
    direct = d <= direct_max_parameters && estimated_dense_bytes <= max_dense_bytes
    diagonal_data = reduced_design_diagonal(Xtrain, Ztrain) ./ n
    diagonal_beta = stacked_beta_diagonal(Ztrain, m) ./ n
    A = nothing
    gram_data = nothing
    gram_beta = nothing
    rhs = Vector{Float64}()
    if direct
        A = reduced_design(Xtrain, Ztrain, collect(1:n))
        gram_data = Matrix(transpose(A) * A) ./ n
        gram_beta = Matrix(stacked_beta_gram(Ztrain, collect(1:n), m)) ./ n
        rhs = Vector(transpose(A) * ytrain) ./ n
    else
        rhs = _data_transpose_product(Xtrain, Ztrain, ytrain) ./ n
    end
    elapsed = (time_ns() - t0) / 1e9
    return AdaptiveQuadraticWorkspace(
        Xtrain=Xtrain, Ztrain=Ztrain, ytrain=ytrain, indices=collect(idx), tau=tau,
        availability_lag=availability_lag, m=m, q=q, d=d, n=n, direct=direct,
        A=A, gram_data=gram_data, gram_beta=gram_beta, rhs=rhs,
        diagonal_data=diagonal_data, diagonal_beta=diagonal_beta,
        preparation_seconds=elapsed,
    )
end

function _normal_product(work::AdaptiveQuadraticWorkspace, theta::AbstractVector,
                         lambda::Float64, penalty::Symbol,
                         lambda_beta0::Float64, lambda_V::Float64,
                         jitter::Float64)
    length(theta) == work.d || throw(DimensionMismatch("theta length differs from workspace"))
    beta0 = @view theta[1:work.m]
    V = reshape(@view(theta[(work.m + 1):end]), work.m, work.q)
    beta_rows = work.Ztrain * transpose(V)
    beta_rows .+= transpose(beta0)
    pred = vec(sum(work.Xtrain .* beta_rows; dims=2))
    out = _data_transpose_product(work.Xtrain, work.Ztrain, pred) ./ work.n
    if penalty == :stacked_beta
        gb = vec(sum(beta_rows; dims=1)) ./ work.n
        gV = transpose(beta_rows) * work.Ztrain ./ work.n
        out .+= lambda .* vcat(gb, vec(gV))
    elseif penalty == :parameters
        @views out[1:work.m] .+= lambda_beta0 .* beta0
        @views out[(work.m + 1):end] .+= lambda_V .* vec(V)
    elseif penalty == :identity
        out .+= lambda .* theta
    else
        throw(ArgumentError("unknown penalty $(penalty)"))
    end
    jitter > 0 && (out .+= jitter .* theta)
    return out
end


"""Return a numerically stable Cholesky factor and the diagonal shift used."""
function _stable_spd_cholesky(G::AbstractMatrix{<:Real};
                              jitter::Float64=1e-12,
                              max_attempts::Int=8)
    size(G, 1) == size(G, 2) || throw(DimensionMismatch("G must be square"))
    base = Symmetric(Matrix{Float64}(G))
    factor = cholesky(base; check=false)
    issuccess(factor) && return factor, 0.0

    scale = max(maximum(abs, diag(G)), 1.0)
    d = size(G, 1)
    identity = Matrix{Float64}(I, d, d)
    for attempt in 0:(max_attempts - 1)
        shift = jitter * scale * 10.0^attempt
        shifted = Symmetric(Matrix{Float64}(G) + shift .* identity)
        factor = cholesky(shifted; check=false)
        issuccess(factor) && return factor, shift
    end
    return nothing, NaN
end

"""
Prepare the exact dual kernel for the stacked-beta penalty.

For `w_t = [1; z_t]`, the stacked-coefficient Gram matrix is
`R = (W'W/n) ⊗ I_m`.  When `R` is positive definite and `lambda > 0`,

    (A'A/n + lambda R) theta = A'y/n

can be solved in the observation-space dual without forming the `d × d` Gram matrix:

    (A R^{-1} A' + n lambda I) alpha = y,
    theta = R^{-1} A' alpha.

The kernel is `(W H^{-1} W') .* (X X')`, where `H = W'W/n`.  This avoids
normal-equation PCG stagnation in the publication sweeps with `m=25,50` or
`tau=20,25`, while remaining algebraically identical to the stated quadratic objective.
"""
function _prepare_stacked_beta_dual(work::AdaptiveQuadraticWorkspace;
                                    factor_jitter::Float64=1e-12,
                                    dual_max_rows::Int=3500,
                                    decompose::Bool=true)
    work.n <= dual_max_rows || return nothing
    p = work.q + 1
    p <= work.n || return nothing

    started = time_ns()
    W = hcat(ones(Float64, work.n), work.Ztrain)
    H = Matrix(transpose(W) * W) ./ work.n
    factor, shift = _stable_spd_cholesky(H; jitter=factor_jitter)
    factor === nothing && return nothing

    # If a material shift was needed, use PCG instead of silently changing the
    # scientific regularizer. A tiny round-off shift is harmless and is recorded.
    scale = max(maximum(abs, diag(H)), 1.0)
    shift <= 1e-10 * scale || return nothing

    # H = U'U and Q = W/U, so Q*Q' = W*H^{-1}*W'.
    Q = W / factor.U
    kernel = Q * transpose(Q)
    kernel .*= work.Xtrain * transpose(work.Xtrain)
    kernel .= 0.5 .* (kernel .+ transpose(kernel))

    eigenvalues = nothing
    eigenvectors = nothing
    projected_y = nothing
    if decompose
        decomposition = eigen(Symmetric(kernel))
        eigenvalues = max.(Vector{Float64}(decomposition.values), 0.0)
        eigenvectors = Matrix{Float64}(decomposition.vectors)
        projected_y = transpose(eigenvectors) * work.ytrain
    end
    elapsed = (time_ns() - started) / 1e9

    return (
        W=W,
        H_factor=factor,
        H_shift=shift,
        kernel=kernel,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_y=projected_y,
        preparation_seconds=elapsed,
    )
end

function _pcg(op::Function, b::AbstractVector;
              diagonal::AbstractVector,
              x0::Union{Nothing,AbstractVector}=nothing,
              rtol::Float64=1e-9,
              atol::Float64=1e-12,
              maxiter::Int=max(1000, min(20000, 4 * length(b))),
              residual_recompute_interval::Int=25)
    n = length(b)

    length(diagonal) == n ||
        throw(DimensionMismatch("preconditioner has wrong length"))

    rtol >= 0 ||
        throw(ArgumentError("rtol must be nonnegative"))

    atol >= 0 ||
        throw(ArgumentError("atol must be nonnegative"))

    maxiter >= 1 ||
        throw(ArgumentError("maxiter must be positive"))

    residual_recompute_interval >= 0 ||
        throw(ArgumentError(
            "residual_recompute_interval must be nonnegative"
        ))

    rhs = Float64.(b)

    x = if x0 === nothing
        zeros(Float64, n)
    else
        length(x0) == n ||
            throw(DimensionMismatch("initial guess has wrong length"))

        Float64.(x0)
    end

    preconditioner = Float64.(diagonal)

    all(isfinite, preconditioner) ||
        throw(ArgumentError(
            "preconditioner contains a nonfinite value"
        ))

    all(preconditioner .> 0.0) ||
        throw(ArgumentError(
            "PCG requires a strictly positive diagonal preconditioner"
        ))

    # Use the standard right-hand-side-relative stopping rule. In particular,
    # a warm start must not make the requested tolerance arbitrarily stricter.
    r = rhs .- op(x)

    initial_residual_norm = norm(r)
    rhs_norm = norm(rhs)
    target = max(atol, rtol * rhs_norm)

    if initial_residual_norm <= target
        return x, 0, true, initial_residual_norm
    end

    z = r ./ preconditioner
    p = copy(z)

    rz = dot(r, z)

    if !isfinite(rz) || rz <= 0.0
        return x, 0, false, initial_residual_norm
    end

    residual_norm = initial_residual_norm
    iteration = 0

    for k in 1:maxiter
        Ap = op(p)
        denominator = dot(p, Ap)

        # For an SPD operator, p'Ap must be positive. Do not reject a
        # small but valid positive value merely because p itself is small.
        if !isfinite(denominator) || denominator <= 0.0
            # Recompute the true residual before declaring breakdown.
            r .= rhs .- op(x)
            residual_norm = norm(r)

            if residual_norm <= target
                return x, k - 1, true, residual_norm
            end

            # Restart from the preconditioned residual. This can recover
            # from loss of conjugacy caused by finite-precision arithmetic.
            z .= r ./ preconditioner
            p .= z
            rz = dot(r, z)

            if !isfinite(rz) || rz <= 0.0
                return x, k - 1, false, residual_norm
            end

            continue
        end

        alpha = rz / denominator

        if !isfinite(alpha)
            r .= rhs .- op(x)
            residual_norm = norm(r)
            return x, k - 1, residual_norm <= target, residual_norm
        end

        x .+= alpha .* p

        restarted = false

        # The short recurrence is inexpensive, but it accumulates rounding
        # error. Periodically recompute b - A*x and restart the direction.
        if residual_recompute_interval > 0 &&
           k % residual_recompute_interval == 0
            r .= rhs .- op(x)
            restarted = true
        else
            r .-= alpha .* Ap
        end

        residual_norm = norm(r)
        iteration = k

        if residual_norm <= target
            # Confirm convergence using the true residual rather than only
            # the recursively updated residual.
            r .= rhs .- op(x)
            residual_norm = norm(r)

            if residual_norm <= target
                return x, iteration, true, residual_norm
            end

            restarted = true
        end

        z .= r ./ preconditioner
        rz_new = dot(r, z)

        if !isfinite(rz_new) || rz_new <= 0.0
            r .= rhs .- op(x)
            residual_norm = norm(r)
            return x, iteration, residual_norm <= target, residual_norm
        end

        if restarted
            p .= z
        else
            beta = rz_new / rz

            if !isfinite(beta)
                r .= rhs .- op(x)
                residual_norm = norm(r)
                return x, iteration, residual_norm <= target, residual_norm
            end

            p .= z .+ beta .* p
        end

        rz = rz_new
    end

    # Always report the true final residual.
    r .= rhs .- op(x)
    residual_norm = norm(r)

    return x, iteration, residual_norm <= target, residual_norm
end
function _quadratic_objective(work::AdaptiveQuadraticWorkspace, theta::AbstractVector,
                              lambda::Float64, penalty::Symbol,
                              lambda_beta0::Float64, lambda_V::Float64)
    beta0 = @view theta[1:work.m]
    V = reshape(@view(theta[(work.m + 1):end]), work.m, work.q)
    beta_rows = work.Ztrain * transpose(V)
    beta_rows .+= transpose(beta0)
    pred = vec(sum(work.Xtrain .* beta_rows; dims=2))
    value = mean(abs2, work.ytrain .- pred)
    if penalty == :stacked_beta
        value += lambda * sum(abs2, beta_rows) / work.n
    elseif penalty == :parameters
        value += lambda_beta0 * sum(abs2, beta0) + lambda_V * sum(abs2, V)
    elseif penalty == :identity
        value += lambda * sum(abs2, theta)
    else
        throw(ArgumentError("unknown penalty $(penalty)"))
    end
    return Float64(value)
end

function _fit_quadratic_dual(work::AdaptiveQuadraticWorkspace, lambda::Real,
                             dual_state;
                             penalty::Symbol=:stacked_beta,
                             lambda_beta0::Real=lambda,
                             lambda_V::Real=lambda,
                             residual_rtol::Float64=1e-8)
    penalty == :stacked_beta ||
        throw(ArgumentError("dual solver is available only for penalty=:stacked_beta"))
    lambda > 0 || throw(ArgumentError("dual solver requires lambda > 0"))
    dual_state === nothing && throw(ArgumentError("dual solver state is unavailable"))

    l = Float64(lambda)
    started = time_ns()
    dual_shift = work.n * l
    alpha = Vector{Float64}()
    solver = ""
    condition_number = NaN
    if dual_state.eigenvalues === nothing
        system = copy(dual_state.kernel)
        @inbounds for i in axes(system, 1)
            system[i, i] += dual_shift
        end
        alpha, linear_solver = _stable_linear_solve(system, work.ytrain; jitter=1e-12)
        solver = "dual_$(linear_solver)_stacked_beta"
        size(system, 1) <= 1200 && (condition_number = _condition_number(system))
    else
        denominators = dual_state.eigenvalues .+ dual_shift
        minimum(denominators) > 0 || error("dual system is not positive definite")
        alpha = dual_state.eigenvectors * (dual_state.projected_y ./ denominators)
        solver = "dual_eigen_stacked_beta"
        smallest = minimum(denominators)
        largest = maximum(denominators)
        condition_number = largest / max(smallest, eps(Float64))
    end

    weighted_history = reshape(alpha, :, 1) .* dual_state.W
    gradient_matrix = transpose(work.Xtrain) * weighted_history
    # Materialize the logical m-by-(q+1) transpose before vectorizing so the
    # parameter order is exactly [beta0; vec(V)] in column-major order.
    theta_matrix = Matrix(transpose(
        dual_state.H_factor \ transpose(gradient_matrix)))
    theta = vec(theta_matrix)

    residual = _normal_product(work, theta, l, penalty,
        Float64(lambda_beta0), Float64(lambda_V), 0.0) .- work.rhs
    residual_norm = norm(residual)
    relative_residual = residual_norm / max(norm(work.rhs), eps(Float64))
    relative_residual <= residual_rtol || error(
        "dual stacked-beta solve failed residual check: " *
        "relative_residual=$(relative_residual), tolerance=$(residual_rtol), " *
        "parameters=$(work.d), observations=$(work.n)")

    solve_seconds = (time_ns() - started) / 1e9
    beta0, V = unpack_theta(theta, work.m, work.tau)
    objective = _quadratic_objective(work, theta, l, penalty,
        Float64(lambda_beta0), Float64(lambda_V))
    return AdaptiveRidgeModel(
        beta0=beta0, V=V, theta=theta, tau=work.tau, lambda=l,
        penalty=penalty, objective_variant=:quadratic,
        availability_lag=work.availability_lag, warmup=:static,
        objective=objective,
        fit_seconds=work.preparation_seconds + dual_state.preparation_seconds + solve_seconds,
        preparation_seconds=work.preparation_seconds + dual_state.preparation_seconds,
        solve_seconds=solve_seconds,
        linear_solver=solver,
        condition_number=condition_number,
        iterations=1, converged=true, residual_norm=residual_norm,
        n_parameters=work.d,
    )
end

function _fit_quadratic_workspace(work::AdaptiveQuadraticWorkspace, lambda::Real;
                                  penalty::Symbol=:stacked_beta,
                                  lambda_beta0::Real=lambda,
                                  lambda_V::Real=lambda,
                                  jitter::Float64=1e-10,
                                  cg_rtol::Float64=1e-9,
                                  cg_atol::Float64=1e-12,
                                  cg_maxiter::Int=max(1000, min(20000, 4 * work.d)),
                                  initial_theta::Union{Nothing,AbstractVector}=nothing,
                                  allow_dual::Bool=true,
                                  dual_state=nothing,
                                  dual_max_rows::Int=3500)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    l = Float64(lambda)
    l0 = Float64(lambda_beta0)
    lV = Float64(lambda_V)

    if !work.direct && allow_dual && penalty == :stacked_beta && l > 0
        state = dual_state === nothing ?
            _prepare_stacked_beta_dual(work; dual_max_rows=dual_max_rows,
                decompose=false) : dual_state
        if state !== nothing
            return _fit_quadratic_dual(work, l, state;
                penalty=penalty, lambda_beta0=l0, lambda_V=lV,
                residual_rtol=max(1e-8, 10 * cg_rtol))
        end
    end

    t0 = time_ns()
    solver = ""
    iterations = 0
    converged = true
    residual_norm = 0.0
    condition_number = NaN
    theta = Vector{Float64}()
    if work.direct
        G = copy(work.gram_data::Matrix{Float64})
        if penalty == :stacked_beta
            G .+= l .* (work.gram_beta::Matrix{Float64})
        elseif penalty == :parameters
            for j in 1:work.m
                G[j, j] += l0
            end
            for j in (work.m + 1):work.d
                G[j, j] += lV
            end
        elseif penalty == :identity
            for j in 1:work.d
                G[j, j] += l
            end
        else
            throw(ArgumentError("unknown penalty $(penalty)"))
        end
        theta, solver = _stable_linear_solve(G, work.rhs; jitter=jitter)
        residual_norm = norm(G * theta - work.rhs)
        work.d <= 1200 && (condition_number = _condition_number(G))
    else
        diagonal = copy(work.diagonal_data)
        if penalty == :stacked_beta
            diagonal .+= l .* work.diagonal_beta
        elseif penalty == :parameters
            @views diagonal[1:work.m] .+= l0
            @views diagonal[(work.m + 1):end] .+= lV
        elseif penalty == :identity
            diagonal .+= l
        else
            throw(ArgumentError("unknown penalty $(penalty)"))
        end
        diagonal .+= jitter
        op = theta_in -> _normal_product(work, theta_in, l, penalty, l0, lV, jitter)
        theta, iterations, converged, residual_norm = _pcg(op, work.rhs;
            diagonal=diagonal, x0=initial_theta, rtol=cg_rtol, atol=cg_atol,
            maxiter=cg_maxiter)
        solver = "matrix_free_pcg"
        if !converged
            relative_residual =
                residual_norm / max(norm(work.rhs), eps(Float64))

            error(
                "matrix-free PCG did not converge: " *
                "residual=$(residual_norm), " *
                "relative_residual=$(relative_residual), " *
                "iterations=$(iterations), " *
                "rtol=$(cg_rtol), " *
                "atol=$(cg_atol), " *
                "maxiter=$(cg_maxiter), " *
                "parameters=$(work.d)"
            )
        end
    end
    solve_seconds = (time_ns() - t0) / 1e9
    beta0, V = unpack_theta(theta, work.m, work.tau)
    objective = _quadratic_objective(work, theta, l, penalty, l0, lV)
    return AdaptiveRidgeModel(
        beta0=beta0, V=V, theta=theta, tau=work.tau, lambda=l,
        penalty=penalty, objective_variant=:quadratic,
        availability_lag=work.availability_lag, warmup=:static,
        objective=objective, fit_seconds=work.preparation_seconds + solve_seconds,
        preparation_seconds=work.preparation_seconds, solve_seconds=solve_seconds,
        linear_solver=solver, condition_number=condition_number,
        iterations=iterations, converged=converged, residual_norm=residual_norm,
        n_parameters=work.d,
    )
end

"""
Fit the smooth quadratic implementation in the reduced parameterization. This is kept
separate from the paper's exact norm-plus-norm robust counterpart.
"""
function fit_adaptive_quadratic(X::AbstractMatrix, y::AbstractVector,
                                idx::AbstractVector{Int}, tau::Int, lambda::Real;
                                availability_lag::Int=1,
                                event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                penalty::Symbol=:stacked_beta,
                                lambda_beta0::Real=lambda,
                                lambda_V::Real=lambda,
                                normalize_history::Bool=false,
                                direct_max_parameters::Int=3500,
                                max_dense_bytes::Int=750_000_000,
                                jitter::Float64=1e-10,
                                cg_rtol::Float64=1e-9,
                                cg_atol::Float64=1e-12,
                                cg_maxiter::Int=0,
                                allow_dual::Bool=true,
                                dual_max_rows::Int=3500)
    work = prepare_adaptive_quadratic_workspace(X, y, idx, tau;
        availability_lag=availability_lag, event_ids=event_ids,
        normalize_history=normalize_history,
        direct_max_parameters=direct_max_parameters,
        max_dense_bytes=max_dense_bytes)
    maxiter = cg_maxiter > 0 ? cg_maxiter : max(1000, min(20000, 4 * work.d))
    return _fit_quadratic_workspace(work, lambda;
        penalty=penalty, lambda_beta0=lambda_beta0, lambda_V=lambda_V,
        jitter=jitter, cg_rtol=cg_rtol, cg_atol=cg_atol,
        cg_maxiter=maxiter, allow_dual=allow_dual,
        dual_max_rows=dual_max_rows)
end

"""Fit a complete regularization path while reusing the causal design and Gram matrices."""
function fit_adaptive_quadratic_path(work::AdaptiveQuadraticWorkspace,
                                     lambdas::AbstractVector{<:Real};
                                     penalty::Symbol=:stacked_beta,
                                     jitter::Float64=1e-10,
                                     cg_rtol::Float64=1e-9,
                                     cg_atol::Float64=1e-12,
                                     cg_maxiter::Int=0,
                                     allow_dual::Bool=true,
                                     dual_max_rows::Int=3500)
    all(Float64.(lambdas) .>= 0) || throw(ArgumentError("lambdas must be nonnegative"))
    order = sortperm(Float64.(lambdas); rev=true)
    models = Vector{AdaptiveRidgeModel}(undef, length(lambdas))
    warm = nothing
    maxiter = cg_maxiter > 0 ? cg_maxiter : max(1000, min(20000, 4 * work.d))
    dual_state = if !work.direct && allow_dual && penalty == :stacked_beta &&
                    any(Float64(lambda) > 0 for lambda in lambdas)
        _prepare_stacked_beta_dual(work; dual_max_rows=dual_max_rows,
            decompose=true)
    else
        nothing
    end
    for i in order
        model = _fit_quadratic_workspace(work, Float64(lambdas[i]);
            penalty=penalty, jitter=jitter, cg_rtol=cg_rtol,
            cg_atol=cg_atol, cg_maxiter=maxiter, initial_theta=warm,
            allow_dual=allow_dual, dual_state=dual_state,
            dual_max_rows=dual_max_rows)
        models[i] = model
        warm = model.theta
    end
    return models
end


function _adaptive_norm_components(work::AdaptiveQuadraticWorkspace,
                                   theta::AbstractVector)
    length(theta) == work.d || throw(DimensionMismatch("theta length differs from workspace"))
    beta0 = @view theta[1:work.m]
    V = reshape(@view(theta[(work.m + 1):end]), work.m, work.q)
    beta_rows = work.Ztrain * transpose(V)
    beta_rows .+= transpose(beta0)
    prediction = vec(sum(work.Xtrain .* beta_rows; dims=2))
    residual = work.ytrain .- prediction
    return norm(residual), norm(beta_rows), residual, beta_rows
end

function _adaptive_norm_stationarity(work::AdaptiveQuadraticWorkspace,
                                     theta::AbstractVector,
                                     lambda::Float64,
                                     smoothing::Float64)
    residual_norm, coefficient_norm, _, _ =
        _adaptive_norm_components(work, theta)
    residual_scale = hypot(residual_norm, smoothing)
    coefficient_scale = hypot(coefficient_norm, smoothing)
    effective_lambda = lambda * residual_scale / coefficient_scale

    # The gradient of the smoothed objective is zero if and only if
    #
    #   (G_data + effective_lambda * G_beta) * theta = rhs.
    #
    # For direct workspaces, evaluate the certificate with the same stored Gram
    # matrices used by the factorization. Recomputing A'A*theta and B'B*theta
    # matrix-free can have a materially higher round-off floor on the strongly
    # collinear energy design, even when the solved normal system is accurate.
    # Non-direct workspaces retain the matrix-free product because dense Grams are
    # deliberately unavailable.
    normal_residual = if work.direct
        gram_data = work.gram_data::Matrix{Float64}
        gram_beta = work.gram_beta::Matrix{Float64}
        (gram_data * theta) .+
            effective_lambda .* (gram_beta * theta) .-
            work.rhs
    else
        _normal_product(
            work, theta, effective_lambda, :stacked_beta,
            effective_lambda, effective_lambda, 0.0,
        ) .- work.rhs
    end
    relative_stationarity = norm(normal_residual) /
        max(norm(work.rhs), eps(Float64))
    return relative_stationarity, effective_lambda,
           residual_norm, coefficient_norm
end

"""Return the accepted IRLS stopping reason, or the empty string if not certified."""
function _robust_norm_convergence_reason(iteration::Int,
                                         relative_theta::Float64,
                                         relative_objective::Float64,
                                         stationarity_residual::Float64;
                                         min_iterations::Int,
                                         iterate_rtol::Float64,
                                         objective_rtol::Float64,
                                         stationarity_rtol::Float64,
                                         practical_stationarity_rtol::Float64)
    iteration >= min_iterations || return ""
    all(isfinite, (relative_theta, relative_objective,
                   stationarity_residual)) || return ""

    iterate_and_objective =
        relative_theta <= iterate_rtol &&
        relative_objective <= objective_rtol
    strict_stationarity = stationarity_residual <= stationarity_rtol
    practical_stationarity =
        relative_objective <= objective_rtol &&
        stationarity_residual <= practical_stationarity_rtol

    if iterate_and_objective && strict_stationarity
        return "iterate_objective_and_strict_stationarity"
    elseif strict_stationarity
        return "strict_smoothed_stationarity"
    elseif iterate_and_objective
        return "iterate_and_objective"
    elseif practical_stationarity
        # This certificate is deliberately conjunctive: a slightly relaxed
        # stationarity residual is accepted only after the smoothed objective has
        # stabilized at the strict publication tolerance. It prevents raw theta
        # motion in nearly unidentified directions from causing false failures.
        return "objective_and_practical_stationarity"
    end
    return ""
end

"""
Fit the paper-aligned nonsquared objective

    ||y - Aθ||₂ + λ ||Bθ||₂

with a smoothed IRLS majorization. Every outer iteration solves the corresponding
quadratic problem in the same reduced parameterization used by
`fit_adaptive_quadratic`.

Convergence is certified by one of three explicit conditions:

1. successive parameters and the smoothed objective both stabilize;
2. the normalized first-order stationarity residual of the smoothed convex
   objective falls below `irls_stationarity_rtol`; or
3. the smoothed objective stabilizes at the strict objective tolerance while the
   stationarity residual is below the separately configured practical tolerance.

The third certificate is deliberately conjunctive: a practical stationarity
residual is accepted only after the objective has stabilized. This prevents false
failures in flat, weakly identified directions without accepting objective
stagnation alone. The independent Gurobi SOCP implementation remains the exact
verification solver.
"""
function fit_adaptive_robust_norm(work::AdaptiveQuadraticWorkspace, lambda::Real;
                                  smoothing::Float64=1e-8,
                                  irls_rtol::Float64=1e-8,
                                  irls_objective_rtol::Float64=1e-8,
                                  irls_stationarity_rtol::Float64=1e-8,
                                  irls_practical_stationarity_rtol::Float64=1e-7,
                                  irls_maxiter::Int=1000,
                                  irls_min_iterations::Int=2,
                                  jitter::Float64=1e-10,
                                  cg_rtol::Float64=1e-9,
                                  cg_atol::Float64=1e-12,
                                  cg_maxiter::Int=0,
                                  allow_dual::Bool=true,
                                  dual_max_rows::Int=3500,
                                  initial_theta::Union{Nothing,AbstractVector}=nothing,
                                  fallback_solver::Symbol=:none,
                                  gurobi_output_flag::Int=0,
                                  gurobi_time_limit::Real=Inf,
                                  gurobi_threads::Int=1,
                                  gurobi_optimality_tolerance::Float64=1e-8,
                                  require_convergence::Bool=true)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    smoothing > 0 || throw(ArgumentError("smoothing must be positive"))
    irls_rtol > 0 || throw(ArgumentError("irls_rtol must be positive"))
    irls_objective_rtol > 0 ||
        throw(ArgumentError("irls_objective_rtol must be positive"))
    irls_stationarity_rtol > 0 ||
        throw(ArgumentError("irls_stationarity_rtol must be positive"))
    irls_practical_stationarity_rtol >= irls_stationarity_rtol ||
        throw(ArgumentError(
            "irls_practical_stationarity_rtol must be at least irls_stationarity_rtol"))
    irls_maxiter >= 1 || throw(ArgumentError("irls_maxiter must be positive"))
    1 <= irls_min_iterations <= irls_maxiter ||
        throw(ArgumentError("irls_min_iterations must be in 1:irls_maxiter"))
    fallback_solver in (:none, :gurobi) ||
        throw(ArgumentError("fallback_solver must be :none or :gurobi"))
    gurobi_threads >= 1 ||
        throw(ArgumentError("gurobi_threads must be positive"))
    gurobi_optimality_tolerance > 0 ||
        throw(ArgumentError("gurobi_optimality_tolerance must be positive"))

    maxiter = cg_maxiter > 0 ? cg_maxiter : max(1000, min(20000, 4 * work.d))
    started = time_ns()
    dual_state = if !work.direct && allow_dual
        _prepare_stacked_beta_dual(work; dual_max_rows=dual_max_rows,
            decompose=true)
    else
        nothing
    end

    theta = if initial_theta === nothing
        initializer = _fit_quadratic_workspace(work, max(Float64(lambda), 1e-8);
            penalty=:stacked_beta, jitter=jitter, cg_rtol=cg_rtol,
            cg_atol=cg_atol, cg_maxiter=maxiter,
            allow_dual=allow_dual, dual_state=dual_state,
            dual_max_rows=dual_max_rows)
        copy(initializer.theta)
    else
        length(initial_theta) == work.d ||
            throw(DimensionMismatch("initial_theta length differs from workspace"))
        Float64.(initial_theta)
    end

    previous_smoothed_objective = Inf
    exact_objective = Inf
    smoothed_objective = Inf
    final_quadratic = nothing
    total_solve_seconds = 0.0
    converged = false
    convergence_reason = ""
    outer_iterations = 0
    relative_theta = Inf
    relative_objective = Inf
    stationarity_residual = Inf

    for iteration in 1:irls_maxiter
        residual_norm, coefficient_norm, _, _ =
            _adaptive_norm_components(work, theta)
        effective_lambda = Float64(lambda) *
            hypot(residual_norm, smoothing) /
            hypot(coefficient_norm, smoothing)

        quadratic = _fit_quadratic_workspace(work, effective_lambda;
            penalty=:stacked_beta, jitter=jitter, cg_rtol=cg_rtol,
            cg_atol=cg_atol, cg_maxiter=maxiter, initial_theta=theta,
            allow_dual=allow_dual, dual_state=dual_state,
            dual_max_rows=dual_max_rows)
        total_solve_seconds += quadratic.solve_seconds
        new_theta = quadratic.theta

        stationarity_residual, _, new_residual_norm, new_coefficient_norm =
            _adaptive_norm_stationarity(
                work, new_theta, Float64(lambda), smoothing)

        exact_objective =
            new_residual_norm + Float64(lambda) * new_coefficient_norm
        smoothed_objective =
            hypot(new_residual_norm, smoothing) +
            Float64(lambda) * hypot(new_coefficient_norm, smoothing)
        relative_theta =
            norm(new_theta - theta) / max(norm(theta), 1.0)
        relative_objective = isfinite(previous_smoothed_objective) ?
            abs(previous_smoothed_objective - smoothed_objective) /
                max(abs(previous_smoothed_objective), 1.0) : Inf

        theta = copy(new_theta)
        final_quadratic = quadratic
        previous_smoothed_objective = smoothed_objective
        outer_iterations = iteration

        convergence_reason = _robust_norm_convergence_reason(
            iteration, relative_theta, relative_objective,
            stationarity_residual;
            min_iterations=irls_min_iterations,
            iterate_rtol=irls_rtol,
            objective_rtol=irls_objective_rtol,
            stationarity_rtol=irls_stationarity_rtol,
            practical_stationarity_rtol=
                irls_practical_stationarity_rtol,
        )
        if !isempty(convergence_reason)
            converged = true
            break
        end
    end

    outer_change = max(relative_theta, relative_objective)
    if !converged && fallback_solver == :gurobi
        # A failed IRLS certificate is not silently relaxed.  Instead, solve the
        # original nonsmoothed convex objective exactly as a second-order-cone
        # program and retain the final IRLS diagnostics for provenance.
        irls_elapsed = (time_ns() - started) / 1e9
        exact = try
            fit_adaptive_norm_gurobi(work, Float64(lambda);
                output_flag=gurobi_output_flag,
                time_limit=gurobi_time_limit,
                gurobi_threads=gurobi_threads,
                optimality_tolerance=gurobi_optimality_tolerance)
        catch err
            message = sprint(showerror, err, catch_backtrace())
            error(
                "robust-norm IRLS did not converge and the exact Gurobi " *
                "fallback failed: tau=$(work.tau), lambda=$(Float64(lambda)). " *
                "IRLS relative_theta=$(relative_theta), " *
                "relative_objective=$(relative_objective), " *
                "stationarity_residual=$(stationarity_residual). " *
                "Gurobi error: $(message)"
            )
        end
        exact.fit_seconds += irls_elapsed
        exact.solve_seconds += total_solve_seconds
        exact.linear_solver = "irls_limit_then_gurobi_socp"
        exact.iterations = outer_iterations
        exact.residual_norm = outer_change
        exact.stationarity_residual = stationarity_residual
        exact.relative_objective_change = relative_objective
        exact.convergence_reason =
            "gurobi_socp_fallback_after_irls_limit"
        final_quadratic !== nothing &&
            (exact.condition_number = final_quadratic.condition_number)
        return exact
    elseif !converged && require_convergence
        error(
            "robust-norm IRLS did not converge: " *
            "tau=$(work.tau), lambda=$(Float64(lambda)), " *
            "iterations=$(irls_maxiter), " *
            "relative_theta=$(relative_theta), " *
            "relative_objective=$(relative_objective), " *
            "stationarity_residual=$(stationarity_residual), " *
            "iterate_tolerance=$(irls_rtol), " *
            "objective_tolerance=$(irls_objective_rtol), " *
            "strict_stationarity_tolerance=$(irls_stationarity_rtol), " *
            "practical_stationarity_tolerance=$(irls_practical_stationarity_rtol), " *
            "smoothing=$(smoothing), fallback_solver=$(fallback_solver)"
        )
    elseif !converged
        convergence_reason = "iteration_limit"
        @warn "robust-norm IRLS reached the iteration limit" tau=work.tau lambda=Float64(lambda) iterations=outer_iterations relative_change=outer_change stationarity_residual=stationarity_residual
    end

    final_quadratic === nothing &&
        error("robust-norm IRLS produced no quadratic iterate")
    beta0, V = unpack_theta(theta, work.m, work.tau)
    total_seconds = (time_ns() - started) / 1e9 + work.preparation_seconds
    return AdaptiveRidgeModel(
        beta0=beta0,
        V=V,
        theta=theta,
        tau=work.tau,
        lambda=Float64(lambda),
        penalty=:norm_stacked_beta,
        objective_variant=:norm_irls,
        availability_lag=work.availability_lag,
        warmup=:static,
        objective=Float64(exact_objective),
        fit_seconds=total_seconds,
        preparation_seconds=work.preparation_seconds,
        solve_seconds=total_solve_seconds,
        linear_solver="irls+" * final_quadratic.linear_solver,
        condition_number=final_quadratic.condition_number,
        iterations=outer_iterations,
        converged=converged,
        residual_norm=outer_change,
        stationarity_residual=stationarity_residual,
        relative_objective_change=relative_objective,
        convergence_reason=convergence_reason,
        n_parameters=work.d,
    )
end

function fit_adaptive_robust_norm(X::AbstractMatrix, y::AbstractVector,
                                  idx::AbstractVector{Int}, tau::Int, lambda::Real;
                                  availability_lag::Int=1,
                                  event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                  normalize_history::Bool=false,
                                  direct_max_parameters::Int=3500,
                                  max_dense_bytes::Int=750_000_000,
                                  kwargs...)
    work = prepare_adaptive_quadratic_workspace(X, y, idx, tau;
        availability_lag=availability_lag,
        event_ids=event_ids,
        normalize_history=normalize_history,
        direct_max_parameters=direct_max_parameters,
        max_dense_bytes=max_dense_bytes)
    return fit_adaptive_robust_norm(work, lambda; kwargs...)
end

"""Fit a robust-norm regularization path while reusing one causal design workspace."""
function fit_adaptive_robust_norm_path(work::AdaptiveQuadraticWorkspace,
                                       lambdas::AbstractVector{<:Real}; kwargs...)
    all(Float64.(lambdas) .>= 0) ||
        throw(ArgumentError("lambdas must be nonnegative"))
    order = sortperm(Float64.(lambdas); rev=true)
    models = Vector{AdaptiveRidgeModel}(undef, length(lambdas))
    warm = nothing
    for i in order
        lambda = Float64(lambdas[i])
        model = try
            fit_adaptive_robust_norm(work, lambda;
                initial_theta=warm, kwargs...)
        catch err
            message = sprint(showerror, err, catch_backtrace())
            error(
                "robust-norm path candidate failed: " *
                "tau=$(work.tau), lambda=$(lambda). " *
                "Original error: $(message)"
            )
        end
        models[i] = model
        warm = model.theta
    end
    return models
end


function predict_adaptive(model::AdaptiveRidgeModel, X::AbstractMatrix,
                          y_history::AbstractVector, idx::AbstractVector{Int};
                          event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                          normalize_history::Bool=false)
    Z, _ = build_error_histories(X, y_history, model.tau;
        availability_lag=model.availability_lag,
        event_ids=event_ids,
        normalize=normalize_history)
    preds = Vector{Float64}(undef, length(idx))
    for (r, t) in enumerate(idx)
        preds[r] = prediction_from_history(model.beta0, model.V,
            @view(X[t, :]), @view(Z[t, :]))
    end
    return preds
end

function fit_static_ridge(X::AbstractMatrix, y::AbstractVector,
                          idx::AbstractVector{Int}, lambda::Real;
                          jitter::Float64=1e-10)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    isempty(idx) && throw(ArgumentError("training indices cannot be empty"))
    t0 = time_ns()
    Xt = Matrix{Float64}(X[idx, :])
    yt = Float64.(y[idx])
    n = length(idx)
    m = size(X, 2)
    G = transpose(Xt) * Xt / n + Float64(lambda) * Matrix{Float64}(I, m, m)
    b = transpose(Xt) * yt / n
    w, _ = _stable_linear_solve(G, b; jitter=jitter)
    return StaticRidgeModel(weights=w, lambda=Float64(lambda),
        fit_seconds=(time_ns() - t0) / 1e9)
end

predict_static(model::StaticRidgeModel, X::AbstractMatrix,
               idx::AbstractVector{Int}) = Matrix(X[idx, :]) * model.weights


"""
Fail fast on a missing Gurobi installation or license by solving a tiny second-order
cone problem. Publication profiles that request the exact robust-norm objective call
this before launching any long synthetic or real-world stages.
"""
function gurobi_preflight(; output_flag::Int=0, threads::Int=1)
    try
        model = Model(Gurobi.Optimizer)

        set_optimizer_attribute(model, "OutputFlag", output_flag)
        set_optimizer_attribute(model, "Threads", threads)

        # Continuous QCP/SOCP models are solved by Gurobi's barrier method.
        # Tighten the QCP-specific convergence tolerance for this tiny,
        # well-scaled diagnostic model.
        set_optimizer_attribute(model, "BarQCPConvTol", 1e-10)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-8)
        set_optimizer_attribute(model, "NumericFocus", 1)

        @variable(model, radius >= 0)
        @variable(model, x[1:2])

        @constraint(model, x[1] == 1.0)
        @constraint(model, x[2] == 0.0)
        @constraint(model, [radius; x] in SecondOrderCone())

        @objective(model, Min, radius)

        optimize!(model)

        status = termination_status(model)
        pstatus = primal_status(model)

        is_solved_and_feasible(model; allow_almost=true) ||
            error(
                "Gurobi did not return a solved feasible model: " *
                "termination_status=$(status), " *
                "primal_status=$(pstatus), " *
                "raw_status=$(raw_status(model))"
            )

        radius_value = Float64(value(radius))
        x_value = Vector{Float64}(value.(x))
        objective = Float64(objective_value(model))

        expected_radius = norm(x_value)
        cone_violation = max(expected_radius - radius_value, 0.0)
        fixed_violation = maximum(abs.(x_value .- [1.0, 0.0]))
        objective_mismatch = abs(objective - radius_value)

        # This check is deliberately looser than the tightened solver tolerance.
        # It detects a genuinely incorrect solution without rejecting normal
        # floating-point barrier output.
        check_tolerance = 1e-5

        numerically_valid =
            isfinite(radius_value) &&
            all(isfinite, x_value) &&
            isfinite(objective) &&
            abs(radius_value - 1.0) <= check_tolerance &&
            cone_violation <= check_tolerance &&
            fixed_violation <= check_tolerance &&
            objective_mismatch <= check_tolerance

        numerically_valid || error(
            "Gurobi returned a solved model, but the numerical preflight " *
            "checks failed: " *
            "termination_status=$(status), " *
            "primal_status=$(pstatus), " *
            "raw_status=$(raw_status(model)), " *
            "radius=$(repr(radius_value)), " *
            "x=$(repr(x_value)), " *
            "objective=$(repr(objective)), " *
            "cone_violation=$(repr(cone_violation)), " *
            "fixed_violation=$(repr(fixed_violation)), " *
            "objective_mismatch=$(repr(objective_mismatch)), " *
            "check_tolerance=$(check_tolerance)"
        )

        return Dict{String,Any}(
            "status" => string(status),
            "primal_status" => string(pstatus),
            "raw_status" => raw_status(model),
            "objective" => objective,
            "radius" => radius_value,
            "x" => x_value,
            "cone_violation" => cone_violation,
            "fixed_violation" => fixed_violation,
            "objective_mismatch" => objective_mismatch,
            "check_tolerance" => check_tolerance,
            "threads" => threads,
        )
    catch err
        message = sprint(showerror, err, catch_backtrace())
        error(
            "Gurobi preflight failed. If a Gurobi license banner appeared, " *
            "the license was successfully initialized and the remaining " *
            "failure is in model solution or numerical validation. " *
            "Original error: " * message
        )
    end
end
"""Solve the paper's exact nonsquared `||residual||₂ + λ||stacked β||₂` SOCP."""
function fit_adaptive_norm_gurobi(work::AdaptiveQuadraticWorkspace,
                                  lambda::Real;
                                  output_flag::Int=0,
                                  time_limit::Real=Inf,
                                  gurobi_threads::Int=1,
                                  optimality_tolerance::Float64=1e-8)
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    gurobi_threads >= 1 || throw(ArgumentError("gurobi_threads must be positive"))
    optimality_tolerance > 0 ||
        throw(ArgumentError("optimality_tolerance must be positive"))
    total_t0 = time_ns()
    Xt = work.Xtrain
    Zt = work.Ztrain
    yt = work.ytrain
    n, m = size(Xt)
    q = size(Zt, 2)
    d = m + m * q
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", output_flag)
    set_optimizer_attribute(model, "Threads", gurobi_threads)
    set_optimizer_attribute(model, "OptimalityTol", optimality_tolerance)
    set_optimizer_attribute(model, "FeasibilityTol",
        min(1e-8, optimality_tolerance))
    # QCP/SOCP models are solved by barrier, whose relevant convergence
    # tolerance is BarQCPConvTol rather than the simplex OptimalityTol.
    set_optimizer_attribute(model, "BarQCPConvTol",
        optimality_tolerance)
    set_optimizer_attribute(model, "NumericFocus", 1)
    isfinite(time_limit) &&
        set_optimizer_attribute(model, "TimeLimit", Float64(time_limit))
    @variable(model, beta0[1:m])
    @variable(model, V[1:m, 1:q])
    @variable(model, residual_radius >= 0)
    @variable(model, coefficient_radius >= 0)
    @expression(model, beta[r=1:n, j=1:m],
        beta0[j] + sum(V[j, k] * Zt[r, k] for k in 1:q))
    @expression(model, residual[r=1:n],
        yt[r] - sum(Xt[r, j] * beta[r, j] for j in 1:m))
    residual_vector = [residual[r] for r in 1:n]
    coefficient_vector = [beta[r, j] for r in 1:n for j in 1:m]
    @constraint(model, [residual_radius; residual_vector] in SecondOrderCone())
    @constraint(model, [coefficient_radius; coefficient_vector] in SecondOrderCone())
    @objective(model, Min,
        residual_radius + Float64(lambda) * coefficient_radius)
    solve_t0 = time_ns()
    optimize!(model)
    solve_seconds = (time_ns() - solve_t0) / 1e9
    status = termination_status(model)
    status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
        error("Gurobi SOCP failed with status $(status)")
    has_values(model) || error("Gurobi SOCP returned no primal solution")
    b0 = value.(beta0)
    Vm = value.(V)
    theta = vcat(b0, vec(Vm))

    # Recompute the objective and cone violations from the returned parameters.
    beta_values = Zt * transpose(Vm)
    beta_values .+= transpose(b0)
    residual_values = yt .- vec(sum(Xt .* beta_values; dims=2))
    residual_norm = norm(residual_values)
    coefficient_norm = norm(beta_values)
    computed_objective = residual_norm + Float64(lambda) * coefficient_norm
    reported_objective = objective_value(model)
    check_tolerance = max(1e-7, 20 * optimality_tolerance)
    objective_mismatch = abs(reported_objective - computed_objective) /
        max(1.0, abs(computed_objective))
    residual_violation = max(residual_norm - value(residual_radius), 0.0)
    coefficient_violation =
        max(coefficient_norm - value(coefficient_radius), 0.0)
    objective_mismatch <= check_tolerance ||
        error("Gurobi SOCP objective mismatch $(objective_mismatch) exceeds " *
              "$(check_tolerance)")
    max(residual_violation, coefficient_violation) <= check_tolerance ||
        error("Gurobi SOCP cone violation exceeds $(check_tolerance)")

    elapsed = (time_ns() - total_t0) / 1e9
    return AdaptiveRidgeModel(
        beta0=b0, V=Vm, theta=theta, tau=work.tau,
        lambda=Float64(lambda), penalty=:norm_stacked_beta,
        objective_variant=:norm, availability_lag=work.availability_lag,
        warmup=:static, objective=computed_objective,
        fit_seconds=elapsed + work.preparation_seconds,
        preparation_seconds=max(elapsed - solve_seconds, 0.0) +
            work.preparation_seconds,
        solve_seconds=solve_seconds, linear_solver="gurobi_socp",
        condition_number=NaN, iterations=0, converged=true,
        residual_norm=0.0, stationarity_residual=NaN,
        relative_objective_change=NaN,
        convergence_reason="gurobi_socp_optimal",
        n_parameters=d,
    )
end

function fit_adaptive_norm_gurobi(X::AbstractMatrix, y::AbstractVector,
                                  idx::AbstractVector{Int}, tau::Int,
                                  lambda::Real;
                                  availability_lag::Int=1,
                                  event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                  output_flag::Int=0,
                                  time_limit::Real=Inf,
                                  gurobi_threads::Int=1,
                                  optimality_tolerance::Float64=1e-8,
                                  direct_max_parameters::Int=3500,
                                  max_dense_bytes::Int=750_000_000)
    work = prepare_adaptive_quadratic_workspace(X, y, idx, tau;
        availability_lag=availability_lag, event_ids=event_ids,
        direct_max_parameters=direct_max_parameters,
        max_dense_bytes=max_dense_bytes)
    return fit_adaptive_norm_gurobi(work, lambda;
        output_flag=output_flag, time_limit=time_limit,
        gurobi_threads=gurobi_threads,
        optimality_tolerance=optimality_tolerance)
end

"""
Verify the reduced quadratic representation against a literal JuMP model with one
`β_t` variable per time point. This is a verification experiment, not the production
solver.
"""
function verify_reduced_against_jump(X::AbstractMatrix, y::AbstractVector,
                                     idx::AbstractVector{Int}, tau::Int, lambda::Real;
                                     availability_lag::Int=1,
                                     event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                     output_flag::Int=0,
                                     gurobi_threads::Int=1)
    fast = fit_adaptive_quadratic(X, y, idx, tau, lambda;
        availability_lag=availability_lag, event_ids=event_ids,
        penalty=:stacked_beta, direct_max_parameters=typemax(Int))
    _, Z, _ = causal_design(X, y, tau, idx;
        availability_lag=availability_lag, event_ids=event_ids)
    yt = Float64.(y[idx])
    n = length(idx)
    m = size(X, 2)
    q = m * tau
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", output_flag)
    set_optimizer_attribute(model, "Threads", gurobi_threads)
    @variable(model, beta0[1:m])
    @variable(model, V[1:m, 1:q])
    @variable(model, beta[1:n, 1:m])
    @constraint(model, [r=1:n, j=1:m],
        beta[r, j] == beta0[j] + sum(V[j, k] * Z[idx[r], k] for k in 1:q))
    @objective(model, Min,
        sum((yt[r] - sum(X[idx[r], j] * beta[r, j] for j in 1:m))^2 for r in 1:n) / n +
        Float64(lambda) * sum(beta[r, j]^2 for r in 1:n, j in 1:m) / n)
    optimize!(model)
    termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ||
        error("verification model did not solve")
    b0 = value.(beta0)
    Vm = value.(V)
    p_fast = predict_adaptive(fast, X, y, idx; event_ids=event_ids)
    p_jump = [dot(@view(X[idx[r], :]), b0 + Vm * @view(Z[idx[r], :])) for r in 1:n]
    return Dict{String,Any}(
        "max_prediction_difference" => maximum(abs.(p_fast .- p_jump)),
        "relative_prediction_difference" => norm(p_fast - p_jump) / max(norm(p_jump), eps()),
        "beta0_difference" => norm(fast.beta0 - b0),
        "V_difference" => norm(fast.V - Vm),
        "jump_objective" => objective_value(model),
        "fast_objective" => fast.objective,
        "relative_objective_difference" => abs(fast.objective - objective_value(model)) /
            max(abs(objective_value(model)), eps()),
    )
end
