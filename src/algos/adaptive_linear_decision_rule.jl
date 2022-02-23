function get_X_Z_y(args, X, y, T)
    """
    Input: training data X and corresponding labels y ; how many time-steps from the past to be used
    Output: the past features X with past targets y as a Z training data (no present features)
    """
    n, p = size(X)
    #T past time steps * p features + T targets
    Z = ones(n-T, T*p+T)
    for i=T+1:n
        for t=1:T
            if args["err_rule"]
                Z[i-T,1+p*(t-1):p*t] = X[i-t,:].-y[i-t]
            else
                Z[i-T,1+p*(t-1):p*t] = X[i-t,:]
            end
        end
        Z[i-T, (p*T+1):end] = y[i-T:i-1]
    end
    return X[T+1:end,:], Z, y[T+1:end]
end

function adaptive_ridge_regression_exact(args, X, y, ρ_β0, ρ_V0, T, N0)

    #Version with actual robust equivalence
    #The formula for the regularization is different
    #Considers there is potentially a stable part in the time series
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    X, Z, y = get_X_Z_y(args, X, y, T)
    println("X", X[4,:])
    println("Z", Z[4,:])
    println("y", y[1:4])

    N, P = size(X)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, V0[j=1:P, k=1:T*P+T])
    @variable(model, t>=0)
    @variable(model, β[t=1:N,j=1:P])

    # If no stable part, then no reg
    if N0 == 1
        ρ_β0 = 0
        N0 = 2
    end

    # Add objective
    @objective(model, Min, t + 1/(N0-1)*ρ_β0 * sum(β0[j]^2 for j=1:P)
                            + 1/(N-N0+1)*ρ_V0 * sum(β[t,k]^2 for t=N0:N for k=1:P)
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(transpose(X[i, :])*β[i,:]))^2 for i=1:N))

    #stable and adaptive part
    @constraint(model, [i=N0:N], β[i,:] .== β0+V0*Z[i,:])
    @constraint(model, [i=1:N0-1], β[i,:] .== β0)

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(V0), getvalue.(β)
end

function adaptive_ridge_regression_exact_no_stable(args, X, y, ρ_β, ρ_β0, ρ_V0, T)

    #Version with actual robust equivalence
    #The formula for the regularization is different
    #no stable part
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    X, Z, y = get_X_Z_y(args, X, y, T)
    println("X", X[4,:])
    println("Z", Z[4,:])
    println("y", y[1:4])

    N, P = size(X)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, V0[j=1:P, k=1:T*P+T])
    @variable(model, t>=0)
    @variable(model, β[t=1:N,j=1:P])

    # Add objective
    @objective(model, Min, t + 1/N*ρ_β * sum(β[t,k]^2 for t=1:N for k=1:P)
                           + ρ_β0 * sum(β0[j]^2 for j=1:P) + ρ_V0 * sum(V0[j,k]^2 for j=1:P for k=1:T*P+T) ###IMPORTANT TO ADD OR REMOVE
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(transpose(X[i, :])*β[i,:]))^2 for i=1:N))

    #linear decision rule for adaptive RO
    @constraint(model, [i=1:N], β[i,:] .== β0.+V0*Z[i,:])

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(V0), getvalue.(β)
end


function adaptive_ridge_regression_exact_Vt(args, X, y, ρ_β0, ρ_V0, T, N0)

    #Version with actual robust equivalence
    #The formula for the regularization is different
    #Vt can move
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    X, Z, y = get_X_Z_y(args, X, y, T)

    N, P = size(X)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, Vt[t=1:N,j=1:P, k=1:T*P+T])
    @variable(model, t>=0)
    @variable(model, β[t=1:N,j=1:P])

    # If no stable part, then no reg
    if N0 == 1
        ρ_β0 = 0
        N0 = 2
    end

    # Add objective
    @objective(model, Min, t + 1/(N0-1)*ρ_β0 * sum(β0[j]^2 for j=1:P)
                            + 1/(N-N0+1)*ρ_V0 * sum(β[t,k]^2 for t=N0:N for k=1:P)
                            + 1/(N-N0+1)*ρ_V0 * sum((Vt[t+1,k,j]-Vt[t,k,j])^2 for t=N0:(N-1) for k=1:P for j=1:T*P+T)
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(transpose(X[i, :])*β[i,:]))^2 for i=1:N))

    #stable and adaptive part
    @constraint(model, [i=N0:N], β[i,:] .== β0+Vt[i,:,:]*Z[i,:])
    @constraint(model, [i=1:N0-1], β[i,:] .== β0)

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(Vt), getvalue.(β)
end



function adaptive_ridge_regression_standard(args, X, y, ρ_β0, ρ_V0, T)

#     Adaptive ridge: does not run fast
# Regulairzation on beta and V separately

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    X, Z, y = get_X_Z_y(args, X, y, T)

    N, P = size(X)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, V0[j=1:P, k=1:T*P+T])
    @variable(model, t>=0)

    # Add objective
    @objective(model, Min, t + ρ_β0 * sum(β0[j]^2 for j=1:P)
                            + ρ_V0 * sum(V0[j,k]^2 for j=1:P for k=1:T*P+T)
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(X[i, j] * (β0[j]
                 + sum(V0[j, l] * Z[i, l] for l=1:(T * P + T))
                 )
                for j=1:P))^2
        for i=1:N))

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(V0)
end


function adaptive_ridge_regression_slowly_varying(args, X, y, ρ_β0, ρ_V0, T)

    #Version with actual robust equivalence
    #The formula for the regularization is different
    #no stable part
    #Slowly varying version
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    X, Z, y = get_X_Z_y(args, X, y, T)

    N, P = size(X)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, V0[j=1:P, k=1:T*P+T])
    @variable(model, t>=0)
    @variable(model, β[t=1:N,j=1:P])

    # Add objective
    @objective(model, Min, t + 1/N * ρ_β0 * sum(β[t,k]^2 for t=1:N for k=1:P)
                            + 1/(N-1) * ρ_V0 * sum((β[t,k]-β[t-1,k])^2 for t=2:N for k=1:P)
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(transpose(X[i, :])*β[i,:]))^2 for i=1:N))

    #linear decision rule for adaptive RO
    @constraint(model, [i=1:N], β[i,:] .== β0.+V0*Z[i,:])

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(V0)
end


##########################################

function adaptive_ridge_regression_exact_no_stable_hurricane(args, X, Z, y, ρ_β, ρ_β0, ρ_V0)

    #Version with actual robust equivalence
    #The formula for the regularization is different
    #no stable part
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    N, P = size(X)
    T = size(Z,2)
    # Add variables

    @variable(model, β0[j=1:P])
    @variable(model, V0[j=1:P, k=1:T])
    @variable(model, t>=0)
    @variable(model, β[t=1:N,j=1:P])

    # Add objective
    @objective(model, Min, t + 1/N*ρ_β * sum(β[t,k]^2 for t=1:N for k=1:P)
                           + ρ_β0 * sum(β0[j]^2 for j=1:P) + ρ_V0 * sum(V0[j,k]^2 for j=1:P for k=1:T) ###IMPORTANT TO ADD OR REMOVE
    )

    @constraint(model, t >= 1 / N * sum((y[i] - sum(transpose(X[i, :])*β[i,:]))^2 for i=1:N))

    #linear decision rule for adaptive RO
    @constraint(model, [i=1:N], β[i,:] .== β0.+V0*Z[i,:])

    optimize!(model);
    return objective_value(model), getvalue.(β0), getvalue.(V0), getvalue.(β)
end