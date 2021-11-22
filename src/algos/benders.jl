


function get_Y(X, t)
    "
    Create the vector of data for the dual problem
    "
    T, p = size(X)
    Y = zeros(1,T*p)
    Y[(t-1)*p+1:t*p] = X[t,:]
    return Y
end

function get_Z(X)
    T, p = size(X)
    Z = zeros(T, T*p)
    for t=1:T
        Z[t,:] = get_Y(X,t)
    end
    return Z
end

function get_A(X, t)
    T, p = size(X)
    A = zeros(p,T*p)
    A[:,(t-1)*p+1:t*p] = 1 * Matrix(I, p, p)
    return A
end


function solve_model_benders(m)
    "Solve the Benders master problem
    "
    optimize!(m)
    U_OA = objective_value(m)
    #println("U_OA from solve ", U_OA)
    return value.(m[:β]), value.(m[:α]), U_OA
end

function S_primal(X, y, β0, epsilon, delta)

    n, p = size(X)

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Add variables
    @variable(model, β[i=1:n,j=1:p])
    @variable(model, b[i=1:n]>=0)

    # Add objective
    @objective(model, Min, sum(b[i] for i=1:n))

    #@constraint(model,[i=1:n], y .- dot(X,β) .<= b)
    #@constraint(model,[i=1:n],-y .+ dot(X,β) .<= b)


    @constraint(model, res_plus[i=1:n],  - y[i] + dot(X[i,:],β[i,:]) <= b[i])
    @constraint(model, res_minus[i=1:n],  y[i] - dot(X[i,:],β[i,:]) <= b[i])

    @constraint(model, diff_plus[i=2:n],   β[i,:] .- β[i-1,:] .<= delta)
    @constraint(model, diff_minus[i=2:n], - β[i,:] .+ β[i-1,:] .<= delta)

    @constraint(model, diff_0_plus[i=1:n],   β[i,:] .- β0 .<= epsilon)
    @constraint(model, diff_0_minus[i=1:n], - β[i,:] .+ β0 .<= epsilon)

    optimize!(model);

    return objective_value(model), getvalue.(β)
end


function R(X, D_min, D_max, β0, epsilon, delta)
    "
    Full dual problem
    "
    T, p = size(X)
    Z = get_Z(X)

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))#Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "NonConvex", 2)

    # Add variables
    @variable(model, λ[i=1:2, j=1:T] >= 0)
    @variable(model, ν[i=1:2, j=1:T-1, k=1:p]>=0)
    @variable(model, μ[i=1:2, j=1:T, k=1:p]>=0)

    @variable(model, y[j=1:T])


    @constraint(model,[t=1:T], λ[1,:] .+ λ[2,:] .== 1)


    @constraint(model, transpose(λ[2,:])*Z-transpose(λ[1,:])*Z
                        + sum(transpose(ν[1,t,:])*(get_A(X, t+1).-get_A(X, t)) for t=1:T-1)
                        + sum(transpose(ν[2,t,:])*(-get_A(X, t+1).+get_A(X, t)) for t=1:T-1)
                        + sum(transpose(μ[1,t,:])*get_A(X,t) for t=1:T)
                        - sum(transpose(μ[2,t,:])*get_A(X,t) for t=1:T) .== 0)

    #y in uncertainty set
    @constraint(model, [1:T], D_min .<= y)
    @constraint(model, [1:T], y .<= D_max)

    # Add objective
    @objective(model, Max, 2*dot(λ[1,:],y) - sum(y)
                            - delta * sum(sum(ν[1,t,i]+ν[2,t,i] for i=1:p) for t=1:T-1)
                            - sum(dot(epsilon .+ β0, μ[1,t,:]) for t = 1:T)
                            - sum(dot(epsilon .- β0, μ[2,t,:]) for t = 1:T)) #
    optimize!(model)
    return objective_value(model), getvalue.(y), getvalue.(λ), getvalue.(ν), getvalue.(μ)
end

function master_problem(X0, Xt, y0, D_min, D_max, threshold = 0.1, epsilon = 0.1, delta = 0.1, reg = 1, ρ = 1, max_cuts = 10, verbose=0)
    n, p = size(X0)
    T, p = size(Xt)
    #Z = get_Z(X0)
    L_BD = -10000
    U_BD = 10000
    cuts = 0

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))#Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Add variables
    @variable(model, α)
    @variable(model, β[j=1:p])

    #Warm start for β

    β_val0 = l2_regression(X0, y0, ρ)#Random.rand(p)#
    #β_val0 = Random.rand(p)


    #Initialization
    _, y_val0, λ_val0, ν_val0, μ_val0 = R(Xt, D_min, D_max, β_val0, epsilon, delta)

    #First constraint
    @constraint(model, α >= 2*dot(λ_val0[1,:],y_val0) - sum(y_val0)
                            - delta * sum(sum(ν_val0[1,t,i]+ν_val0[2,t,i] for i=1:p) for t=1:T-1)
                            - sum(dot(epsilon .+ β, μ_val0[1,t,:]) for t = 1:T)
                            - sum(dot(epsilon .- β, μ_val0[2,t,:]) for t = 1:T))

    # Add objective
    @objective(model, Min, 1/n*sum((y0[i]-sum(X0[i,j]*β[j] for j=1:p))^2 for i=1:n) + reg*α + ρ*sum(β[j]^2 for j=1:p))

    while cuts < max_cuts && U_BD - L_BD > threshold
        if verbose
            println("Lower: ", L_BD, " Upper: ", U_BD)
        end
        cuts += 1

        #Solve current Master Problem
        β_val, α_val, L_BD = solve_model_benders(model)
        U_OA, y_val, λ_val, ν_val, μ_val = R(Xt, D_min, D_max, β_val, epsilon, delta)

        U_BD = 1/n*sum((y0[i]-sum(X0[i,j]*β_val[j] for j=1:p))^2 for i=1:n) + reg*U_OA + ρ*sum(β_val[j]^2 for j=1:p)

        if U_BD - L_BD > threshold
            @constraint(model, α >= 2*dot(λ_val[1,:],y_val) - sum(y_val)
                            - delta * sum(sum(ν_val[1,t,i]+ν_val[2,t,i] for i=1:p) for t=1:T-1)
                            - sum(dot(epsilon .+ β, μ_val[1,t,:]) for t = 1:T)
                            - sum(dot(epsilon .- β, μ_val[2,t,:]) for t = 1:T))
            if verbose
                println("Cut added")
                println("y_val: ", y_val)
            end
        end
    end
    optimize!(model)
    β_val, α_val, L_BD = solve_model_benders(model)
    U_OA, y_val, λ_val, ν_val, μ_val = R(Xt, D_min, D_max, β_val, epsilon, delta)
    if verbose
        println("Final model Obj value: ", objective_value(model))
        println("Lower: ", L_BD, " Upper: ", U_BD)
        println("Final y: ", y_val)
    end
    return objective_value(model), getvalue.(β), getvalue.(α), y_val#, getvalue.(ν), getvalue.(μ)
end