# This whole code is not used anymore.



function master_primal_l2(X0, Xt, y0, Dmin, Dmax, epsilon, delta, reg, ρ, ϵ_l2, δ_l2, β0_fix = false, β0_val = 0)
    ## Version where the adaptive part is L1 loss
    M = 1000
    n0, p = size(X0)
    n, _ = size(Xt)

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Add variables
    @variable(model, β[i=1:n,j=1:p])
    @variable(model, b[i=1:n]>=0)
    @variable(model, β0[j=1:p])
    @variable(model, z[i=1:n], Bin)
    @variable(model, y[i=1:n])
    # Add objective
    @objective(model, Min, 1/n0*sum((y0[i]-sum(X0[i,j]*β0[j] for j=1:p))^2 for i=1:n0)
        + reg*sum(b[i] for i=1:n) + ρ*sum(β0[j]^2 for j=1:p))

    #@constraint(model,[i=1:n], y .- dot(X,β) .<= b)
    #@constraint(model,[i=1:n],-y .+ dot(X,β) .<= b)

    if β0_fix
        @constraint(model, β0 .== β0_val)
    end
    @constraint(model, res_plus_min[i=1:n],  - y[i] + dot(Xt[i,:],β[i,:]) <= b[i])
    @constraint(model, res_minus_min[i=1:n],  y[i] - dot(Xt[i,:],β[i,:]) <= b[i])

    @constraint(model,  y .== Dmax.*z + Dmin.*(1 .-z))
    #@constraint(model,  y .== Dmax.*(1 .-z) + Dmin.*z)

    #@constraint(model, bigM[i=1:n], (dot(Xt[i,:],β[i,:])-Dmin[i])^2 + M*z[i] >= (dot(Xt[i,:],β[i,:])-Dmax[i])^2)
    @constraint(model, bigM[i=1:n], (dot(Xt[i,:],β[i,:]) - Dmin[i])^2 - (dot(Xt[i,:],β[i,:])-Dmax[i])^2 + M*z[i] >= 0)


    #@constraint(model, bigM2[i=1:n],  (dot(Xt[i,:],β[i,:])-Dmax[i])^2 + M*(1-z[i]) >= (dot(Xt[i,:],β[i,:])-Dmin[i])^2)

    @constraint(model, diff_plus[i=2:n],   β[i,:] .- β[i-1,:] .<= delta)
    @constraint(model, diff_minus[i=2:n], - β[i,:] .+ β[i-1,:] .<= delta)

    @constraint(model, diff_0_plus[i=1:n],   β[i,:] .- β0 .<= epsilon)
    @constraint(model, diff_0_minus[i=1:n], - β[i,:] .+ β0 .<= epsilon)

    @constraint(model, sq_0[i=1:n], sum((β[i,:] .- β0).^2) .<= ϵ_l2)
    @constraint(model, sq_t[i=2:n], sum((β[i,:] .- β[i-1,:]).^2) .<= δ_l2)

    #@constraint(model, abs_0[i=1:n], sum(abs.(β[i,:] .- β0)) .<= ϵ_l1)
    #@constraint(model, abs_t[i=2:n], sum(abs.(β[i,:] .- β[i-1,:])) .<= δ_l1)

    optimize!(model);
    #println("SUM ", sum(getvalue.(b)))
    #println("Y ", getvalue.(y))
    return objective_value(model), getvalue.(β), getvalue.(β0)
end




function master_primal_l2_ridge(X0, Xt, y0, Dmin, Dmax, epsilon, delta, reg, ρ, ϵ_l2, δ_l2, β0_fix = false, β0_val = 0)
    ## Version where the adaptive part is L2 loss.
    M = 1000
    n0, p = size(X0)
    n, _ = size(Xt)

    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Add variables
    @variable(model, β[i=1:n,j=1:p])
    @variable(model, β0[j=1:p])

    #z models if y takes Dmin or Dmax
    @variable(model, z[i=1:n], Bin)
    @variable(model, y[i=1:n])
    # Add objective
    #TODO CHANGE THE REGULARIZATION
    @objective(model, Min, 1/n0*sum((y0[i]-sum(X0[i,j]*β0[j] for j=1:p))^2 for i=1:n0)
        + 1/n*sum((y[i] - dot(Xt[i,:],β[i,:]))^2 for i=1:n) + ρ*sum(β0[j]^2 for j=1:p) + ρ/n*sum(sum(β[i,j]^2 for j=1:p) for i=1:n))

    #@constraint(model,[i=1:n], y .- dot(X,β) .<= b)
    #@constraint(model,[i=1:n],-y .+ dot(X,β) .<= b)

    if β0_fix
        @constraint(model, β0 .== β0_val)
    end

    @constraint(model,  y .== Dmax.*z + Dmin.*(1 .-z))
    #@constraint(model,  y .== Dmax.*(1 .-z) + Dmin.*z)

    #@constraint(model, bigM[i=1:n], (dot(Xt[i,:],β[i,:])-Dmin[i])^2 + M*z[i] >= (dot(Xt[i,:],β[i,:])-Dmax[i])^2)
    @constraint(model, bigM[i=1:n], (dot(Xt[i,:],β[i,:]) - Dmin[i])^2 - (dot(Xt[i,:],β[i,:])-Dmax[i])^2 + M*z[i] >= 0)


    #@constraint(model, bigM2[i=1:n],  (dot(Xt[i,:],β[i,:])-Dmax[i])^2 + M*(1-z[i]) >= (dot(Xt[i,:],β[i,:])-Dmin[i])^2)

    @constraint(model, diff_plus[i=2:n],   β[i,:] .- β[i-1,:] .<= delta)
    @constraint(model, diff_minus[i=2:n], - β[i,:] .+ β[i-1,:] .<= delta)

    @constraint(model, diff_0_plus[i=1:n],   β[i,:] .- β0 .<= epsilon)
    @constraint(model, diff_0_minus[i=1:n], - β[i,:] .+ β0 .<= epsilon)

    @constraint(model, sq_0[i=1:n], sum((β[i,:] .- β0).^2) .<= ϵ_l2)
    @constraint(model, sq_t[i=2:n], sum((β[i,:] .- β[i-1,:]).^2) .<= δ_l2)

    #@constraint(model, abs_0[i=1:n], sum(abs.(β[i,:] .- β0)) .<= ϵ_l1)
    #@constraint(model, abs_t[i=2:n], sum(abs.(β[i,:] .- β[i-1,:])) .<= δ_l1)

    optimize!(model);
    #println("SUM ", sum(getvalue.(b)))
    #println("Y ", getvalue.(y))
    return objective_value(model), getvalue.(β), getvalue.(β0)
end

