function l2_regression(X, y, rho; solver_output=0)
    n,p = size(X)

    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", solver_output)
    set_optimizer_attribute(model, "NonConvex", 2)

    @variable(model,beta[j=1:p])
    @variable(model, sse>=0)
    #@variable(model, reg>=0)
    @constraint(model, sum((y[i]-sum(X[i,j]*beta[j] for j=1:p))^2 for i=1:n) <= sse)
    #@constraint(model, sum(beta[j]^2 for j=1:p)<=reg)
    @objective(model,Min, 1/n*sse + rho*sum(beta[j]^2 for j=1:p))

    optimize!(model)
    #println("Obj ", objective_value(model))
    return value.(beta)
end