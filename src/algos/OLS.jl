function l2_regression(X, y, rho, rho_stat; solver_output=0)
    n,p = size(X)

    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", solver_output)
    #Should work without but uncomment in case
    #set_optimizer_attribute(model, "NonConvex", 2)

    @variable(model,beta[j=1:p])
    @variable(model, sse>=0)
    #@variable(model, reg>=0)
    @constraint(model, sum((y[i]-sum(X[i,j]*beta[j] for j=1:p))^2 for i=1:n) <= sse)
    #@constraint(model, sum(beta[j]^2 for j=1:p)<=reg)
    @objective(model,Min, 1/n*sse + rho*sum(beta[j]^2 for j=1:p)
#                                   + rho_stat*(#sqrt(
#                                   2*rho_stat/(n*n)
#                                         * sum(
#                                               ((y[i]-sum(X[i,j]*beta[j] for j=1:p))^2 - 1/n * sum((y[i]-sum(X[i,j]*beta[j] for j=1:p))^2 for i=1:n))^2
#                                         for i=1:n)
#                                         )
                                        )

    optimize!(model)
    #println("Obj ", objective_value(model))
    return value.(beta)
end



