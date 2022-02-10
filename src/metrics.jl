function R2(y_true, y_test)
    SSR = sum(abs2.(y_true.-y_test))
    SST = sum(abs2.(y_true.-mean(y_true)))
    return 1 - SSR/SST
end

function R2_err(err, y_true)
    SSR = sum(abs2.(err))
    SST = sum(abs2.(y_true.-mean(y_true)))
    return 1 - SSR/SST
end

function get_metrics(args, method, err, yt_true)
    #TODO ADD saving mechanism
    MAE = mean(err)
    #CVAR_05 = compute_CVaR(err, 0.05)
    #CVAR_15 = compute_CVaR(err, 0.15)
    R2 = R2_err(err, yt_true)
    MAPE = 100*sum(abs.(err./yt_true))/size(err)[1]
    RMSE = sqrt(sum(abs2.(err))/size(err)[1])
    println("MAE : ", MAE)
    println("MAPE : ", MAPE)
    println("RMSE : ", RMSE)
    println("R2 : ", R2)
    #println("CVAR 0.05 :", CVAR_05)
    #println("CVAR 0.15 :", CVAR_15)
    CVAR_05, CVAR_15 = 0, 0
    add_Dataframe(args, method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15)
end


function compute_CVaR(errs, α_risk)
#     '''
#     Compute the Conditional Value at Risk
#     :param X: Either the data matrix X or the errors
#     :param y: the target values
#     :param alpha: the risk value
#     :param beta:
#     :param b0:
#     :param errs: whether you want to input the data matrix X or the errors
#     :return: the model and the CVaR
#     '''
    n = size(errs)[1]
    # Create model
    model = Model(with_optimizer(Gurobi.Optimizer, GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)

    # Add variables
    @variable(model, τ)
    @variable(model, z[1:n] >= 0)

    # Add objective
    @objective(model, Min, sum(z)/(α_risk * n) + τ)

    @constraint(model, [1:n], errs .- τ .<= z)

    optimize!(model)

    return objective_value(model)
end



function add_Dataframe(args, method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15)

#     results = DataFrame(Dataset = String[], End_id = Int64[],
#         Rho = Float64[], Rho_V = Float64[], Past = Float64[], Num_past = Float64[], Val = Float64[],
#         Train_test_split = Float64[], Method = String[], MAE = Float64[], MAPE = Float64[], RMSE = Float64[], R2 = Float64[], CVAR_05 = Float64[], CVAR_15 = Float64[])

    results = DataFrame(CSV.File("results.csv"))

    push!(results, (args["data"], args["end-id"], args["rho"], args["rho_V"], args["past"], args["num-past"], args["val"], args["train_test_split"], method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15))
    CSV.write("results.csv", results)

#             CSV.write(filename, df; append=true, writeheader=false)
#             CSV.write(string("sparsity_", data_set,"_", parameter_range), DataFrame(obj_values))
#             CSV.write(string("sparsity_", data_set,"_", parameter_range), DataFrame(best_parameters) ; append=true)
#             CSV.write(string("sparsity_", data_set,"_", parameter_range), DataFrame(all_accuracies) ; append=true)
end
