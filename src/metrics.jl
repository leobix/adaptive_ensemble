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

function MAPE(err, yt_true)
    return 100*sum(abs.(err./yt_true))/size(err)[1]
end

function get_best_model_errors(yt_true, Xt, mean_y, std_y)
"""
Determines the best model in hindsight, wrt MAPE
"""
    n = size(Xt, 2)
    val = size(yt_true, 1)
    best_err = [abs(yt_true[s]-(Xt[s,1].*std_y.+mean_y)) for s=1:val]
    best_MAPE = MAPE(best_err, yt_true)
    for i=2:n
        err = [abs(yt_true[s]-(Xt[s,i].*std_y.+mean_y)) for s=1:val]
        new_MAPE = MAPE(err, yt_true)
        if new_MAPE < best_MAPE
            best_err = err
            best_MAPE = new_MAPE
        end
    end
    return best_err
end

function get_metrics(args, method, err, yt_true, time = 0)
    #TODO ADD saving mechanism
    MAE = mean(err)
    R2 = R2_err(err, yt_true)
    MAPE = 100*sum(abs.(err./yt_true))/size(err)[1]
    RMSE = sqrt(sum(abs2.(err))/size(err)[1])
    len_test = size(err)[1]
    println("Length Test Set: ", len_test)
    println("MAE : ", MAE)
    println("MAPE : ", MAPE)
    println("RMSE : ", RMSE)
    println("R2 : ", R2)
    if args["CVAR"]
        CVAR_05 = compute_CVaR(err, 0.05)
        CVAR_15 = compute_CVaR(err, 0.15)
        println("CVAR 0.05 :", CVAR_05)
        println("CVAR 0.15 :", CVAR_15)
    else
        CVAR_05, CVAR_15 = 0, 0
    end
    add_Dataframe(args, method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, len_test, time)
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



function add_Dataframe(args, method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, len_test, time)
    filename = "results_4_22/"
    if args["data"] == "synthetic"
        try
            results = DataFrame(CSV.File(filename*"results_"*args["data"]*"_"*string(args["seed"])*".csv"))
            push!(results, (args["data"], args["train_length"], len_test,
                args["std_pert"], args["bias_range"], args["std_range"], args["bias_drift"], args["std_drift"], args["y_bias_drift"], args["y_std_drift"], args["period"], args["N_models"], args["seed"], args["T"],
                args["end-id"], args["rho_beta"], args["rho"], args["rho_V"], args["past"], args["num-past"], args["val"], args["train_test_split"], method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, time))
            CSV.write(filename*"results_"*args["data"]*"_"*string(args["seed"])*".csv", results)
        catch e
            results = DataFrame(Dataset = String[], Train_Length = Int64[], Test_Length = Int64[], Std_Pert_y = Float64[], Bias_Range = Float64[], Std_Range = Float64[], Bias_Drift_range = Float64[], Std_Drift_Range = Float64[],
                y_Bias_Drift_range = Float64[], y_Std_Drift_Range = Float64[],
                Period = Int64[], N_models = Int64[], Seed = Int64[], T = Int64[], End_id = Int64[],
                Rho_beta = Float64[], Rho = Float64[], Rho_V = Float64[], Past = Float64[], Num_past = Float64[], Val = Float64[],
                Train_test_split = Float64[], Method = String[], MAE = Float64[], MAPE = Float64[], RMSE = Float64[], R2 = Float64[], CVAR_05 = Float64[], CVAR_15 = Float64[], Time = Int64[])

            push!(results, (args["data"], args["train_length"], len_test, args["std_pert"], args["bias_range"], args["std_range"], args["bias_drift"], args["std_drift"], args["y_bias_drift"], args["y_std_drift"], args["period"], args["N_models"], args["seed"], args["T"],
                args["end-id"], args["rho_beta"], args["rho"], args["rho_V"], args["past"], args["num-past"], args["val"], args["train_test_split"], method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, time))
            CSV.write(filename*"results_"*args["data"]*"_"*string(args["seed"])*".csv", results)
        end
    else
        try
            results = DataFrame(CSV.File(filename*"results_"*args["data"]*".csv"))
            push!(results, (args["data"], args["train_length"], len_test, args["end-id"], args["rho_beta"], args["rho"], args["rho_V"], args["past"], args["num-past"], args["val"], args["train_test_split"], method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, time))
            #CSV.write("results_3_29/results_"*args["data"]*"_"*string(args["seed"])*".csv", results)
            CSV.write(filename*"results_"*args["data"]*".csv", results)
        catch e
            results = DataFrame(Dataset = String[], Train_Length = Int64[], Test_Length = Int64[], End_id = Int64[],
                Rho_beta = Float64[], Rho = Float64[], Rho_V = Float64[], Past = Float64[], Num_past = Float64[], Val = Float64[],
                Train_test_split = Float64[], Method = String[], MAE = Float64[], MAPE = Float64[], RMSE = Float64[], R2 = Float64[], CVAR_05 = Float64[], CVAR_15 = Float64[], Time = Int64[])

            push!(results, (args["data"], args["train_length"], len_test, args["end-id"], args["rho_beta"], args["rho"], args["rho_V"], args["past"], args["num-past"], args["val"], args["train_test_split"], method, MAE, MAPE, RMSE, R2, CVAR_05, CVAR_15, time))
            CSV.write(filename*"results_"*args["data"]*".csv", results)
        end
    end

end
