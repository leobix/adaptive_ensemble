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
    errors = Array{Float64, 2}(undef, val, n)
    errors[:, 1] = [yt_true[s]-(Xt[s,1].*std_y.+mean_y) for s=1:val]
    for i=2:n
        err = [abs(yt_true[s]-(Xt[s,i].*std_y.+mean_y)) for s=1:val]
        new_MAPE = MAPE(err, yt_true)
        println("Model ", i, " MAPE : ", new_MAPE)
        errors[:, i] = [yt_true[s]-(Xt[s,i].*std_y.+mean_y) for s=1:val]
        if new_MAPE < best_MAPE
            best_err = err
            best_MAPE = new_MAPE
        end
    end
    # Write errors without depending on DataFrames
    try
        # Create a NamedTuple-of-vectors table for CSV.write
        cols = (; (Symbol("col" * string(i)) => errors[:, i] for i in 1:size(errors, 2))...)
        if !ispath("results_beta")
            mkpath("results_beta")
        end
        CSV.write("results_beta/model_errors.csv", cols)
    catch e
        @warn "Failed to write model_errors.csv" e
    end
    return best_err
end

function get_metrics(args, method, err, yt_true, time = 0)
    ```
    Given the errors made by a given model, and the true values, computes the different metrics.
    Also saves the values in the dataframe
    ```
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
    model = Model(optimizer_with_attributes(Gurobi.Optimizer))
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
    # Resolve results directory from CLI args (defaults to "results")
    results_dir = haskey(args, "results_dir") ? args["results_dir"] : "results"
    if !ispath(results_dir)
        mkpath(results_dir)
    end
    if args["data"] == "synthetic"
        # Compose row for synthetic experiments
        row = (
            Dataset = String(args["data"]),
            Train_Length = Int(args["train_length"]),
            Test_Length = Int(len_test),
            Std_Pert_y = Float64(args["std_pert"]),
            Bias_Range = Float64(args["bias_range"]),
            Std_Range = Float64(args["std_range"]),
            Bias_Drift_range = Float64(args["bias_drift"]),
            Std_Drift_Range = Float64(args["std_drift"]),
            y_Bias_Drift_range = Float64(get(args, "y_bias_drift", 0.0)),
            y_Std_Drift_Range = Float64(get(args, "y_std_drift", 0.0)),
            p_Bernoulli_discrete = Float64(get(args, "p_ber", 1.0)),
            Period = Int(args["period"]),
            N_models = Int(args["N_models"]),
            Seed = Int(args["seed"]),
            T = Int(args["T"]),
            End_id = Int(args["end-id"]),
            Rho_beta = Float64(args["rho_beta"]),
            Rho = Float64(args["rho"]),
            Rho_V = Float64(args["rho_V"]),
            Past = Int(args["past"]),
            Num_past = Int(args["num-past"]),
            Val = Int(args["val"]),
            Train_test_split = Float64(args["train_test_split"]),
            Method = String(method),
            MAE = Float64(MAE),
            MAPE = Float64(MAPE),
            RMSE = Float64(RMSE),
            R2 = Float64(R2),
            CVAR_05 = Float64(CVAR_05),
            CVAR_15 = Float64(CVAR_15),
            Time = Int(time)
        )
        outfile = joinpath(results_dir, "results_" * args["data"] * "_" * string(args["seed"]) * ".csv")
        CSV.write(outfile, [row]; append=ispath(outfile))
    else
        param_combo = "0"
        try
            param_combo = string(args["param_combo"])
        catch e
        end
        row = (
            Dataset = String(args["data"]),
            Train_Length = Int(args["train_length"]),
            Test_Length = Int(len_test),
            End_id = Int(args["end-id"]),
            Rho_beta = Float64(args["rho_beta"]),
            Rho = Float64(args["rho"]),
            Rho_V = Float64(args["rho_V"]),
            Past = Int(args["past"]),
            Num_past = Int(args["num-past"]),
            Val = Int(args["val"]),
            Train_test_split = Float64(args["train_test_split"]),
            Method = String(method),
            MAE = Float64(MAE),
            MAPE = Float64(MAPE),
            RMSE = Float64(RMSE),
            R2 = Float64(R2),
            CVAR_05 = Float64(CVAR_05),
            CVAR_15 = Float64(CVAR_15),
            Time = Int(time)
        )
        outfile = joinpath(results_dir, "results_" * args["data"] * "_" * param_combo * ".csv")
        CSV.write(outfile, [row]; append=ispath(outfile))
    end

end
