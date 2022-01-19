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

function get_metrics(err, yt_true)
    #TODO ADD saving mechanism
    MAE = mean(err)
    CVAR_05 = compute_CVaR(err, 0.05)
    CVAR_15 = compute_CVaR(err, 0.15)
    R2 = R2_err(err, yt_true)
    MAPE = sum(abs.(err./yt_true))*100/size(err)[1]
    RMSE = sqrt(sum(abs2(err)))/size(err)[1]
    println("MAE : ", MAE)
    println("CVAR 0.05 :", CVAR_05)
    println("CVAR 0.15 :", CVAR_15)
    println("R2 : ", R2)
    println("MAPE : ", MAPE)
    println("RMSE : ", RMSE)
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


function eval_method(X, y, y_true, split_, past, num_past, val, uncertainty, ϵ_inf, δ_inf, last_yT,
        ϵ_l2, δ_l2, ρ, reg, max_cuts, verbose,
        fix_β0, more_data_for_β0, benders, ridge)

    threshold_benders = 0.01
    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end
    #TODO check split_index with max(split inex, 1)
    X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), num_past*past, val, uncertainty, last_yT)

    β_list0 = zeros(val, p)
    β_listt = zeros(val, p)
    β_listl2 = zeros(val, p)
    β_l2_init = l2_regression(X0,y0,ρ);
    for s=1:val

        #TODO check split_index with max(split inex, 1)
        if more_data_for_β0
            X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), s+(num_past-1)*past, past-1, uncertainty, last_yT)
        else
            X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(s+split_index-num_past*past+1, 1), (num_past-1)*past, past-1, uncertainty, last_yT)
        end


        if benders
            ##TODO handle fix_beta0
            obj, β0_val, α, y_val = master_problem(X0, Xt, y0, D_min, D_max, threshold_benders, ϵ_inf, δ_inf, reg, ρ, max_cuts, verbose)
            _, βt_val = S_primal(Xt, y_val, β0_val, ϵ_inf, δ_inf);
        else
            if ridge
                obj, βt_val, β0_val = master_primal_l2_ridge(X0, Xt, y0, D_min, D_max, ϵ_inf, δ_inf, reg, ρ, ϵ_l2, δ_l2, fix_β0, β_l2_init)
            else
                obj, βt_val, β0_val = master_primal_l2(X0, Xt, y0, D_min, D_max, ϵ_inf, δ_inf, reg, ρ, ϵ_l2, δ_l2, fix_β0, β_l2_init)
            end
        end

        β_listt[s,:] = βt_val[past-1,:]
        β_list0[s,:] = β0_val
        β_l2 = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
        β_listl2[s,:] = β_l2

    end

    X0, y0, Xt, yt, _, D_min, D_max = prepare_data_from_y(X, y, 1, split_index, val, uncertainty, last_yT)
    _, _, _, _, yt_true, _, _ = prepare_data_from_y(X, y_true, 1, split_index, val, uncertainty, last_yT)

    err_0 = [abs(yt_true[s]-dot(Xt[s,:],β_list0[s,:])) for s=1:val]
    err_t = [abs(yt_true[s]-dot(Xt[s,:],β_listt[s,:])) for s=1:val]
    err_baseline = [abs(yt_true[s]-dot(Xt[s,:],β_l2_init)) for s=1:val]
    err_l2 = [abs(yt_true[s]-dot(Xt[s,:],β_listl2[s,:])) for s=1:val]

    #TODO check get_metrics
    println("\n### β0 Baseline ###")
    get_metrics(err_baseline, yt_true)

    println("\n### β0 Baseline Retrained ###")
    get_metrics(err_l2, yt_true)

    println("\n### β0 Adaptive ###")
    get_metrics(err_0, yt_true)

    println("\n### βt Adaptive ###")
    get_metrics(err_t, yt_true)

end



