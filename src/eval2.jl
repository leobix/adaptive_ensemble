include("algos/multi_armed_bandits.jl")
include("algos/passive_agressive.jl")


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
    MAPE = 100*sum(abs.(err./yt_true))/size(err)[1]
    RMSE = sqrt(sum(abs2.(err))/size(err)[1])
    println("MAE : ", MAE)
    println("MAPE : ", MAPE)
    println("RMSE : ", RMSE)
    println("R2 : ", R2)
    println("CVAR 0.05 :", CVAR_05)
    println("CVAR 0.15 :", CVAR_15)
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
    #THE BASELINE is stronger if I put X0,Xt than X0 alone
    β_l2_init = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);

    # Version where we have a stable part for beta0 and the rest linear adaptive
    β_list_linear_adaptive0 = zeros(val, p)
    β_list_linear_adaptive_pure = zeros(val, p)
    β_list_linear_adaptive_pure_Vt = zeros(val, p)
    β_list_linear_adaptive_trained_one = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard = zeros(val, p)


    _, β0_0, V0_0, _ = adaptive_ridge_regression_exact(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)
    _, β0_1, V0_1 = adaptive_ridge_regression_standard(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past)

    #TODO ADD version with only one first training and no stable part

    # Version where we have a stable part for beta0 and we add more data every time to the beta 0

    ## Version where we have a stable part for beta0 and we add more data every time to the adaptive

    # Version where we adapt from the beginning and we don't add data every time

    # Version where we adapt from the beginning and we add data every time

    β_list_bandits_t = zeros(val, p)
    β_list_bandits_all = zeros(val, p)
    β_list_PA = zeros(val, p)
    β_PA = β_l2_init

    for s=1:val

        #TODO check split_index with max(split inex, 1)
        if more_data_for_β0
         #CHECK NO CHEATING WITH S
            X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), s+(num_past-1)*past, past-1, uncertainty, last_yT)
        else
        # I checked NO CHEATING WITH S because later on we start at split_index+2 to evaluate
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
        #Line to code the input X and y for the linear adaptive rule for more data case

        if more_data_for_β0
            X_adaptive = X[max(split_index-num_past*(past+1)+1,1):split_index+s,:]
            y_adaptive = y[max(split_index-num_past*(past+1)+1,1):split_index+s,:]
            N0_adaptive = (num_past-1)*past + 1 + s
        else
        #CHECK IF we give more of the data to beta 0 or beta t, for now to beta 0
            #Line to code the input X and y for the linear adaptive rule for no more data case
            X_adaptive = X[max(s+split_index-num_past*(past+1)+1,1):split_index+s,:]
            y_adaptive = y[max(s+split_index-num_past*(past+1)+1,1):split_index+s,:]
            X__, Z__, y__ = get_X_Z_y(X_adaptive, y_adaptive, past)
            #println("Z ADAPT", Z__[end,:])
            #println("X ADAPT", X__[end,:])
            #We always keep the same number of elements for the time-varying part
            N0_adaptive = (num_past-1)*past
        end
        #Solve the problem for beta 0 case
        #CHECK past or past-1

        #obj, β_linear_adaptive0_0, V0_adaptive0, β_linear_adaptive0_t = adaptive_ridge_regression_exact(X_adaptive, y_adaptive, ρ, ρ, past, N0_adaptive)
        obj, β_linear_adaptive0_0, V0_adaptive0, β_linear_adaptive0_t = adaptive_ridge_regression_exact(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 10)

        #Solve the problem for pure adaptive case
#         obj, β_linear_adaptive_pure_0, V0_adaptive_pure, β_linear_adaptive_pure_t = adaptive_ridge_regression_exact(X_adaptive, y_adaptive, ρ, ρ, past, 1)
        obj, β_linear_adaptive_pure_0, V0_adaptive_pure, β_linear_adaptive_pure_t = adaptive_ridge_regression_exact(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)

        obj, β_linear_adaptive_pure_0_Vt, Vt_adaptive_pure, _ = adaptive_ridge_regression_exact_Vt(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)

        #Line to get Z_{t+1}: CHECKED
        X_for_Z = X[split_index-past+s+1:split_index+s+1,:]
        y_for_Z = y[split_index-past+s+1:split_index+s+1,:]
        X_, Z_test, y_ = get_X_Z_y(X_for_Z, y_for_Z, past)

        β_linear_adaptive0_test = β_linear_adaptive0_0 + V0_adaptive0 * Z_test[1,:]
        β_linear_adaptive_pure_test = β_linear_adaptive_pure_0 + V0_adaptive_pure * Z_test[1,:]
        β_linear_adaptive_pure_test_Vt = β_linear_adaptive_pure_0_Vt + Vt_adaptive_pure[end,:,:] * Z_test[1,:]

        #BASELINES
        β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt), vcat(y0,yt))
        β_list_bandits_t[s,:] = compute_bandit_weights(Xt, yt)
        β_PA = compute_PA_weights(0.001, β_PA, Matrix(Xt)[end,:], yt[end])
        β_list_PA[s,:] = β_PA


        β_listt[s,:] = βt_val[past-1,:] #TODO check with end
        β_list0[s,:] = β0_val
        β_l2 = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
        β_listl2[s,:] = β_l2
        β_list_linear_adaptive0[s,:] = β_linear_adaptive0_test
        β_list_linear_adaptive_pure[s,:] = β_linear_adaptive_pure_test
        β_list_linear_adaptive_pure_Vt[s,:] = β_linear_adaptive_pure_test_Vt
        β_list_linear_adaptive_trained_one[s,:] = β0_0 + V0_0 * Z_test[1,:]
        #println("SIZES ", size(β0_0), "SIZES ", size(V0_0), "SIZES ", size(Z_test))
        β_list_linear_adaptive_trained_one_standard[s,:] = β0_1 + V0_1 * Z_test[1, :]
        #println("SIZES ", β_list_linear_adaptive_trained_one_standard)


    end

    #TODO Best underlying model

    X0, y0, Xt, yt, _, D_min, D_max = prepare_data_from_y(X, y, 1, split_index, val, uncertainty, last_yT)
    _, _, _, _, yt_true, _, _ = prepare_data_from_y(X, y_true, 1, split_index, val, uncertainty, last_yT)
    #println(yt_true)
    #pred = [sum(Xt[i, j] * (β0_1[i,j] + sum(V0_1[i, j, :] .* Z[i, :])) for j=1:P) for i=1:val]
    err_mean = [abs(yt_true[s]-mean(Xt[s,:])) for s=1:val]
    err_0 = [abs(yt_true[s]-dot(Xt[s,:],β_list0[s,:])) for s=1:val]
    err_bandit_full = [abs(yt_true[s]-dot(Xt[s,:],β_list_bandits_all[s,:])) for s=1:val]
    err_bandit_t = [abs(yt_true[s]-dot(Xt[s,:],β_list_bandits_t[s,:])) for s=1:val]
    err_PA = [abs(yt_true[s]-dot(Xt[s,:],β_list_PA[s,:])) for s=1:val]
    err_t = [abs(yt_true[s]-dot(Xt[s,:],β_listt[s,:])) for s=1:val]
    err_baseline = [abs(yt_true[s]-dot(Xt[s,:],β_l2_init)) for s=1:val]
    err_l2 = [abs(yt_true[s]-dot(Xt[s,:],β_listl2[s,:])) for s=1:val]
    err_linear_adaptive0 = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive0[s,:])) for s=1:val]
    err_linear_adaptive_pure = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_pure[s,:])) for s=1:val]
    err_linear_adaptive_pure_Vt = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_pure_Vt[s,:])) for s=1:val]
    err_linear_adaptive_trained_one = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:])) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:])) for s=1:val] #[abs(y_true[i]-pred[i]) for i=1:val]#

    #TODO check get_metrics
    println("\n### Mean Baseline ###")
    get_metrics(err_mean, yt_true)

    println("\n### Bandits Full Baseline ###")
    get_metrics(err_bandit_full, yt_true)

    println("\n### Bandits Only Last T Baseline ###")
    get_metrics(err_bandit_t, yt_true)

    println("\n### Passive-Aggressive Baseline ###")
    ### The Beta 0 that is originating from the adaptive formulation
    get_metrics(err_PA, yt_true)

    println("\n### β0 Baseline ###")
    get_metrics(err_baseline, yt_true)

    println("\n### β0 Baseline Retrained ###")
    get_metrics(err_l2, yt_true)

    println("\n### β0 Fully Adaptive ###")
    ### The Beta 0 that is originating from the adaptive formulation
    get_metrics(err_0, yt_true)

    println("\n### βt Fully Adaptive ###")
    ### Using Beta t+1 = Beta t, with Beta t that is originating from the adaptive formulation
    get_metrics(err_t, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with Stable Part ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with stable part
    get_metrics(err_linear_adaptive0, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(err_linear_adaptive_pure, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part Vt ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(err_linear_adaptive_pure_Vt, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(err_linear_adaptive_trained_one, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(err_linear_adaptive_trained_one_standard, yt_true)

end



