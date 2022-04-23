include("algos/multi_armed_bandits.jl")
include("algos/passive_agressive.jl")
include("algos/benders.jl")
include("algos/master_primal.jl")
include("algos/OLS.jl")
include("algos/adaptive_linear_decision_rule.jl")
include("metrics.jl")

using Dates

#TODO: COMPUTE ALL METRICS IN THE ORIGINAL SPACE, in particular MAPE

function eval_method(args, X, y, y_true, split_, past, num_past, val, mean_y, std_y)

    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end

    X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), min(num_past*past, split_index), val, args["uncertainty"], args["last_yT"])
    println("There are ", size(X)[1], " samples in total.")
    println("Number of samples in train set: ", size(y0))
    args["train_length"] = size(y0)[1]
    println("We start training at index ", max(split_index-num_past*past+1, 1))
    println("We test between index ", max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index), " and ",  max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index)+val+1)

    β_list0 = zeros(val, p)
    β_listt = zeros(val, p)
    β_listl2 = zeros(val, p)

    #### Standard Ridge Regression
    start = now()
    β_l2_init = l2_regression(X0,y0,args["rho"], 0); #0 is for beta_stat which I removed
    l2_regression_time = (now() - start).value
    println("Time L2 Regression ", l2_regression_time)

    #β_list_linear_adaptive_pure_Vt = zeros(val, p)

    β_list_linear_adaptive_trained_one = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard = zeros(val, p)


    β_list_linear_adaptive_trained_one_err_rule = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard_err_rule = zeros(val, p)

    #### ARO Ridge + standard. The regularization is applied to all reg coefs, including a separate reg on beta and V
    args["err_rule"] = false
    start = now()
    _, β0_0, V0_0, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, args["rho_beta"], args["rho"], args["rho_V"], past)
    arole_allreg_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_allreg_regression_time)

    #### ARO Ridge + standard. The regularization is applied to beta and V only not the time varying beta
    start = now()
    _, β0_1, V0_1 = adaptive_ridge_regression_exact_no_stable(args, X0, y0, 0, args["rho"], args["rho_V"], past)
    arole_beta0andV_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_beta0andV_regression_time)

    #### We use the error of the forecasts in the past timesteps instead of the forecasts themselves.
    #### ARO Ridge + Error rule for Z instead of the values of the past forecasts
    args["err_rule"] = true

    start = now()
    _, β0_0_err_rule, V0_0_err_rule, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, args["rho_beta"], args["rho"], args["rho_V"], past)
    arole_allreg_errorrule_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_allreg_errorrule_regression_time)

    start = now()
    _, β0_1_err_rule, V0_1_err_rule, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, 0, args["rho"], args["rho_V"], past)
    arole_beta0andV_errorrule_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_beta0andV_errorrule_regression_time)
    #_, β0_1, V0_1 = adaptive_ridge_regression_standard(args, X0, y0, args["rho"], args["rho_V"], past)

    #TODO: Uncomment
    #obj, β_linear_adaptive_pure_0_Vt, Vt_adaptive_pure, _ = adaptive_ridge_regression_exact_Vt(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)

    β_list_bandits_t = zeros(val, p-1)
    β_list_bandits_all = zeros(val, p-1)
    β_list_PA = zeros(val, p)
    #IMPORTANT: We initialize with equal weights but we could also initialize with l2 weights
    β_PA = ones(p)/(p)#β_l2_init[2:end]
    println("Optimization finished. Evaluation starts.")

    last_timesteps = zeros(val)

    for s=1:val

        #TODO check split_index with max(split inex, 1) and CHECK the MIN
        #The min ensures we remain in bounds.
        X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(s+split_index-num_past*past+1, 1), min((num_past-1)*past,split_index-past+s), past-1, args["uncertainty"], args["last_yT"])

        #Line to get Z_{t+1}
        X_for_Z = X[split_index-past+s+1:split_index+s+1,:]
        X_for_Z[:,1] .= 1
        y_for_Z = y[split_index-past+s+1:split_index+s+1,:]

        args["err_rule"] = false
        X_, Z_test, y_ = get_X_Z_y(args, X_for_Z, y_for_Z, past)
        args["err_rule"] = true
        _, Z_test_err_rule, _ = get_X_Z_y(args, X_for_Z, y_for_Z, past)

        #BASELINES
        β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt)[:,2:end], vcat(y0,yt))
        β_list_bandits_t[s,:] = compute_bandit_weights(Xt[:,2:end], yt)
        β_PA = compute_PA_weights(args["rho_beta"], β_PA, Matrix(Xt)[end,1:end], yt[end])
        β_list_PA[s,:] = β_PA
        #β_l2 = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
        #β_listl2[s,:] = β_l2

        #TODO Add if needed
        #β_list_linear_adaptive_pure_Vt[s,:] = β_linear_adaptive_pure_0_Vt + Vt_adaptive_pure[end,:,:] * Z_test[1,:]
        β_list_linear_adaptive_trained_one[s,:] = β0_0 + V0_0 * Z_test[1,:]
        β_list_linear_adaptive_trained_one_standard[s,:] = β0_1 + V0_1 * Z_test[1, :]

        β_list_linear_adaptive_trained_one_err_rule[s,:] = β0_0_err_rule + V0_0_err_rule * Z_test_err_rule[1,:]
        β_list_linear_adaptive_trained_one_standard_err_rule[s,:] = β0_1_err_rule + V0_1_err_rule * Z_test_err_rule[1, :]

        last_timesteps[s] = Z_test[1,end]
    end

    #TODO Best underlying model
    println("Evaluation finished. Metrics start.")
    X0, y0, Xt, yt, _, D_min, D_max = prepare_data_from_y(X, y, 1, split_index, val, args["uncertainty"], args["last_yT"])
    _, _, _, _, yt_true, _, _ = prepare_data_from_y(X, y_true, 1, split_index, val, args["uncertainty"], args["last_yT"])

    # Unstandardize for metrics
    yt_true = yt_true.*std_y.+mean_y

    # Unstandardize predictions as well
    # The reason why we put 2:end is because the first element is the intercept term (1)
    err_mean = [abs(yt_true[s]-(mean(Xt[s,2:end]).*std_y.+mean_y)) for s=1:val]
    err_last_timestep = [abs(yt_true[s]-(last_timesteps[s].*std_y.+mean_y)) for s=2:val]
    err_best_model = get_best_model_errors(yt_true, Xt, mean_y, std_y)
    err_bandit_full = [abs(yt_true[s]-(dot(Xt[s,2:end],β_list_bandits_all[s,:]).*std_y.+mean_y)) for s=1:val]
    err_bandit_t = [abs(yt_true[s]-(dot(Xt[s,2:end],β_list_bandits_t[s,:]).*std_y.+mean_y)) for s=1:val]
    err_PA = [abs(yt_true[s]-(dot(Xt[s,1:end],β_list_PA[s,:]).*std_y.+mean_y)) for s=1:val]
    err_baseline = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init).*std_y.+mean_y)) for s=1:val]
    #err_l2 = [abs(yt_true[s]-dot(Xt[s,:],β_listl2[s,:])) for s=1:val]

    #TODO: Uncomment
    #err_linear_adaptive_pure_Vt = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_pure_Vt[s,:])) for s=1:val]
    err_linear_adaptive_trained_one = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y)) for s=1:val]

    err_linear_adaptive_trained_one_err_rule = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_err_rule[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard_err_rule = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard_err_rule[s,:]).*std_y.+mean_y)) for s=1:val]

    #TODO check get_metrics
    println("\n### Mean Baseline ###")
    get_metrics(args, "mean", err_mean, yt_true)

    println("\n### Last Timestep Baseline ###")
    get_metrics(args, "last_timestep", err_last_timestep, yt_true[2:end])

    println("\n### Best Model Baseline ###")
    get_metrics(args, "best_model", err_best_model, yt_true)

    println("\n### Bandits Full Baseline ###")
    get_metrics(args, "bandits_full", err_bandit_full, yt_true)

    println("\n### Bandits Only Last T Baseline ###")
    get_metrics(args, "bandits_recent", err_bandit_t, yt_true)

    println("\n### Passive-Aggressive Baseline ###")
    ### The Beta 0 that is originating from the adaptive formulation
    get_metrics(args, "PA", err_PA, yt_true)

    println("\n### β0 Baseline ###")
    get_metrics(args, "ridge", err_baseline, yt_true, l2_regression_time)

#     println("\n### β0 Baseline Retrained ###")
#     get_metrics(err_l2, yt_true)

    #TODO: Uncomment
#     println("\n### βt Linear Decision Rule Adaptive with NO Stable Part Vt ###")
#     ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
#     get_metrics(err_linear_adaptive_pure_Vt, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact", err_linear_adaptive_trained_one, yt_true, arole_allreg_regression_time)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard", err_linear_adaptive_trained_one_standard, yt_true, arole_beta0andV_regression_time)

    #SAME AS LAST 2, BUT WITH ERROR RULES i.e., instead of forecast values we use the previous errors of the models
    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE + ERROR RULE for Z ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact_err_rule", err_linear_adaptive_trained_one_err_rule, yt_true, arole_allreg_errorrule_regression_time)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD + ERROR RULE for Z ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard_err_rule", err_linear_adaptive_trained_one_standard_err_rule, yt_true, arole_beta0andV_errorrule_regression_time)

end



function eval_method_hurricane(args, X, Z, y, y_true, split_, past, num_past, val, mean_y, std_y)

    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end

    X0, Z0, y0, Xt, Zt, yt, yt_true, D_min, D_max = prepare_data_from_y_hurricane(X, Z, y, max(split_index-num_past*past+1, 1), min(num_past*past, split_index), val, args["uncertainty"], args["last_yT"])
    println("Training data X0 size ", size(X0))
    println("Testing data Xt size ", size(Xt))
    println("Z ", size(Z0))
    println("y ", size(y0))
    println("There are ", size(X)[1], " samples in total.")
    println("We start training at index ", max(split_index-num_past*past+1, 1))
    println("We test between index ", max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index), " and ",  max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index)+val+1)

    β_list0 = zeros(val, p)
    β_listt = zeros(val, p)
    β_listl2 = zeros(val, p)


    β_l2_init = l2_regression(X0,y0,args["rho_beta"], 0);
    ## Uncomment for statistical error regularization
    #β_l2_init_stat = l2_regression(X0,y0,args["rho_beta"], args["rho_stat"]);

    β_list_linear_adaptive_trained_one = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard = zeros(val, p)

    _, β0_0, V0_0, _ = adaptive_ridge_regression_exact_no_stable_hurricane(args, X0, Z0, y0, args["rho_beta"], args["rho"], args["rho_V"])
    _, β0_1, V0_1 = adaptive_ridge_regression_exact_no_stable_hurricane(args, X0, Z0, y0, 0, args["rho"], args["rho_V"])
#     println("Beta 0", β0_0)
#     println("V 0 ", V0_0)

    β_list_bandits_t = zeros(val, p-1)
    β_list_bandits_all = zeros(val, p-1)
    β_list_PA = zeros(val, p)
    #IMPORTANT: We initialize with equal weights but we could also initialize with l2 weights
    β_PA = ones(p)/(p)#β_l2_init[2:end]
    println("Optimization finished. Evaluation starts.")

    #SOLVE PROBLEM WITH s=1
    for s=1:val
        #BASELINES
        if s < 5
            β_list_bandits_all[s,:] = ones(p-1)/(p-1)
        else
            β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt[1:s-4,:])[:,1:end-1], vcat(y0,yt[1:s-4]))
            #do not take the intercept term into account
            #β_list_bandits_t[s,:] = compute_bandit_weights(Xt[1:s,1:end-1], yt[s])
            β_PA = compute_PA_weights(0.01, β_PA, Matrix(Xt)[s-4,:], yt[s-4])
        end
        β_list_PA[s,:] = β_PA

        β_list_linear_adaptive_trained_one[s,:] = β0_0 + V0_0 * Zt[s,:]
        β_list_linear_adaptive_trained_one_standard[s,:] = β0_1 + V0_1 * Zt[s,:]

#         println("Iter ", s)
#         println(β_list_linear_adaptive_trained_one_standard[s,:])
#         println()
    end

    # Unstandardize for metrics

    yt_true = yt_true.*std_y.+mean_y

    # Unstandardize predictions as well
    # The reason why we put 1:end-1 is because the last element is the intercept term (1)
    err_mean = [abs(yt_true[s]-(mean(Xt[s,1:end-1]).*std_y.+mean_y)) for s=1:val]
    err_best_model = get_best_model_errors(yt_true, Xt, mean_y, std_y)
    err_bandit_full = [abs(yt_true[s]-(dot(Xt[s,1:end-1],β_list_bandits_all[s,:]).*std_y.+mean_y)) for s=1:val]
    err_last_timestep = [abs(yt_true[s]-(Zt[s,end].*std_y.+mean_y)) for s=2:val]
    err_PA = [abs(yt_true[s]-(dot(Xt[s,:],β_list_PA[s,:]).*std_y.+mean_y)) for s=1:val]
    err_baseline = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init).*std_y.+mean_y)) for s=1:val]
    #err_baseline_stat = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init_stat).*std_y.+mean_y)) for s=1:val]

    err_linear_adaptive_trained_one = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y)) for s=1:val]

    #TODO check get_metrics
    println("\n### Mean Baseline ###")
    get_metrics(args, "mean", err_mean, yt_true)

    println("\n### Best Model Baseline ###")
    get_metrics(args, "best_model", err_best_model, yt_true)

    println("\n### Bandits Full Baseline ###")
    get_metrics(args, "bandits_full", err_bandit_full, yt_true)

    println("\n### Last Timestep Baseline ###")
    get_metrics(args, "last_timestep", err_last_timestep, yt_true[2:end])

    println("\n### Passive-Aggressive Baseline ###")
    ### The Beta 0 that is originating from the adaptive formulation
    get_metrics(args, "PA", err_PA, yt_true)

    println("\n### Ridge Baseline ###")
    get_metrics(args, "ridge", err_baseline, yt_true)

#     println("\n### Ridge + Stat Baseline ###")
#     get_metrics(args, "ridge_stat", err_baseline_stat, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact", err_linear_adaptive_trained_one, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard", err_linear_adaptive_trained_one_standard, yt_true)

end