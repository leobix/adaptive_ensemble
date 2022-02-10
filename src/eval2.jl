include("algos/multi_armed_bandits.jl")
include("algos/passive_agressive.jl")
include("algos/benders.jl")
include("algos/master_primal.jl")
include("algos/OLS.jl")
include("algos/adaptive_linear_decision_rule.jl")
include("metrics.jl")

#TODO: COMPUTE ALL METRICS IN THE ORIGINAL SPACE, in particular MAPE

function eval_method(args, X, y, y_true, split_, past, num_past, val, uncertainty, ϵ_inf, δ_inf, last_yT,
        ϵ_l2, δ_l2, ρ, reg, max_cuts, verbose,
        fix_β0, more_data_for_β0, benders, ridge, mean_y, std_y)

    threshold_benders = 0.01
    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end

    X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), min(num_past*past, split_index), val, uncertainty, last_yT)
    println("There are ", size(X)[1], " samples in total.")
    println("We start training at index ", max(split_index-num_past*past+1, 1))
    println("We test between index ", max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index), " and ",  max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index)+val+1)

    β_list0 = zeros(val, p)
    β_listt = zeros(val, p)
    β_listl2 = zeros(val, p)
    #THE BASELINE is stronger if I put X0,Xt than X0 alone
    #β_l2_init = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
    β_l2_init = l2_regression(X0,y0,ρ);

    #β_list_linear_adaptive_pure_Vt = zeros(val, p)
    β_list_linear_adaptive_trained_one = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard = zeros(val, p)


    #_, β0_0, V0_0, _ = adaptive_ridge_regression_exact_no_stable(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past)
    _, β0_0, V0_0, _ = adaptive_ridge_regression_exact_no_stable(X0, y0, ρ, args["rho_V"], past)
    _, β0_1, V0_1 = adaptive_ridge_regression_standard(X0, y0, ρ, args["rho_V"], past)

    #TODO: Uncomment
    #obj, β_linear_adaptive_pure_0_Vt, Vt_adaptive_pure, _ = adaptive_ridge_regression_exact_Vt(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)

    β_list_bandits_t = zeros(val, p-1)
    β_list_bandits_all = zeros(val, p-1)
    β_list_PA = zeros(val, p)
    #IMPORTANT: We initialize with equal weights but we could also initialize with l2 weights
    β_PA = ones(p)/(p)#β_l2_init[2:end]
    println("Optimization finished. Evaluation starts.")


    for s=1:val

        #TODO check split_index with max(split inex, 1)
        if more_data_for_β0
            X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(split_index-num_past*past+1, 1), s+(num_past-1)*past, past-1, uncertainty, last_yT)
        else
            X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(s+split_index-num_past*past+1, 1), (num_past-1)*past, past-1, uncertainty, last_yT)
        end

        #TODO: evaluate also if we change Vt regularly
        #obj, β_linear_adaptive_pure_0_Vt, Vt_adaptive_pure, _ = adaptive_ridge_regression_exact_Vt(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)


        #Line to get Z_{t+1}
        X_for_Z = X[split_index-past+s+1:split_index+s+1,:]
        X_for_Z[:,1] .= 1
        y_for_Z = y[split_index-past+s+1:split_index+s+1,:]
        X_, Z_test, y_ = get_X_Z_y(X_for_Z, y_for_Z, past)

        #BASELINES
        β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt)[:,2:end], vcat(y0,yt))
        β_list_bandits_t[s,:] = compute_bandit_weights(Xt[:,2:end], yt)
        β_PA = compute_PA_weights(0.001, β_PA, Matrix(Xt)[end,1:end], yt[end])
        β_list_PA[s,:] = β_PA
        #β_l2 = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
        #β_listl2[s,:] = β_l2

        #TODO Add if needed
        #β_list_linear_adaptive_pure_Vt[s,:] = β_linear_adaptive_pure_0_Vt + Vt_adaptive_pure[end,:,:] * Z_test[1,:]
        β_list_linear_adaptive_trained_one[s,:] = β0_0 + V0_0 * Z_test[1,:]
        β_list_linear_adaptive_trained_one_standard[s,:] = β0_1 + V0_1 * Z_test[1, :]

    end

    #TODO Best underlying model
    println("Evaluation finished. Metrics start.")
    X0, y0, Xt, yt, _, D_min, D_max = prepare_data_from_y(X, y, 1, split_index, val, uncertainty, last_yT)
    _, _, _, _, yt_true, _, _ = prepare_data_from_y(X, y_true, 1, split_index, val, uncertainty, last_yT)

    # Unstandardize for metrics
    yt_true = yt_true.*std_y.+mean_y

    # Unstandardize predictions as well
    # The reason why we put 2:end is because the first element is the intercept term (1)
    err_mean = [abs(yt_true[s]-(mean(Xt[s,2:end]).*std_y.+mean_y)) for s=1:val]
    err_bandit_full = [abs(yt_true[s]-(dot(Xt[s,2:end],β_list_bandits_all[s,:]).*std_y.+mean_y)) for s=1:val]
    err_bandit_t = [abs(yt_true[s]-(dot(Xt[s,2:end],β_list_bandits_t[s,:]).*std_y.+mean_y)) for s=1:val]
    err_PA = [abs(yt_true[s]-(dot(Xt[s,1:end],β_list_PA[s,:]).*std_y.+mean_y)) for s=1:val]
    err_baseline = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init).*std_y.+mean_y)) for s=1:val]
    #err_l2 = [abs(yt_true[s]-dot(Xt[s,:],β_listl2[s,:])) for s=1:val]

    #TODO: Uncomment
    #err_linear_adaptive_pure_Vt = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_pure_Vt[s,:])) for s=1:val]
    err_linear_adaptive_trained_one = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y)) for s=1:val]

    #TODO check get_metrics
    println("\n### Mean Baseline ###")
    get_metrics(args, "mean", err_mean, yt_true)

    println("\n### Bandits Full Baseline ###")
    get_metrics(args, "bandits_full", err_bandit_full, yt_true)

    println("\n### Bandits Only Last T Baseline ###")
    get_metrics(args, "bandits_recent", err_bandit_t, yt_true)

    println("\n### Passive-Aggressive Baseline ###")
    ### The Beta 0 that is originating from the adaptive formulation
    get_metrics(args, "PA", err_PA, yt_true)

    println("\n### β0 Baseline ###")
    get_metrics(args, "ridge", err_baseline, yt_true)

#     println("\n### β0 Baseline Retrained ###")
#     get_metrics(err_l2, yt_true)

    #TODO: Uncomment
#     println("\n### βt Linear Decision Rule Adaptive with NO Stable Part Vt ###")
#     ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
#     get_metrics(err_linear_adaptive_pure_Vt, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact", err_linear_adaptive_trained_one, yt_true)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard", err_linear_adaptive_trained_one_standard, yt_true)

end



