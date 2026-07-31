include("algos/multi_armed_bandits.jl")
include("algos/hedge.jl")
include("algos/rls.jl")
include("algos/passive_agressive.jl")
include("algos/benders.jl")
include("algos/master_primal.jl")
include("algos/OLS.jl")
include("algos/adaptive_linear_decision_rule.jl")
include("algos/fedformer.jl")
include("metrics.jl")

using Dates
using DelimitedFiles
using CSV, DataFrames


function eval_method(args, X, y, y_true, split_, past, num_past, val, mean_y, std_y)

    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end


    # Notice the small adjustments to make sure the indices work
    training_index_begin = max(split_index-num_past*past+1, 1) #first index of training data
    training_index_end = min(num_past*past, split_index) #last index of training data (included)

    # Prepare all data based on the desired splits
    X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, training_index_begin, training_index_end, val, args["uncertainty"], args["last_yT"])

    println("There are ", size(X)[1], " samples in total.")
    println("Number of samples in train set: ", size(y0))
    args["train_length"] = size(y0)[1]
    println("We start training at index ", training_index_begin)
    #TODO Make a check that this is corresponding to training_index_end+1
    println("We test between index ", max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index), " and ",  max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index)+val+1)

    #initialize the lists to store all the weights optimized for each time step (can be constant)
    #TODO In fact useless, check
    #β_list0 = zeros(val, p)
    #β_listt = zeros(val, p)
    #β_listl2 = zeros(val, p)

    #### Standard Ridge Regression
    start = now()
    β_l2_init = l2_regression(X0,y0,args["rho"], 0); #0 is for beta_stat which I removed
    l2_regression_time = (now() - start).value
    println("Time L2 Regression ", l2_regression_time)

    #β_list_linear_adaptive_pure_Vt = zeros(val, p)

    #Initialize for the adaptive ridge
    β_list_linear_adaptive_trained_one = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard = zeros(val, p)
    β_list_linear_adaptive_trained_one_err_rule = zeros(val, p)
    β_list_linear_adaptive_trained_one_standard_err_rule = zeros(val, p)

    #### ARO Ridge + standard. The regularization is applied to all reg coefs, including an additional reg on beta and V
    args["err_rule"] = false
    start = now()
    _, β0_0, V0_0, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, args["rho_beta"], args["rho"], args["rho_V"], past)
    arole_allreg_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_allreg_regression_time)

    #### ARO Ridge + standard. The regularization is applied to beta 0 and V only from the decision rule not the time varying beta.
    start = now()
    _, β0_1, V0_1 = adaptive_ridge_regression_exact_no_stable(args, X0, y0, 0, args["rho"], args["rho_V"], past)
    arole_beta0andV_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_beta0andV_regression_time)

    #### We use the error of the forecasts in the past timesteps instead of the forecasts themselves.
    #### ARO Ridge + Error rule for Z instead of the values of the past forecasts
    # The regularization is applied to all reg coefs, including an additional reg on beta and V
    args["err_rule"] = true

    start = now()
    _, β0_0_err_rule, V0_0_err_rule, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, args["rho_beta"], args["rho"], args["rho_V"], past)
    arole_allreg_errorrule_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_allreg_errorrule_regression_time)

    # Now, the regularization is applied to beta 0 and V only from the decision rule not the time varying beta.
    start = now()
    _, β0_1_err_rule, V0_1_err_rule, _ = adaptive_ridge_regression_exact_no_stable(args, X0, y0, 0, args["rho"], args["rho_V"], past)
    arole_beta0andV_errorrule_regression_time = (now() - start).value
    println("Time Adaptive Regression Standard ", arole_beta0andV_errorrule_regression_time)
    #_, β0_1, V0_1 = adaptive_ridge_regression_standard(args, X0, y0, args["rho"], args["rho_V"], past)

    #TODO: Uncomment
    #obj, β_linear_adaptive_pure_0_Vt, Vt_adaptive_pure, _ = adaptive_ridge_regression_exact_Vt(vcat(X0,Xt), vcat(y0,yt), ρ, ρ, past, 1)

    #Initialize the different benchmarks
    β_list_bandits_t = zeros(val, p-1)
    β_list_bandits_all = zeros(val, p-1)
    β_list_hedge = zeros(val, p-1)
    β_list_PA = zeros(val, p)
    #IMPORTANT: We initialize with equal weights but we could also initialize with l2 weights
    β_PA = ones(p)/(p)#β_l2_init[2:end]
    println("Optimization finished. Evaluation starts.")

    last_timesteps = zeros(val)

    # FEDformer (train once on the training window)
    fedformer_pred = nothing
    if haskey(args, "fedformer") && args["fedformer"]
        train_start = training_index_begin
        train_end = min(training_index_begin + training_index_end - 1, split_index)
        fedformer_pred = fedformer_train_predict(args, X, y, train_start, train_end, split_index, val)
    end

    for s=1:val

        #TODO check split_index with max(split index, 1) and CHECK the MIN
        #The min and max ensure we remain in bounds.
        X0, y0, Xt, yt, yt_true, D_min, D_max = prepare_data_from_y(X, y, max(s+split_index-num_past*past+1, 1), min((num_past-1)*past,split_index-past+s), past-1, args["uncertainty"], args["last_yT"])

        #Line to get Z_{t+1}
        X_for_Z = X[split_index-past+s+1:split_index+s+1,:]
        X_for_Z[:,1] .= 1
        y_for_Z = y[split_index-past+s+1:split_index+s+1,:]

        #computation of Z_t changes depending on if we use forecasts and targets in the decision rule or forecasts - targets
        args["err_rule"] = false
        X_, Z_test, y_ = get_X_Z_y(args, X_for_Z, y_for_Z, past)
        args["err_rule"] = true
        _, Z_test_err_rule, _ = get_X_Z_y(args, X_for_Z, y_for_Z, past)

        #BASELINES
        β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt)[:,2:end], vcat(y0,yt))
        β_list_bandits_t[s,:] = compute_bandit_weights(Xt[:,2:end], yt)
        β_PA = compute_PA_weights(args["rho_beta"], β_PA, Matrix(Xt)[end,1:end], yt[end])
        β_list_PA[s,:] = β_PA

        #A possibility is to retrain L2 at each time step with the new data but too expensive to compute
        #β_l2 = l2_regression(vcat(X0,Xt),vcat(y0,yt),ρ);
        #β_listl2[s,:] = β_l2

        #TODO Add if needed
        #β_list_linear_adaptive_pure_Vt[s,:] = β_linear_adaptive_pure_0_Vt + Vt_adaptive_pure[end,:,:] * Z_test[1,:]

        #Coefficients for Adaptive Ridge
        β_list_linear_adaptive_trained_one[s,:] = β0_0 + V0_0 * Z_test[1,:]
        β_list_linear_adaptive_trained_one_standard[s,:] = β0_1 + V0_1 * Z_test[1, :]

        β_list_linear_adaptive_trained_one_err_rule[s,:] = β0_0_err_rule + V0_0_err_rule * Z_test_err_rule[1,:]
        β_list_linear_adaptive_trained_one_standard_err_rule[s,:] = β0_1_err_rule + V0_1_err_rule * Z_test_err_rule[1, :]

        last_timesteps[s] = Z_test[1,end]
    end

    println("Evaluation finished. Evaluation start.")
    #TODO Check if I can remove the second line since I always use y_true now
    X0, y0, Xt, yt, _, D_min, D_max = prepare_data_from_y(X, y, 1, split_index, val, args["uncertainty"], args["last_yT"])
    _, _, _, _, yt_true, _, _ = prepare_data_from_y(X, y_true, 1, split_index, val, args["uncertainty"], args["last_yT"])

    # Unstandardize target for computing metrics
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


    err_linear_adaptive_trained_one = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y)) for s=1:val]

    err_linear_adaptive_trained_one_err_rule = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_err_rule[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard_err_rule = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard_err_rule[s,:]).*std_y.+mean_y)) for s=1:val]

    #Uncomment if needed
    #err_linear_adaptive_pure_Vt = [abs(yt_true[s]-dot(Xt[s,:],β_list_linear_adaptive_pure_Vt[s,:])) for s=1:val]
    #err_l2 = [abs(yt_true[s]-dot(Xt[s,:],β_listl2[s,:])) for s=1:val]

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

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact", err_linear_adaptive_trained_one, yt_true, arole_allreg_regression_time)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard", err_linear_adaptive_trained_one_standard, yt_true, arole_beta0andV_regression_time)

    if fedformer_pred !== nothing
        err_fedformer = [abs(yt_true[s]-(fedformer_pred[s].*std_y.+mean_y)) for s=1:val]
        println("\n### FEDformer ###")
        get_metrics(args, "fedformer", err_fedformer, yt_true)
        if haskey(args, "fedformer_save_preds") && args["fedformer_save_preds"]
            save_array_as_csv(args, reshape(fedformer_pred, :, 1), "results_beta/", "fedformer_preds")
        end
    end

    #SAME AS LAST 2, BUT WITH ERROR RULES i.e., instead of forecast values we use the previous errors of the models
    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE + ERROR RULE for Z ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_exact_err_rule", err_linear_adaptive_trained_one_err_rule, yt_true, arole_allreg_errorrule_regression_time)

    println("\n### βt Linear Decision Rule Adaptive with NO Stable Part and Trained ONCE STANDARD + ERROR RULE for Z ###")
    ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
    get_metrics(args, "adaptive_ridge_standard_err_rule", err_linear_adaptive_trained_one_standard_err_rule, yt_true, arole_beta0andV_errorrule_regression_time)

    #println("\n### β0 Baseline Retrained ###")
    #get_metrics(err_l2, yt_true)

    #TODO: Uncomment
#     println("\n### βt Linear Decision Rule Adaptive with NO Stable Part Vt ###")
#     ### Using Beta t+1 = Beta 0 + V0*Z_{t+1}, with Beta 0, V0 that is originating from the linear adaptive formulation with NO stable part
#     get_metrics(err_linear_adaptive_pure_Vt, yt_true)
end



function eval_method_hurricane(args, X, Z, y, y_true, split_, past, num_past, val, mean_y, std_y)
    """
    Specific function for the hurricane forecasting use case. Requires some adjustments because we predict t+24h and can't use the targets until they are would have been available in real time.
    """
    n, p = size(X)
    split_index = floor(Int,n*split_)
    if val == -1
        val = size(X)[1]
    end

    training_index_begin = max(split_index-num_past*past+1, 1)
    training_index_end = min(num_past*past, split_index)
    X0, Z0, y0, Xt, Zt, yt, yt_true, D_min, D_max = prepare_data_from_y_hurricane(X, Z, y, training_index_begin, training_index_end, val, args["uncertainty"], args["last_yT"])
    println("Training data X0 size ", size(X0))
    println("Testing data Xt size ", size(Xt))
    println("Z ", size(Z0))
    println("y ", size(y0))
    println("There are ", size(X)[1], " samples in total.")
    println("We start training at index ", max(split_index-num_past*past+1, 1))
    println("We test between index ", max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index), " and ",  max(split_index-num_past*past+1, 1)+1+min(num_past*past, split_index)+val+1)
    args["train_length"] = size(y0)[1]


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
    β_list_hedge_ewa = zeros(val, p-1)
    β_list_rls = zeros(val, p-1)
    #IMPORTANT: We initialize with equal weights but we could also initialize with l2 weights
    β_PA = ones(p)/(p)#β_l2_init[2:end]
    println("Optimization finished. Evaluation starts.")

    #SOLVE PROBLEM WITH s=1
    hedge_eta = 0.0
    try
        hedge_eta = args["hedge_eta"]
    catch e
        hedge_eta = 0.0
    end
    w_ewa = ones(p-1)/(p-1)
    λ_rls = 0.99
    try
        λ_rls = args["rls_lambda"]
    catch e
        λ_rls = 0.99
    end
    w_rls, P_rls = rls_init(p-1; δ=1000.0)
    # Warm start RLS with initial training window
    for r in 1:size(X0,1)
        x_r = vec(X0[r, 1:end-1])
        y_r = y0[r]
        w_rls, P_rls = rls_update(w_rls, P_rls, x_r, y_r, λ_rls)
    end

    # FEDformer (train once on the training window)
    fedformer_pred = nothing
    if haskey(args, "fedformer") && args["fedformer"]
        train_start = training_index_begin
        train_end = min(training_index_begin + training_index_end - 1, split_index)
        fedformer_pred = fedformer_train_predict(args, X, y, train_start, train_end, split_index, val)
    end
    for s=1:val
        #BASELINES
        #The reason why 5 in particular, is because the first 4 samples represent 4*6h of predictions and we are meant to predict 24h in advance so there is a lag to take into account.
        #Note that with hurricanes the samples are not necessarily contiguous, and that for a next hurricane we reuse the best weights obtained from the previous one.
        if s < 5
            β_list_bandits_all[s,:] = ones(p-1)/(p-1)
            β_list_hedge[s,:] = ones(p-1)/(p-1)
            β_list_hedge_ewa[s,:] = w_ewa
            β_list_rls[s,:] = w_rls
        else
            β_list_bandits_all[s,:] = compute_bandit_weights(vcat(X0,Xt[1:s-4,:])[:,1:end-1], vcat(y0,yt[1:s-4]))
            if hedge_eta > 0
                β_list_hedge[s,:] = compute_hedge_weights(vcat(X0,Xt[1:s-4,:])[:,1:end-1], vcat(y0,yt[1:s-4]); η=hedge_eta)
            else
                β_list_hedge[s,:] = compute_hedge_weights(vcat(X0,Xt[1:s-4,:])[:,1:end-1], vcat(y0,yt[1:s-4]))
            end
            # Dynamic Hedge (EWA): update with most recent usable sample (lag 4)
            m = p-1
            η_s = hedge_eta > 0 ? hedge_eta : sqrt(8log(m)/max(s,1))
            x_s = Matrix(Xt)[s-4, 1:end-1]
            y_s = yt[s-4]
            losses_s = abs2.(x_s .- y_s)
            w_ewa .*= exp.(-η_s .* losses_s)
            Zs = sum(w_ewa)
            if Zs == 0 || !isfinite(Zs)
                w_ewa .= 1/m
            else
                w_ewa ./= Zs
            end
            β_list_hedge_ewa[s,:] = w_ewa
            #do not take the intercept term into account
            #β_list_bandits_t[s,:] = compute_bandit_weights(Xt[1:s,1:end-1], yt[s])
            β_PA = compute_PA_weights(args["rho_beta"], β_PA, Matrix(Xt)[s-4,:], yt[s-4])
            # RLS update
            w_rls, P_rls = rls_update(w_rls, P_rls, vec(x_s), y_s, λ_rls)
            β_list_rls[s,:] = w_rls
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
    err_hedge = [abs(yt_true[s]-(dot(Xt[s,1:end-1],β_list_hedge[s,:]).*std_y.+mean_y)) for s=1:val]
    err_hedge_ewa = [abs(yt_true[s]-(dot(Xt[s,1:end-1],β_list_hedge_ewa[s,:]).*std_y.+mean_y)) for s=1:val]
    err_rls = [abs(yt_true[s]-(dot(Xt[s,1:end-1],β_list_rls[s,:]).*std_y.+mean_y)) for s=1:val]
    err_last_timestep = [abs(yt_true[s]-(Zt[s,end].*std_y.+mean_y)) for s=2:val]
    err_PA = [abs(yt_true[s]-(dot(Xt[s,:],β_list_PA[s,:]).*std_y.+mean_y)) for s=1:val]
    err_baseline = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init).*std_y.+mean_y)) for s=1:val]
     
    #err_baseline_stat = [abs(yt_true[s]-(dot(Xt[s,:],β_l2_init_stat).*std_y.+mean_y)) for s=1:val]

    err_linear_adaptive_trained_one = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one[s,:]).*std_y.+mean_y)) for s=1:val]
    err_linear_adaptive_trained_one_standard = [abs(yt_true[s]-(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y)) for s=1:val]
    
    
    pred_adaptive = [(dot(Xt[s,:],β_list_linear_adaptive_trained_one_standard[s,:]).*std_y.+mean_y) for s=1:val]
    pred_truth = [yt_true[s] for s=1:val]
    #pred_all = [(Xt[s,1].*std_y.+mean_y) for s=1:val]
    
    
    filename = "results_11_12_id14_16/"# results_11_11/
    
          CSV.write(filename*"pred_adaptive_"*args["data"]*"_trainlength_"*string(args["train_length"])*"_rhobeta_"*string(args["rho_beta"])*"_rho_"*string(args["rho"])*"_rhoV_"*string(args["rho_V"])*"_combo_"*string(args["param_combo"])*".csv", DataFrame(:pred => pred_adaptive))
    CSV.write(filename*"pred_truth_"*args["data"]*"_trainlength_"*string(args["train_length"])*"_"*string(args["param_combo"])*".csv", DataFrame(:pred_truth => pred_truth))
    #println(pred_all)
    #println(DataFrame(pred_all))
    #CSV.write(filename*"pred_adaptive_"*args["data"]*"_"*args["param_combo"]*".csv", pred_adaptive)
    #CSV.write(filename*"pred_truth_"*args["data"]*"_"*param_combo*".csv", pred_truth)
    #CSV.write(filename*"pred_all_"*args["data"]*"_"*param_combo*".csv", pred_all)

    #TODO check get_metrics
    println("\n### Mean Baseline ###")
    get_metrics(args, "mean", err_mean, yt_true)

    println("\n### Best Model Baseline ###")
    get_metrics(args, "best_model", err_best_model, yt_true)

    println("\n### Bandits Full Baseline ###")
    get_metrics(args, "bandits_full", err_bandit_full, yt_true)

    println("\n### Hedge Baseline ###")
    get_metrics(args, "hedge", err_hedge, yt_true)

    println("\n### Hedge (EWA) Baseline ###")
    get_metrics(args, "hedge-ewa", err_hedge_ewa, yt_true)

    println("\n### RLS (Forgetting) Baseline ###")
    get_metrics(args, "rls", err_rls, yt_true)

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

    if fedformer_pred !== nothing
        err_fedformer = [abs(yt_true[s]-(fedformer_pred[s].*std_y.+mean_y)) for s=1:val]
        println("\n### FEDformer ###")
        get_metrics(args, "fedformer", err_fedformer, yt_true)
        if haskey(args, "fedformer_save_preds") && args["fedformer_save_preds"]
            save_array_as_csv(args, reshape(fedformer_pred, :, 1), "results_beta/", "fedformer_preds")
        end
    end

end
