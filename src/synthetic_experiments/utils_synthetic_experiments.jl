using Statistics, StatsBase, Distributions
####TODO ADD SEED
function create_y(T, period, bias, σ_pert)
    d = Normal(0, σ_pert)
    y_base = [sin(2*pi*period*t/T) for t=1:2T]
    y = y_base.+rand(d, size(y_base))
    return y
end

function create_ensemble_values(y, N_models, bias_range, std_range, δ_pert, σ_pert, total_drift_additive)
    #uniform distribution of models biases
    T = size(y)[1]
    d = Uniform(-bias_range, bias_range)

    #We just add an intercept term
    N_models = N_models + 1
    #sample distribution of models biases
    biases = rand(d, N_models)

    #uniform distribution of models stds
    s = Uniform(0, std_range)

    #sample distribution of models biases
    variances = rand(s, N_models)

    #Create the features X by taking y and adding noise.
    X = zeros(T, N_models)

    for i=1:N_models
        d = Normal(biases[i], variances[i])
        X[:,i] = y.+ rand(d, T)
    end

    if total_drift_additive
        #### Perturbations on base learners because of data drift
        d = Normal(0, δ_pert)
        biases_perturb = rand(d, N_models)
        d = Uniform(0, σ_pert)
        variances_perturb = rand(d, N_models)
        for i=1:N_models
            d = Normal(biases_perturb[i], variances_perturb[i])
            X[:,i] = X[:,i] .+ [t/T for t=1:T].*rand(d, T)
        end
    end

    #intercept term
    X[:,1] .= 1
    return X
end