using Statistics, StatsBase, Distributions, Random

#TODO Add drift on y as well!!!!!
function create_y(T, period, bias, σ_pert, seed)
    Random.seed!(seed)
    d = Normal(0, σ_pert)
    y_base = [sin(2*pi*period*t/T) for t=1:2T]
    y = y_base.+rand(d, size(y_base))
    return y
end

#TODO add correlation between the normal variables
#TODO add concept drift in test set only
function create_ensemble_values(y, N_models, bias_range, std_range, δ_pert, σ_pert, total_drift_additive, y_bias_drift, y_std_drift, seed, p_ber = 1)
    Random.seed!(seed)
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
        bernoulli = Binomial(1,p_ber)
        for i=1:N_models
            d = Normal(biases_perturb[i], variances_perturb[i])
            if p_ber < 1
                X[:,i] = X[:,i] .+ rand(d, T).*rand(bernoulli, T)
            else
                X[:,i] = X[:,i] .+ [t/T for t=1:T].*rand(d, T)
            end
        end
    end
    if y_std_drift+y_bias_drift>0
        d_y = Normal(y_bias_drift, y_std_drift)
        y = y .+ [t/T for t=1:T].*rand(d_y, T)
    end
    #intercept term
    X[:,1] .= 1
    return X, y
end



# using Distributions
# mean = [2.,3.]
# C = [0.2 0; 0 0.3]
# d = MvNormal(mean, C)
# x = rand(d, 2000)
