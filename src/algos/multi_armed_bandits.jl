function compute_bandit_weights(X, y)
    # Exp3 algorithm
    #Compute the multi-armed-bandit optimal weights for choosing the expert to follow
    #The sum of the weights is equal to 1
    #Expects the entire data as input and updates at each timestep the experts weights

    n, p = size(X)

    #optimal eta
    η = sqrt(8log(p)/n)

    β = ones(p)/n
    regrets = zeros(p)

    #compute the regrets (losses) of each expert model
    for i = 1:p
        regrets[i] = sum(abs2.(X[:,i]-y))
    end

    #compute the corresponding new weight of each model
    for i = 1:p
        β[i] = exp(-η*regrets[i])/sum(exp(-η*regrets[i]) for i=1:p)
    end

    # return weights
    return β
end
