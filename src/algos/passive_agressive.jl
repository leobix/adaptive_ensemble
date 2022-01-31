#=
passive_agressive:
- Julia version: 
- Author: leonardboussioux
- Date: 2022-01-31
=#

function compute_PA_weights(ϵ, w_t, X_t, y_t)

    #Crammer et al 2006
    p = size(X_t)

    #loss incurred last time step
    l_t = max(abs(transpose(w_t)*X_t - y_t)-ϵ, 0)
    τ_t = l_t / sum(abs2.(X_t))

    #updating the weights
    w_new = w_t + sign(y_t-transpose(w_t)*X_t)*τ_t*X_t
    regrets = zeros(p)

    return w_new
end

