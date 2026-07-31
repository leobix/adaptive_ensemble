# Julia-native comparison methods used by the publication suite. All sequential
# methods predict first and update only when the corresponding outcome is available.

function _stable_softmax(v::AbstractVector{<:Real})
    z = Float64.(v)
    z .-= maximum(z)
    e = exp.(clamp.(z, -700.0, 0.0))
    s = sum(e)
    return isfinite(s) && s > 0 ? e ./ s : fill(1.0 / length(z), length(z))
end

ensemble_mean_predictions(X::AbstractMatrix, idx::AbstractVector{Int}) =
    vec(mean(@view(X[idx, :]); dims=2))

function _member_selection_score(metric::Symbol, y::AbstractVector, prediction::AbstractVector)
    metric == :rmse && return rmse(y, prediction)
    metric == :mae && return mae(y, prediction)
    metric == :cvar_05 && return empirical_cvar(abs.(y .- prediction), 0.05)
    throw(ArgumentError("member-selection metric must be rmse, mae, or cvar_05"))
end

function oracle_best_member_predictions(X::AbstractMatrix, y::AbstractVector,
                                        idx::AbstractVector{Int}; metric::Symbol=:rmse)
    isempty(idx) && throw(ArgumentError("member selection requires at least one observation"))
    scores = Float64[]
    truth = @view y[idx]
    for j in axes(X, 2)
        prediction = @view X[idx, j]
        push!(scores, _member_selection_score(metric, truth, prediction))
    end
    j = argmin(scores)
    return Vector{Float64}(@view X[idx, j]), j, scores[j]
end

function _new_event(event_ids, t::Int, previous_t::Int)
    event_ids === nothing && return false
    previous_t < 1 && return false
    return event_ids[t] != event_ids[previous_t]
end

function _eligible_updates(last_updated::Int, t::Int, availability_lag::Int)
    stop = t - availability_lag
    stop <= last_updated && return 1:0
    return (last_updated + 1):stop
end

function _available_initial_indices(init_idx::AbstractVector{Int},
                                    pred_idx::AbstractVector{Int},
                                    availability_lag::Int)
    isempty(init_idx) && return Int[]
    isempty(pred_idx) && return collect(init_idx)
    cutoff = first(pred_idx) - availability_lag
    return [j for j in init_idx if j <= cutoff]
end

function _training_loss_scale(X::AbstractMatrix, y::AbstractVector,
                              idx::AbstractVector{Int}; probability::Float64=0.95)
    isempty(idx) && return 1.0
    errors = vec(Float64.(X[idx, :]) .- reshape(Float64.(y[idx]), :, 1))
    losses = errors .^ 2
    return max(quantile(losses, probability), 1e-6)
end

"""Full-information exponentially weighted forecaster (Hedge), not bandit Exp3."""
function hedge_predictions(X::AbstractMatrix, y::AbstractVector,
                           init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                           eta::Float64=0.5, window::Int=0,
                           availability_lag::Int=1,
                           event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                           reset_on_event::Bool=false)
    validate_xy(X, y)
    eta >= 0 || throw(ArgumentError("eta must be nonnegative"))
    window >= 0 || throw(ArgumentError("window must be nonnegative"))
    m = size(X, 2)
    losses = zeros(Float64, m)
    queues = [Float64[] for _ in 1:m]
    update_count = 0

    function update!(j::Int)
        ell = (Float64.(X[j, :]) .- y[j]).^2
        losses .+= ell
        update_count += 1
        if window > 0
            for k in 1:m
                push!(queues[k], ell[k])
                if length(queues[k]) > window
                    losses[k] -= popfirst!(queues[k])
                end
            end
        end
        return nothing
    end

    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    foreach(update!, eligible_init)
    global_losses = copy(losses)
    global_queues = [copy(queue) for queue in queues]
    global_count = update_count

    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    last_updated = isempty(eligible_init) ? 0 : maximum(eligible_init)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            losses .= global_losses
            for k in 1:m
                empty!(queues[k])
                append!(queues[k], global_queues[k])
            end
            update_count = global_count
            last_updated = t - 1
        end
        for j in _eligible_updates(last_updated, t, availability_lag)
            if event_ids === nothing || event_ids[j] == event_ids[t]
                update!(j)
            end
            last_updated = j
        end
        effective_count = window > 0 ? min(update_count, window) : update_count
        scale = max(effective_count, 1)
        w = _stable_softmax((-eta / scale) .* losses)
        @views weights[r, :] .= w
        preds[r] = dot(w, @view X[t, :])
        previous_t = t
    end
    return preds, weights
end

"""
True bandit-feedback Exp3. At each forecast origin the algorithm samples one member from
its exploration mixture and predicts with that member only. Feedback is delayed according
to `availability_lag`; the historical initialization is replayed with the same causal rule.
Only the sampled member's clipped squared loss enters the importance-weighted update.
"""
function exp3_predictions(X::AbstractMatrix, y::AbstractVector,
                          init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                          gamma::Float64=0.10,
                          availability_lag::Int=1,
                          event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                          reset_on_event::Bool=false,
                          seed::Int=2026,
                          loss_scale::Union{Nothing,Float64}=nothing)
    validate_xy(X, y)
    0 < gamma <= 1 || throw(ArgumentError("gamma must be in (0,1]"))
    availability_lag >= 1 || throw(ArgumentError("availability_lag must be positive"))
    m = size(X, 2)
    n = size(X, 1)
    rng = MersenneTwister(seed)
    weights_state = ones(Float64, m)
    chosen_arms = zeros(Int, n)
    chosen_probabilities = zeros(Float64, n)
    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    scale = loss_scale === nothing ? _training_loss_scale(X, y, eligible_init) : Float64(loss_scale)
    scale > 0 || throw(ArgumentError("loss_scale must be positive"))

    probabilities() = (1.0 - gamma) .* (weights_state ./ sum(weights_state)) .+ gamma / m

    function draw_arm!(t::Int)
        p = probabilities()
        arm = rand(rng, Categorical(p))
        chosen_arms[t] = arm
        chosen_probabilities[t] = p[arm]
        return arm, p
    end

    function update!(t::Int)
        arm = chosen_arms[t]
        arm == 0 && return nothing
        sampled_probability = chosen_probabilities[t]
        sampled_probability > 0 || error(
            "Exp3 update missing the sampling probability at time $(t)")
        loss = min((Float64(X[t, arm]) - Float64(y[t]))^2 / scale, 1.0)
        estimated_loss = loss / max(sampled_probability, eps(Float64))
        weights_state[arm] *= exp(-gamma * estimated_loss / m)
        total = sum(weights_state)
        if !(isfinite(total) && total > 0)
            fill!(weights_state, 1.0)
        elseif maximum(weights_state) < 1e-200 || maximum(weights_state) > 1e200
            weights_state ./= maximum(weights_state)
        end
        return nothing
    end

    # Replay training origins without seeing outcomes before they are verifiable.
    last_updated = 0
    for t in init_idx
        for j in _eligible_updates(last_updated, t, availability_lag)
            update!(j)
            last_updated = j
        end
        draw_arm!(t)
    end
    if !isempty(pred_idx)
        for j in _eligible_updates(last_updated, first(pred_idx), availability_lag)
            update!(j)
            last_updated = j
        end
    end
    global_weights = copy(weights_state)

    preds = zeros(Float64, length(pred_idx))
    probability_rows = Matrix{Float64}(undef, length(pred_idx), m)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            weights_state .= global_weights
            last_updated = t - 1
        elseif r > 1
            for j in _eligible_updates(last_updated, t, availability_lag)
                if event_ids === nothing || event_ids[j] == event_ids[t]
                    update!(j)
                end
                last_updated = j
            end
        end
        arm, p = draw_arm!(t)
        @views probability_rows[r, :] .= p
        preds[r] = Float64(X[t, arm])
        previous_t = t
    end
    return preds, probability_rows
end

"""Fixed Share: Hedge plus a uniform share step to track a changing best expert."""
function fixed_share_predictions(X::AbstractMatrix, y::AbstractVector,
                                 init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                                 eta::Float64=0.5, share::Float64=0.02,
                                 availability_lag::Int=1,
                                 event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                 reset_on_event::Bool=false)
    0 <= share < 1 || throw(ArgumentError("share must be in [0,1)"))
    eta >= 0 || throw(ArgumentError("eta must be nonnegative"))
    m = size(X, 2)
    w = fill(1.0 / m, m)
    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    scale = _training_loss_scale(X, y, eligible_init)

    function update!(j::Int)
        ell = clamp.((Float64.(X[j, :]) .- y[j]).^2 ./ scale, 0.0, 1.0)
        w .*= exp.(-eta .* ell)
        total = sum(w)
        w .= (isfinite(total) && total > 0) ? w ./ total : fill(1.0 / m, m)
        w .= (1 - share) .* w .+ share / m
        return nothing
    end

    foreach(update!, eligible_init)
    global_w = copy(w)
    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    last_updated = isempty(eligible_init) ? 0 : maximum(eligible_init)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            w .= global_w
            last_updated = t - 1
        end
        for j in _eligible_updates(last_updated, t, availability_lag)
            if event_ids === nothing || event_ids[j] == event_ids[t]
                update!(j)
            end
            last_updated = j
        end
        @views weights[r, :] .= w
        preds[r] = dot(w, @view X[t, :])
        previous_t = t
    end
    return preds, weights
end

"""
Passive-Aggressive regression (PA-I). The policy starts from equal ensemble weights and
replays only outcomes available before the first requested prediction. This avoids the
historical code's undocumented Static-Ridge initialization and preserves predict-then-update
semantics under delayed feedback.
"""
function passive_aggressive_predictions(X::AbstractMatrix, y::AbstractVector,
                                        init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                                        C::Float64=1.0, epsilon::Float64=0.0,
                                        availability_lag::Int=1,
                                        event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                                        reset_on_event::Bool=false)
    C > 0 || throw(ArgumentError("C must be positive"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    m = size(X, 2)
    w = fill(1.0 / m, m)

    function update!(j::Int)
        x = @view X[j, :]
        error_value = y[j] - dot(w, x)
        loss = max(0.0, abs(error_value) - epsilon)
        loss == 0 && return nothing
        denominator = dot(x, x)
        denominator <= eps(Float64) && return nothing
        step = min(C, loss / denominator)
        w .+= sign(error_value) * step .* x
        return nothing
    end

    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    foreach(update!, eligible_init)
    global_w = copy(w)
    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    last_updated = isempty(eligible_init) ? 0 : maximum(eligible_init)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            w .= global_w
            last_updated = t - 1
        end
        for j in _eligible_updates(last_updated, t, availability_lag)
            if event_ids === nothing || event_ids[j] == event_ids[t]
                update!(j)
            end
            last_updated = j
        end
        preds[r] = dot(w, @view X[t, :])
        @views weights[r, :] .= w
        previous_t = t
    end
    return preds, weights
end

function rls_predictions(X::AbstractMatrix, y::AbstractVector,
                         init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                         forgetting::Float64=0.99, ridge::Float64=1e-3,
                         availability_lag::Int=1,
                         event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                         reset_on_event::Bool=false)
    0 < forgetting <= 1 || throw(ArgumentError("forgetting must be in (0,1]"))
    ridge > 0 || throw(ArgumentError("ridge must be positive"))
    m = size(X, 2)
    w = zeros(Float64, m)
    P = Matrix{Float64}(I, m, m) / ridge

    function update!(j::Int)
        x = Vector{Float64}(@view X[j, :])
        Px = P * x
        denominator = forgetting + dot(x, Px)
        denominator > eps(Float64) || return nothing
        gain = Px / denominator
        w .+= gain * (y[j] - dot(x, w))
        P .= (P .- gain * transpose(x) * P) ./ forgetting
        P .= 0.5 .* (P .+ transpose(P))
        return nothing
    end

    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    foreach(update!, eligible_init)
    global_w = copy(w)
    global_P = copy(P)
    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    last_updated = isempty(eligible_init) ? 0 : maximum(eligible_init)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            w .= global_w
            P .= global_P
            last_updated = t - 1
        end
        for j in _eligible_updates(last_updated, t, availability_lag)
            if event_ids === nothing || event_ids[j] == event_ids[t]
                update!(j)
            end
            last_updated = j
        end
        preds[r] = dot(w, @view X[t, :])
        @views weights[r, :] .= w
        previous_t = t
    end
    return preds, weights
end

"""
Rolling Ridge using rank-one sufficient-statistic updates. This produces the same model as
refitting Ridge on the most recent eligible `window` observations, while avoiding a complete
matrix rebuild at every forecast origin.
"""
function rolling_ridge_predictions(X::AbstractMatrix, y::AbstractVector,
                                   init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                                   window::Int=168, lambda::Float64=1e-3,
                                   availability_lag::Int=1,
                                   event_ids::Union{Nothing,AbstractVector{Int}}=nothing)
    window >= 2 || throw(ArgumentError("window must be at least 2"))
    lambda >= 0 || throw(ArgumentError("lambda must be nonnegative"))
    isempty(init_idx) && throw(ArgumentError("rolling ridge requires an initialization set"))
    m = size(X, 2)
    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    isempty(eligible_init) && throw(ArgumentError(
        "rolling ridge has no outcomes available before the first prediction"))
    global_model = fit_static_ridge(X, y, eligible_init, lambda)
    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    queue = Int[]
    gram = zeros(Float64, m, m)
    rhs = zeros(Float64, m)
    last_added = 0
    previous_t = 0

    function add_observation!(j::Int)
        x = Vector{Float64}(@view X[j, :])
        gram .+= x * transpose(x)
        rhs .+= x .* y[j]
        push!(queue, j)
        return nothing
    end

    function remove_oldest!()
        j = popfirst!(queue)
        x = Vector{Float64}(@view X[j, :])
        gram .-= x * transpose(x)
        rhs .-= x .* y[j]
        return nothing
    end

    function rebuild!(t::Int)
        empty!(queue)
        fill!(gram, 0.0)
        fill!(rhs, 0.0)
        stop = t - availability_lag
        if stop >= 1
            start = max(1, stop - window + 1)
            for j in start:stop
                if event_ids === nothing || event_ids[j] == event_ids[t]
                    add_observation!(j)
                end
            end
        end
        last_added = max(stop, 0)
        return nothing
    end

    for (r, t) in enumerate(pred_idx)
        if r == 1 || _new_event(event_ids, t, previous_t)
            rebuild!(t)
        else
            stop = t - availability_lag
            if stop > last_added
                for j in (last_added + 1):stop
                    if event_ids === nothing || event_ids[j] == event_ids[t]
                        add_observation!(j)
                    end
                end
                last_added = stop
            end
            while length(queue) > window
                remove_oldest!()
            end
        end
        current_weights = global_model.weights
        if length(queue) >= max(2, m)
            count = length(queue)
            system = gram ./ count + lambda * Matrix{Float64}(I, m, m)
            target = rhs ./ count
            current_weights, _ = _stable_linear_solve(system, target)
        end
        @views weights[r, :] .= current_weights
        preds[r] = dot(current_weights, @view X[t, :])
        previous_t = t
    end
    return preds, weights
end

"""Dynamic model averaging over ensemble members with discounted probabilities."""
function dma_predictions(X::AbstractMatrix, y::AbstractVector,
                         init_idx::AbstractVector{Int}, pred_idx::AbstractVector{Int};
                         forgetting::Float64=0.99, variance_forgetting::Float64=0.97,
                         availability_lag::Int=1,
                         event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                         reset_on_event::Bool=false)
    0 < forgetting <= 1 || throw(ArgumentError("forgetting must be in (0,1]"))
    0 < variance_forgetting <= 1 || throw(ArgumentError("variance_forgetting must be in (0,1]"))
    m = size(X, 2)
    probabilities = fill(1.0 / m, m)
    variances = ones(Float64, m)
    eligible_init = _available_initial_indices(init_idx, pred_idx, availability_lag)
    if !isempty(eligible_init)
        errors = (Matrix(X[eligible_init, :]) .- Float64.(y[eligible_init])).^2
        variances .= vec(mean(errors; dims=1)) .+ 1e-8
        log_probability = -0.5 .* log.(variances) .-
            0.5 .* vec(sum(errors; dims=1)) ./ (length(eligible_init) .* variances)
        probabilities .= _stable_softmax(log_probability)
    end

    function update!(j::Int)
        error_value = Float64.(X[j, :]) .- y[j]
        log_likelihood = -0.5 .* (log.(2pi .* variances) .+ (error_value.^2) ./ variances)
        prior = probabilities .^ forgetting
        prior ./= max(sum(prior), 1e-300)
        probabilities .= _stable_softmax(log.(max.(prior, 1e-300)) .+ log_likelihood)
        variances .= variance_forgetting .* variances .+
            (1 - variance_forgetting) .* (error_value.^2)
        variances .= max.(variances, 1e-10)
        return nothing
    end

    global_probabilities = copy(probabilities)
    global_variances = copy(variances)
    preds = zeros(Float64, length(pred_idx))
    weights = Matrix{Float64}(undef, length(pred_idx), m)
    last_updated = isempty(eligible_init) ? 0 : maximum(eligible_init)
    previous_t = isempty(init_idx) ? 0 : maximum(init_idx)
    for (r, t) in enumerate(pred_idx)
        if reset_on_event && _new_event(event_ids, t, previous_t)
            probabilities .= global_probabilities
            variances .= global_variances
            last_updated = t - 1
        end
        for j in _eligible_updates(last_updated, t, availability_lag)
            if event_ids === nothing || event_ids[j] == event_ids[t]
                update!(j)
            end
            last_updated = j
        end
        prior = probabilities .^ forgetting
        prior ./= max(sum(prior), 1e-300)
        @views weights[r, :] .= prior
        preds[r] = dot(prior, @view X[t, :])
        previous_t = t
    end
    return preds, weights
end

function _feature_standardizer(F::AbstractMatrix, idx::AbstractVector{Int})
    c=vec(mean(@view(F[idx,:]);dims=1)); s=vec(std(@view(F[idx,:]);dims=1))
    s[.!isfinite.(s) .| (s .< 1e-8)] .= 1.0
    return c,s
end

_standardize_features(F,c,s) = (Float64.(F) .- transpose(c)) ./ transpose(s)

function fit_neural_gate(X::AbstractMatrix,y::AbstractVector,idx::AbstractVector{Int},tau::Int;
                         availability_lag::Int=1,
                         event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                         hidden::Int=16,epochs::Int=250,batch_size::Int=128,
                         learning_rate::Float64=2e-3,weight_decay::Float64=1e-4,
                         seed::Int=2026)
    rng=MersenneTwister(seed); t0=time_ns(); m=size(X,2)
    Z,_=build_error_histories(X,y,tau;availability_lag=availability_lag,event_ids=event_ids)
    c,s=_feature_standardizer(Z,idx); F=_standardize_features(Z,c,s); q=size(F,2)
    W1=0.05.*randn(rng,hidden,q); b1=zeros(hidden); W2=0.05.*randn(rng,m,hidden); b2=zeros(m)
    params=(W1,b1,W2,b2); mt=map(x->zeros(size(x)),params); vt=map(x->zeros(size(x)),params)
    step=0; order=collect(idx)
    for _ in 1:epochs
        shuffle!(rng,order)
        for start in 1:batch_size:length(order)
            batch=order[start:min(start+batch_size-1,length(order))]; nb=length(batch)
            gW1=zeros(size(W1)); gb1=zeros(size(b1)); gW2=zeros(size(W2)); gb2=zeros(size(b2))
            for t in batch
                f=@view F[t,:]; x=@view X[t,:]
                a=W1*f+b1; h=tanh.(a); logits=W2*h+b2; w=_stable_softmax(logits)
                pred=dot(w,x); dp=2.0*(pred-y[t])/nb
                gl=dp .* w .* (Float64.(x).-pred)
                gW2 .+= gl*transpose(h); gb2 .+= gl
                gh=transpose(W2)*gl; ga=gh.*(1 .- h.^2)
                gW1 .+= ga*transpose(f); gb1 .+= ga
            end
            gW1 .+= 2*weight_decay.*W1; gW2 .+= 2*weight_decay.*W2
            grads=(gW1,gb1,gW2,gb2); step+=1
            for k in eachindex(params)
                mt[k] .= 0.9.*mt[k] .+ 0.1.*grads[k]
                vt[k] .= 0.999.*vt[k] .+ 0.001.*(grads[k].^2)
                mhat=mt[k]./(1-0.9^step); vhat=vt[k]./(1-0.999^step)
                params[k] .-= learning_rate.*mhat./(sqrt.(vhat).+1e-8)
            end
        end
    end
    return NeuralGateModel(W1=W1,b1=b1,W2=W2,b2=b2,tau=tau,availability_lag=availability_lag,
        feature_center=c,feature_scale=s,epochs=epochs,fit_seconds=(time_ns()-t0)/1e9,
        hidden=hidden,learning_rate=learning_rate,weight_decay=weight_decay)
end

function predict_neural_gate(model::NeuralGateModel,X::AbstractMatrix,y::AbstractVector,
                             idx::AbstractVector{Int};event_ids::Union{Nothing,AbstractVector{Int}}=nothing)
    Z,_=build_error_histories(X,y,model.tau;availability_lag=model.availability_lag,event_ids=event_ids)
    F=_standardize_features(Z,model.feature_center,model.feature_scale); m=size(X,2)
    preds=zeros(Float64,length(idx)); weights=Matrix{Float64}(undef,length(idx),m)
    for (r,t) in enumerate(idx)
        h=tanh.(model.W1*@view(F[t,:])+model.b1); w=_stable_softmax(model.W2*h+model.b2)
        preds[r]=dot(w,@view X[t,:]); @views weights[r,:].=w
    end
    return preds,weights
end

_tree_predict(node::MetaLeaf,x) = node.value
function _tree_predict(node::MetaSplit,x)
    return x[node.feature] <= node.threshold ? _tree_predict(node.left,x) : _tree_predict(node.right,x)
end

function _best_tree_split(F::AbstractMatrix,r::AbstractVector,idx::Vector{Int},min_leaf::Int,n_thresholds::Int)
    rv = @view r[idx]
    mu = mean(rv)
    base = sum(abs2, rv .- mu)
    best_gain = 0.0
    best = (0, 0.0, Int[], Int[])
    for j in axes(F,2)
        vals=Float64.(@view F[idx,j]); thresholds=unique([quantile(vals,q) for q in range(0.1,0.9;length=n_thresholds)])
        for th in thresholds
            left=Int[];right=Int[]
            for i in idx
                (F[i,j] <= th ? push!(left,i) : push!(right,i))
            end
            (length(left)<min_leaf || length(right)<min_leaf) && continue
            rleft = @view r[left]
            rright = @view r[right]
            ml = mean(rleft)
            mr = mean(rright)
            sse = sum(abs2, rleft .- ml) + sum(abs2, rright .- mr)
            gain=base-sse
            gain>best_gain && (best_gain=gain;best=(j,Float64(th),left,right))
        end
    end
    return best_gain,best
end

function _fit_meta_tree(F::AbstractMatrix,r::AbstractVector,idx::Vector{Int},depth::Int,min_leaf::Int,n_thresholds::Int)
    depth<=0 && return MetaLeaf(value=mean(@view r[idx]))
    gain,(j,th,left,right)=_best_tree_split(F,r,idx,min_leaf,n_thresholds)
    (gain<=1e-12 || j==0) && return MetaLeaf(value=mean(@view r[idx]))
    return MetaSplit(feature=j,threshold=th,
        left=_fit_meta_tree(F,r,left,depth-1,min_leaf,n_thresholds),
        right=_fit_meta_tree(F,r,right,depth-1,min_leaf,n_thresholds))
end

function fit_boosted_meta(X::AbstractMatrix,y::AbstractVector,idx::AbstractVector{Int},tau::Int;
                          availability_lag::Int=1,
                          event_ids::Union{Nothing,AbstractVector{Int}}=nothing,
                          n_trees::Int=150,max_depth::Int=2,min_leaf::Int=20,
                          learning_rate::Float64=0.05,n_thresholds::Int=15)
    t0=time_ns(); Z,_=build_error_histories(X,y,tau;availability_lag=availability_lag,event_ids=event_ids)
    Fraw=hcat(Float64.(X),Z); c,s=_feature_standardizer(Fraw,idx); F=_standardize_features(Fraw,c,s)
    base=mean(@view y[idx]); fitted=fill(base,length(y)); trees=AbstractMetaTreeNode[]; ids=collect(idx)
    for _ in 1:n_trees
        residual=Float64.(y).-fitted
        tree=_fit_meta_tree(F,residual,ids,max_depth,min_leaf,n_thresholds)
        push!(trees,tree)
        for i in idx
            fitted[i]+=learning_rate*_tree_predict(tree,@view F[i,:])
        end
    end
    return BoostedMetaModel(base_value=base,trees=trees,learning_rate=learning_rate,tau=tau,
        availability_lag=availability_lag,feature_center=c,feature_scale=s,
        fit_seconds=(time_ns()-t0)/1e9,max_depth=max_depth,min_leaf=min_leaf)
end

function predict_boosted_meta(model::BoostedMetaModel,X::AbstractMatrix,y::AbstractVector,
                              idx::AbstractVector{Int};event_ids::Union{Nothing,AbstractVector{Int}}=nothing)
    Z,_=build_error_histories(X,y,model.tau;availability_lag=model.availability_lag,event_ids=event_ids)
    F=_standardize_features(hcat(Float64.(X),Z),model.feature_center,model.feature_scale)
    preds=fill(model.base_value,length(idx))
    for (r,t) in enumerate(idx), tree in model.trees
        preds[r]+=model.learning_rate*_tree_predict(tree,@view F[t,:])
    end
    return preds
end
