module GradientBoosting

using ..DecisionTreeEnsembler

mutable struct GBRTModel
    trees::Vector{DecisionTreeEnsembler.TreeNode}
    b0::Float64
    lr::Float64
end

function fit_gbrt(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                  n_estimators::Int=30, learning_rate::Float64=0.1,
                  max_depth::Int=2, min_leaf::Int=5, num_thresholds::Int=25)
    n, p = size(X)
    b0 = mean(y)
    F = fill(b0, n)
    trees = DecisionTreeEnsembler.TreeNode[]
    for t in 1:n_estimators
        r = y .- F
        tree = DecisionTreeEnsembler.fit_decision_tree(X, r; max_depth=max_depth, min_leaf=min_leaf, num_thresholds=num_thresholds)
        push!(trees, tree)
        # Update stage predictions
        for i in 1:n
            F[i] += learning_rate * DecisionTreeEnsembler.predict_decision_tree(tree, view(X, i, :))
        end
    end
    return GBRTModel(trees, b0, learning_rate)
end

function predict(model::GBRTModel, x::AbstractVector{<:Real})
    yhat = model.b0
    for tr in model.trees
        yhat += model.lr * DecisionTreeEnsembler.predict_decision_tree(tr, x)
    end
    return yhat
end

function predict(model::GBRTModel, X::AbstractMatrix{<:Real})
    n = size(X, 1)
    yhat = fill(model.b0, n)
    for tr in model.trees
        for i in 1:n
            yhat[i] += model.lr * DecisionTreeEnsembler.predict_decision_tree(tr, view(X, i, :))
        end
    end
    return yhat
end

end # module

