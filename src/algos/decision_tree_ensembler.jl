module DecisionTreeEnsembler

using Statistics

mutable struct TreeNode
    is_leaf::Bool
    prediction::Float64
    feature::Int
    threshold::Float64
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
end

TreeNode(pred::Float64) = TreeNode(true, pred, 0, 0.0, nothing, nothing)

function _best_split(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; min_leaf::Int=5, num_thresholds::Int=25)
    n, p = size(X)
    best_feat, best_thr = 0, 0.0
    best_mse = Inf
    best_left, best_right = nothing, nothing
    if n < 2 * min_leaf
        return best_feat, best_thr, best_mse, best_left, best_right
    end
    for j in 1:p
        xj = X[:, j]
        xmin, xmax = extrema(xj)
        if xmin == xmax
            continue
        end
        # generate candidate thresholds uniformly between min and max
        for k in 1:num_thresholds
            thr = xmin + (k/(num_thresholds+1)) * (xmax - xmin)
            left_idx = findall(<=(thr), xj)
            right_idx = findall(>(thr), xj)
            if length(left_idx) < min_leaf || length(right_idx) < min_leaf
                continue
            end
            yl = y[left_idx]
            yr = y[right_idx]
            μl = mean(yl); μr = mean(yr)
            mse_l = sum((yl .- μl).^2)
            mse_r = sum((yr .- μr).^2)
            mse = mse_l + mse_r
            if mse < best_mse
                best_mse = mse
                best_feat = j
                best_thr = thr
                best_left = left_idx
                best_right = right_idx
            end
        end
    end
    return best_feat, best_thr, best_mse, best_left, best_right
end

function _fit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; depth::Int, max_depth::Int, min_leaf::Int, num_thresholds::Int)
    n = size(X, 1)
    # stopping conditions
    if depth >= max_depth || n < 2 * min_leaf
        return TreeNode(mean(y))
    end

    feat, thr, mse, left_idx, right_idx = _best_split(X, y; min_leaf=min_leaf, num_thresholds=num_thresholds)
    if feat == 0 || left_idx === nothing || right_idx === nothing
        return TreeNode(mean(y))
    end

    left_node = _fit(X[left_idx, :], y[left_idx]; depth=depth+1, max_depth=max_depth, min_leaf=min_leaf, num_thresholds=num_thresholds)
    right_node = _fit(X[right_idx, :], y[right_idx]; depth=depth+1, max_depth=max_depth, min_leaf=min_leaf, num_thresholds=num_thresholds)

    return TreeNode(false, 0.0, feat, thr, left_node, right_node)
end

function fit_decision_tree(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; max_depth::Int=2, min_leaf::Int=5, num_thresholds::Int=25)
    return _fit(X, y; depth=0, max_depth=max_depth, min_leaf=min_leaf, num_thresholds=num_thresholds)
end

function predict_decision_tree(node::TreeNode, x::AbstractVector{<:Real})
    curr = node
    while !curr.is_leaf
        if x[curr.feature] <= curr.threshold
            curr = curr.left::TreeNode
        else
            curr = curr.right::TreeNode
        end
    end
    return curr.prediction
end

end # module

