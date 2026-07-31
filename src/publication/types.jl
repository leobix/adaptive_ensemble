Base.@kwdef struct SplitSpec
    train::UnitRange{Int}
    validation::UnitRange{Int}
    test::UnitRange{Int}
end

Base.@kwdef struct DatasetBundle
    name::String
    X::Matrix{Float64}
    y::Vector{Float64}
    member_names::Vector{String}
    split::SplitSpec
    event_ids::Union{Nothing,Vector{Int}} = nothing
    availability_lag::Int = 1
    units::String = "standardized"
    metadata::Dict{String,Any} = Dict{String,Any}()
end

struct Standardizer
    center::Float64
    scale::Float64
end

"""Reusable sufficient statistics for one `(data, indices, τ)` quadratic fit path."""
Base.@kwdef struct AdaptiveQuadraticWorkspace
    Xtrain::Matrix{Float64}
    Ztrain::Matrix{Float64}
    ytrain::Vector{Float64}
    indices::Vector{Int}
    tau::Int
    availability_lag::Int
    m::Int
    q::Int
    d::Int
    n::Int
    direct::Bool
    A::Union{Nothing,Matrix{Float64}} = nothing
    gram_data::Union{Nothing,Matrix{Float64}} = nothing
    gram_beta::Union{Nothing,Matrix{Float64}} = nothing
    rhs::Vector{Float64}
    diagonal_data::Vector{Float64}
    diagonal_beta::Vector{Float64}
    preparation_seconds::Float64
end

Base.@kwdef mutable struct AdaptiveRidgeModel
    beta0::Vector{Float64}
    V::Matrix{Float64}
    theta::Vector{Float64}
    tau::Int
    lambda::Float64
    penalty::Symbol
    objective_variant::Symbol
    availability_lag::Int
    warmup::Symbol
    objective::Float64
    fit_seconds::Float64
    preparation_seconds::Float64 = 0.0
    solve_seconds::Float64 = 0.0
    linear_solver::String
    condition_number::Float64
    iterations::Int = 0
    converged::Bool = true
    residual_norm::Float64 = 0.0
    stationarity_residual::Float64 = NaN
    relative_objective_change::Float64 = NaN
    convergence_reason::String = ""
    n_parameters::Int = 0
end

Base.@kwdef struct StaticRidgeModel
    weights::Vector{Float64}
    lambda::Float64
    fit_seconds::Float64
end

Base.@kwdef mutable struct NeuralGateModel
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    tau::Int
    availability_lag::Int
    feature_center::Vector{Float64}
    feature_scale::Vector{Float64}
    epochs::Int
    fit_seconds::Float64
    hidden::Int
    learning_rate::Float64
    weight_decay::Float64
end

abstract type AbstractMetaTreeNode end
Base.@kwdef struct MetaLeaf <: AbstractMetaTreeNode
    value::Float64
end
Base.@kwdef struct MetaSplit <: AbstractMetaTreeNode
    feature::Int
    threshold::Float64
    left::AbstractMetaTreeNode
    right::AbstractMetaTreeNode
end
Base.@kwdef struct BoostedMetaModel
    base_value::Float64
    trees::Vector{AbstractMetaTreeNode}
    learning_rate::Float64
    tau::Int
    availability_lag::Int
    feature_center::Vector{Float64}
    feature_scale::Vector{Float64}
    fit_seconds::Float64
    max_depth::Int = 2
    min_leaf::Int = 20
end

Base.@kwdef struct RunPaths
    root::String
    tables::String
    figures::String
    predictions::String
    models::String
    logs::String
    checkpoints::String
    manifests::String
end

const DEFAULT_METHODS = [
    "adaptive_ridge", "adaptive_ridge_quadratic", "adaptive_robust_norm", "static_ridge",
    "rolling_ridge", "rls", "hedge", "exp3", "fixed_share", "passive_aggressive",
    "dma", "neural_gate", "boosted_meta", "ensemble_mean",
    "oracle_best_member", "validation_best_member"
]
