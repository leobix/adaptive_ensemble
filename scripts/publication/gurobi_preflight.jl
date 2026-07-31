repo = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(repo, "src", "AdaptiveEnsemblePublication.jl"))
using .AdaptiveEnsemblePublication

println(AdaptiveEnsemblePublication.gurobi_preflight())
