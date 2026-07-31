#!/usr/bin/env julia
repo = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo, "src", "AdaptiveEnsemblePublication.jl"))
using .AdaptiveEnsemblePublication
AdaptiveEnsemblePublication.publication_main(vcat(["manuscript-sweeps"], ARGS))
