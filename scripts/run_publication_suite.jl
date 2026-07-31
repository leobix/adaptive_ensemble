#!/usr/bin/env julia

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "src", "AdaptiveEnsemblePublication.jl"))
using .AdaptiveEnsemblePublication

AdaptiveEnsemblePublication.publication_main(vcat(["suite"], ARGS))
