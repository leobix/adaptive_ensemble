const SYNTHETIC_REGIMES = [
    "stationary", "gradual", "abrupt", "periodic", "persistent_bias",
    "member_degradation", "recurring", "variance_burst", "discrete_gaussian"
]

function _correlated_noise(rng::AbstractRNG, n::Int, m::Int, rho::Float64)
    0.0 <= rho < 1.0 || throw(ArgumentError("rho must be in [0,1)"))
    common = randn(rng, n)
    idiosyncratic = randn(rng, n, m)
    return sqrt(rho) .* reshape(common, n, 1) .+ sqrt(1.0 - rho) .* idiosyncratic
end

function _bounded_interval(a::Int, b::Int, n::Int)
    lo = clamp(a, 1, n)
    hi = clamp(b, lo, n)
    return lo:hi
end

"""
Generate one publication-scale synthetic data set. The default target and the `gradual`
regime reproduce the manuscript design: 4,000 observations, a period-500 sine target,
Gaussian target noise with standard deviation 0.1, member biases sampled uniformly from
[-0.5,0.5], member error standard deviations sampled uniformly from [0,0.5], and a
linearly amplified Gaussian drift term. The other named regimes implement the reviewer-
requested abrupt, periodic, persistent-bias, member-specific, recurring, and burst changes.
"""
function generate_synthetic_regime(regime::AbstractString, seed::Int;
                                   n::Int=4000, m::Int=10, period::Int=500,
                                   target_noise::Float64=0.1,
                                   rho::Float64=0.0,
                                   train_end::Int=2000, val_end::Int=3000,
                                   sigma_drift::Float64=0.5,
                                   s_drift::Float64=0.5,
                                   p_drift::Float64=0.5)
    regime in SYNTHETIC_REGIMES || throw(ArgumentError("unknown regime $(regime)"))
    n >= 100 || throw(ArgumentError("n is too small"))
    m >= 2 || throw(ArgumentError("m must be at least 2"))
    period >= 2 || throw(ArgumentError("period must be at least 2"))
    1 <= train_end < val_end < n || throw(ArgumentError("invalid split endpoints"))

    rng = MersenneTwister(seed)
    time = collect(1:n)
    signal = sin.(2pi .* time ./ period)
    y = signal .+ target_noise .* randn(rng, n)

    member_bias = rand(rng, m) .- 0.5
    member_sigma = 0.5 .* rand(rng, m)
    base_noise = _correlated_noise(rng, n, m, rho)
    errors = transpose(member_bias) .+ base_noise .* transpose(member_sigma)

    metadata = Dict{String,Any}(
        "regime" => string(regime),
        "seed" => seed,
        "n" => n,
        "m" => m,
        "rho" => rho,
        "target_period" => period,
        "target_noise" => target_noise,
        "member_bias" => member_bias,
        "member_sigma" => member_sigma,
        "sigma_drift" => sigma_drift,
        "s_drift" => s_drift,
        "p_drift" => p_drift,
        "change_points" => Int[],
    )

    if regime == "gradual"
        # Manuscript-faithful linearly amplified drift.
        drift_bias = sigma_drift .* randn(rng, m)
        drift_sigma = s_drift .* rand(rng, m)
        for i in 1:n
            scale = i / n
            @views errors[i, :] .+= scale .* (drift_bias .+ drift_sigma .* randn(rng, m))
        end
        metadata["drift_bias"] = drift_bias
        metadata["drift_sigma"] = drift_sigma
    elseif regime == "abrupt"
        cp = val_end + 1
        quality = abs.(member_bias) .+ member_sigma
        previously_good = sortperm(quality)[1:max(1, div(m, 3))]
        previously_bad = sortperm(quality; rev=true)[1:max(1, div(m, 3))]
        shift = 0.8 .* sign.(randn(rng, length(previously_good)))
        @views errors[cp:end, previously_good] .+= transpose(shift)
        @views errors[cp:end, previously_good] .+= 0.45 .* randn(rng, n - cp + 1, length(previously_good))
        @views errors[cp:end, previously_bad] .*= 0.35
        metadata["change_points"] = [cp]
        metadata["degraded_members"] = previously_good
        metadata["improved_members"] = previously_bad
    elseif regime == "periodic"
        drift_period = max(48, round(Int, 0.06n))
        amplitudes = 0.55 .* randn(rng, m)
        phases = 2pi .* rand(rng, m)
        for i in 1:n, j in 1:m
            errors[i, j] += amplitudes[j] * sin(2pi * i / drift_period + phases[j])
        end
        metadata["drift_period"] = drift_period
        metadata["periodic_amplitudes"] = amplitudes
    elseif regime == "persistent_bias"
        cp = val_end + 1
        affected = sortperm(abs.(member_bias) .+ member_sigma)[1:max(1, div(m, 2))]
        shifts = 0.65 .* sign.(randn(rng, length(affected)))
        @views errors[cp:end, affected] .+= transpose(shifts)
        metadata["change_points"] = [cp]
        metadata["affected_members"] = affected
        metadata["bias_shifts"] = shifts
    elseif regime == "member_degradation"
        test_length = n - val_end
        cp = val_end + max(1, round(Int, 0.20test_length))
        affected = sortperm(abs.(member_bias) .+ member_sigma)[1:max(1, div(m, 3))]
        severity = 0.8 .+ 0.3 .* rand(rng, length(affected))
        for i in cp:n
            ramp = (i - cp) / max(n - cp, 1)
            @views errors[i, affected] .+= ramp .* severity
            @views errors[i, affected] .+= (0.10 + 0.80ramp) .* randn(rng, length(affected))
        end
        metadata["change_points"] = [cp]
        metadata["affected_members"] = affected
    elseif regime == "recurring"
        block = max(50, round(Int, 0.05n))
        group_a = collect(1:cld(m, 2))
        group_b = setdiff(collect(1:m), group_a)
        changes = Int[]
        for i in 1:n
            state_a = isodd(div(i - 1, block) + 1)
            if state_a
                @views errors[i, group_a] .*= 0.25
                !isempty(group_b) && (@views errors[i, group_b] .*= 1.70)
            else
                @views errors[i, group_a] .*= 1.70
                !isempty(group_b) && (@views errors[i, group_b] .*= 0.25)
            end
            i > 1 && mod(i - 1, block) == 0 && push!(changes, i)
        end
        metadata["block_length"] = block
        metadata["change_points"] = changes
        metadata["group_a"] = group_a
        metadata["group_b"] = group_b
    elseif regime == "discrete_gaussian"
        0.0 <= p_drift <= 1.0 || throw(ArgumentError("p_drift must be in [0,1]"))
        drift_bias = sigma_drift .* randn(rng, m)
        drift_sigma = s_drift .* rand(rng, m)
        active = rand(rng, n) .< p_drift
        for i in 1:n
            if active[i]
                @views errors[i, :] .+= drift_bias .+ drift_sigma .* randn(rng, m)
            end
        end
        metadata["active_drift_count"] = count(active)
        metadata["drift_bias"] = drift_bias
        metadata["drift_sigma"] = drift_sigma
        metadata["change_points"] = findall(active[2:end] .!= active[1:(end - 1)]) .+ 1
    elseif regime == "variance_burst"
        affected = collect(1:max(1, div(m, 2)))
        test_length = n - val_end
        centers = [val_end + round(Int, f * test_length) for f in (0.20, 0.50, 0.80)]
        halfwidths = [max(10, round(Int, f * test_length)) for f in (0.03, 0.04, 0.03)]
        bursts = UnitRange{Int}[]
        for (center, halfwidth) in zip(centers, halfwidths)
            interval = _bounded_interval(center - halfwidth, center + halfwidth, n)
            push!(bursts, interval)
            @views errors[interval, affected] .+= 1.2 .* randn(rng, length(interval), length(affected))
        end
        metadata["bursts"] = [[first(r), last(r)] for r in bursts]
        metadata["change_points"] = [first(r) for r in bursts]
        metadata["affected_members"] = affected
    end

    X = Float64.(y .+ errors)
    split = SplitSpec(
        train=1:train_end,
        validation=(train_end + 1):val_end,
        test=(val_end + 1):n,
    )
    bundle = DatasetBundle(
        name="synthetic_$(regime)",
        X=X,
        y=Float64.(y),
        member_names=["member_$(j)" for j in 1:m],
        split=split,
        availability_lag=1,
        units="signal units",
        metadata=metadata,
    )
    validate_split(bundle.split, n)
    return bundle
end

function generate_correlation_stress(seed::Int, rho::Float64; kwargs...)
    return generate_synthetic_regime("recurring", seed; rho=rho, kwargs...)
end
