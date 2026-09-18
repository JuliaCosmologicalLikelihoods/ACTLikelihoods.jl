"""
    test/test_nuisance.jl

The nuisance parameter container, its vector layout, and the official ACT DR6
priors.
"""

using ForwardDiff

@testset "ACT DR6 nuisance — free parameter set matches the official config" begin
    @test length(ACT_DR6_FREE_PARAMETERS) == 29
    @test fieldnames(ACTDR6Nuisance) == ACT_DR6_FREE_PARAMETERS
    @test allunique(ACT_DR6_FREE_PARAMETERS)

    foreground_names = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p, :a_c, :a_s,
                        :beta_s, :a_gtt, :a_gte, :a_gee, :a_psee, :a_pste, :xi)
    calibration_names = (:calG_all,
                         :cal_dr6_pa4_f220, :cal_dr6_pa5_f090, :cal_dr6_pa5_f150,
                         :cal_dr6_pa6_f090, :cal_dr6_pa6_f150,
                         :calE_dr6_pa5_f090, :calE_dr6_pa5_f150,
                         :calE_dr6_pa6_f090, :calE_dr6_pa6_f150)
    shift_names = Tuple(Symbol("bandint_shift_", channel)
                        for channel in ACTLikelihoods.ACT_DR6_CHANNELS)

    @test length(foreground_names) == 14
    @test length(calibration_names) == 10
    @test length(shift_names) == 5
    @test ACT_DR6_FREE_PARAMETERS == (foreground_names..., calibration_names...,
                                      shift_names...)

    # beta_c is derived and calE_dr6_pa4_f220 is fixed: neither is free.
    @test !(:beta_c in ACT_DR6_FREE_PARAMETERS)
    @test !(:calE_dr6_pa4_f220 in ACT_DR6_FREE_PARAMETERS)
    @test ACT_DR6_FIXED_PARAMETERS.calE_dr6_pa4_f220 == 1.0
    @test ACT_DR6_FIXED_PARAMETERS.T_d == 9.60
    @test ACT_DR6_FIXED_PARAMETERS.alpha_dT == -0.6
    @test ACT_DR6_FIXED_PARAMETERS.alpha_dE == -0.4
    @test ACT_DR6_FIXED_PARAMETERS.alpha_p == 1.0
    @test ACT_DR6_FIXED_PARAMETERS.alpha_s == 1.0
    @test ACT_DR6_FIXED_PARAMETERS.T_effd == 19.6
    @test ACT_DR6_FIXED_PARAMETERS.beta_d == 1.5
    @test isdisjoint(keys(ACT_DR6_FIXED_PARAMETERS), ACT_DR6_FREE_PARAMETERS)
end

@testset "ACT DR6 nuisance — vector round-trip is exact" begin
    p = ACTDR6Nuisance()
    x = parameter_vector(p)
    @test length(x) == 29
    @test eltype(x) === Float64
    @test ACTDR6Nuisance(x) == p
    @test parameter_vector(ACTDR6Nuisance(x)) == x

    # The documented ordering is the one the vector actually uses.
    for (index, name) in enumerate(ACT_DR6_FREE_PARAMETERS)
        @test x[index] === getfield(p, name)
    end

    # Round-trip of an arbitrary point, bit for bit.
    arbitrary = collect(range(-1.5, 3.25; length=29))
    @test parameter_vector(ACTDR6Nuisance(arbitrary)) == arbitrary

    @test_throws DimensionMismatch ACTDR6Nuisance(zeros(28))
    @test_throws DimensionMismatch ACTDR6Nuisance(zeros(30))
end

@testset "ACT DR6 nuisance — construction from NamedTuple and Dict" begin
    reference = ACTDR6Nuisance()
    @test ACTDR6Nuisance(ACT_DR6_REFERENCE_NUISANCE) == reference
    @test ACTDR6Nuisance(free_parameters(reference)) == reference

    symbol_dict = Dict(pairs(free_parameters(reference)))
    @test ACTDR6Nuisance(symbol_dict) == reference
    string_dict = Dict(String(k) => v for (k, v) in pairs(free_parameters(reference)))
    @test ACTDR6Nuisance(string_dict) == reference

    # Partial construction fills from the reference point.
    partial = ACTDR6Nuisance((a_tSZ = 4.0,))
    @test partial.a_tSZ == 4.0
    @test partial.a_kSZ == reference.a_kSZ

    # Copy-with-overrides.
    modified = ACTDR6Nuisance(reference; a_tSZ = 4.0, xi = 0.1)
    @test modified.a_tSZ == 4.0
    @test modified.xi == 0.1
    @test modified.beta_p == reference.beta_p

    # Unknown names must be rejected rather than silently dropped.
    @test_throws ArgumentError ACTDR6Nuisance((a_tsz = 4.0,))
    @test_throws ArgumentError ACTDR6Nuisance(Dict("not_a_parameter" => 1.0))

    # Fixed constants may be present but must carry their official value.
    @test ACTDR6Nuisance(merge(free_parameters(reference), (T_d = 9.60,))) == reference
    @test_throws ArgumentError ACTDR6Nuisance(merge(free_parameters(reference),
                                                     (T_d = 9.70,)))
    @test_throws ArgumentError ACTDR6Nuisance(merge(free_parameters(reference),
                                                     (calE_dr6_pa4_f220 = 0.99,)))
end

@testset "ACT DR6 nuisance — beta_c is derived from beta_p" begin
    p = ACTDR6Nuisance(; beta_p = 2.25)
    full = NamedTuple(p)
    @test full.beta_c == 2.25
    @test full.beta_c === full.beta_p
    @test haskey(full, :beta_c)

    # A contradictory beta_c is an error, not a silent override.
    @test_throws ArgumentError ACTDR6Nuisance(merge(free_parameters(p), (beta_c = 1.9,)))
    @test ACTDR6Nuisance(merge(free_parameters(p), (beta_c = 2.25,))) == p

    # ... including when beta_p is *not* also supplied, where the comparison is
    # against the inherited default. Checking only the both-supplied case let a
    # contradictory beta_c through silently.
    reference_beta_p = ACTDR6Nuisance().beta_p
    @test_throws ArgumentError ACTDR6Nuisance((beta_c = 999.0,))
    @test_throws ArgumentError ACTDR6Nuisance(Dict("beta_c" => 999.0))
    @test ACTDR6Nuisance((beta_c = reference_beta_p,)) == ACTDR6Nuisance()
    @test_throws ArgumentError ACTDR6Nuisance(p; beta_c = 1.9)
    @test ACTDR6Nuisance(p; beta_c = 2.25) == p
end

@testset "ACT DR6 nuisance — copy-with-overrides validates non-free names" begin
    p = ACTDR6Nuisance()

    # A fixed constant supplied as an override used to be accepted and then
    # discarded, so the call silently returned an unchanged point.
    @test_throws ArgumentError ACTDR6Nuisance(p; T_d = 999.0)
    @test_throws ArgumentError ACTDR6Nuisance(p; calE_dr6_pa4_f220 = 0.99)
    @test_throws ArgumentError ACTDR6Nuisance(p; alpha_dT = 0.0)
    @test_throws ArgumentError ACTDR6Nuisance(; T_d = 999.0)

    # Agreeing values remain acceptable, and remain no-ops.
    @test ACTDR6Nuisance(p; T_d = ACT_DR6_FIXED_PARAMETERS.T_d) == p
    @test ACTDR6Nuisance(p; calE_dr6_pa4_f220 = 1.0) == p

    # Genuinely unknown names are still rejected.
    @test_throws ArgumentError ACTDR6Nuisance(p; a_tsz = 4.0)

    # And a real override still works.
    @test ACTDR6Nuisance(p; a_tSZ = 4.0).a_tSZ == 4.0
end

@testset "ACT DR6 nuisance — NamedTuple carries free, derived and fixed values" begin
    p = ACTDR6Nuisance()
    full = NamedTuple(p)
    for name in ACT_DR6_FREE_PARAMETERS
        @test full[name] === getfield(p, name)
    end
    for name in keys(ACT_DR6_FIXED_PARAMETERS)
        @test full[name] == ACT_DR6_FIXED_PARAMETERS[name]
    end
    @test length(full) == 29 + length(ACT_DR6_FIXED_PARAMETERS) + 1

    # free_parameters must exclude the fixed constants and beta_c.
    free = free_parameters(p)
    @test keys(free) == ACT_DR6_FREE_PARAMETERS
    @test !haskey(free, :beta_c)
    @test !haskey(free, :T_d)
end

@testset "ACT DR6 nuisance — AD scalars survive construction" begin
    x = parameter_vector(ACTDR6Nuisance())
    dual_vector = ForwardDiff.Dual.(x, 1.0)
    p = ACTDR6Nuisance(dual_vector)
    @test p isa ACTDR6Nuisance{<:ForwardDiff.Dual}
    @test parameter_vector(p) == dual_vector
    @test ForwardDiff.value(NamedTuple(p).beta_c) == ACTDR6Nuisance().beta_p
    @test NamedTuple(p).T_d == ACT_DR6_FIXED_PARAMETERS.T_d

    # And a gradient flows through the whole container.
    objective(v) = sum(parameter_vector(ACTDR6Nuisance(v)) .^ 2)
    @test ForwardDiff.gradient(objective, x) ≈ 2 .* x
end

@testset "ACT DR6 priors — values come from the official configuration" begin
    @test ACT_DR6_GAUSSIAN_PRIORS.calG_all == (1.0, 0.003)
    @test ACT_DR6_GAUSSIAN_PRIORS.cal_dr6_pa4_f220 == (1.0, 0.013)
    @test ACT_DR6_GAUSSIAN_PRIORS.cal_dr6_pa5_f090 == (1.0, 0.0016)
    @test ACT_DR6_GAUSSIAN_PRIORS.cal_dr6_pa5_f150 == (1.0, 0.0020)
    @test ACT_DR6_GAUSSIAN_PRIORS.cal_dr6_pa6_f090 == (1.0, 0.0018)
    @test ACT_DR6_GAUSSIAN_PRIORS.cal_dr6_pa6_f150 == (1.0, 0.0024)
    @test ACT_DR6_GAUSSIAN_PRIORS.bandint_shift_dr6_pa4_f220 == (0.0, 3.6)
    @test ACT_DR6_GAUSSIAN_PRIORS.bandint_shift_dr6_pa5_f090 == (0.0, 1.0)
    @test ACT_DR6_GAUSSIAN_PRIORS.bandint_shift_dr6_pa5_f150 == (0.0, 1.3)
    @test ACT_DR6_GAUSSIAN_PRIORS.bandint_shift_dr6_pa6_f090 == (0.0, 1.2)
    @test ACT_DR6_GAUSSIAN_PRIORS.bandint_shift_dr6_pa6_f150 == (0.0, 1.1)
    @test ACT_DR6_GAUSSIAN_PRIORS.a_gtt == (7.95, 0.32)
    @test ACT_DR6_GAUSSIAN_PRIORS.a_gte == (0.423, 0.03)
    @test ACT_DR6_GAUSSIAN_PRIORS.a_gee == (0.1681, 0.017)
    @test length(ACT_DR6_GAUSSIAN_PRIORS) == 14

    @test ACT_DR6_UNIFORM_PRIORS.a_tSZ == (0.0, 10.0)
    @test ACT_DR6_UNIFORM_PRIORS.beta_s == (-3.5, -1.5)
    @test ACT_DR6_UNIFORM_PRIORS.xi == (0.0, 0.2)
    @test ACT_DR6_UNIFORM_PRIORS.calE_dr6_pa5_f090 == (0.9, 1.1)
    @test length(ACT_DR6_UNIFORM_PRIORS) == 18

    # Every prior names a real free parameter, and every free parameter has at
    # least one prior.
    named = union(keys(ACT_DR6_GAUSSIAN_PRIORS), keys(ACT_DR6_UNIFORM_PRIORS))
    @test issubset(named, Set(ACT_DR6_FREE_PARAMETERS))
    @test Set(ACT_DR6_FREE_PARAMETERS) == Set(named)
end

@testset "ACT DR6 priors — logprior, prior_chi2 and bounds" begin
    p = ACTDR6Nuisance()

    expected_chi2 = 0.0
    for name in keys(ACT_DR6_GAUSSIAN_PRIORS)
        location, scale = ACT_DR6_GAUSSIAN_PRIORS[name]
        expected_chi2 += ((getfield(p, name) - location) / scale)^2
    end
    @test prior_chi2(p) ≈ expected_chi2
    @test prior_chi2(p) ≈ prior_chi2(parameter_vector(p))
    @test prior_chi2(p) > 0

    expected_logprior = -expected_chi2 / 2
    for name in keys(ACT_DR6_GAUSSIAN_PRIORS)
        expected_logprior -= log(ACT_DR6_GAUSSIAN_PRIORS[name][2] * sqrt(2 * pi))
    end
    for name in keys(ACT_DR6_UNIFORM_PRIORS)
        lower, upper = ACT_DR6_UNIFORM_PRIORS[name]
        expected_logprior -= log(upper - lower)
    end
    @test logprior(p) ≈ expected_logprior
    @test logprior(p) ≈ logprior(parameter_vector(p))

    # Outside a uniform range the prior vanishes.
    @test logprior(ACTDR6Nuisance(p; xi = 0.5)) == -Inf
    @test prior_chi2(ACTDR6Nuisance(p; xi = 0.5)) == Inf
    @test logprior(ACTDR6Nuisance(p; beta_s = -4.0)) == -Inf
    @test logprior(ACTDR6Nuisance(p; calE_dr6_pa5_f090 = 1.5)) == -Inf
    @test isfinite(logprior(ACTDR6Nuisance(p; xi = 0.19)))

    # The Gaussian part responds to each of its parameters.
    for name in keys(ACT_DR6_GAUSSIAN_PRIORS)
        location, scale = ACT_DR6_GAUSSIAN_PRIORS[name]
        at_peak = ACTDR6Nuisance(p; NamedTuple{(name,)}((location,))...)
        off_peak = ACTDR6Nuisance(p; NamedTuple{(name,)}((location + scale,))...)
        @test logprior(at_peak) > logprior(off_peak)
    end

    @test_throws DimensionMismatch logprior(zeros(10))
    @test_throws DimensionMismatch prior_chi2(zeros(10))
end

@testset "ACT DR6 priors — non-finite values are outside the support" begin
    reference = parameter_vector(ACTDR6Nuisance())

    # `value < lower || value > upper` is false for NaN, so a naive bounds test
    # hands a NaN proposal a finite prior instead of rejecting it. Every
    # non-finite category must be rejected, for every priored parameter, in both
    # the uniform-only, Gaussian-only and combined cases.
    uniform_only = [name for name in ACT_DR6_FREE_PARAMETERS
                    if haskey(ACT_DR6_UNIFORM_PRIORS, name) &&
                       !haskey(ACT_DR6_GAUSSIAN_PRIORS, name)]
    gaussian_only = [name for name in ACT_DR6_FREE_PARAMETERS
                     if haskey(ACT_DR6_GAUSSIAN_PRIORS, name) &&
                        !haskey(ACT_DR6_UNIFORM_PRIORS, name)]
    both = [name for name in ACT_DR6_FREE_PARAMETERS
            if haskey(ACT_DR6_GAUSSIAN_PRIORS, name) &&
               haskey(ACT_DR6_UNIFORM_PRIORS, name)]

    @test !isempty(uniform_only)
    @test !isempty(gaussian_only)
    @test length(both) == 3          # the three galactic-dust amplitudes
    @test length(uniform_only) + length(gaussian_only) + length(both) == 29

    for name in (uniform_only..., gaussian_only..., both...)
        index = findfirst(==(name), ACT_DR6_FREE_PARAMETERS)
        for bad in (NaN, Inf, -Inf)
            x = copy(reference)
            x[index] = bad
            @test logprior(x) == -Inf
            @test prior_chi2(x) == Inf
            # The struct entry point must agree with the vector one.
            p = ACTDR6Nuisance(x)
            @test logprior(p) == -Inf
            @test prior_chi2(p) == Inf
        end
    end

    # The regression the review reported, spelled out: a uniform-only parameter
    # set to NaN used to return a finite density.
    xi_index = findfirst(==(:xi), ACT_DR6_FREE_PARAMETERS)
    nan_xi = copy(reference)
    nan_xi[xi_index] = NaN
    @test !isfinite(logprior(nan_xi))
    @test !isfinite(prior_chi2(nan_xi))

    # An all-NaN vector is rejected, not summed.
    @test logprior(fill(NaN, 29)) == -Inf
    @test prior_chi2(fill(NaN, 29)) == Inf

    # And the reference point itself is still finite.
    @test isfinite(logprior(reference))
    @test isfinite(prior_chi2(reference))
end

@testset "ACT DR6 priors — differentiable inside the support" begin
    x = parameter_vector(ACTDR6Nuisance())
    gradient = ForwardDiff.gradient(logprior, x)
    @test all(isfinite, gradient)

    # Uniform-only parameters have zero prior gradient in the interior;
    # Gaussian ones do not.
    for (index, name) in enumerate(ACT_DR6_FREE_PARAMETERS)
        if haskey(ACT_DR6_GAUSSIAN_PRIORS, name)
            location, scale = ACT_DR6_GAUSSIAN_PRIORS[name]
            @test gradient[index] ≈ -(x[index] - location) / scale^2
        else
            @test gradient[index] == 0
        end
    end
end

@testset "ACT DR6 nuisance — display" begin
    text = sprint(show, MIME"text/plain"(), ACTDR6Nuisance())
    @test occursin("29 free ACT DR6 parameters", text)
    @test occursin("beta_c = beta_p", text)
    @test occursin("a_tSZ", text)
    @test occursin("ACTDR6Nuisance", sprint(show, ACTDR6Nuisance()))
end
