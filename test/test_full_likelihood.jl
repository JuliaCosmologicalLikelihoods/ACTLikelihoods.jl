"""
    test/test_full_likelihood.jl

End-to-end parity of the full multifrequency likelihood at the reference point.
"""

const REFERENCE_SCALARS = reference_scalars()
const BASELINE_FOREGROUNDS = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)
const BASELINE_PREDICTION = predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS,
                                    REFERENCE_NUISANCE)

@testset "ACT DR6 full likelihood — baseline model vector" begin
    reference = read_reference_vector(REFERENCE_DIR, "model_vector.txt")
    @test length(BASELINE_PREDICTION) == 1651
    @test BASELINE_PREDICTION ≈ reference rtol=1e-10 atol=1e-11

    residual_reference = read_reference_vector(REFERENCE_DIR, "residual.txt")
    @test LIKE.observed .- BASELINE_PREDICTION ≈ residual_reference rtol=1e-10 atol=1e-11

    # Every spectrum block must land in a physically sensible range; an all-zero
    # or NaN block anywhere would otherwise hide inside the global comparison.
    for spectrum in LIKE.spectra
        block = BASELINE_PREDICTION[spectrum.output_indices]
        @test all(isfinite, block)
        @test !all(iszero, block)
        @test maximum(abs, block) < 1e5
    end
end

@testset "ACT DR6 full likelihood — chi-square and log likelihood" begin
    value = chi2(LIKE, BASELINE_PREDICTION)
    @test value ≈ 1592.0584129991541 rtol=1e-10
    @test value ≈ REFERENCE_SCALARS["chi2"] rtol=1e-10

    @test loglikelihood(LIKE, BASELINE_PREDICTION) ≈ -value / 2
    @test loglikelihood(LIKE, BASELINE_PREDICTION) ≈ -796.0292064995771 rtol=1e-10
    @test loglikelihood(LIKE, BASELINE_PREDICTION) ≈
        REFERENCE_SCALARS["data_only_loglikelihood"] rtol=1e-10
end

@testset "ACT DR6 full likelihood — Gaussian normalization convention" begin
    # loglikelihood is data only. The upstream convention adds a fixed constant.
    constant = gaussian_normalization(LIKE)
    @test constant ≈ -2145.1713776754927 rtol=1e-12
    @test constant ≈ REFERENCE_SCALARS["gaussian_log_normalization"] rtol=1e-12

    upstream = loglikelihood(LIKE, BASELINE_PREDICTION) + constant
    @test upstream ≈ -2941.2005841750693 rtol=1e-12
    @test upstream ≈ REFERENCE_SCALARS["upstream_loglike"] rtol=1e-12

    # The constant must not depend on the parameters.
    shifted = predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS,
                      ACTDR6Nuisance(REFERENCE_NUISANCE; calG_all = 1.01))
    @test gaussian_normalization(LIKE) == constant
    @test chi2(LIKE, shifted) != chi2(LIKE, BASELINE_PREDICTION)
end

@testset "ACT DR6 full likelihood — posterior assembly" begin
    data_only = loglikelihood(LIKE, BASELINE_PREDICTION)
    @test logposterior(LIKE, BASELINE_PREDICTION, REFERENCE_NUISANCE) ≈
        data_only + logprior(REFERENCE_NUISANCE)
    @test logposterior(LIKE, BASELINE_PREDICTION,
                       parameter_vector(REFERENCE_NUISANCE)) ≈
        data_only + logprior(REFERENCE_NUISANCE)

    # Priors must never leak into the data-only likelihood.
    @test loglikelihood(LIKE, BASELINE_PREDICTION) == -chi2(LIKE, BASELINE_PREDICTION) / 2
end

@testset "ACT DR6 full likelihood — prevalidated index path agrees" begin
    fast = predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS,
                   NamedTuple(REFERENCE_NUISANCE), CMB_INDICES)
    @test fast == BASELINE_PREDICTION
end

@testset "ACT DR6 full likelihood — NamedTuple and Dict parameters agree" begin
    named = NamedTuple(REFERENCE_NUISANCE)
    dictionary = Dict(pairs(named))
    from_named = foregrounds(FOREGROUND_MODEL, LIKE, named)
    from_dict = foregrounds(FOREGROUND_MODEL, LIKE, dictionary)
    @test from_named.TT == from_dict.TT
    @test from_named.TE == from_dict.TE
    @test from_named.EE == from_dict.EE
    @test predict(LIKE, REFERENCE_CMB, from_dict, dictionary) == BASELINE_PREDICTION
end

@testset "ACT DR6 full likelihood — every varied calibration is required" begin
    named = NamedTuple(REFERENCE_NUISANCE)

    # A missing calibration must not silently default to unity: that would let
    # `predict` return a wrong model for an incomplete container, while
    # foreground assembly rejects the very same container.
    polarization_channels = [channel for channel in LIKE.channels
                             if channel != "dr6_pa4_f220"]
    @test !isempty(polarization_channels)

    required = [:calG_all]
    append!(required, Symbol.("cal_" .* LIKE.channels))
    append!(required, Symbol.("calE_" .* polarization_channels))

    for name in required
        incomplete = Base.structdiff(named, NamedTuple{(name,)})
        @test !haskey(incomplete, name)
        @test_throws ArgumentError predict(LIKE, REFERENCE_CMB,
                                           BASELINE_FOREGROUNDS, incomplete)
        @test_throws ArgumentError predict(LIKE, REFERENCE_CMB,
                                           BASELINE_FOREGROUNDS,
                                           Dict(pairs(incomplete)))
    end

    # The reported reproduction: a container carrying only calG_all.
    @test_throws ArgumentError ACTLikelihoods._map_calibration((calG_all = 1.0,),
                                                              :T, "dr6_pa5_f090")
    @test ACTLikelihoods._map_calibration(named, :T, "dr6_pa5_f090") ==
        inv(named.calG_all) / named.cal_dr6_pa5_f090

    # calT is the one genuinely implicit quantity — ACT DR6 fixes it at 1 and
    # carries no calT_* parameter — so the :T leg needs nothing further.
    @test !any(startswith(String(name), "calT") for name in keys(named))

    # A complete container is of course still accepted.
    @test predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS, named) ==
        BASELINE_PREDICTION
end

@testset "ACT DR6 full likelihood — one key contract for dictionaries" begin
    named = NamedTuple(REFERENCE_NUISANCE)
    symbol_dict = Dict{Any, Float64}(pairs(named))
    string_dict = Dict{Any, Float64}(String(k) => v for (k, v) in pairs(named))

    # `ACTDR6Nuisance` accepts String keys, so model evaluation must resolve
    # them identically rather than reporting the parameters missing.
    @test ACTDR6Nuisance(string_dict) == REFERENCE_NUISANCE
    from_string = foregrounds(FOREGROUND_MODEL, LIKE, string_dict)
    @test from_string.TT == BASELINE_FOREGROUNDS.TT
    @test from_string.TE == BASELINE_FOREGROUNDS.TE
    @test from_string.EE == BASELINE_FOREGROUNDS.EE
    @test predict(LIKE, REFERENCE_CMB, from_string, string_dict) ==
        BASELINE_PREDICTION
    @test predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS, symbol_dict) ==
        BASELINE_PREDICTION

    # A String-keyed dictionary missing a calibration fails the same way a
    # Symbol-keyed one does.
    incomplete = copy(string_dict)
    delete!(incomplete, "cal_dr6_pa5_f090")
    @test_throws ArgumentError predict(LIKE, REFERENCE_CMB,
                                       BASELINE_FOREGROUNDS, incomplete)

    # Containers that are neither a NamedTuple nor an AbstractDict are rejected
    # with a clear message rather than a MethodError deep in the model.
    @test_throws ArgumentError predict(LIKE, REFERENCE_CMB,
                                       BASELINE_FOREGROUNDS, 1.0)
    @test_throws ArgumentError foregrounds(FOREGROUND_MODEL, LIKE, 1.0)
end

@testset "ACT DR6 full likelihood — calE_dr6_pa4_f220 is inert" begin
    # ACT DR6 uses no pa4 f220 polarization channel, which is why the official
    # configuration fixes this parameter. Demonstrate that it cannot move the
    # model at all, rather than asserting it from the YAML alone.
    named = NamedTuple(REFERENCE_NUISANCE)
    perturbed = merge(named, (calE_dr6_pa4_f220 = 1.25,))
    @test predict(LIKE, REFERENCE_CMB, BASELINE_FOREGROUNDS, perturbed) ==
        BASELINE_PREDICTION
end
