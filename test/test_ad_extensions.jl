"""
    test/test_ad_extensions.jl

Unit tests for the custom pullbacks this package defines, with arbitrary
cotangents. These extensions are never gated: if `ChainRulesCore` or `Mooncake`
fails to load, the suite fails.
"""

using ADTypes
using ChainRulesCore
using DifferentiationInterface
using FiniteDifferences
using ForwardDiff
using Mooncake
using Random

@testset "ACT extensions — both extensions are loaded" begin
    @test Base.get_extension(ACTLikelihoods, :ACTLikelihoodsChainRulesCoreExt) !== nothing
    @test Base.get_extension(ACTLikelihoods, :ACTLikelihoodsMooncakeExt) !== nothing
end

@testset "Fixed ACT covariance solve — pullback with arbitrary cotangents" begin
    rng = Random.MersenneTwister(20260917)
    factor = LowerTriangular([2.0 0.0 0.0; 0.3 1.7 0.0; -0.2 0.4 1.3])
    residual = [0.5, -0.7, 1.1]

    whitened, pullback = ChainRulesCore.rrule(
        ACTLikelihoods._fixed_lower_solve, factor, residual,
    )
    @test whitened ≈ factor \ residual

    # Arbitrary cotangents, not just a convenient one. The factor is a fixed
    # data product, so its cotangent must be NoTangent every time.
    for trial in 1:8
        cotangent = trial == 1 ? [0.2, 0.4, -0.3] : randn(rng, 3)
        function̄, factor̄, residual̄ = pullback(cotangent)
        @test function̄ isa ChainRulesCore.NoTangent
        @test factor̄ isa ChainRulesCore.NoTangent
        @test residual̄ ≈ transpose(factor) \ cotangent

        # The pullback must be the exact adjoint of the forward map.
        direction = randn(rng, 3)
        @test dot(cotangent, factor \ direction) ≈ dot(residual̄, direction) rtol=1e-12
    end

    # Thunked cotangents (Mooncake passes these) must be handled.
    _, _, thunked = pullback(ChainRulesCore.Thunk(() -> [0.2, 0.4, -0.3]))
    @test thunked ≈ transpose(factor) \ [0.2, 0.4, -0.3]

    objective(x) = sum(abs2, ACTLikelihoods._fixed_lower_solve(factor, x))
    forward = DifferentiationInterface.gradient(objective, AutoForwardDiff(), residual)
    reverse = DifferentiationInterface.gradient(objective, AutoMooncake(; config=nothing),
                                                residual)
    finite = DifferentiationInterface.gradient(
        objective, AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1)), residual)
    @test reverse ≈ forward rtol=1e-12
    @test reverse ≈ finite rtol=1e-7 atol=1e-9
end

@testset "ACT projection kernel — pullback with arbitrary cotangents" begin
    rng = Random.MersenneTwister(20260918)
    calibrations = ACTLikelihoods._spectrum_calibrations(LIKE, NamedTuple(REFERENCE_NUISANCE))
    baseline_foregrounds = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)
    offset = Int(first(CMB_INDICES))

    arguments = (LIKE, REFERENCE_CMB.TT, REFERENCE_CMB.TE, REFERENCE_CMB.EE,
                 baseline_foregrounds.TT, baseline_foregrounds.TE,
                 baseline_foregrounds.EE, calibrations, offset)

    model, pullback = ChainRulesCore.rrule(ACTLikelihoods._act_projection, arguments...)
    @test model == ACTLikelihoods._act_projection(arguments...)
    @test model == predict(LIKE, REFERENCE_CMB, baseline_foregrounds, REFERENCE_NUISANCE)

    # The kernel is linear in the CMB and the foregrounds, so the pullback must
    # be its exact adjoint: <v̄, J·u> == <Jᵀ·v̄, u> for arbitrary v̄ and u.
    for trial in 1:5
        cotangent = randn(rng, length(model))
        tangents = pullback(cotangent)
        @test tangents[1] isa ChainRulesCore.NoTangent   # the function itself
        @test tangents[2] isa ChainRulesCore.NoTangent   # the likelihood: released data
        @test tangents[end] isa ChainRulesCore.NoTangent # the CMB index offset

        cmb_TT̄, cmb_TĒ, cmb_EĒ = tangents[3], tangents[4], tangents[5]
        fg_TT̄, fg_TĒ, fg_EĒ = tangents[6], tangents[7], tangents[8]
        calibrations̄ = tangents[9]

        # The kernel is exactly linear in the CMB blocks, the foreground blocks
        # and the calibrations, so the Jacobian-vector product is the kernel
        # itself applied to the direction. That makes the adjoint identity an
        # exact check rather than a finite-difference one.
        zero_cmb = (zero(REFERENCE_CMB.TT), zero(REFERENCE_CMB.TE), zero(REFERENCE_CMB.EE))
        zero_fg = (zero(baseline_foregrounds.TT), zero(baseline_foregrounds.TE),
                   zero(baseline_foregrounds.EE))

        direction_TT = randn(rng, length(REFERENCE_CMB.TT))
        direction_TE = randn(rng, length(REFERENCE_CMB.TE))
        direction_EE = randn(rng, length(REFERENCE_CMB.EE))
        applied = ACTLikelihoods._act_projection(
            LIKE, direction_TT, direction_TE, direction_EE, zero_fg...,
            calibrations, offset,
        )
        @test dot(cotangent, applied) ≈
            dot(cmb_TT̄, direction_TT) + dot(cmb_TĒ, direction_TE) +
            dot(cmb_EĒ, direction_EE) rtol=1e-12

        # ... and through the foreground blocks, which the kernel adds to the
        # CMB, so they must receive exactly the same cotangent.
        direction_fg = (randn(rng, size(baseline_foregrounds.TT)),
                        randn(rng, size(baseline_foregrounds.TE)),
                        randn(rng, size(baseline_foregrounds.EE)))
        applied_fg = ACTLikelihoods._act_projection(
            LIKE, zero_cmb..., direction_fg..., calibrations, offset,
        )
        @test dot(cotangent, applied_fg) ≈
            dot(fg_TT̄, direction_fg[1]) + dot(fg_TĒ, direction_fg[2]) +
            dot(fg_EĒ, direction_fg[3]) rtol=1e-12

        for (cmb̄, fḡ) in ((cmb_TT̄, fg_TT̄), (cmb_TĒ, fg_TĒ), (cmb_EĒ, fg_EĒ))
            @test sum(fḡ) ≈ sum(@view cmb̄[offset:(offset + length(LIKE.ells) - 1)]) rtol=1e-10
        end

        # And through the calibrations, which the model is also linear in.
        direction = randn(rng, length(calibrations))
        applied_calibration = ACTLikelihoods._act_projection(
            LIKE, REFERENCE_CMB.TT, REFERENCE_CMB.TE, REFERENCE_CMB.EE,
            baseline_foregrounds.TT, baseline_foregrounds.TE, baseline_foregrounds.EE,
            direction, offset,
        )
        @test dot(cotangent, applied_calibration) ≈ dot(calibrations̄, direction) rtol=1e-12

        # Nothing outside the window support may receive a cotangent: the
        # released windows are exactly zero there.
        @test all(iszero, @view cmb_TT̄[1:(offset - 1)])
    end

    # Thunked cotangents, as Mooncake passes them.
    thunked = pullback(ChainRulesCore.Thunk(() -> ones(length(model))))
    plain = pullback(ones(length(model)))
    @test thunked[3] ≈ plain[3]
    @test thunked[9] ≈ plain[9]
end
