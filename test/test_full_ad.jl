"""
    test/test_full_ad.jl

Automatic differentiation of the complete public path.

ForwardDiff is the numerical oracle, prepared Mooncake is the primary
reverse-mode requirement, and finite differences provide an independent
directional check. Nothing here is conditional.
"""

using ADTypes
using DifferentiationInterface
using FiniteDifferences
using ForwardDiff
using Mooncake
using Random

const MOONCAKE = AutoMooncake(; config=nothing)
const FDM = AutoFiniteDifferences(; fdm=FiniteDifferences.central_fdm(5, 1))

"""
    full_loglikelihood(x)

The complete public path as a function of the 29-element free nuisance vector:
nuisance container -> foreground assembly (bandpass shifts, chromatic beams,
SEDs, angular templates) -> calibration -> window convolution -> covariance
solve -> data-only log likelihood.
"""
function full_loglikelihood(x)
    p = ACTDR6Nuisance(x)
    fg = foregrounds(FOREGROUND_MODEL, LIKE, p)
    prediction = predict(LIKE, REFERENCE_CMB, fg, NamedTuple(p), CMB_INDICES)
    return loglikelihood(LIKE, prediction)
end

"""
    prepared_mooncake_gradient(f, x) -> (gradient, preparation)

Reverse-mode gradient through a prepared DifferentiationInterface cache, which
is the documented way this package expects Mooncake to be used.
"""
function prepared_mooncake_gradient(f, x)
    preparation = DifferentiationInterface.prepare_gradient(f, MOONCAKE, x)
    gradient = similar(x)
    DifferentiationInterface.gradient!(f, gradient, preparation, MOONCAKE, x)
    return gradient, preparation
end

const AD_REFERENCE_POINT = parameter_vector(REFERENCE_NUISANCE)

@testset "ACT DR6 AD — all 29 free nuisance parameters" begin
    x = copy(AD_REFERENCE_POINT)
    forward = ForwardDiff.gradient(full_loglikelihood, x)
    reverse, preparation = prepared_mooncake_gradient(full_loglikelihood, x)

    @test length(forward) == 29
    @test all(isfinite, forward)
    @test all(isfinite, reverse)
    @test reverse ≈ forward rtol=1e-6 atol=1e-7

    # No parameter may be silently ignored by either mode.
    @test all(!iszero, forward)
    @test all(!iszero, reverse)

    # Independent directional check by finite differences, in a random
    # direction, so no single coordinate can accidentally agree.
    rng = Random.MersenneTwister(1651)
    direction = randn(rng, 29)
    scale = max.(abs.(x), 1e-3)
    direction .*= scale
    directional(t) = full_loglikelihood(x .+ t .* direction)
    finite = FiniteDifferences.central_fdm(5, 1)(directional, 0.0)
    @test finite ≈ dot(forward, direction) rtol=1e-5 atol=1e-5

    # Prepared cache must be reusable at a different parameter point.
    moved = copy(x)
    moved[1] += 0.35            # a_tSZ
    moved[15] *= 1.002          # calG_all
    moved[25] -= 0.75           # bandint_shift_dr6_pa4_f220
    reused = similar(moved)
    DifferentiationInterface.gradient!(full_loglikelihood, reused, preparation,
                                       MOONCAKE, moved)
    @test reused ≈ ForwardDiff.gradient(full_loglikelihood, moved) rtol=1e-6 atol=1e-7
    @test !(reused ≈ reverse)

    # And the cache must not have captured the first point.
    repeated = similar(x)
    DifferentiationInterface.gradient!(full_loglikelihood, repeated, preparation,
                                       MOONCAKE, x)
    @test repeated ≈ reverse rtol=1e-12
end

@testset "ACT DR6 AD — parameter subgroups" begin
    x = copy(AD_REFERENCE_POINT)
    groups = (
        foregrounds = 1:14,
        calibration = 15:24,
        bandpass_shifts = 25:29,
    )
    for (name, indices) in pairs(groups)
        @testset "$name only" begin
            subvector = x[indices]
            function subobjective(v)
                full = similar(v, 29)
                full .= eltype(v).(x)
                full[indices] = v
                return full_loglikelihood(full)
            end
            forward = ForwardDiff.gradient(subobjective, subvector)
            reverse, _ = prepared_mooncake_gradient(subobjective, subvector)
            @test all(isfinite, forward)
            @test reverse ≈ forward rtol=1e-6 atol=1e-7
            @test all(!iszero, forward)
            @test forward ≈ ForwardDiff.gradient(full_loglikelihood, x)[indices] rtol=1e-8
        end
    end
end

@testset "ACT DR6 AD — foreground assembly alone" begin
    # Differentiate the foreground arrays themselves, so a failure localizes to
    # assembly rather than to the window or covariance stages.
    rng = Random.MersenneTwister(4141)
    weights_TT = randn(rng, 5, 5)
    weights_TE = randn(rng, 5, 5)
    weights_EE = randn(rng, 5, 5)
    sample_ells = (1, 37, 500, 2999, 8500)

    function foreground_functional(x)
        fg = foregrounds(FOREGROUND_MODEL, LIKE, ACTDR6Nuisance(x))
        total = zero(eltype(fg.TT))
        for index in sample_ells, i in 1:5, j in 1:5
            total += weights_TT[i, j] * fg.TT[i, j, index]
            total += weights_TE[i, j] * fg.TE[i, j, index]
            total += weights_EE[i, j] * fg.EE[i, j, index]
        end
        return total
    end

    x = copy(AD_REFERENCE_POINT)
    forward = ForwardDiff.gradient(foreground_functional, x)
    reverse, _ = prepared_mooncake_gradient(foreground_functional, x)
    @test all(isfinite, forward)
    @test reverse ≈ forward rtol=1e-6 atol=1e-8

    # Calibration parameters cannot affect the foregrounds.
    @test all(iszero, forward[15:24])
    # Foreground and bandpass-shift parameters must.
    @test any(!iszero, forward[1:14])
    @test all(!iszero, forward[25:29])
end

@testset "ACT DR6 AD — calibration only" begin
    fixed_foregrounds = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)
    calibration_indices = 15:24
    x = copy(AD_REFERENCE_POINT)

    function calibration_objective(v)
        full = similar(v, 29)
        full .= eltype(v).(x)
        full[calibration_indices] = v
        prediction = predict(LIKE, REFERENCE_CMB, fixed_foregrounds,
                             NamedTuple(ACTDR6Nuisance(full)), CMB_INDICES)
        return loglikelihood(LIKE, prediction)
    end

    subvector = x[calibration_indices]
    forward = ForwardDiff.gradient(calibration_objective, subvector)
    reverse, _ = prepared_mooncake_gradient(calibration_objective, subvector)
    finite = DifferentiationInterface.gradient(calibration_objective, FDM, subvector)
    @test reverse ≈ forward rtol=1e-8 atol=1e-9
    @test forward ≈ finite rtol=1e-5 atol=1e-4
    @test all(!iszero, forward)
end

@testset "ACT DR6 AD — bandpass shifts only" begin
    shift_indices = 25:29
    x = copy(AD_REFERENCE_POINT)

    function shift_objective(v)
        full = similar(v, 29)
        full .= eltype(v).(x)
        full[shift_indices] = v
        return full_loglikelihood(full)
    end

    subvector = x[shift_indices]
    forward = ForwardDiff.gradient(shift_objective, subvector)
    reverse, _ = prepared_mooncake_gradient(shift_objective, subvector)
    finite = DifferentiationInterface.gradient(shift_objective, FDM, subvector)
    @test reverse ≈ forward rtol=1e-7 atol=1e-8
    @test forward ≈ finite rtol=1e-4 atol=1e-4
    @test all(!iszero, forward)
end

@testset "ACT DR6 AD — CMB amplitude scalings" begin
    # Differentiate with respect to TT/TE/EE input amplitudes, exercising the
    # CMB side of the window convolution and the covariance solve.
    fixed_foregrounds = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)
    nuisance = NamedTuple(REFERENCE_NUISANCE)

    function amplitude_objective(scales)
        cmb = ACTCMBTheory(REFERENCE_CMB.ell,
                           REFERENCE_CMB.TT .* scales[1],
                           REFERENCE_CMB.TE .* scales[2],
                           REFERENCE_CMB.EE .* scales[3])
        prediction = predict(LIKE, cmb, fixed_foregrounds, nuisance, CMB_INDICES)
        return loglikelihood(LIKE, prediction)
    end

    scales = [1.0, 1.0, 1.0]
    forward = ForwardDiff.gradient(amplitude_objective, scales)
    reverse, _ = prepared_mooncake_gradient(amplitude_objective, scales)
    finite = DifferentiationInterface.gradient(amplitude_objective, FDM, scales)
    @test all(isfinite, forward)
    @test reverse ≈ forward rtol=1e-8 atol=1e-9
    @test forward ≈ finite rtol=1e-6 atol=1e-5
    @test all(!iszero, forward)
end

@testset "ACT DR6 AD — chi2 and posterior are differentiable end to end" begin
    x = copy(AD_REFERENCE_POINT)

    chi2_objective(v) = chi2(LIKE, predict(
        LIKE, REFERENCE_CMB, foregrounds(FOREGROUND_MODEL, LIKE, ACTDR6Nuisance(v)),
        NamedTuple(ACTDR6Nuisance(v)), CMB_INDICES))
    forward_chi2 = ForwardDiff.gradient(chi2_objective, x)
    reverse_chi2, _ = prepared_mooncake_gradient(chi2_objective, x)
    @test reverse_chi2 ≈ forward_chi2 rtol=1e-6 atol=1e-7
    @test forward_chi2 ≈ -2 .* ForwardDiff.gradient(full_loglikelihood, x) rtol=1e-10

    posterior_objective(v) = full_loglikelihood(v) + logprior(v)
    forward_posterior = ForwardDiff.gradient(posterior_objective, x)
    reverse_posterior, _ = prepared_mooncake_gradient(posterior_objective, x)
    @test all(isfinite, forward_posterior)
    @test reverse_posterior ≈ forward_posterior rtol=1e-6 atol=1e-7
    @test forward_posterior ≈
        ForwardDiff.gradient(full_loglikelihood, x) .+ ForwardDiff.gradient(logprior, x) rtol=1e-10
end

@testset "ACT DR6 AD — the simple documented path is reverse-differentiable" begin
    # `predict(like, cmb, fg, p)` resolves the CMB index range itself. That
    # resolution used to compare two integer vectors with `==`, which lowers to
    # a `memcmp` foreigncall Mooncake cannot trace, so the README's own example
    # failed while the prevalidated-index path in the rest of this file passed.
    # Keep both paths covered.
    function simple_path(x)
        p = ACTDR6Nuisance(x)
        fg = foregrounds(FOREGROUND_MODEL, LIKE, p)
        return loglikelihood(LIKE, predict(LIKE, REFERENCE_CMB, fg, p))
    end

    x = copy(AD_REFERENCE_POINT)
    @test simple_path(x) ≈ full_loglikelihood(x)

    forward = ForwardDiff.gradient(simple_path, x)
    reverse, _ = prepared_mooncake_gradient(simple_path, x)
    @test all(isfinite, reverse)
    @test reverse ≈ forward rtol=1e-6 atol=1e-7
    @test reverse ≈ ForwardDiff.gradient(full_loglikelihood, x) rtol=1e-6 atol=1e-7
end

@testset "ACT DR6 AD — gradients are finite at the physical fiducial point" begin
    x = copy(AD_REFERENCE_POINT)
    gradient = ForwardDiff.gradient(full_loglikelihood, x)
    @test all(isfinite, gradient)
    @test isfinite(full_loglikelihood(x))
    @test full_loglikelihood(x) ≈ -796.0292064995771 rtol=1e-10

    # The fiducial point is strictly inside the prior support.
    @test isfinite(logprior(x))
    @test isfinite(prior_chi2(x))
end
