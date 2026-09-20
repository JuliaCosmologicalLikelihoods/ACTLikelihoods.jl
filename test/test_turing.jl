"""
    test/test_turing.jl

The Turing extension. The likelihood is stated with `~`, not `@addlogprob!`, so
these tests check the three properties that choice buys: the distribution's
`logpdf` is exactly this package's own likelihood, the model's log joint is
exactly priors plus likelihood, and the priors are the official ones rather
than a hand-copied restatement.
"""

using ADTypes
using Turing
using Distributions
using Random

# Computed here rather than reused from another test file, so this file does
# not depend on include order.
const TURING_FOREGROUNDS = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)

const TURING_EXT = Base.get_extension(ACTLikelihoods, :ACTLikelihoodsTuringExt)

@testset "ACT Turing extension — loaded" begin
    @test TURING_EXT !== nothing
end

@testset "ACT Turing extension — logpdf is the package likelihood" begin
    prediction = predict(LIKE, REFERENCE_CMB, TURING_FOREGROUNDS, REFERENCE_NUISANCE)
    d = TURING_EXT.ACTDR6Bandpowers(LIKE, prediction)

    @test length(d) == 1651
    # Exactly, not approximately: the distribution evaluates the same covariance
    # solve `chi2` uses, so the only difference from `loglikelihood` is the
    # Gaussian normalization this package deliberately keeps separate.
    @test logpdf(d, LIKE.observed) ==
        loglikelihood(LIKE, prediction) + gaussian_normalization(LIKE)
    # ... which is the upstream act_dr6_mflike convention.
    @test logpdf(d, LIKE.observed) ≈ -2941.2005841750693 rtol=1e-12

    # A different vector must give a different, finite answer: the distribution
    # must not ignore its argument and echo `like.observed`.
    shifted = LIKE.observed .+ 1.0
    @test isfinite(logpdf(d, shifted))
    @test logpdf(d, shifted) != logpdf(d, LIKE.observed)

    @test_throws DimensionMismatch TURING_EXT.ACTDR6Bandpowers(LIKE, prediction[1:10])

    # Sampling works, which is what makes prior predictive checks possible.
    draw = rand(Random.MersenneTwister(20260919), d)
    @test length(draw) == 1651
    @test all(isfinite, draw)
end

@testset "ACT Turing extension — priors come from the official tables" begin
    priors = TURING_EXT.ACT_DR6_PRIOR_DISTRIBUTIONS
    @test length(priors) == 29
    @test keys(priors) == ACT_DR6_FREE_PARAMETERS

    for name in ACT_DR6_FREE_PARAMETERS
        distribution = priors[name]
        gaussian = get(ACT_DR6_GAUSSIAN_PRIORS, name, nothing)
        uniform = get(ACT_DR6_UNIFORM_PRIORS, name, nothing)
        if gaussian !== nothing && uniform !== nothing
            @test distribution isa Truncated
            @test minimum(distribution) == uniform[1]
            @test maximum(distribution) == uniform[2]
        elseif gaussian !== nothing
            @test distribution isa Normal
            @test (mean(distribution), std(distribution)) == gaussian
        else
            @test distribution isa Uniform
            @test (minimum(distribution), maximum(distribution)) == uniform
        end
    end
end

@testset "ACT Turing extension — log joint is priors plus likelihood" begin
    model = TURING_EXT.act_dr6_model(LIKE, FOREGROUND_MODEL, REFERENCE_CMB)
    values = (; (name => getfield(REFERENCE_NUISANCE, name)
                 for name in ACT_DR6_FREE_PARAMETERS)...)

    prediction = predict(LIKE, REFERENCE_CMB, TURING_FOREGROUNDS, REFERENCE_NUISANCE)
    likelihood_term = loglikelihood(LIKE, prediction) + gaussian_normalization(LIKE)
    prior_term = sum(logpdf(TURING_EXT.ACT_DR6_PRIOR_DISTRIBUTIONS[name],
                            getfield(REFERENCE_NUISANCE, name))
                     for name in ACT_DR6_FREE_PARAMETERS)

    @test Turing.DynamicPPL.logjoint(model, values) ≈ prior_term + likelihood_term rtol=1e-12
    @test isfinite(Turing.DynamicPPL.logjoint(model, values))
end

@testset "ACT Turing extension — NUTS samples" begin
    model = TURING_EXT.act_dr6_model(LIKE, FOREGROUND_MODEL, REFERENCE_CMB)
    start = (; (name => getfield(REFERENCE_NUISANCE, name)
                for name in ACT_DR6_FREE_PARAMETERS)...)
    Random.seed!(20260919)
    # `initial_params` is not optional in practice: a draw from the priors can
    # put a bandpass shift where the passband normalization is not finite, and
    # CMBForegrounds throws there rather than returning -Inf, which kills the
    # chain instead of rejecting the proposal.
    chain = sample(model, NUTS(2, 0.65; adtype=AutoForwardDiff()), 2;
                   initial_params=start, progress=false, verbose=false)
    @test size(chain, 1) == 2
    parameters = [k for k in keys(chain) if occursin("Parameter", string(k))]
    @test length(parameters) == 29
end
