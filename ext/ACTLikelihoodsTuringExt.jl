"""
    ACTLikelihoodsTuringExt

Turing.jl support for the full ACT DR6 multifrequency likelihood.

Loaded automatically when `ACTLikelihoods`, `Turing` and `Distributions` are all
present. It provides [`ACTDR6Bandpowers`](@ref), a `Distribution` over the 1651
observed bandpowers, so a model states the likelihood with `~`:

```julia
like.observed ~ ACTDR6Bandpowers(like, prediction)
```

rather than injecting a number with `@addlogprob!`. The difference is not
cosmetic: `~` keeps the observation visible to DynamicPPL, so conditioning,
`predict`, prior sampling and log-density decompositions all behave normally.

`logpdf` is evaluated through this package's own covariance solve, the same one
`chi2` uses, so the model's log density is exactly

    loglikelihood(like, prediction) + gaussian_normalization(like)

and reverse mode goes through the registered `_fixed_lower_solve` rule rather
than through a generic dense path.
"""
module ACTLikelihoodsTuringExt

using ACTLikelihoods
using Turing
using Distributions
using LinearAlgebra
using Random

# ACTLikelihoods and Turing/StatsAPI/DynamicPPL both export `predict`,
# `foregrounds` and `loglikelihood`. Inside this module the names must be
# imported explicitly or every use is ambiguous. Users hit the same ambiguity
# when they write `using ACTLikelihoods, Turing`; see the README.
import ACTLikelihoods: _fixed_lower_solve, predict, foregrounds

export ACTDR6Bandpowers, act_dr6_model

"""
    ACTDR6Bandpowers(like, prediction)

The ACT DR6 bandpower likelihood as a multivariate distribution: a Gaussian with
the released fixed covariance and mean `prediction`.

The covariance is released data, never a parameter, so it is carried by
reference and its stored Cholesky factor is reused on every evaluation.
"""
struct ACTDR6Bandpowers{L<:ACTDR6FullLikelihood,P<:AbstractVector} <:
       ContinuousMultivariateDistribution
    like::L
    prediction::P

    function ACTDR6Bandpowers(like::ACTDR6FullLikelihood, prediction::AbstractVector)
        length(prediction) == length(like.observed) || throw(DimensionMismatch(
            "prediction must have $(length(like.observed)) entries, got $(length(prediction))",
        ))
        return new{typeof(like), typeof(prediction)}(like, prediction)
    end
end

Base.length(d::ACTDR6Bandpowers) = length(d.like.observed)
Base.eltype(::Type{<:ACTDR6Bandpowers{<:Any,P}}) where {P} = eltype(P)

function Distributions._logpdf(d::ACTDR6Bandpowers, x::AbstractVector{<:Real})
    residual = x .- d.prediction
    whitened = _fixed_lower_solve(d.like.covariance_cholesky.L, residual)
    return -dot(whitened, whitened) / 2 + gaussian_normalization(d.like)
end

# Sampling is cheap because the lower Cholesky factor is already stored: a draw
# is `mean + L * z`. This is what makes prior predictive checks possible.
function Distributions._rand!(rng::Random.AbstractRNG, d::ACTDR6Bandpowers,
                              x::AbstractVector{<:Real})
    z = randn(rng, length(d))
    x .= d.prediction .+ d.like.covariance_cholesky.L * z
    return x
end

"""
    ACT_DR6_PRIOR_DISTRIBUTIONS

One `Distribution` per free parameter, in [`ACT_DR6_FREE_PARAMETERS`](@ref)
order, built **from the package's own prior tables** rather than restated by
hand, so the Turing model and [`logprior`](@ref) cannot drift apart.

A parameter carrying only a range becomes `Uniform`; one carrying only a normal
prior becomes `Normal`; the three galactic-dust amplitudes carry both and become
`truncated(Normal(...), min, max)`.

Note that `truncated` renormalizes, so the model's log prior differs from
`logprior(p)` by a fixed constant. The constant cancels in every posterior ratio
and in every gradient, which is what the test suite checks.
"""
const ACT_DR6_PRIOR_DISTRIBUTIONS = NamedTuple(
    name => begin
        gaussian = get(ACT_DR6_GAUSSIAN_PRIORS, name, nothing)
        uniform = get(ACT_DR6_UNIFORM_PRIORS, name, nothing)
        if gaussian !== nothing && uniform !== nothing
            truncated(Normal(gaussian[1], gaussian[2]), uniform[1], uniform[2])
        elseif gaussian !== nothing
            Normal(gaussian[1], gaussian[2])
        elseif uniform !== nothing
            Uniform(uniform[1], uniform[2])
        else
            error("ACT DR6 parameter $name has no prior")
        end
    end
    for name in ACT_DR6_FREE_PARAMETERS
)

"""
    act_dr6_model(like, foreground_model, cmb)

Turing model for the full ACT DR6 likelihood: the 29 official nuisance priors,
then the bandpower likelihood, all through `~`.

`cmb` is an [`ACTCMBTheory`](@ref) held fixed; sample over cosmology by passing a
different `cmb` per evaluation or by wrapping this model in a larger one.

```julia
like  = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
chain = sample(act_dr6_model(like, model, cmb), NUTS(), 1000)
```
"""
@model function act_dr6_model(like::ACTDR6FullLikelihood,
                              foreground_model::ACTDR6FullForegroundModel,
                              cmb::ACTCMBTheory)
    P = ACT_DR6_PRIOR_DISTRIBUTIONS
    # --- foregrounds (14) ---
    a_tSZ ~ P.a_tSZ
    alpha_tSZ ~ P.alpha_tSZ
    a_kSZ ~ P.a_kSZ
    a_p ~ P.a_p
    beta_p ~ P.beta_p
    a_c ~ P.a_c
    a_s ~ P.a_s
    beta_s ~ P.beta_s
    a_gtt ~ P.a_gtt
    a_gte ~ P.a_gte
    a_gee ~ P.a_gee
    a_psee ~ P.a_psee
    a_pste ~ P.a_pste
    xi ~ P.xi
    # --- calibration (10) ---
    calG_all ~ P.calG_all
    cal_dr6_pa4_f220 ~ P.cal_dr6_pa4_f220
    cal_dr6_pa5_f090 ~ P.cal_dr6_pa5_f090
    cal_dr6_pa5_f150 ~ P.cal_dr6_pa5_f150
    cal_dr6_pa6_f090 ~ P.cal_dr6_pa6_f090
    cal_dr6_pa6_f150 ~ P.cal_dr6_pa6_f150
    calE_dr6_pa5_f090 ~ P.calE_dr6_pa5_f090
    calE_dr6_pa5_f150 ~ P.calE_dr6_pa5_f150
    calE_dr6_pa6_f090 ~ P.calE_dr6_pa6_f090
    calE_dr6_pa6_f150 ~ P.calE_dr6_pa6_f150
    # --- bandpass shifts (5) ---
    bandint_shift_dr6_pa4_f220 ~ P.bandint_shift_dr6_pa4_f220
    bandint_shift_dr6_pa5_f090 ~ P.bandint_shift_dr6_pa5_f090
    bandint_shift_dr6_pa5_f150 ~ P.bandint_shift_dr6_pa5_f150
    bandint_shift_dr6_pa6_f090 ~ P.bandint_shift_dr6_pa6_f090
    bandint_shift_dr6_pa6_f150 ~ P.bandint_shift_dr6_pa6_f150

    # The documented vector constructor, not 29 positional arguments: a vector
    # literal promotes to a common element type, so the model keeps working when
    # the sampler supplies `Dual`s or reverse-mode tracers alongside `Float64`.
    nuisance = ACTDR6Nuisance([
        a_tSZ, alpha_tSZ, a_kSZ, a_p, beta_p, a_c, a_s, beta_s,
        a_gtt, a_gte, a_gee, a_psee, a_pste, xi,
        calG_all, cal_dr6_pa4_f220, cal_dr6_pa5_f090, cal_dr6_pa5_f150,
        cal_dr6_pa6_f090, cal_dr6_pa6_f150,
        calE_dr6_pa5_f090, calE_dr6_pa5_f150, calE_dr6_pa6_f090, calE_dr6_pa6_f150,
        bandint_shift_dr6_pa4_f220, bandint_shift_dr6_pa5_f090,
        bandint_shift_dr6_pa5_f150, bandint_shift_dr6_pa6_f090,
        bandint_shift_dr6_pa6_f150,
    ])
    foregrounds_total = foregrounds(foreground_model, like, nuisance)
    prediction = predict(like, cmb, foregrounds_total, nuisance)

    like.observed ~ ACTDR6Bandpowers(like, prediction)

    return nuisance
end

end # module
