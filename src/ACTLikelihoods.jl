"""
    ACTLikelihoods

Native Julia implementation of the ACT DR6 CMB likelihoods.

# The primary likelihood: full multifrequency TT/TE/ET/EE

[`ACTDR6FullLikelihood`](@ref) is the authoritative ACT DR6 likelihood provided
by this package: the **full, non-marginalized** multifrequency likelihood with
1651 bandpowers over 41 ordered TT/TE/ET/EE spectra across five array-frequency
channels, with explicit foreground, calibration and bandpass-shift modelling.

Its runtime data come from an immutable published artifact (Zenodo
`10.5281/zenodo.22821597`), so the ordinary user path needs no local files:

```julia
like  = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
p     = ACTDR6Nuisance()
cmb   = ACTCMBTheory(ell, Dl_TT, Dl_TE, Dl_EE)

fg    = foregrounds(model, like, p)
pred  = predict(like, cmb, fg, p)
chi2(like, pred)            # 1592.0584129991541 at the reference point
loglikelihood(like, pred)   # data only: -chi2/2
logposterior(like, pred, p) # + official ACT DR6 nuisance priors
```

# The secondary variant: CMB-only, foreground-marginalized

[`ACTDR6Likelihood`](@ref) implements the **foreground-marginalized** ACT DR6
CMB-only likelihood. It is a different, compressed data product with its own
loose local data files; it is not the default, and its results are not evidence
about the full multifrequency likelihood.

# Conventions
- Input `Dls_CMB` are `D_ℓ = ℓ(ℓ+1)/(2π) C_ℓ` in μK²
- `loglikelihood` is data only, `-chi2/2`; see [`gaussian_normalization`](@ref)
- Ordered TE/ET map legs are preserved; spectra are never symmetrized

# References
See the README for the full citation list and the exact upstream provenance
revisions carried by [`act_dr6_provenance`](@ref).
"""
module ACTLikelihoods

using Artifacts
using LinearAlgebra
using DelimitedFiles
using CMBForegrounds
using JSON3
import SHA

# `loglikelihood` and `predict` are extended from StatsAPI rather than defined
# here. Distributions, StatsBase, DynamicPPL and Turing all extend that same
# binding, so defining rival ones made `using ACTLikelihoods, Turing` ambiguous
# for two of this package's main entry points — and broke the Turing extension
# from the inside. StatsAPI has no dependencies of its own.
import StatsAPI: loglikelihood, predict

include("ACT_artifact.jl")
include("ACT_data_loader.jl")
include("ACT_foreground_model.jl")
include("ACT_likelihood.jl")
include("ACT_full_likelihood.jl")
include("ACT_full_foregrounds.jl")
include("ACT_nuisance.jl")

# --- full multifrequency likelihood (primary) ------------------------------
export ACTDR6FullLikelihood, ACTDR6FullForegroundModel
export ACTCMBTheory, ACTForegrounds
export foregrounds, predict, chi2, loglikelihood, gaussian_normalization
export window_matrix

# --- nuisance parameters and priors ----------------------------------------
export ACTDR6Nuisance, parameter_vector, free_parameters
export logprior, prior_chi2, logposterior
export ACT_DR6_FREE_PARAMETERS, ACT_DR6_FIXED_PARAMETERS
export ACT_DR6_GAUSSIAN_PRIORS, ACT_DR6_UNIFORM_PRIORS
export ACT_DR6_REFERENCE_NUISANCE, act_dr6_fiducial_nuisance

# --- artifact and provenance -----------------------------------------------
export act_dr6_tttee_artifact_path, act_dr6_provenance
export validate_act_dr6_metadata, verify_act_dr6_checksums

# --- CMB-only, foreground-marginalized variant (secondary) -----------------
export ACTDR6Likelihood, ACTCMBOnlyFG, ACTCMBOnlyPars
export loglike, build_theory_bandpowers, effective_ells

# `bin_theory_vector`, `apply_calibration` and `compute_foreground_Dls` are
# deliberately *not* exported: `apply_calibration` collides with the
# CMBForegrounds export of the same name. Reach them as
# `ACTLikelihoods.apply_calibration`.

end # module ACTLikelihoods
