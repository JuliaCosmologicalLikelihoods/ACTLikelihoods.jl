"""
    ACTLikelihoodsTuringExt

Turing.jl extension for the ACT DR6 multi-frequency likelihood.

This extension is automatically loaded when both `ACTLikelihoods` and `Turing`
(and `PDMats`) are loaded in the same Julia session. It provides two
`@model` variants:

- `act_dr6_model(cmb_dls; data, model)` — Variant A: uses `Turing.@addlogprob!`
  to inject `loglike(cmb_dls, fg, cal, data, model)` directly. The full
  Gaussian normalization (`data.logp_const`) is included by `loglike` itself.

- `act_dr6_model_canon(cmb_dls; data, model)` — Variant C: native data-form
  `data.data_vec ~ MvNormalCanon(theory, J·theory, J)`. Uses `data.inv_cov`
  directly as the precision matrix — no covariance inversion. The Cholesky
  of J is precomputed once via `PDMat`.

Both variants assume the standard ACT DR6 14-parameter foreground model
plus a single global calibration `calG_all`. All other foreground knobs
(`alpha_dT`, `alpha_dE`, `alpha_p`, `alpha_s`, `T_effd`, `beta_d`, `T_d`)
are left at their `compute_fg_totals` defaults — matching the ACT DR6
baseline configuration described in Beringue+2025 (arXiv:2506.06274).
"""
module ACTLikelihoodsTuringExt

using ACTLikelihoods
using Turing
using LinearAlgebra
using PDMats: PDMat


# ============================================================================
# Shared prior helper — ACT DR6 14 foreground params + global calibration.
# Matches Beringue+2025 (arXiv:2506.06274) Table 1 priors.
# ============================================================================

function _act_dr6_priors()
    return (
        # Hot-gas (Compton-y)
        a_tSZ      = truncated(Normal(3.4, 2.0), 0.0, 50.0),
        alpha_tSZ  = Uniform(-5.0, 5.0),
        a_kSZ      = truncated(Normal(1.5, 2.0), 0.0, 50.0),
        # CIB Poisson + clustered (β_c tied to β_p in ACT baseline)
        a_p        = truncated(Normal(6.0, 3.0), 0.0, 50.0),
        beta_p     = Uniform(0.0, 5.0),
        a_c        = truncated(Normal(4.0, 2.0), 0.0, 50.0),
        # tSZ × CIB correlation
        xi         = Uniform(0.0, 0.2),
        # Radio TT + spectral index (shared across TT/TE/EE)
        a_s        = truncated(Normal(3.0, 2.0), 0.0, 50.0),
        beta_s     = Uniform(-3.5, -1.5),
        # Galactic dust amplitudes (Gaussian Planck-353 priors)
        a_gtt      = truncated(Normal(8.0,  0.4),  0.0, 50.0),
        a_gte      = truncated(Normal(0.42, 0.03), -1.0, 1.0),
        a_gee      = truncated(Normal(0.168, 0.017), 0.0, 1.0),
        # Radio TE / EE
        a_pste     = Uniform(-1.0, 1.0),
        a_psee     = Uniform( 0.0, 1.0),
        # Global calibration (very tight Planck cross-cal)
        calG_all   = truncated(Normal(1.0, 0.003), 0.95, 1.05),
    )
end


# ============================================================================
# Submodel: priors only — flat-namespace NamedTuple of all 15 params.
# Mirrors `hillipop_nuisance_priors` from HillipopTuringExt.
# ============================================================================

@model function act_dr6_nuisance_priors()
    pr = _act_dr6_priors()

    a_tSZ     ~ pr.a_tSZ
    alpha_tSZ ~ pr.alpha_tSZ
    a_kSZ     ~ pr.a_kSZ
    a_p       ~ pr.a_p
    beta_p    ~ pr.beta_p
    a_c       ~ pr.a_c
    xi        ~ pr.xi
    a_s       ~ pr.a_s
    beta_s    ~ pr.beta_s
    a_gtt     ~ pr.a_gtt
    a_gte     ~ pr.a_gte
    a_gee     ~ pr.a_gee
    a_pste    ~ pr.a_pste
    a_psee    ~ pr.a_psee
    calG_all  ~ pr.calG_all

    return (; a_tSZ, alpha_tSZ, a_kSZ, a_p, beta_p, a_c, xi,
              a_s, beta_s, a_gtt, a_gte, a_gee, a_pste, a_psee, calG_all)
end


# ============================================================================
# Helpers: split flat NT into (fg, cal) tuples expected by `loglike`.
# ============================================================================

# 14 foreground parameter symbols passed to `compute_fg_totals`.
const _ACT_FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p, :a_c, :xi,
                       :a_s, :beta_s, :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)

# Global calibration symbol(s).
const _ACT_CAL_KEYS = (:calG_all,)

@inline _split_fg(nt::NamedTuple) = NamedTuple{_ACT_FG_KEYS}(map(k -> getproperty(nt, k), _ACT_FG_KEYS))
@inline _split_cal(nt::NamedTuple) = NamedTuple{_ACT_CAL_KEYS}(map(k -> getproperty(nt, k), _ACT_CAL_KEYS))


# ============================================================================
# Variant A — @addlogprob! (direct, low-allocation)
# ============================================================================

"""
    act_dr6_model(cmb_dls; data=ACT_DR6[].data, model=ACT_DR6[].model)

Return a Turing `@model` that places ACT DR6 priors over the 14 foreground
parameters + `calG_all`, and injects the ACT log-likelihood via
`Turing.@addlogprob!`. The full Gaussian normalization is included via
`data.logp_const` inside `loglike`.

`cmb_dls` must be a `Dict("tt" => ..., "te" => ..., "ee" => ...)` of
CMB D_ℓ vectors on the data ell-grid `data.l_bpws`.
"""
function act_dr6_model(cmb_dls::AbstractDict;
                       data::ACTData = ACTLikelihoods.ACT_DR6[].data,
                       model::ForegroundModel = ACTLikelihoods.ACT_DR6[].model)
    return _act_dr6_model_A(cmb_dls, data, model)
end

@model function _act_dr6_model_A(cmb_dls, data, model)
    nuis ~ to_submodel(act_dr6_nuisance_priors(), false)
    fg   = _split_fg(nuis)
    cal  = _split_cal(nuis)

    Turing.@addlogprob! loglike(cmb_dls, fg, cal, data, model)
end


# ============================================================================
# Variant C — data ~ MvNormalCanon (precision-matrix form, no inversion)
# ============================================================================

"""
    act_dr6_model_canon(cmb_dls; data=ACT_DR6[].data, model=ACT_DR6[].model)

Return a Turing `@model` that scores `data.data_vec ~ MvNormalCanon(theory, …)`
using the inverse covariance `data.inv_cov` directly as the precision matrix.

```
data.data_vec ~ MvNormalCanon(theory, J·theory, J)
```

This is mathematically equivalent to `data ~ MvNormal(theory, Σ)` but
avoids the `O(n³)` covariance inversion entirely. The Cholesky factorization
of `J = Σ⁻¹` is computed once at construction (via `PDMat`) and cached for
all subsequent likelihood evaluations.

Note: `Distributions.MvNormalCanon`'s 3-argument signature is `(μ, h, J)`
where `h = J·μ` is the *potential vector* — the 2-arg form
`MvNormalCanon(h, J)` treats the first argument as `h`, NOT as `μ`.
"""
function act_dr6_model_canon(cmb_dls::AbstractDict;
                              data::ACTData = ACTLikelihoods.ACT_DR6[].data,
                              model::ForegroundModel = ACTLikelihoods.ACT_DR6[].model)
    # Cholesky-factor the precision matrix once. Symmetric() guarantees the
    # PDMat constructor sees a numerically symmetric matrix even if rounding
    # has perturbed `inv_cov` away from exact symmetry.
    J_pd     = PDMat(Symmetric(data.inv_cov))
    data_vec = data.data_vec   # bind to a local for the @model macro
    return _act_dr6_model_C(cmb_dls, data, model, data_vec, J_pd)
end

@model function _act_dr6_model_C(cmb_dls, data, model, data_vec, J_pd)
    nuis ~ to_submodel(act_dr6_nuisance_priors(), false)
    fg   = _split_fg(nuis)
    cal  = _split_cal(nuis)

    fg_totals = compute_fg_totals(fg, model)
    theory    = theory_vector(cmb_dls, fg_totals, cal, data)

    h_pot = J_pd * theory
    data_vec ~ MvNormalCanon(theory, h_pot, J_pd)
end


# ============================================================================
# Debugging / verification utilities
# ============================================================================

# Evaluate the data-likelihood part of `model` after fixing the nuisance
# parameters at the values implied by `nt`.
function _act_dr6_eval_loglik(model, nt::NamedTuple)
    m_fixed = Turing.DynamicPPL.fix(model, nt)
    vi      = Turing.DynamicPPL.VarInfo(m_fixed)
    return Turing.DynamicPPL.getloglikelihood(vi)
end

"""
    act_dr6_loglike_check(nt, cmb_dls; data=ACT_DR6[].data, model=ACT_DR6[].model)

Compare `loglike(cmb_dls, fg, cal, data, model)` against the
`act_dr6_model` evaluation. Both paths include `data.logp_const`, so they
should match to machine precision.
"""
function act_dr6_loglike_check(nt::NamedTuple, cmb_dls::AbstractDict;
                                data::ACTData = ACTLikelihoods.ACT_DR6[].data,
                                model::ForegroundModel = ACTLikelihoods.ACT_DR6[].model)
    fg          = _split_fg(nt)
    cal         = _split_cal(nt)
    logL_direct = loglike(cmb_dls, fg, cal, data, model)

    m           = act_dr6_model(cmb_dls; data=data, model=model)
    logL_turing = _act_dr6_eval_loglik(m, nt)

    return (direct = logL_direct,
            turing = logL_turing,
            diff   = abs(logL_direct - logL_turing))
end

"""
    act_dr6_loglike_check_canon(nt, cmb_dls; data=ACT_DR6[].data, model=ACT_DR6[].model)

Compare `loglike(cmb_dls, fg, cal, data, model)` against the
`act_dr6_model_canon` evaluation. Both paths include the Gaussian
normalization `½ log|J| − ½ N log(2π) = data.logp_const`, so they should
match to machine precision.
"""
function act_dr6_loglike_check_canon(nt::NamedTuple, cmb_dls::AbstractDict;
                                       data::ACTData = ACTLikelihoods.ACT_DR6[].data,
                                       model::ForegroundModel = ACTLikelihoods.ACT_DR6[].model)
    fg          = _split_fg(nt)
    cal         = _split_cal(nt)
    logL_direct = loglike(cmb_dls, fg, cal, data, model)

    m           = act_dr6_model_canon(cmb_dls; data=data, model=model)
    logL_turing = _act_dr6_eval_loglik(m, nt)

    return (direct = logL_direct,
            turing = logL_turing,
            diff   = abs(logL_direct - logL_turing))
end

end # module ACTLikelihoodsTuringExt
