"""
    foreground.jl

Full ACT DR6 foreground model.
Mirrors LAT_MFLike/mflike/foreground.py :: _get_foreground_model_arrays.

Produces D_ℓ arrays for each spectrum (TT, TE, EE) and each frequency
pair (n_exp × n_exp), summed over all foreground components.

All functions are pure — no mutation — compatible with ForwardDiff and Mooncake.
"""

# ------------------------------------------------------------------ #
# ForegroundModel struct — holds precomputed fixed quantities           #
# ------------------------------------------------------------------ #

"""
    ForegroundModel

Pre-loaded templates and raw bandpasses for the ACT DR6 foreground model.

Construct with `ForegroundModel(experiments, raw_T, raw_P, ell)` — templates
are loaded automatically from the fgspectra data directory.

The raw (un-normalized) bandpasses are stored so that bandpass shift
parameters (`bandint_shift_<exp>`) can be applied differentiably at
runtime inside `compute_fg_totals`.

Fields:
- `ell`:         multipole array (Integer vector)
- `ell_0`:       reference multipole = 3000
- `nu_0`:        reference frequency = 150.0 GHz
- `experiments`: experiment names, one per bandpass entry
- `raw_T`:       Vector{RawBand}, length n_exp — raw temperature bandpasses
- `raw_P`:       Vector{RawBand}, length n_exp — raw polarization bandpasses
- `T_tsz`:       tSZ template (Battaglia+2010)
- `T_ksz`:       kSZ template (Battaglia+2013)
- `T_cibc`:      CIB clustered template (Choi+2021)
- `T_szxcib`:    tSZ×CIB cross template (Addison+2012)
"""
struct ForegroundModel{T<:Real}
    ell         :: Vector{Int}
    ell_0       :: Int
    nu_0        :: T
    experiments :: Vector{String}
    raw_T       :: Vector{RawBand{T}}
    raw_P       :: Vector{RawBand{T}}
    # Templates (D_ℓ arrays, 1-indexed as template[ℓ+1])
    T_tsz    :: Vector{Float64}
    T_ksz    :: Vector{Float64}
    T_cibc   :: Vector{Float64}
    T_szxcib :: Vector{Float64}
end

"""
    ForegroundModel(experiments, raw_T, raw_P, ell; ell_0=3000, nu_0=150.0)

Construct a `ForegroundModel` from experiment names and raw (un-normalized) bandpasses.
Templates are loaded from the fgspectra data directory.
"""
function ForegroundModel(
    experiments :: AbstractVector{<:AbstractString},
    raw_T       :: Vector{RawBand{T}},
    raw_P       :: Vector{RawBand{T}},
    ell         :: AbstractVector{<:Integer};
    ell_0::Integer = 3000,
    nu_0::Real     = 150.0,
) where T<:Real
    ForegroundModel{T}(
        Vector{Int}(ell),
        Int(ell_0),
        T(nu_0),
        collect(String, experiments),
        raw_T,
        raw_P,
        load_tsz_template(),
        load_ksz_template(),
        load_cibc_template(),
        load_szxcib_template(),
    )
end

# ------------------------------------------------------------------ #
# Foreground parameter helper                                           #
# ------------------------------------------------------------------ #

"""
    fg_param(p, key, default)

Get a foreground parameter value, with a fallback default.
`p` can be a NamedTuple, Dict, or any indexable type.
"""
fg_param(p::NamedTuple, key::Symbol, default) =
    hasproperty(p, key) ? getproperty(p, key) : default

fg_param(p::AbstractDict, key::Symbol, default) =
    get(p, key, get(p, String(key), default))

# ------------------------------------------------------------------ #
# Main foreground computation                                           #
# ------------------------------------------------------------------ #

"""
    compute_fg_totals(p, model) → (fg_TT, fg_TE, fg_EE)

Compute total foreground D_ℓ for all three spectra.

`p` must be a NamedTuple with the following fields:

**TT:**
  a_kSZ, a_tSZ, alpha_tSZ, a_p, beta_p, T_d,
  a_c, xi, a_s, beta_s, a_gtt

**TE:**
  a_pste, beta_s, a_gte

**EE:**
  a_psee, beta_s, a_gee

**Fixed (read from p with defaults):**
  alpha_dT=-0.6, alpha_dE=-0.4, alpha_p=1.0, alpha_s=1.0,
  T_effd=19.6, beta_d=1.5,
  beta_c=beta_p   # tied to beta_p in ACT DR6 baseline (act_dr6_example.yml)
                  # — pass `beta_c` explicitly only for the extension model

Returns three arrays of shape (n_exp, n_exp, n_ell):
  `fg_TT`, `fg_TE`, `fg_EE`

## Implementation note

Uses the fused per-spectrum assemblers (`assemble_TT/EE/TE` from
CMBForegrounds), which collapse the full spectrum into a single
Mooncake tape entry per spectrum — essential for the 15-parameter
Mooncake gradient to stay at ~26 ms.
"""
function compute_fg_totals(p::NamedTuple, model::ForegroundModel{T}) where {T<:Real}
    ell     = model.ell
    ell_0   = model.ell_0
    nu_0    = model.nu_0

    # Fixed parameters with defaults
    alpha_dT = fg_param(p, :alpha_dT, -0.6)
    alpha_dE = fg_param(p, :alpha_dE, -0.4)
    alpha_p  = fg_param(p, :alpha_p,   1.0)
    alpha_s  = fg_param(p, :alpha_s,   1.0)
    T_effd   = fg_param(p, :T_effd,   19.6)
    beta_d   = fg_param(p, :beta_d,    1.5)
    T_d      = fg_param(p, :T_d,       9.6)

    # ℓ-grid variants for Poisson/radio (use ℓ(ℓ+1) normalization)
    ell_clp   = @. ell * (ell + 1)
    ell_0clp  = ell_0 * (ell_0 + 1)

    # ---- Apply bandpass shifts (differentiable) ---- #
    # bandint_shift_<exp> shifts the frequency grid before renormalization.
    # Use a typed zero fallback so AD paths don't widen to Union{Dual,Float64}.
    # `oftype(shift0, …)` is preferred over `convert(typeof(shift0), …)`
    # because Julia 1.10's inference propagates the concrete type through
    # `oftype` but not through a `typeof` local binding.
    shift0 = zero(p.a_kSZ)
    bands_T = [shift_and_normalize(r, oftype(shift0, fg_param(p, Symbol("bandint_shift_" * exp), shift0)))
               for (r, exp) in zip(model.raw_T, model.experiments)]
    bands_P = [shift_and_normalize(r, oftype(shift0, fg_param(p, Symbol("bandint_shift_" * exp), shift0)))
               for (r, exp) in zip(model.raw_P, model.experiments)]

    # ---- Frequency SED integrals ---- #
    # Each produces a Vector of length n_exp
    eval_sed_typed(f, bands::AbstractVector{Band{BT}}) where {BT<:Real} =
        eval_sed_bands((ν::BT) -> f(ν), bands)

    f_ksz_T  = eval_sed_typed(ν -> constant_sed(ν),                  bands_T)
    f_tsz_T  = eval_sed_typed(ν -> tsz_sed(ν, nu_0),                  bands_T)
    # ACT DR6 baseline ties beta_c = beta_p (act_dr6_example.yml: beta_c: lambda beta_p: beta_p)
    # Allow explicit override via :beta_c only for the (paper-internal) extension model.
    beta_c   = fg_param(p, :beta_c, p.beta_p)
    f_cibp_T = eval_sed_typed(ν -> mbb_sed(ν, nu_0, p.beta_p, T_d),   bands_T)
    f_cibc_T = eval_sed_typed(ν -> mbb_sed(ν, nu_0, beta_c,   T_d),   bands_T)
    f_radio_T = eval_sed_typed(ν -> radio_sed(ν, nu_0, p.beta_s),      bands_T)
    f_dust_T  = eval_sed_typed(ν -> mbb_sed(ν, nu_0, beta_d, T_effd),  bands_T)

    f_radio_P = eval_sed_typed(ν -> radio_sed(ν, nu_0, p.beta_s),      bands_P)
    f_dust_P  = eval_sed_typed(ν -> mbb_sed(ν, nu_0, beta_d, T_effd),  bands_P)

    # ---- ℓ-templates ---- #

    cl_ksz   = ksz_template_scaled(eval_template(model.T_ksz, ell, ell_0), p.a_kSZ)
    cl_tsz   = eval_template_tilt(model.T_tsz, ell, ell_0, p.alpha_tSZ; amp=p.a_tSZ)
    cl_cibc  = eval_template(model.T_cibc, ell, ell_0; amp=p.a_c)
    cl_szxcib = eval_template(model.T_szxcib, ell, ell_0;
                               amp=-p.xi * sqrt(p.a_tSZ * p.a_c))

    cl_cibp  = eval_powerlaw(Float64.(ell_clp), Float64(ell_0clp), alpha_p)
    cl_radio = eval_powerlaw(Float64.(ell_clp), Float64(ell_0clp), alpha_s)
    cl_dustT = eval_powerlaw(Float64.(ell),     500.0,             alpha_dT)
    cl_dustE = eval_powerlaw(Float64.(ell),     500.0,             alpha_dE)

    # ---- Fused assemblers (single rrule per spectrum) ---- #
    # See CMBForegrounds/cross.jl :: assemble_TT/EE/TE.  The fused form
    # replaces a chain of broadcast `α .* X .+ Y` over (n_exp, n_exp, n_ell)
    # arrays, which under Mooncake would generate ~15k tape entries per
    # spectrum (4× `.*` + 4× `.+` × 2 spectra × n_freq² × n_ell elements).
    # The fused rrule collapses this to one tape entry per spectrum.

    fg_TT = assemble_TT(p.a_p, p.a_gtt, p.a_s,
                        f_ksz_T,  f_cibp_T, f_dust_T, f_radio_T,
                        f_tsz_T,  f_cibc_T,
                        cl_ksz,   cl_cibp,  cl_dustT, cl_radio,
                        cl_tsz,   cl_cibc,  cl_szxcib)

    fg_EE = assemble_EE(p.a_psee, p.a_gee,
                        f_radio_P, f_dust_P,
                        cl_radio,  cl_dustE)

    fg_TE = assemble_TE(p.a_pste, p.a_gte,
                        f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                        cl_radio,  cl_dustE)

    return fg_TT, fg_TE, fg_EE
end
