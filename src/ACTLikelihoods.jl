"""
    ACTLikelihoods

Julia implementation of the ACT DR6 multi-frequency CMB likelihood.

Faithfully mirrors the Python `mflike` + `fgspectra` stack, designed for
use with automatic differentiation (ForwardDiff, Mooncake).

## Quick start

Data is downloaded automatically on first use via the `act_dr6_data` Julia
artifact (Zenodo). The package exposes a global ACT DR6 dataset (`ACT_DR6[]`)
populated at module load, and a 3-argument `loglike` convenience method that
uses it.

```julia
using ACTLikelihoods

# CMB theory D_ℓ from your Boltzmann/emulator code (length matches data.l_bpws)
cmb_dls = Dict("tt" => tt, "te" => te, "ee" => ee)

# Foreground + calibration parameters
fg = (a_tSZ=3.5, alpha_tSZ=-0.5, a_kSZ=1.5, a_p=6.0, beta_p=2.2,
      T_d=9.6, a_c=4.0, xi=0.12, a_s=3.0, beta_s=-2.5,
      a_gtt=8.0, a_gte=0.42, a_gee=0.17, a_pste=0.0, a_psee=0.05)
# (ACT DR6 baseline ties beta_c = beta_p; pass `beta_c` only for the extension model)

cal = (calG_all=1.0,)

# Evaluate log-likelihood — uses ACT_DR6[] under the hood
ll = loglike(cmb_dls, fg, cal)

# Or explicit form (useful for custom datasets):
data  = load_data(act_dr6_filtered_dir())
raw_T, raw_P = load_bands(act_dr6_bandpass_dir(), data.experiments)
model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)
ll    = loglike(cmb_dls, fg, cal, data, model)
```
"""
module ACTLikelihoods

using LinearAlgebra
using DelimitedFiles: readdlm
using ChainRulesCore
using CMBForegrounds: ksz_template_scaled, eval_template, eval_template_tilt, eval_powerlaw
# Bandpass machinery now lives in CMBForegrounds (Step 2 of the unification plan).
# ACT keeps its own SED definitions (frequency.jl) for hot-path performance, but
# re-uses the universal Band/RawBand types and integration helpers.
using CMBForegrounds: trapz, RawBand, Band, make_band, point_band,
                      shift_and_normalize, integrate_sed, integrate_tsz,
                      eval_sed_bands
# Cross-spectrum assembly + fused TT/EE/TE assemblers now live in CMBForegrounds
# (Step 4). Their ChainRulesCore rrules + Mooncake @from_chainrules registrations
# also live there (in CMBForegroundsMooncakeExt).
using CMBForegrounds: factorized_cross, factorized_cross_te, correlated_cross,
                      build_szxcib_cl, assemble_TT, assemble_EE, assemble_TE

# ------------------------------------------------------------------ #
# Sub-modules (order matters — each builds on the previous)            #
# ------------------------------------------------------------------ #

include("artifacts.jl")    # Zenodo-backed data artifact (data, cov, bandpasses, templates)
include("frequency.jl")    # SED functions (local copies for hot-path performance)
include("power.jl")        # Template loading and ℓ-template evaluation
include("foreground.jl")   # Full foreground model (ForegroundModel, compute_fg_totals)
include("likelihood.jl")   # ACTData, load_data, theory_vector, loglike

# ------------------------------------------------------------------ #
# Custom rrule for Mooncake performance                                #
# ------------------------------------------------------------------ #
# Cross-spectrum kernel rrules (factorized_cross, factorized_cross_te,
# correlated_cross, assemble_TT/EE/TE) live in CMBForegrounds and are
# registered with Mooncake by the CMBForegroundsMooncakeExt extension.
# Only the ACT-specific theory_vector_core rrule (depends on ACTData)
# remains here.
include("rrules.jl")

using Mooncake: @from_chainrules, MinimalCtx
@from_chainrules MinimalCtx Tuple{typeof(theory_vector_core),
    Vector{Float64}, Vector{Float64}, Vector{Float64},
    Array{Float64,3}, Array{Float64,3}, Array{Float64,3},
    Vector{Float64}, Vector{Float64},
    ACTData}

# ------------------------------------------------------------------ #
# Public API                                                            #
# ------------------------------------------------------------------ #

# frequency.jl
export T_CMB_K, H_OVER_KT
export rj2cmb, cmb2bb, x_cmb
export tsz_sed, mbb_sed, radio_sed, constant_sed

# bandpass machinery (re-exported from CMBForegrounds)
export trapz, RawBand, shift_and_normalize, Band, make_band, point_band,
       integrate_sed, integrate_tsz, eval_sed_bands

# power.jl
export load_template
export load_tsz_template, load_ksz_template, load_cibc_template, load_szxcib_template
export eval_template, eval_template_tilt, eval_powerlaw

# cross.jl
export factorized_cross, factorized_cross_te, correlated_cross, build_szxcib_cl
export assemble_TT, assemble_EE, assemble_TE

# foreground.jl
export ForegroundModel, compute_fg_totals

# likelihood.jl
export SpecMeta, ACTData, load_data, load_bands, theory_vector, theory_vector_core, loglike

# artifacts.jl
export act_dr6_artifact_root, act_dr6_filtered_dir, act_dr6_bandpass_dir, act_dr6_template_dir
export ACTDR6Dataset, ACT_DR6

# ------------------------------------------------------------------ #
# Module-level state populated at load time                            #
# ------------------------------------------------------------------ #
#
# `__init__` is called once per Julia session, after the precompiled
# image is loaded. It is the correct place to do anything that depends
# on per-machine paths — including triggering download/extraction of
# the lazy `act_dr6_data` Zenodo artifact.
#
# We use a single wrapper struct (`ACTDR6Dataset`) holding all four
# data structs (likelihood data, T-bandpasses, P-bandpasses, foreground
# model), exposed via one `Ref` populated by `__init__`.

"""
    ACTDR6Dataset

Bundle of all ACT DR6 likelihood inputs that live in the Zenodo artifact:

- `data`  — `ACTData` (data vector, inverse covariance, windows, spec metadata)
- `raw_T` — temperature `RawBand` per experiment array
- `raw_P` — polarization `RawBand` per experiment array
- `model` — default `ForegroundModel{Float64}` built from `raw_T`, `raw_P`,
            `data.l_bpws`, with templates loaded from the artifact

Populated at module load time by `__init__` and accessed via the global
`ACT_DR6[]`. Used as the default arguments to the convenience method
`loglike(cmb_dls, fg_params, cal)`.
"""
struct ACTDR6Dataset
    data  :: ACTData
    raw_T :: Vector{RawBand{Float64}}
    raw_P :: Vector{RawBand{Float64}}
    model :: ForegroundModel{Float64}
end

const ACT_DR6 = Ref{ACTDR6Dataset}()
const ACT_DR6_INIT_ERROR = Ref{Any}(nothing)

function __init__()
    try
        data = load_data(act_dr6_filtered_dir())
        raw_T, raw_P = load_bands(act_dr6_bandpass_dir(), data.experiments)
        model = ForegroundModel(
            data.experiments,
            raw_T,
            raw_P,
            data.l_bpws;
            ell_0 = data.ell_0,
            nu_0  = data.nu_0,
        )
        ACT_DR6[] = ACTDR6Dataset(data, raw_T, raw_P, model)
    catch err
        ACT_DR6_INIT_ERROR[] = err
        @warn "ACTLikelihoods: failed to load default ACT DR6 dataset at module init. \
               The 3-argument `loglike(cmb_dls, fg, cal)` form will be unavailable. \
               Use the explicit 5-argument form, or call `ACTLikelihoods.reload_act_dr6!()` \
               after fixing the underlying issue." exception=(err, catch_backtrace())
    end
    return nothing
end

"""
    reload_act_dr6!() → ACTDR6Dataset

Re-attempt loading the default ACT DR6 dataset into `ACT_DR6[]`. Useful after
a failed `__init__` (e.g. artifact not yet downloaded, network issue at first
import). Throws if loading still fails.
"""
function reload_act_dr6!()
    data = load_data(act_dr6_filtered_dir())
    raw_T, raw_P = load_bands(act_dr6_bandpass_dir(), data.experiments)
    model = ForegroundModel(
        data.experiments, raw_T, raw_P, data.l_bpws;
        ell_0 = data.ell_0, nu_0 = data.nu_0,
    )
    ACT_DR6[] = ACTDR6Dataset(data, raw_T, raw_P, model)
    ACT_DR6_INIT_ERROR[] = nothing
    return ACT_DR6[]
end

# Convenience method: resolve the (data, model) defaults from ACT_DR6[].
# Lets users call `loglike(cmb_dls, fg_params, cal)` without explicitly
# passing the structs. Emits a clear error if the default dataset failed
# to load at module init (instead of an opaque `UndefRefError`).
function loglike(cmb_dls, fg_params, cal)
    if !isassigned(ACT_DR6)
        err = ACT_DR6_INIT_ERROR[]
        msg = "ACTLikelihoods.ACT_DR6[] is not initialized — the default ACT DR6 \
               dataset failed to load at package init. Use the 5-argument form \
               `loglike(cmb_dls, fg, cal, data, model)`, or call \
               `ACTLikelihoods.reload_act_dr6!()` after fixing the underlying issue."
        err === nothing ? error(msg) : error(msg * "\nOriginal init error: " * sprint(showerror, err))
    end
    return loglike(cmb_dls, fg_params, cal, ACT_DR6[].data, ACT_DR6[].model)
end

end # module ACTLikelihoods
