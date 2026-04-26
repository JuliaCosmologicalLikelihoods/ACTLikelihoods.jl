"""
    likelihood.jl

ACT DR6 multi-frequency CMB likelihood.
Mirrors LAT_MFLike/mflike/mflike.py :: _MFLike.

Data is loaded from the pre-extracted text files in
`CosmologicalLikelihoods/ACTLikelihoods.jl/data/ACT_DR6_TTTEEE_filtered/`.

Implements:
  - `SpecMeta`     — per-spectrum metadata + window matrix
  - `ACTData`      — full pre-loaded data struct
  - `load_data`    — reads text-file directory, returns ACTData
  - `theory_vector`— CMB + FG + calibration + binning
  - `loglike`      — full Gaussian log-likelihood

All functions are pure — no mutation — compatible with ForwardDiff and Mooncake.
"""

using LinearAlgebra: dot, cholesky, Hermitian, logdet, diag

# ------------------------------------------------------------------ #
# Spectrum metadata                                                    #
# ------------------------------------------------------------------ #

"""
    SpecMeta

Metadata + bandpower window for one observed cross-spectrum.

Fields:
- `pol`:        "tt", "te", or "ee"
- `t1`, `t2`:   experiment array names (e.g. "dr6_pa5_f090")
- `hasYX_xsp`:  true when this is an ET/EB/BE spectrum (transposed frequency pair)
- `ids`:        1-indexed positions in the data_vec for this spectrum's bins
- `W`:          bandpower window matrix, shape (n_ell, n_bins)
                Row k corresponds to ell = l_bpws[k].
                Binning: ps_vec = W' * theory_dl
"""
struct SpecMeta
    pol       :: String
    t1        :: String
    t2        :: String
    hasYX_xsp :: Bool
    ids       :: Vector{Int}
    W         :: Matrix{Float64}    # (n_ell, n_bins)
end

# ------------------------------------------------------------------ #
# ACTData struct                                                        #
# ------------------------------------------------------------------ #

"""
    ACTData

Pre-loaded ACT DR6 likelihood data.

Fields:
- `data_vec`:    observed bandpower data vector, length n_bins_total
- `inv_cov`:     inverse covariance matrix, shape (n_bins_total, n_bins_total)
- `logp_const`:  Gaussian normalisation: −N/2·log(2π) − ½·log(det Σ)
- `l_bpws`:      ell grid for the full theory vector, length n_ell
                 Starts at ℓ=2 (window row 0 ↔ ℓ=2).
- `spec_meta`:   Vector{SpecMeta}, one per observed cross-spectrum
- `experiments`: ordered list of unique array names  (e.g. ["dr6_pa4_f220", ...])
- `nu_0`:        reference frequency (150.0 GHz)
- `ell_0`:       reference multipole (3000)
"""
struct ACTData
    data_vec    :: Vector{Float64}
    inv_cov     :: Matrix{Float64}
    logp_const  :: Float64
    l_bpws      :: Vector{Int}
    spec_meta   :: Vector{SpecMeta}
    experiments :: Vector{String}
    exp_idx     :: Dict{String,Int}   # cached: experiment name → 1-based index in `experiments`
    nu_0        :: Float64
    ell_0       :: Int
end

# ------------------------------------------------------------------ #
# Data loading                                                          #
# ------------------------------------------------------------------ #

"""
    load_data(filtered_dir) → ACTData

Load the ACT DR6 likelihood data from a text-file directory.

Expected layout:
    filtered_dir/
      data_vec.txt       — 1-column, n_bins rows
      cov.txt            — n_bins × n_bins (for logp_const computation)
      inv_cov.txt        — n_bins × n_bins
      spec_meta.txt      — rows: "<name> <n_bins_spec>"
      windows/
        <name>.txt       — n_ell × n_bins_spec  (column = window weight per ell)

The ell grid runs from ℓ=2 to ℓ=2+n_ell−1.

The `windows` directory uses the same name as each row in spec_meta.txt.
Duplicate TE entries in spec_meta.txt indicate ET spectra: the second
occurrence of a (t1, t2, pol=TE) triple is marked `hasYX_xsp=true`.
"""
function load_data(filtered_dir::AbstractString)
    # 1. Data vector
    data_vec = Float64.(vec(readdlm(joinpath(filtered_dir, "data_vec.txt"))))

    # 2. Covariance → logp_const, then inv_cov
    cov     = Float64.(readdlm(joinpath(filtered_dir, "cov.txt")))
    C       = cholesky(Hermitian(cov))
    N       = length(data_vec)
    logp_const = -0.5 * N * log(2π) - 0.5 * logdet(C)

    inv_cov = Float64.(readdlm(joinpath(filtered_dir, "inv_cov.txt")))

    # 3. Parse spec_meta.txt
    sm_raw  = readdlm(joinpath(filtered_dir, "spec_meta.txt"), String)
    n_spec  = size(sm_raw, 1)

    # Track seen (t1, t2, pol) triples to detect ET (second-occurrence TE)
    seen = Dict{Tuple{String,String,String}, Int}()

    # Collect unique experiments in order of first appearance
    exp_set = String[]

    spec_meta  = SpecMeta[]
    id_offset  = 0

    for row in 1:n_spec
        raw_name = sm_raw[row, 1]
        n_bins   = parse(Int, sm_raw[row, 2])

        # Parse: "dr6_pa4_f220_x_dr6_pa5_f090_TT"
        parts = split(raw_name, "_x_"; limit=2)
        t1    = String(parts[1])

        # parts[2] = "dr6_pa5_f090_TT"
        rest_parts = split(parts[2], "_")
        pol        = lowercase(String(rest_parts[end]))
        t2         = join(rest_parts[1:end-1], "_")

        # Detect ET (second occurrence of same t1, t2, te triple)
        key        = (t1, t2, pol)
        count      = get(seen, key, 0) + 1
        seen[key]  = count
        hasYX_xsp  = (pol == "te") && (count == 2)

        # Load window matrix from file
        # Both TE and ET share the same window file name
        win_path = joinpath(filtered_dir, "windows", raw_name * ".txt")
        W        = Float64.(readdlm(win_path))   # (n_ell, n_bins_spec)

        ids = collect((id_offset + 1):(id_offset + n_bins))
        id_offset += n_bins

        # Collect unique experiments
        for exp in (t1, t2)
            if exp ∉ exp_set
                push!(exp_set, exp)
            end
        end

        push!(spec_meta, SpecMeta(pol, t1, t2, hasYX_xsp, ids, W))
    end

    # ell grid: window rows correspond to ell = 2, 3, ..., 2+n_ell-1
    n_ell  = size(spec_meta[1].W, 1)
    @assert(all(size(m.W, 1) == n_ell for m in spec_meta),
        "All window matrices must have the same number of ell rows; got $(unique(size(m.W,1) for m in spec_meta))")
    l_bpws = collect(2:(2 + n_ell - 1))

    exp_idx = Dict(exp => i for (i, exp) in enumerate(exp_set))

    return ACTData(
        data_vec, inv_cov, logp_const,
        l_bpws, spec_meta, exp_set, exp_idx,
        150.0, 3000,
    )
end

# ------------------------------------------------------------------ #
# Bandpass loading helper                                               #
# ------------------------------------------------------------------ #

"""
    load_bands(bandpass_dir, experiments) → (raw_T, raw_P)

Load temperature (s0) and polarization (s2) raw bandpasses from text files.

Each file is two-column: `nu [GHz]  bandpass_transmission`.
Returns `(raw_T, raw_P)` — both `Vector{RawBand{Float64}}` of length n_exp,
in the same order as `experiments`.

The bandpasses are intentionally left un-normalized so that
`bandint_shift_<exp>` parameters can be applied differentiably at runtime
via `shift_and_normalize`.
"""
function load_bands(bandpass_dir::AbstractString,
                    experiments::AbstractVector{<:AbstractString})
    raw_T = RawBand{Float64}[]
    raw_P = RawBand{Float64}[]
    for exp in experiments
        for (lst, suffix) in ((raw_T, "_s0"), (raw_P, "_s2"))
            path = joinpath(bandpass_dir, exp * suffix * ".txt")
            bp_data = Float64.(readdlm(path))
            push!(lst, RawBand{Float64}(bp_data[:, 1], bp_data[:, 2]))
        end
    end
    return raw_T, raw_P
end

# ------------------------------------------------------------------ #
# Calibration                                                           #
# ------------------------------------------------------------------ #

"""
    calibration_factors(cal, experiments) → (cal_t, cal_e)

Compute per-experiment effective calibration multipliers.

In mflike, D_th is divided by the calibration product.  Here we return
1/cal so that `theory_dl = (cmb + fg) * cf` with `cf = cal_t[ν₁] * cal_t[ν₂]`.

`cal` may be a NamedTuple or Dict with any subset of:
  - `calG_all`      : global calibration (default 1.0)
  - `cal_<exp>`     : per-array relative calibration
  - `calE_<exp>`    : E-mode polarization efficiency
"""
function calibration_factors(cal, experiments::AbstractVector{<:AbstractString})
    cal_t_vec, cal_e_vec = _calibration_vectors(cal, experiments)
    cal_t = Dict(exp => cal_t_vec[i] for (i, exp) in enumerate(experiments))
    cal_e = Dict(exp => cal_e_vec[i] for (i, exp) in enumerate(experiments))
    return cal_t, cal_e
end

function _calibration_vectors(cal, experiments::AbstractVector{<:AbstractString})
    calG = inv(_get_cal(cal, :calG_all, 1.0))
    CalT = typeof(calG)
    one_cal = one(calG)

    n_exp = length(experiments)
    cal_t_vec = Vector{CalT}(undef, n_exp)
    cal_e_vec = Vector{CalT}(undef, n_exp)

    @inbounds for i in eachindex(experiments)
        exp = experiments[i]
        cT = convert(CalT, _get_cal(cal, Symbol("cal_" * exp), one_cal))
        cE = convert(CalT, _get_cal(cal, Symbol("calE_" * exp), one_cal))
        cal_t_vec[i] = calG / cT
        cal_e_vec[i] = cal_t_vec[i] / cE
    end

    return cal_t_vec, cal_e_vec
end

_get_cal(cal::NamedTuple, key::Symbol, default) =
    hasproperty(cal, key) ? getproperty(cal, key) : default
_get_cal(cal::AbstractDict, key::Symbol, default) =
    get(cal, key, get(cal, String(key), default))

# ------------------------------------------------------------------ #
# Theory vector                                                         #
# ------------------------------------------------------------------ #

"""
    theory_vector(cmb_dls, fg_totals, cal, data) → Vector

Assemble the full theory bandpower vector:
1. CMB + foreground D_ℓ for each frequency pair
2. Apply calibration
3. Bin with bandpower window matrices (W' * theory_dl)

Arguments:
- `cmb_dls`:   Dict "tt"/"te"/"ee" → D_ℓ Vector of length n_ell (starts at ℓ=2)
- `fg_totals`: Tuple (fg_TT, fg_TE, fg_EE), each (n_exp, n_exp, n_ell)
- `cal`:       NamedTuple or Dict of calibration params
- `data`:      ACTData

Returns a Vector{T} of length n_bins_total.
"""
function theory_vector(
    cmb_dls   :: AbstractDict{<:AbstractString, <:AbstractVector{<:Number}},
    fg_totals :: Tuple{<:AbstractArray{<:Number,3}, <:AbstractArray{<:Number,3}, <:AbstractArray{<:Number,3}},
    cal,
    data      :: ACTData,
)
    fg_TT, fg_TE, fg_EE = fg_totals
    experiments         = data.experiments

    # Per-experiment calibration vectors (gradient flows through here).
    # 1/cal convention matches calibration_factors().
    cal_t_vec, cal_e_vec = _calibration_vectors(cal, experiments)

    return theory_vector_core(
        cmb_dls["tt"], cmb_dls["te"], cmb_dls["ee"],
        fg_TT, fg_TE, fg_EE,
        cal_t_vec, cal_e_vec,
        data,
    )
end

"""
    theory_vector_core(cmb_tt, cmb_te, cmb_ee,
                       fg_TT, fg_TE, fg_EE,
                       cal_t_vec, cal_e_vec, data) → Vector

All-positional core with a hand-written ChainRulesCore.rrule (see rrules.jl).
Calibration factors are passed as Vectors indexed by `data.experiments`.
"""
function theory_vector_core(
    cmb_tt    :: AbstractVector{<:Number},
    cmb_te    :: AbstractVector{<:Number},
    cmb_ee    :: AbstractVector{<:Number},
    fg_TT     :: AbstractArray{<:Number,3},
    fg_TE     :: AbstractArray{<:Number,3},
    fg_EE     :: AbstractArray{<:Number,3},
    cal_t_vec :: AbstractVector{<:Number},
    cal_e_vec :: AbstractVector{<:Number},
    data      :: ACTData,
)
    exp_idx = data.exp_idx

    ps_parts = map(data.spec_meta) do m
        i = exp_idx[m.t1]
        j = exp_idx[m.t2]
        # ET: swap frequency indices for fg lookup (same window, different fg column)
        fi = m.hasYX_xsp ? j : i
        fj = m.hasYX_xsp ? i : j

        if m.pol == "tt"
            dl = cmb_tt .+ fg_TT[fi, fj, :]
            cf = cal_t_vec[i] * cal_t_vec[j]
        elseif m.pol == "te"
            dl = cmb_te .+ fg_TE[fi, fj, :]
            cf = m.hasYX_xsp ? cal_e_vec[i] * cal_t_vec[j] :
                                cal_t_vec[i] * cal_e_vec[j]
        else   # ee
            dl = cmb_ee .+ fg_EE[fi, fj, :]
            cf = cal_e_vec[i] * cal_e_vec[j]
        end

        m.W' * (dl .* cf)   # (n_bins_spec,)
    end

    return reduce(vcat, ps_parts)
end

# ------------------------------------------------------------------ #
# Log-likelihood                                                        #
# ------------------------------------------------------------------ #

"""
    loglike(cmb_dls, fg_params, cal, data, model) → Float64

Full ACT DR6 Gaussian log-likelihood:

    ln ℒ = −½ (D_th − D_data)ᵀ Σ⁻¹ (D_th − D_data) + const

Arguments:
- `cmb_dls`:   Dict "tt"/"te"/"ee" → D_ℓ Vector on ell grid `data.l_bpws`
- `fg_params`: foreground parameters; see `compute_fg_totals`
- `cal`:       calibration parameters
- `data`:      ACTData
- `model`:     ForegroundModel
"""
function loglike(
    cmb_dls   :: AbstractDict,
    fg_params,
    cal,
    data      :: ACTData,
    model     :: ForegroundModel,
)
    fg_totals = compute_fg_totals(fg_params, model)
    ps_vec    = theory_vector(cmb_dls, fg_totals, cal, data)
    δ         = data.data_vec .- ps_vec
    chi2      = dot(δ, data.inv_cov * δ)
    return -0.5 * chi2 + data.logp_const
end
