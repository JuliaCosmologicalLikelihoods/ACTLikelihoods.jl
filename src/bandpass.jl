"""
    bandpass.jl

Bandpass normalization and frequency-integrated SED evaluation.
Mirrors the `_bandpass_construction` and `eval_bandpass` logic in
LAT_MFLike/mflike/foreground.py and fgspectra/frequency.py.

All functions are pure (no mutation), compatible with ForwardDiff and Mooncake.
"""

# ------------------------------------------------------------------ #
# Trapezoid integration                                                #
# Written explicitly so AD can differentiate through it               #
# ------------------------------------------------------------------ #

"""
    trapz(x, y)

Trapezoidal integration of `y` over `x`.
Both must be AbstractVector of the same length.
Pure functional — differentiable via ForwardDiff and Mooncake.
"""
function trapz(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "trapz: x and y must have the same length"
    s = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for i in 1:(length(x)-1)
        s += (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s / 2
end

# ------------------------------------------------------------------ #
# RawBand — un-normalized passband for bandpass-shift support          #
# ------------------------------------------------------------------ #

"""
    RawBand{T<:Real}

Stores the raw (un-normalized) frequency grid and passband transmission
for one detector array.  Used when bandpass shift parameters are varied,
so that normalization is recomputed at runtime (differentiably).

Fields:
- `nu`:  frequency grid [GHz]
- `bp`:  raw transmission (proportional to τ(ν)/ν², RJ convention)
"""
struct RawBand{T<:Real}
    nu :: Vector{T}
    bp :: Vector{T}
end

"""
    shift_and_normalize(raw, shift) → Band

Apply a frequency shift `shift` [GHz] to `raw.nu` and return a fully
normalized `Band`.  Differentiable w.r.t. `shift` through `cmb2bb` and `trapz`.
"""
function shift_and_normalize(raw::RawBand, shift::S) where {S<:Real}
    T   = promote_type(eltype(raw.nu), S)
    nu_s = raw.nu .+ shift
    make_band(convert(Vector{T}, nu_s), convert(Vector{T}, raw.bp))
end

# ------------------------------------------------------------------ #
# Band struct — holds a normalized passband for one experiment          #
# ------------------------------------------------------------------ #

"""
    Band

Holds the frequency array and normalized transmission for one experiment/channel.

Fields:
- `nu`:       frequency grid [GHz], length n_freq
- `norm_bp`:  normalized transmission τ̃(ν) = bp·∂B/∂T / ∫ bp·∂B/∂T dν
              (or `nothing` for a Dirac-delta / monochromatic channel)
- `nu_eff`:   effective (central) frequency [GHz], used when norm_bp is nothing

When `norm_bp` is not nothing, SEDs are integrated over the band.
When `norm_bp` is nothing, the SED is evaluated at `nu_eff`.
"""
struct Band{T<:Real}
    nu      :: Vector{T}      # frequency grid
    norm_bp :: Vector{T}      # normalized transmission (same length as nu)
    nu_eff  :: T              # effective frequency (weighted center)
    monofreq :: Bool          # true if Dirac-delta (single-sample band)
end

"""
    make_band(nu, bp)

Construct a `Band` from raw frequency grid `nu` (GHz) and passband
transmission `bp` (proportional to τ(ν)/ν²; already in the RJ convention
as stored in the SACC file).

The normalization factor is:
    τ̃(ν) = bp(ν)·∂B_ν/∂T / ∫ bp(ν)·∂B_ν/∂T dν

where ∂B/∂T ∝ cmb2bb(ν) (see frequency.jl).
"""
function make_band(nu::AbstractVector{T}, bp::AbstractVector{T}) where T<:Real
    if length(nu) == 1
        # Monochromatic: Dirac-delta passband, no integration
        return Band{T}(Vector{T}(nu), Vector{T}(bp), nu[1], true)
    end
    w     = bp .* cmb2bb.(nu)
    norm  = trapz(nu, w)
    norm_bp = w ./ norm
    nu_eff  = nu[argmax(bp)]   # approximate center; exact value only needed for display
    return Band{T}(Vector{T}(nu), norm_bp, T(nu_eff), false)
end

# ------------------------------------------------------------------ #
# SED evaluation over a band                                           #
# ------------------------------------------------------------------ #

"""
    integrate_sed(sed_fn, band)

Integrate a scalar SED function `sed_fn(ν)` over the normalized bandpass,
returning a single effective SED value:

    ∫ SED(ν) · τ̃(ν) dν   (trapezoidal)

For monochromatic bands, returns `sed_fn(band.nu_eff)` directly.
`sed_fn` must accept a scalar Float.
"""
function integrate_sed(sed_fn, band::Band{T}) where {T<:Real}
    if band.monofreq
        νmono::T = band.nu[1]
        return sed_fn(νmono)
    end

    ν1::T = band.nu[1]
    y1 = sed_fn(ν1) * band.norm_bp[1]
    y = Vector{typeof(y1)}(undef, length(band.nu))
    y[1] = y1
    @inbounds for i in 2:length(band.nu)
        ν::T = band.nu[i]
        y[i] = sed_fn(ν) * band.norm_bp[i]
    end
    return trapz(band.nu, y)
end

"""
    integrate_tsz(band, nu_0)

Integrate the normalized tSZ SED over a band.
This avoids higher-order closures in hot AD/JET paths.
"""
function integrate_tsz(band::Band{T}, nu_0::S) where {T<:Real,S<:Real}
    if band.monofreq
        return tsz_sed(band.nu[1], nu_0)
    end
    y = tsz_sed(band.nu, nu_0) .* band.norm_bp
    return trapz(band.nu, y)
end

"""
    eval_sed_bands(sed_fn, bands)

Evaluate a SED over an array of `Band`s, returning a vector of length
`n_exp` with one integrated SED value per experiment.

`sed_fn` is a function ν → SED(ν) (scalar → scalar).
"""
function eval_sed_bands(sed_fn, bands::AbstractVector{Band{T}}) where {T<:Real}
    n = length(bands)
    @assert n > 0 "eval_sed_bands: empty band collection"

    b1::Band{T} = bands[1]
    v1 = integrate_sed(sed_fn, b1)
    vals = Vector{typeof(v1)}(undef, n)
    vals[1] = v1
    @inbounds for i in 2:n
        b::Band{T} = bands[i]
        vals[i] = integrate_sed(sed_fn, b)
    end
    return vals
end
