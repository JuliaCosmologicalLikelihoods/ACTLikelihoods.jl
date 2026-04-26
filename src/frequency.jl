"""
    frequency.jl

Frequency-dependent foreground SED functions.
Mirrors fgspectra/frequency.py.

All SEDs return a scalar or vector of the same shape as `nu`.
All are normalized so that SED(nu_0, nu_0, ...) = 1.
All are pure functions — no mutation — compatible with ForwardDiff and Mooncake.
"""

# ------------------------------------------------------------------ #
# Physical constants and CMB temperature                              #
# ------------------------------------------------------------------ #

const T_CMB_K     = 2.72548          # K
const H_OVER_KT   = 6.62607015e-34 * 1e9 / (1.380649e-23 * T_CMB_K)
# h [J·s] × 1e9 [GHz→Hz] / (k_B [J/K] × T_CMB [K])
# so H_OVER_KT * nu [GHz] = hν/(k_B T_CMB) = x (dimensionless)

# ------------------------------------------------------------------ #
# Unit conversion helpers                                              #
# ------------------------------------------------------------------ #

"""
    x_cmb(nu)

Dimensionless frequency ratio x = hν/(k_B T_CMB), with ν in GHz.
"""
@inline x_cmb(nu) = H_OVER_KT * nu

"""
    rj2cmb(nu)

Rayleigh-Jeans to CMB thermodynamic units conversion factor.
= (expm1(x)/x)² / exp(x)   where x = hν/(k_B T_CMB)

Used to convert flux-density SEDs (defined in RJ units) to K_CMB.
"""
function rj2cmb(nu::T) where T<:Real
    x = x_cmb(nu)
    return (expm1(x) / x)^2 / exp(x)
end

function rj2cmb(nu::AbstractVector)
    return rj2cmb.(nu)
end

"""
    cmb2bb(nu)

Proportional to ∂B_ν/∂T|_{T_CMB}, used to normalize passbands.
= exp(x) * (ν·x / expm1(x))²   where x = hν/(k_B T_CMB)

This is the `_cmb2bb` function in foreground.py.
Numerical constants (2k³T²/c²h²) are omitted — they cancel in ratios.
"""
function cmb2bb(nu::T) where T<:Real
    x = x_cmb(nu)
    return exp(x) * (nu * x / expm1(x))^2
end

function cmb2bb(nu::AbstractVector)
    return cmb2bb.(nu)
end

# ------------------------------------------------------------------ #
# tSZ SED                                                              #
# ------------------------------------------------------------------ #

"""
    tsz_f(nu)

Non-relativistic tSZ spectral function (in K_CMB):
f(ν) = x·coth(x/2) - 4,   x = hν/(k_B T_CMB)

Not normalized. Use `tsz_sed` for the normalized ratio.
"""
function tsz_f(nu::T) where T<:Real
    x = x_cmb(nu)
    return x / tanh(x / 2) - 4
end

"""
    tsz_sed(nu, nu_0)

Thermal SZ SED normalized at reference frequency `nu_0` (GHz).
Returns f_tSZ(ν) / f_tSZ(ν₀).

Scalar or vector `nu` supported.
"""
tsz_sed(nu::T,                 nu_0::S) where {T<:Real,S<:Real} = tsz_f(nu)   / tsz_f(nu_0)
tsz_sed(nu::AbstractVector{T}, nu_0::S) where {T<:Real,S<:Real} = tsz_f.(nu) ./ tsz_f(nu_0)

# ------------------------------------------------------------------ #
# Modified Blackbody (MBB) SED                                         #
# Used for: CIB Poisson, CIB clustered, Galactic dust                  #
# ------------------------------------------------------------------ #

"""
    mbb_sed(nu, nu_0, beta, temp)

Modified blackbody SED normalized at `nu_0` (GHz), in K_CMB:

μ(ν) / μ(ν₀) = (ν/ν₀)^(β+1) · [expm1(x₀) / expm1(x)] · [rj2cmb(ν) / rj2cmb(ν₀)]

where x = hν·10⁹/(k_B·T_d), x₀ = hν₀·10⁹/(k_B·T_d).

Arguments:
  nu    — frequency in GHz (scalar or vector)
  nu_0  — reference frequency in GHz
  beta  — MBB spectral index
  temp  — dust temperature in K
"""
function mbb_sed(nu::T, nu_0::Real, beta::Real, temp::Real) where T<:Real
    # Planck function exponent (note: nu in GHz, so factor 1e9 in x)
    H_OVER_K = 6.62607015e-34 * 1e9 / 1.380649e-23   # h [J·s] × 1e9 / k_B [J/K]
    x   = H_OVER_K * nu   / temp
    x_0 = H_OVER_K * nu_0 / temp
    mbb_ratio = (nu / nu_0)^(beta + 1) * expm1(x_0) / expm1(x)
    rj_ratio  = rj2cmb(nu) / rj2cmb(nu_0)
    return mbb_ratio * rj_ratio
end

function mbb_sed(nu::AbstractVector, nu_0::Real, beta::Real, temp::Real)
    return mbb_sed.(nu, nu_0, beta, temp)
end

# ------------------------------------------------------------------ #
# Radio / Power-law SED                                                #
# ------------------------------------------------------------------ #

"""
    radio_sed(nu, nu_0, beta)

Power-law SED in flux units, converted to K_CMB:
= (ν/ν₀)^β · [rj2cmb(ν) / rj2cmb(ν₀)]

Used for unresolved radio sources. `beta` is typically in [-3.5, -1.5].
"""
radio_sed(nu::Real,           nu_0::Real, beta::Real) = (nu/nu_0)^beta * rj2cmb(nu) / rj2cmb(nu_0)
radio_sed(nu::AbstractVector, nu_0::Real, beta::Real) = (nu ./ nu_0) .^ beta .* rj2cmb.(nu) ./ rj2cmb(nu_0)

# ------------------------------------------------------------------ #
# Constant SED (kSZ — blackbody, frequency-independent in K_CMB)       #
# ------------------------------------------------------------------ #

"""
    constant_sed(nu)

Frequency-independent SED (returns 1.0 for any frequency).
Used for kSZ, which is a blackbody signal (no frequency scaling in K_CMB).
"""
constant_sed(::Real)           = 1.0
constant_sed(nu::AbstractVector) = ones(eltype(nu), length(nu))
