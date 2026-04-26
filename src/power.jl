"""
    power.jl

Power spectrum templates and ell-dependent foreground functions.
Mirrors fgspectra/power.py.

Templates are D_ℓ arrays (NOT C_ℓ), stored as Vectors indexed by ℓ.
Array index convention: template[ℓ+1] = D_ℓ  (1-based Julia indexing).

All eval functions are pure and AD-compatible.
ℓ is always an integer — not a free parameter — so template lookups
are not differentiated.
"""

# Foreground templates are bundled in the ACT DR6 Zenodo artifact
# (see src/artifacts.jl, Artifacts.toml). The templates directory is resolved
# lazily on first call so that users can override it via the lower-level
# `load_template(path)` form if needed.
fgspectra_data_path() = act_dr6_template_dir()

# ------------------------------------------------------------------ #
# Template loading                                                      #
# ------------------------------------------------------------------ #

"""
    load_template(filename) → Vector{Float64}

Load a two-column `.dat` file (ℓ, D_ℓ) and return a Vector indexed
so that `template[ℓ+1] = D_ℓ`.

Rows need not be sorted; gaps are left as zero.
"""
function load_template(filename::AbstractString)
    data = readdlm(filename)
    ells = Int.(round.(data[:, 1]))
    dls  = Float64.(data[:, 2])
    arr  = zeros(Float64, maximum(ells) + 2)   # +2: 0-indexed ell → 1-indexed +1
    for (ell, dl) in zip(ells, dls)
        arr[ell + 1] = dl
    end
    return arr
end

"""
    load_tsz_template() → Vector{Float64}

Thermal SZ D_ℓ template at 150 GHz (Battaglia+2010), ℓ ∈ [2, 10000].
"""
load_tsz_template() =
    load_template(joinpath(fgspectra_data_path(), "cl_tsz_150_bat.dat"))

"""
    load_ksz_template() → Vector{Float64}

Kinematic SZ D_ℓ template — reionization only (Battaglia+2013), ℓ ∈ [2, 10000].
"""
load_ksz_template() =
    load_template(joinpath(fgspectra_data_path(), "cl_ksz_bat.dat"))

"""
    load_cibc_template() → Vector{Float64}

CIB clustered D_ℓ template (Choi+2021), ℓ ∈ [2, 13000].
"""
load_cibc_template() =
    load_template(joinpath(fgspectra_data_path(), "cl_cib_Choi2020.dat"))

"""
    load_szxcib_template() → Vector{Float64}

tSZ×CIB cross-correlation D_ℓ template (Addison+2012), ℓ ∈ [2, 9999].
"""
load_szxcib_template() =
    load_template(joinpath(fgspectra_data_path(), "cl_sz_x_cib.dat"))

# ------------------------------------------------------------------ #
# Template evaluation                                                   #
# ------------------------------------------------------------------ #

"""
    eval_template(T, ell, ell_0; amp=1.0)

Evaluate a template at multipoles `ell`, normalized to 1 at `ell_0`:
    amp × T[ℓ] / T[ℓ₀]

`ell` is a Vector{Int}. `ell_0` is an Int.
"""
function eval_template(T::AbstractVector, ell::AbstractVector{<:Integer},
                        ell_0::Integer; amp::Real=1.0)
    norm = T[ell_0 + 1]
    return amp .* T[ell .+ 1] ./ norm
end

"""
    eval_template_tilt(T, ell, ell_0, alpha; amp=1.0)

Template rescaled by a power law (used for tSZ tilt α_tSZ):
    amp × T[ℓ] / T[ℓ₀] × (ℓ/ℓ₀)^α

Mirrors `PowerLawRescaledTemplate` in fgspectra.
"""
function eval_template_tilt(T::AbstractVector, ell::AbstractVector{<:Integer},
                              ell_0::Integer, alpha::Real; amp::Real=1.0)
    base = eval_template(T, ell, ell_0; amp=amp)
    tilt = (ell ./ ell_0) .^ alpha
    return base .* tilt
end

"""
    eval_powerlaw(ell, ell_0, alpha; amp=1.0)

Simple power law in ℓ:
    amp × (ℓ/ℓ₀)^α

Used for Poisson CIB, radio, and galactic dust ℓ-dependence.
`ell` can be any numeric vector (Int or Float for ℓ×(ℓ+1) quantities).
"""
function eval_powerlaw(ell::AbstractVector, ell_0::Real, alpha::Real; amp::Real=1.0)
    return amp .* (ell ./ ell_0) .^ alpha
end
