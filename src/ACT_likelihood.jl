"""
    ACT_likelihood.jl

Core data structures, constructor, and likelihood evaluation for the ACT DR6 CMB likelihood.

The main struct `ACTDR6Likelihood` pre-stores:
- The concatenated data band-power vector
- The Cholesky decomposition of the covariance matrix
- Per-spectrum window function matrices
- Spectral layout metadata
- A foreground model instance (any type implementing `compute_foreground_Dls`)

The `loglike` function computes:
    log L = -0.5 (d - t)ᵀ C⁻¹ (d - t)
via Cholesky solve, avoiding explicit matrix inversion.
"""

# ---------------------------------------------------------------------------
# Typed parameter containers
# ---------------------------------------------------------------------------

"""
    ACTCMBOnlyPars{T}

Statically-typed nuisance parameter container for the CMB-only ACT DR6 likelihood.

| Field   | Physical meaning                              |
|---------|-----------------------------------------------|
| `A_act` | Overall temperature calibration (divides by A²)|
| `P_act` | Polarization calibration (TE ÷ P, EE ÷ P²)    |
"""
struct ACTCMBOnlyPars{T}
    A_act :: T
    P_act :: T
end

ACTCMBOnlyPars(A_act::Real, P_act::Real) = ACTCMBOnlyPars(promote(A_act, P_act)...)

function Base.convert(::Type{Dict{Symbol,T}}, p::ACTCMBOnlyPars{T}) where T
    return Dict{Symbol,T}(:A_act => p.A_act, :P_act => p.P_act)
end


# ---------------------------------------------------------------------------
# Main likelihood struct
# ---------------------------------------------------------------------------

"""
    ACTDR6Likelihood{FG}

Pre-computed data structure for the ACT DR6 Gaussian CMB likelihood.

All quantities that can be computed once (Cholesky decomposition, ell grids,
spectrum metadata) are stored here. The `loglike` function then only needs
to perform window-function convolution, foreground addition, and a Cholesky solve.

# Type parameter
- `FG`: foreground model type — `ACTCMBOnlyFG` for the CMB-only data release

# Fields
| Field          | Type                     | Description |
|----------------|--------------------------|-------------|
| `data_vector`  | `Vector{Float64}`        | Concatenated band powers (N_total,) |
| `cov_chol`     | `Cholesky{Float64,...}`  | Cholesky of the covariance |
| `windows`      | `Vector{Matrix{Float64}}`| W[i] has shape (N_ell, N_bins[i]) |
| `fg_model`     | `FG`                     | Foreground model instance |
| `spec_order`   | `Vector{String}`         | e.g. ["TT","TE","EE"] |
| `spec_types`   | `Vector{String}`         | e.g. ["TT","TE","EE"] |
| `N_bins`       | `Vector{Int}`            | Bins per spectrum |
| `N_ell`        | `Int`                    | Theory ell bins |
| `ell_min`      | `Int`                    | 2 |
| `ell_max`      | `Int`                    | ell_min + N_ell - 1 |
| `ells`         | `Vector{Float64}`        | [ell_min..ell_max] |
"""
struct ACTDR6Likelihood{FG}
    # Data
    data_vector :: Vector{Float64}
    cov_chol    :: Cholesky{Float64, Matrix{Float64}}
    windows     :: Vector{Matrix{Float64}}
    # Foreground model
    fg_model    :: FG
    # Spectral layout
    spec_order  :: Vector{String}
    spec_types  :: Vector{String}
    N_bins      :: Vector{Int}
    N_ell       :: Int
    ell_min     :: Int
    ell_max     :: Int
    ells        :: Vector{Float64}
end


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    ACTDR6Likelihood(data_dir; kwargs...) -> ACTDR6Likelihood{ACTCMBOnlyFG}

Load the ACT DR6 CMB-only likelihood from `data_dir`.

# Keyword arguments
- `yaml_file`: path to YAML descriptor (auto-detected if `nothing`)
- `ell_min`: minimum theory ℓ (default: 2)
- `fg_model`: foreground model instance (default: `ACTCMBOnlyFG()`)
"""
function ACTDR6Likelihood(data_dir::AbstractString;
                          yaml_file::Union{Nothing,AbstractString}=nothing,
                          ell_min::Int=2,
                          fg_model=ACTCMBOnlyFG())
    d = load_act_data(data_dir; yaml_file=yaml_file, ell_min=ell_min)
    return ACTDR6Likelihood(
        d.data_vector,
        d.cov_chol,
        d.windows,
        fg_model,
        d.spec_order,
        d.spec_types,
        d.N_bins,
        d.N_ell,
        ell_min,
        ell_min + d.N_ell - 1,
        d.ells,
    )
end


# ---------------------------------------------------------------------------
# Theory vector construction helpers
# ---------------------------------------------------------------------------

"""
    bin_theory_vector(Dls_long, windows, N_ell, N_specs) -> Vector

Apply window functions to the long theory Dℓ vector to obtain model band powers.

# Arguments
- `Dls_long`: concatenated theory Dℓ, length `N_specs * N_ell`
- `windows`: `Vector{Matrix{Float64}}`, `windows[i]` has shape `(N_ell, N_bins[i])`
- `N_ell`: number of theory ell bins per spectrum block
- `N_specs`: number of spectrum blocks

# Returns
- Concatenated model band powers, length `sum(N_bins)`
"""
function bin_theory_vector(Dls_long::AbstractVector,
                            windows::Vector{Matrix{Float64}},
                            N_ell::Int,
                            N_specs::Int)
    T = eltype(Dls_long)
    bands = T[]
    for i in 1:N_specs
        ib = (i - 1) * N_ell + 1 : i * N_ell
        Di = Dls_long[ib]       # (N_ell,)
        Wi = windows[i]         # (N_ell, N_bins[i])
        # Di' * Wi = (N_bins[i],) — one number per band-power bin
        bp = Di' * Wi           # (1, N_bins[i]) → we want a vector
        append!(bands, vec(bp))
    end
    return bands
end


"""
    apply_calibration(Dls_long, A_act, P_act, spec_types, N_ell) -> Vector

Apply overall (A_act) and polarization (P_act) calibration to the theory Dℓ vector.

The ACT DR6 calibration model (matching candl's CalibrationSingleScalarSquared
and PolarisationCalibrationDivision) divides by:
- TT: `A_act²`
- TE: `A_act² × P_act`
- EE: `A_act² × P_act²`

# Arguments
- `Dls_long`: concatenated theory Dℓ, length `N_specs * N_ell`
- `A_act`: overall temperature calibration parameter
- `P_act`: polarization calibration parameter
- `spec_types`: `Vector{String}` of spectrum types, length `N_specs`
- `N_ell`: number of theory ells per block

# Returns
- Calibrated Dℓ vector of the same length as `Dls_long`
"""
function apply_calibration(Dls_long::AbstractVector,
                            A_act, P_act,
                            spec_types::Vector{String},
                            N_ell::Int)
    T = promote_type(eltype(Dls_long), typeof(A_act), typeof(P_act))
    out = similar(Dls_long, T)
    A2 = A_act * A_act
    for (i, st) in enumerate(spec_types)
        ib = (i - 1) * N_ell + 1 : i * N_ell
        if st == "TT"
            out[ib] = Dls_long[ib] ./ A2
        elseif st == "TE"
            out[ib] = Dls_long[ib] ./ (A2 * P_act)
        elseif st == "EE"
            out[ib] = Dls_long[ib] ./ (A2 * P_act * P_act)
        else
            out[ib] = Dls_long[ib]
        end
    end
    return out
end


"""
    build_theory_bandpowers(like, params, Dls_CMB) -> Vector

Compute the full model band-power vector:

1. Unpack `Dls_CMB` into a long theory Dℓ vector (one block per spectrum)
2. Add foreground contributions from `like.fg_model`
3. Apply calibration (A_act, P_act)
4. Convolve with window functions → band powers

# Arguments
- `like`: `ACTDR6Likelihood` struct
- `params`: `NamedTuple` or `Dict{Symbol}` of nuisance parameters
- `Dls_CMB`: `NamedTuple` with keys `:TT`, `:TE`, `:EE` (or other
             spectra matching `like.spec_types`), each a `Vector` of
             theory Dℓ values starting at ℓ=`like.ell_min`.
             For multi-frequency data, each key should match the spectrum type.

# Returns
- `Vector` of model band powers, length `sum(like.N_bins)`
"""
function build_theory_bandpowers(like::ACTDR6Likelihood, params, Dls_CMB)
    N_specs = length(like.spec_order)

    # ---- Unpack theory Dls into the long vector ----
    # For CMB-only: spec_types = ["TT","TE","EE"] → repeat each Dls block
    # For multi-freq: same spec_type may repeat; use spec_order as key
    T_cmb = promote_type(
        eltype(Dls_CMB.TT), eltype(Dls_CMB.TE), eltype(Dls_CMB.EE),
        typeof(get(params, :A_act, 1.0))
    )

    Dls_long = zeros(T_cmb, N_specs * like.N_ell)

    for (i, (spec, st)) in enumerate(zip(like.spec_order, like.spec_types))
        ib = (i - 1) * like.N_ell + 1 : i * like.N_ell
        # Select the right CMB Dℓ block based on spectrum type
        if st == "TT"
            Dls_long[ib] = Dls_CMB.TT[1:like.N_ell]
        elseif st == "TE"
            Dls_long[ib] = Dls_CMB.TE[1:like.N_ell]
        elseif st == "EE"
            Dls_long[ib] = Dls_CMB.EE[1:like.N_ell]
        else
            # Unknown type: leave as zero (will be filled by foreground model)
        end
    end

    # ---- Add foreground contributions ----
    fg_Dls = compute_foreground_Dls(
        like.fg_model, params, like.ells, like.spec_order, like.N_ell)
    Dls_long = Dls_long .+ fg_Dls

    # ---- Apply calibration ----
    A_act = T_cmb(get(params, :A_act, 1.0))
    P_act = T_cmb(get(params, :P_act, 1.0))
    Dls_long = apply_calibration(Dls_long, A_act, P_act, like.spec_types, like.N_ell)

    # ---- Window function convolution → band powers ----
    return bin_theory_vector(Dls_long, like.windows, like.N_ell, N_specs)
end


# ---------------------------------------------------------------------------
# Log-likelihood evaluation
# ---------------------------------------------------------------------------

"""
    loglike(like::ACTDR6Likelihood, params, Dls_CMB) -> Float64

Compute the Gaussian log-likelihood for the ACT DR6 likelihood:

    log L = -0.5 × (d - t)ᵀ C⁻¹ (d - t)

Uses the pre-computed Cholesky factorisation for numerical stability and speed.
The result is the log-probability (not the negative log-likelihood), so callers
who want to *minimise* should negate the return value.

# Arguments
- `like`: `ACTDR6Likelihood` with pre-loaded data and covariance
- `params`: `NamedTuple` or `Dict{Symbol}` of nuisance parameter values.
            Required keys: `:A_act`, `:P_act` (for CMB-only).
            Additional foreground parameters are not used by this CMB-only model.
- `Dls_CMB`: `NamedTuple` with keys `:TT`, `:TE`, `:EE`, each a `Vector`
             of theory Dℓ values (in μK²) starting at ℓ=`like.ell_min`.

# Returns
- `Float64` log-likelihood value (or the corresponding AD dual type)

# Example
```julia
like = ACTDR6Likelihood("path/to/ACT_DR6_CMB_only_v0")
params = (A_act=1.0, P_act=1.0)
Dls_CMB = (TT=camb_cls.TT, TE=camb_cls.TE, EE=camb_cls.EE)
ll = loglike(like, params, Dls_CMB)
```
"""
function loglike(like::ACTDR6Likelihood, params, Dls_CMB)
    # Build model band powers
    theory = build_theory_bandpowers(like, params, Dls_CMB)

    # Residual
    δ = like.data_vector .- theory

    # Cholesky solve: y = L⁻¹ δ  (forward substitution with lower triangular L)
    y = like.cov_chol.L \ δ

    return -0.5 * dot(y, y)
end


"""
    (like::ACTDR6Likelihood)(params, Dls_CMB) -> Float64

Callable interface: returns the **negative** log-likelihood (for minimisers).
Equivalent to `-loglike(like, params, Dls_CMB)`.
"""
(like::ACTDR6Likelihood)(params, Dls_CMB) = -loglike(like, params, Dls_CMB)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

"""
    chi2(like::ACTDR6Likelihood, params, Dls_CMB) -> Float64

Return the chi-squared statistic: `(d - t)ᵀ C⁻¹ (d - t)`.
Equivalent to `-2 * loglike(like, params, Dls_CMB)`.
"""
function chi2(like::ACTDR6Likelihood, params, Dls_CMB)
    return -2.0 * loglike(like, params, Dls_CMB)
end


"""
    effective_ells(like::ACTDR6Likelihood) -> Vector{Float64}

Compute the effective (bin-centre) ℓ values for each band-power bin:
    ℓ_eff[b] = ells' W[:, b]

Returns a concatenated vector of length `sum(like.N_bins)`.
"""
function effective_ells(like::ACTDR6Likelihood)
    eff = Float64[]
    for (i, W) in enumerate(like.windows)
        # ells is a (N_ell,) vector; W is (N_ell, N_bins[i])
        eff_i = like.ells' * W   # (1, N_bins[i])
        append!(eff, vec(eff_i))
    end
    return eff
end


"""
    Base.show(io, like::ACTDR6Likelihood)

Human-readable summary of the likelihood.
"""
function Base.show(io::IO, like::ACTDR6Likelihood)
    N_total   = sum(like.N_bins)
    FG        = typeof(like.fg_model)
    spec_str  = join(like.spec_order, ", ")
    bins_str  = join(like.N_bins, " + ")
    println(io, "ACTDR6Likelihood{$FG}")
    println(io, "  ℓ range     : $(like.ell_min) – $(like.ell_max)")
    println(io, "  Theory bins : $(like.N_ell)")
    println(io, "  Spectra     : $spec_str")
    print(io,   "  Band powers : $bins_str = $N_total total")
end
