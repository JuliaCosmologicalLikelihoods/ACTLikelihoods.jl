"""
    ACT_nuisance.jl

Statically typed nuisance-parameter container, official ACT DR6 priors, and the
posterior assembly for the full multifrequency likelihood.

Everything here is derived from the official ACT DR6 configuration:

* `act_dr6_mflike/act_dr6_mflike/params_systematics.yaml` — calibration and
  bandpass-shift priors, and the fixed `calE_dr6_pa4_f220 = 1.0`
* `act_dr6_mflike/examples/act_dr6_example.yml` — the sampled foreground ranges,
  the derived `beta_c = beta_p`, the Gaussian dust priors, and `T_d = 9.60`
* `LAT_MFLike/mflike/{fg_TT,fg_TE,fg_EE,Foreground}.yaml` — the remaining fixed
  model constants

No prior is invented here.
"""

# ---------------------------------------------------------------------------
# Free parameters
# ---------------------------------------------------------------------------

"""
    ACT_DR6_FREE_PARAMETERS

Deterministic ordering of the 29 free baseline nuisance parameters, matching the
field order of [`ACTDR6Nuisance`](@ref) and the layout used by
[`parameter_vector`](@ref).

1–14 foregrounds, 15–24 calibration, 25–29 bandpass shifts.

`beta_c` is **not** free: the official configuration sets `beta_c = beta_p`.
`calE_dr6_pa4_f220` is **not** free either: ACT DR6 uses no pa4 f220
polarization channel, so the official configuration fixes it at `1.0` and the
parameter cannot enter the model. See [`ACT_DR6_FIXED_PARAMETERS`](@ref).
"""
const ACT_DR6_FREE_PARAMETERS = (
    # --- foregrounds (14) ---
    :a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p, :a_c, :a_s, :beta_s,
    :a_gtt, :a_gte, :a_gee, :a_psee, :a_pste, :xi,
    # --- calibration (10) ---
    :calG_all,
    :cal_dr6_pa4_f220, :cal_dr6_pa5_f090, :cal_dr6_pa5_f150,
    :cal_dr6_pa6_f090, :cal_dr6_pa6_f150,
    :calE_dr6_pa5_f090, :calE_dr6_pa5_f150,
    :calE_dr6_pa6_f090, :calE_dr6_pa6_f150,
    # --- bandpass shifts (5) ---
    :bandint_shift_dr6_pa4_f220, :bandint_shift_dr6_pa5_f090,
    :bandint_shift_dr6_pa5_f150, :bandint_shift_dr6_pa6_f090,
    :bandint_shift_dr6_pa6_f150,
)

"Number of free baseline nuisance parameters (29)."
const ACT_DR6_N_FREE = length(ACT_DR6_FREE_PARAMETERS)

"""
    ACT_DR6_FIXED_PARAMETERS

Model constants held fixed by the official ACT DR6 configuration. These are not
inference parameters and are kept strictly separate from
[`ACT_DR6_FREE_PARAMETERS`](@ref).

| name | value | source |
|---|---|---|
| `T_d` | `9.60` | `act_dr6_example.yml` (overrides the MFLike default `9.7`) |
| `alpha_dT` | `-0.6` | `fg_TT.yaml` |
| `alpha_dE` | `-0.4` | `fg_TE.yaml` / `fg_EE.yaml` |
| `alpha_p` | `1.0` | `fg_TT.yaml` |
| `alpha_s` | `1.0` | `Foreground.yaml` |
| `T_effd` | `19.6` | `Foreground.yaml` |
| `beta_d` | `1.5` | `Foreground.yaml` |
| `calE_dr6_pa4_f220` | `1.0` | `params_systematics.yaml` (no pa4 f220 polarization) |
"""
const ACT_DR6_FIXED_PARAMETERS = (
    T_d = 9.60,
    alpha_dT = -0.6,
    alpha_dE = -0.4,
    alpha_p = 1.0,
    alpha_s = 1.0,
    T_effd = 19.6,
    beta_d = 1.5,
    calE_dr6_pa4_f220 = 1.0,
)

"""
    ACTDR6Nuisance{T} <: Any

Statically typed container for the 29 free ACT DR6 nuisance parameters.

Field order is [`ACT_DR6_FREE_PARAMETERS`](@ref) and is the documented vector
layout: `parameter_vector(p)[i]` is the parameter named
`ACT_DR6_FREE_PARAMETERS[i]`.

The type parameter `T` propagates whatever scalar it is built from, so
`ForwardDiff.Dual` inputs and reverse-mode tracers survive construction,
`NamedTuple` conversion and round-tripping unchanged.

# Construction
```julia
ACTDR6Nuisance()                          # validated reference point
ACTDR6Nuisance(vector)                    # from a 29-element vector
ACTDR6Nuisance(nt::NamedTuple)            # by name; missing names take the reference value
ACTDR6Nuisance(dict::AbstractDict)        # by name, Symbol or String keys
ACTDR6Nuisance(p; a_tSZ = 4.0)            # copy with overrides
```

# Conversion
```julia
parameter_vector(p)          # 29-element Vector{T}, documented ordering
NamedTuple(p)                # free + derived `beta_c` + fixed constants
free_parameters(p)           # the 29 free values only, as a NamedTuple
```
"""
struct ACTDR6Nuisance{T<:Real}
    # --- foregrounds (14) ---
    a_tSZ::T
    alpha_tSZ::T
    a_kSZ::T
    a_p::T
    beta_p::T
    a_c::T
    a_s::T
    beta_s::T
    a_gtt::T
    a_gte::T
    a_gee::T
    a_psee::T
    a_pste::T
    xi::T
    # --- calibration (10) ---
    calG_all::T
    cal_dr6_pa4_f220::T
    cal_dr6_pa5_f090::T
    cal_dr6_pa5_f150::T
    cal_dr6_pa6_f090::T
    cal_dr6_pa6_f150::T
    calE_dr6_pa5_f090::T
    calE_dr6_pa5_f150::T
    calE_dr6_pa6_f090::T
    calE_dr6_pa6_f150::T
    # --- bandpass shifts (5) ---
    bandint_shift_dr6_pa4_f220::T
    bandint_shift_dr6_pa5_f090::T
    bandint_shift_dr6_pa5_f150::T
    bandint_shift_dr6_pa6_f090::T
    bandint_shift_dr6_pa6_f150::T
end

@assert fieldcount(ACTDR6Nuisance) == ACT_DR6_N_FREE
@assert fieldnames(ACTDR6Nuisance) == ACT_DR6_FREE_PARAMETERS

"""
    ACT_DR6_REFERENCE_NUISANCE

The validated reference nuisance point. Evaluated with the ACT DR6 best-fit CMB
spectra it reproduces the upstream `chi2 = 1592.058412999159`, and it is the
point every baseline fixture in `validation/fixtures/` is generated at.

This is a *reference*, not a posterior maximum; it is used as the default
fiducial because it is the point at which parity is established.
"""
const ACT_DR6_REFERENCE_NUISANCE = (
    a_tSZ = 3.50114277,
    alpha_tSZ = -0.4597721879,
    a_kSZ = 0.986604682,
    a_p = 7.647742104,
    beta_p = 1.86490755,
    a_c = 3.805341822,
    a_s = 2.886594272,
    beta_s = -2.7567784,
    a_gtt = 7.974213801,
    a_gte = 0.4184588365,
    a_gee = 0.1676466062,
    a_psee = 0.003755819497,
    a_pste = -0.02500092711,
    xi = 0.06424293336,
    calG_all = 1.001567048,
    cal_dr6_pa4_f220 = 0.9808084654,
    cal_dr6_pa5_f090 = 1.000098497,
    cal_dr6_pa5_f150 = 0.9991342522,
    cal_dr6_pa6_f090 = 0.9998031382,
    cal_dr6_pa6_f150 = 1.001407626,
    calE_dr6_pa5_f090 = 0.9874026803,
    calE_dr6_pa5_f150 = 0.9975776488,
    calE_dr6_pa6_f090 = 0.9975750142,
    calE_dr6_pa6_f150 = 0.9968551529,
    bandint_shift_dr6_pa4_f220 = 6.399328024,
    bandint_shift_dr6_pa5_f090 = -0.2911716302,
    bandint_shift_dr6_pa5_f150 = -1.056426408,
    bandint_shift_dr6_pa6_f090 = 0.3121747872,
    bandint_shift_dr6_pa6_f150 = -0.4252785128,
)

@assert keys(ACT_DR6_REFERENCE_NUISANCE) == ACT_DR6_FREE_PARAMETERS

"""
    act_dr6_fiducial_nuisance() -> ACTDR6Nuisance{Float64}

The fiducial (reference) nuisance point as an [`ACTDR6Nuisance`](@ref).
"""
act_dr6_fiducial_nuisance() = ACTDR6Nuisance(ACT_DR6_REFERENCE_NUISANCE)

function ACTDR6Nuisance(values::AbstractVector{T}) where {T<:Real}
    length(values) == ACT_DR6_N_FREE || throw(DimensionMismatch(
        "ACT DR6 nuisance vector must have $(ACT_DR6_N_FREE) entries, got $(length(values))",
    ))
    Base.require_one_based_indexing(values)
    return ACTDR6Nuisance{T}(ntuple(i -> values[i], Val(ACT_DR6_N_FREE))...)
end

_nuisance_lookup(source::NamedTuple, name::Symbol) =
    haskey(source, name) ? source[name] : nothing
function _nuisance_lookup(source::AbstractDict, name::Symbol)
    haskey(source, name) && return source[name]
    key = String(name)
    haskey(source, key) && return source[key]
    return nothing
end

function _nuisance_from_lookup(source, defaults)
    values = map(ACT_DR6_FREE_PARAMETERS) do name
        value = _nuisance_lookup(source, name)
        value === nothing ? defaults[name] : value
    end
    return ACTDR6Nuisance(promote(values...)...)
end

function _reject_unknown(source, names)
    unknown = [String(name) for name in names
               if !(name in ACT_DR6_FREE_PARAMETERS) &&
                  !(name in keys(ACT_DR6_FIXED_PARAMETERS)) && name !== :beta_c]
    isempty(unknown) || throw(ArgumentError(
        "unknown ACT DR6 nuisance parameter(s): $(join(sort(unknown), ", ")). " *
        "Free parameters are $(join(String.(ACT_DR6_FREE_PARAMETERS), ", ")).",
    ))
    return nothing
end

"""
    ACTDR6Nuisance(source::NamedTuple; kwargs...)
    ACTDR6Nuisance(source::AbstractDict; kwargs...)

Build a nuisance point by name. Names absent from `source` take their
[`ACT_DR6_REFERENCE_NUISANCE`](@ref) value. Fixed model constants and the
derived `beta_c` may appear in `source` but carry no freedom, so they are
validated and then dropped: a fixed constant must equal its official value, and
`beta_c` must equal the *effective* `beta_p` — the one in `source`, or the
inherited default when `source` does not set it. Any other unknown name is an
error.
"""
function ACTDR6Nuisance(source::NamedTuple)
    _reject_unknown(source, keys(source))
    _check_fixed_consistency(source, ACT_DR6_REFERENCE_NUISANCE)
    return _nuisance_from_lookup(source, ACT_DR6_REFERENCE_NUISANCE)
end

function ACTDR6Nuisance(source::AbstractDict)
    _reject_unknown(source, Symbol.(keys(source)))
    _check_fixed_consistency(source, ACT_DR6_REFERENCE_NUISANCE)
    return _nuisance_from_lookup(source, ACT_DR6_REFERENCE_NUISANCE)
end

"""
    _check_fixed_consistency(source, defaults)

Validate the names that `source` may carry but that are not free parameters.

`defaults` is the parameter set the names absent from `source` fall back to, so
a supplied `beta_c` is compared against the *effective* `beta_p` — the one in
`source` if given, otherwise the one inherited from `defaults`. Checking it only
when `beta_p` is also supplied would let a contradictory `beta_c` pass unnoticed.
"""
function _check_fixed_consistency(source, defaults)
    for name in keys(ACT_DR6_FIXED_PARAMETERS)
        value = _nuisance_lookup(source, name)
        value === nothing && continue
        value == ACT_DR6_FIXED_PARAMETERS[name] || throw(ArgumentError(
            "`$name` is a fixed ACT DR6 model constant " *
            "($(ACT_DR6_FIXED_PARAMETERS[name])); got $value. " *
            "Fixed constants are not free inference parameters.",
        ))
    end
    beta_c = _nuisance_lookup(source, :beta_c)
    if beta_c !== nothing
        supplied_beta_p = _nuisance_lookup(source, :beta_p)
        beta_p = supplied_beta_p === nothing ? defaults[:beta_p] : supplied_beta_p
        beta_c == beta_p || throw(ArgumentError(
            "the ACT DR6 baseline model requires the derived relation beta_c = beta_p; " *
            "got beta_c = $beta_c and beta_p = $beta_p" *
            (supplied_beta_p === nothing ? " (inherited)" : ""),
        ))
    end
    return nothing
end

"""
    ACTDR6Nuisance(p::ACTDR6Nuisance; kwargs...)

Copy `p`, replacing the named free parameters. Fixed constants and `beta_c` are
accepted only when they agree with the result, on the same terms as the
`NamedTuple` constructor; they are never a way to change a fixed quantity.
"""
function ACTDR6Nuisance(p::ACTDR6Nuisance; kwargs...)
    overrides = values(kwargs)
    defaults = free_parameters(p)
    _reject_unknown(overrides, keys(overrides))
    _check_fixed_consistency(overrides, defaults)
    return _nuisance_from_lookup(overrides, defaults)
end

ACTDR6Nuisance(; kwargs...) = ACTDR6Nuisance(act_dr6_fiducial_nuisance(); kwargs...)

"""
    parameter_vector(p::ACTDR6Nuisance) -> Vector

The 29 free parameters in the documented [`ACT_DR6_FREE_PARAMETERS`](@ref)
order. Round-trips exactly: `ACTDR6Nuisance(parameter_vector(p)) == p`.
"""
parameter_vector(p::ACTDR6Nuisance{T}) where {T} = T[_free_tuple(p)...]

"""
    _free_tuple(p::ACTDR6Nuisance) -> NTuple{29}

The free values in the documented order, written out explicitly so that both the
ordering and the element type are statically known. `ACT_DR6_FREE_PARAMETERS` is
asserted against `fieldnames(ACTDR6Nuisance)` at load time, so this tuple cannot
silently drift out of the documented layout.
"""
@inline _free_tuple(p::ACTDR6Nuisance) = (
    p.a_tSZ, p.alpha_tSZ, p.a_kSZ, p.a_p, p.beta_p, p.a_c, p.a_s, p.beta_s,
    p.a_gtt, p.a_gte, p.a_gee, p.a_psee, p.a_pste, p.xi,
    p.calG_all,
    p.cal_dr6_pa4_f220, p.cal_dr6_pa5_f090, p.cal_dr6_pa5_f150,
    p.cal_dr6_pa6_f090, p.cal_dr6_pa6_f150,
    p.calE_dr6_pa5_f090, p.calE_dr6_pa5_f150,
    p.calE_dr6_pa6_f090, p.calE_dr6_pa6_f150,
    p.bandint_shift_dr6_pa4_f220, p.bandint_shift_dr6_pa5_f090,
    p.bandint_shift_dr6_pa5_f150, p.bandint_shift_dr6_pa6_f090,
    p.bandint_shift_dr6_pa6_f150,
)

"""
    free_parameters(p::ACTDR6Nuisance) -> NamedTuple

The 29 free parameters only, without the derived `beta_c` or the fixed
constants.
"""
free_parameters(p::ACTDR6Nuisance) =
    NamedTuple{ACT_DR6_FREE_PARAMETERS}(_free_tuple(p))

"""
    NamedTuple(p::ACTDR6Nuisance)

The complete parameter set the likelihood and foreground model consume: the 29
free parameters, the derived `beta_c = beta_p`, and the fixed model constants of
[`ACT_DR6_FIXED_PARAMETERS`](@ref).
"""
Base.NamedTuple(p::ACTDR6Nuisance) =
    merge(ACT_DR6_FIXED_PARAMETERS, free_parameters(p), (beta_c = p.beta_p,))

Base.:(==)(a::ACTDR6Nuisance, b::ACTDR6Nuisance) = free_parameters(a) == free_parameters(b)
Base.isapprox(a::ACTDR6Nuisance, b::ACTDR6Nuisance; kwargs...) =
    isapprox(parameter_vector(a), parameter_vector(b); kwargs...)

function Base.show(io::IO, ::MIME"text/plain", p::ACTDR6Nuisance{T}) where {T}
    println(io, "ACTDR6Nuisance{", T, "} — ", ACT_DR6_N_FREE, " free ACT DR6 parameters")
    for (name, value) in zip(ACT_DR6_FREE_PARAMETERS, _free_tuple(p))
        println(io, "  ", rpad(String(name), 28), value)
    end
    print(io, "  (derived) beta_c = beta_p = ", p.beta_p)
    return nothing
end

Base.show(io::IO, p::ACTDR6Nuisance{T}) where {T} =
    print(io, "ACTDR6Nuisance{", T, "}(", ACT_DR6_N_FREE, " free parameters)")

# ---------------------------------------------------------------------------
# Official ACT DR6 priors
# ---------------------------------------------------------------------------

"""
    ACT_DR6_GAUSSIAN_PRIORS

Normal priors `(loc, scale)` from the official configuration.

Calibration and bandpass shifts come from `params_systematics.yaml`; the three
galactic-dust amplitude priors are the external `TTdust_prior`, `TEdust_prior`
and `EEdust_prior` blocks of `examples/act_dr6_example.yml`.
"""
const ACT_DR6_GAUSSIAN_PRIORS = (
    a_gtt = (7.95, 0.32),
    a_gte = (0.423, 0.03),
    a_gee = (0.1681, 0.017),
    calG_all = (1.0, 0.003),
    cal_dr6_pa4_f220 = (1.0, 0.013),
    cal_dr6_pa5_f090 = (1.0, 0.0016),
    cal_dr6_pa5_f150 = (1.0, 0.0020),
    cal_dr6_pa6_f090 = (1.0, 0.0018),
    cal_dr6_pa6_f150 = (1.0, 0.0024),
    bandint_shift_dr6_pa4_f220 = (0.0, 3.6),
    bandint_shift_dr6_pa5_f090 = (0.0, 1.0),
    bandint_shift_dr6_pa5_f150 = (0.0, 1.3),
    bandint_shift_dr6_pa6_f090 = (0.0, 1.2),
    bandint_shift_dr6_pa6_f150 = (0.0, 1.1),
)

"""
    ACT_DR6_UNIFORM_PRIORS

Uniform prior ranges `(min, max)` from the `params` block of
`examples/act_dr6_example.yml` (foregrounds) and `params_systematics.yaml`
(polarization efficiencies).

`a_gtt`, `a_gte` and `a_gee` carry *both* a uniform range and a normal prior, as
in the official run configuration, where the dust priors are supplied as
external `prior:` entries on top of the sampled ranges.
"""
const ACT_DR6_UNIFORM_PRIORS = (
    a_tSZ = (0.0, 10.0),
    alpha_tSZ = (-5.0, 5.0),
    a_kSZ = (0.0, 10.0),
    a_p = (0.0, 50.0),
    beta_p = (0.0, 5.0),
    a_c = (0.0, 50.0),
    a_s = (0.0, 50.0),
    beta_s = (-3.5, -1.5),
    a_gtt = (0.0, 50.0),
    a_gte = (0.0, 1.0),
    a_gee = (0.0, 1.0),
    a_psee = (0.0, 1.0),
    a_pste = (-1.0, 1.0),
    xi = (0.0, 0.2),
    calE_dr6_pa5_f090 = (0.9, 1.1),
    calE_dr6_pa5_f150 = (0.9, 1.1),
    calE_dr6_pa6_f090 = (0.9, 1.1),
    calE_dr6_pa6_f150 = (0.9, 1.1),
)

"""
    ACT_DR6_GAUSSIAN_PRIOR_INDEX

`(index, loc, scale)` for every normal prior, with `index` pointing into the
[`ACT_DR6_FREE_PARAMETERS`](@ref) vector layout. Derived from
[`ACT_DR6_GAUSSIAN_PRIORS`](@ref) at load time so the two cannot drift apart.
"""
const ACT_DR6_GAUSSIAN_PRIOR_INDEX = Tuple(
    (findfirst(==(name), ACT_DR6_FREE_PARAMETERS)::Int,
     Float64(ACT_DR6_GAUSSIAN_PRIORS[name][1]),
     Float64(ACT_DR6_GAUSSIAN_PRIORS[name][2]))
    for name in keys(ACT_DR6_GAUSSIAN_PRIORS)
)

"""
    ACT_DR6_UNIFORM_PRIOR_INDEX

`(index, min, max)` for every uniform prior, in the
[`ACT_DR6_FREE_PARAMETERS`](@ref) vector layout.
"""
const ACT_DR6_UNIFORM_PRIOR_INDEX = Tuple(
    (findfirst(==(name), ACT_DR6_FREE_PARAMETERS)::Int,
     Float64(ACT_DR6_UNIFORM_PRIORS[name][1]),
     Float64(ACT_DR6_UNIFORM_PRIORS[name][2]))
    for name in keys(ACT_DR6_UNIFORM_PRIORS)
)

@inline function _check_free_length(x::AbstractVector)
    length(x) == ACT_DR6_N_FREE || throw(DimensionMismatch(
        "ACT DR6 parameter vector must have $(ACT_DR6_N_FREE) entries, got $(length(x))",
    ))
    Base.require_one_based_indexing(x)
    return nothing
end

"""
    logprior(p::ACTDR6Nuisance)
    logprior(x::AbstractVector)

Normalized log prior density of the official ACT DR6 nuisance priors, evaluated
either from a parameter struct or directly from a 29-element vector in the
documented [`ACT_DR6_FREE_PARAMETERS`](@ref) order.

Normal terms contribute `-½((x-μ)/σ)² - log(σ√(2π))`; uniform terms contribute
`-log(max-min)` inside the range and `-Inf` outside. Any non-finite entry —
`NaN`, `Inf` or `-Inf` — puts the point outside the support and returns `-Inf`,
so a bad proposal is rejected here rather than becoming a `NaN` downstream. Parameters carrying both
(the three galactic-dust amplitudes) contribute both, exactly as the official
run configuration combines its sampled ranges with external `prior:` entries.

This is entirely separate from [`loglikelihood`](@ref), which stays data-only.
The uniform bounds make the prior non-differentiable at the range edges;
gradients are meaningful strictly inside the support, where the uniform terms
are constant.
"""
function logprior(x::AbstractVector{T}) where {T<:Real}
    _check_free_length(x)
    total = zero(float(T))
    for (index, lower, upper) in ACT_DR6_UNIFORM_PRIOR_INDEX
        value = x[index]
        # Positive containment: `value < lower || value > upper` is false for
        # `NaN`, so a non-finite proposal would otherwise collect a finite
        # uniform term instead of being rejected.
        (lower <= value <= upper) || return convert(typeof(total), -Inf)
        total -= log(upper - lower)
    end
    for (index, location, scale) in ACT_DR6_GAUSSIAN_PRIOR_INDEX
        residual = (x[index] - location) / scale
        # `NaN` would otherwise propagate into the returned density instead of
        # rejecting the point. `±Inf` already squares to `+Inf`.
        isnan(residual) && return convert(typeof(total), -Inf)
        total -= residual^2 / 2 + log(scale * sqrt(2 * pi))
    end
    return total
end

logprior(p::ACTDR6Nuisance) = logprior(parameter_vector(p))

"""
    prior_chi2(p::ACTDR6Nuisance)
    prior_chi2(x::AbstractVector)

`Σ ((x-μ)/σ)²` over the normal ACT DR6 priors. Uniform priors contribute `0`
inside their range and `Inf` outside, so `prior_chi2` stays the quantity that
adds to the data chi-square. Any non-finite entry, `NaN` included, gives `Inf`.
"""
function prior_chi2(x::AbstractVector{T}) where {T<:Real}
    _check_free_length(x)
    total = zero(float(T))
    for (index, lower, upper) in ACT_DR6_UNIFORM_PRIOR_INDEX
        value = x[index]
        # Positive containment, for the same reason as in `logprior`.
        (lower <= value <= upper) || return convert(typeof(total), Inf)
    end
    for (index, location, scale) in ACT_DR6_GAUSSIAN_PRIOR_INDEX
        residual = (x[index] - location) / scale
        isnan(residual) && return convert(typeof(total), Inf)
        total += residual^2
    end
    return total
end

prior_chi2(p::ACTDR6Nuisance) = prior_chi2(parameter_vector(p))

"""
    logposterior(like::ACTDR6FullLikelihood, prediction, p)

`loglikelihood(like, prediction) + logprior(p)`, where `p` is an
[`ACTDR6Nuisance`](@ref) or a 29-element parameter vector.

Data-only likelihood plus the official ACT DR6 nuisance priors. The fixed
Gaussian normalization is *not* included; add
[`gaussian_normalization`](@ref)`(like)` if the upstream convention is wanted.
"""
logposterior(like::ACTDR6FullLikelihood, prediction::AbstractVector, p::ACTDR6Nuisance) =
    loglikelihood(like, prediction) + logprior(p)

logposterior(like::ACTDR6FullLikelihood, prediction::AbstractVector,
             x::AbstractVector) = loglikelihood(like, prediction) + logprior(x)

# ---------------------------------------------------------------------------
# Integration with the model layer
# ---------------------------------------------------------------------------

foregrounds(model::ACTDR6FullForegroundModel, like::ACTDR6FullLikelihood,
            p::ACTDR6Nuisance) = foregrounds(model, like, NamedTuple(p))

predict(like::ACTDR6FullLikelihood, cmb::ACTCMBTheory, fg::ACTForegrounds,
        p::ACTDR6Nuisance) = predict(like, cmb, fg, NamedTuple(p))

predict(like::ACTDR6FullLikelihood, cmb::ACTCMBTheory, fg::ACTForegrounds,
        p::ACTDR6Nuisance, cmb_indices::AbstractUnitRange{<:Integer}) =
    predict(like, cmb, fg, NamedTuple(p), cmb_indices)
