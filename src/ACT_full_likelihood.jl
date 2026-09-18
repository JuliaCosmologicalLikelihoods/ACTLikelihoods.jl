"""
    ACT_full_likelihood.jl

Data model for the **full, non-marginalized** ACT DR6 multifrequency TT/TE/ET/EE
likelihood. It owns the ordered observation vector, the released bandpower
windows, the map-leg metadata, and the fixed Gaussian covariance.

Foreground assembly is a separate input (`ACT_full_foregrounds.jl`) so that the
released data mapping and the Gaussian likelihood stay independently testable
against original-code fixtures.

Conventions:

* Ordered TE/ET map legs are preserved exactly as released (`symmetrize = false`
  upstream). Spectra are never symmetrized.
* [`loglikelihood`](@ref) is **data only**: `-chi2/2`. The parameter-independent
  Gaussian normalization is available separately as
  [`gaussian_normalization`](@ref), and priors live in `ACT_nuisance.jl`.
"""

struct ACTCMBTheory{L<:AbstractVector{<:Integer}, V<:AbstractVector}
    ell::L
    TT::V
    TE::V
    EE::V

    function ACTCMBTheory(ell::L, TT::V, TE::V, EE::V) where {
        L<:AbstractVector{<:Integer}, V<:AbstractVector,
    }
        n = length(ell)
        n == length(TT) == length(TE) == length(EE) ||
            throw(DimensionMismatch("ell, TT, TE, and EE must have the same length"))
        all(diff(ell) .== 1) ||
            throw(ArgumentError("the CMB multipole grid must be contiguous and increasing"))
        return new{L, V}(ell, TT, TE, EE)
    end
end

struct ACTForegrounds{TT<:AbstractArray{<:Real, 3}, TE<:AbstractArray{<:Real, 3},
                      EE<:AbstractArray{<:Real, 3}}
    TT::TT
    TE::TE
    EE::EE
end

"""
    ACTFullSpectrum

One ordered ACT spectrum: its polarization, its two map legs, its bandpower
window and the slice of the observation vector it fills.

The window is stored packed, not dense. Every ACT DR6 bandpower window is
**exactly zero outside one contiguous run of multipoles**: 97.9% of the released
`(n_ell, n_bin)` matrices are structural zeros and the nonzero run averages 174
of 8500 multipoles. Keeping the runs costs 2.2 MiB across all 41 spectra instead
of 107 MiB, so the projection stops streaming a mostly-zero matrix, the
likelihood object stops carrying one, and reverse mode stops zeroing a cotangent
for one. [`window_matrix`](@ref) reconstructs the released dense form.

The packing is exact, not a truncation: the loader takes the run from the first
to the last nonzero entry of each released column, so no nonzero value is ever
dropped, whether or not the run has interior zeros.
"""
struct ACTFullSpectrum
    polarization::Symbol
    temperature_leg::Int
    polarization_leg::Int
    output_indices::UnitRange{Int}
    # Packed windows, one entry per bandpower of this spectrum:
    band_values::Vector{Float64}   # concatenated nonzero runs
    band_starts::Vector{Int}       # first multipole index of each run
    band_offsets::Vector{Int}      # 1-based offsets into band_values, length n_bin + 1
end

"""
    window_matrix(spectrum, n_ell) -> Matrix{Float64}

Reconstruct the released `(n_ell, n_bin)` dense window matrix of `spectrum` from
its packed bands. Use `n_ell = length(like.ells)`.

Exact: every value the released file carried is present, and every entry this
fills with zero was zero in the released file.
"""
function window_matrix(spectrum::ACTFullSpectrum, n_ell::Integer)
    n_bin = length(spectrum.output_indices)
    window = zeros(Float64, n_ell, n_bin)
    for bin in 1:n_bin
        offset = spectrum.band_offsets[bin]
        width = spectrum.band_offsets[bin + 1] - offset
        start = spectrum.band_starts[bin]
        for step in 0:(width - 1)
            window[start + step, bin] = spectrum.band_values[offset + step]
        end
    end
    return window
end

"""
    _pack_window_bands(window) -> (values, starts, offsets)

Pack a released `(n_ell, n_bin)` window matrix into its nonzero runs.

Bandpower `b` occupies `values[offsets[b]:(offsets[b+1]-1)]` and multiplies
multipole indices `starts[b] .+ (0:(offsets[b+1]-offsets[b]-1))`.

The run is taken from the first to the last nonzero entry of the column, so
dropping everything outside it is exact by construction whether or not the run
has interior zeros; for the released ACT DR6 windows it has none.
"""
function _pack_window_bands(window::AbstractMatrix{Float64})
    n_bin = size(window, 2)
    values = Float64[]
    starts = Vector{Int}(undef, n_bin)
    offsets = Vector{Int}(undef, n_bin + 1)
    offsets[1] = 1
    for bin in 1:n_bin
        column = @view window[:, bin]
        first_index = findfirst(!iszero, column)
        if first_index === nothing
            starts[bin] = 1
        else
            last_index = findlast(!iszero, column)::Int
            starts[bin] = first_index
            append!(values, @view column[first_index:last_index])
        end
        offsets[bin + 1] = length(values) + 1
    end
    return values, starts, offsets
end

"""
    ACTDR6FullLikelihood

The full (non-marginalized) ACT DR6 multifrequency likelihood: 1651 bandpowers
over 41 ordered TT/TE/ET/EE spectra across five array-frequency channels, on the
theory multipole grid `2:8501`.

Construct it with no arguments to use the published Zenodo artifact:

```julia
like = ACTDR6FullLikelihood()
```

# Fields
- `observed`: the ordered 1651-element observation vector
- `covariance_cholesky`: Cholesky factorization of the fixed Gaussian covariance
- `channels`: ordered array-frequency channel names
- `spectra`: ordered per-spectrum window and map-leg metadata
- `ells`: the theory multipole grid the windows are defined on
- `source`: the directory the data were read from
- `log_normalization`: `-N/2·log(2π) - ½·logdet C`, the parameter-independent
  Gaussian constant that this package deliberately keeps out of
  [`loglikelihood`](@ref)
"""
struct ACTDR6FullLikelihood
    observed::Vector{Float64}
    covariance_cholesky::Cholesky{Float64, Matrix{Float64}}
    channels::Vector{String}
    spectra::Vector{ACTFullSpectrum}
    ells::Vector{Int}
    source::String
    log_normalization::Float64
end

"""
    _act_parameter_lookup(params, key) -> value or `nothing`

The single name-lookup contract for ACT nuisance containers, shared by the
foreground assembly and the observation-space mapping so the two cannot
disagree about what a container holds.

`NamedTuple`s are looked up by symbol. `AbstractDict`s accept either a symbol
key or the equivalent `String` key, matching what `ACTDR6Nuisance(::AbstractDict)`
accepts. Anything else is not a parameter container.
"""
@inline _act_parameter_lookup(params::NamedTuple, key::Symbol) =
    haskey(params, key) ? params[key] : nothing

@inline function _act_parameter_lookup(params::AbstractDict, key::Symbol)
    haskey(params, key) && return params[key]
    string_key = String(key)
    haskey(params, string_key) && return params[string_key]
    return nothing
end

_act_parameter_lookup(params, ::Symbol) = throw(ArgumentError(
    "ACT nuisance parameters must be a NamedTuple or AbstractDict, got $(typeof(params))",
))

function _read_full_metadata(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("ACT spectrum metadata is empty: $path"))
    header = split(chomp(first(lines)), '\t')
    required = [
        "index", "polarization", "temperature_leg", "polarization_leg",
        "source_t1", "source_t2", "reversed_cross_spectrum", "number_of_bins",
    ]
    header == required || throw(ArgumentError("unexpected ACT spectrum metadata columns in $path"))

    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        fields = split(chomp(line), '\t')
        length(fields) == length(required) ||
            throw(ArgumentError("malformed ACT spectrum metadata row in $path"))
        polarization = Symbol(uppercase(fields[2]))
        polarization in (:TT, :TE, :EE) ||
            throw(ArgumentError("unsupported ACT polarization $(fields[2])"))
        push!(rows, (
            index = parse(Int, fields[1]),
            polarization = polarization,
            temperature_leg = fields[3],
            polarization_leg = fields[4],
            source_t1 = fields[5],
            source_t2 = fields[6],
            reversed = lowercase(fields[7]) == "true",
            nbin = parse(Int, fields[8]),
        ))
    end
    [row.index for row in rows] == collect(1:length(rows)) ||
        throw(ArgumentError("ACT spectrum metadata indices must be consecutive and one-based"))
    return rows
end

function _full_window_path(windows_directory::AbstractString, row)
    suffix = String(row.polarization)
    candidates = (
        "$(row.source_t1)_x_$(row.source_t2)_$suffix.txt",
        "$(row.source_t1)_$(row.source_t2)_$suffix.txt",
    )
    for candidate in candidates
        path = joinpath(windows_directory, candidate)
        isfile(path) && return path
    end
    throw(ArgumentError("cannot find ACT window for $(row.source_t1), $(row.source_t2), $suffix"))
end

"""
    ACTDR6FullLikelihood()
    ACTDR6FullLikelihood(data_directory; validate_metadata)

Load the full (non-marginalized) ACT DR6 multifrequency likelihood.

The zero-argument form uses the published Zenodo runtime artifact
(`$(ACT_DR6_TTTEEE_DOI)`), downloading it on first use, and always validates
`metadata.json`. It requires no local data tree and is the ordinary user path.

The directory form is retained for development and validation. The directory
must contain `data_vec.txt`, `cov.txt`, `spectrum_metadata.tsv`, and a
`windows/` directory. `spectrum_metadata.tsv` is generated from the original
SACC selection; it retains the ordered TE and ET map legs that cannot be
recovered from nominal frequency labels alone. `validate_metadata` defaults to
`true` when the directory carries a `metadata.json`, and checking it is
mandatory for the published artifact.

Metadata validation covers the artifact identity, `marginalized == false`, the
released dimensions, the channel order, the multipole range, the source SACC
digest and the upstream revisions. The 474 MB `SHA256SUMS` ledger is *not*
hashed here; [`verify_act_dr6_checksums`](@ref) does that in the test suite.

# Example
```julia
like = ACTDR6FullLikelihood()
length(like.observed)   # 1651
length(like.spectra)    # 41
like.channels           # 5 ordered array-frequency channels
```
"""
function ACTDR6FullLikelihood()
    return ACTDR6FullLikelihood(act_dr6_tttee_artifact_path(); validate_metadata=true)
end

function ACTDR6FullLikelihood(
    data_directory::AbstractString;
    validate_metadata::Bool = isfile(joinpath(data_directory, "metadata.json")),
)
    validate_metadata && validate_act_dr6_metadata(data_directory)
    observed = Float64.(vec(readdlm(joinpath(data_directory, "data_vec.txt"))))
    covariance = Float64.(readdlm(joinpath(data_directory, "cov.txt")))
    size(covariance) == (length(observed), length(observed)) ||
        throw(DimensionMismatch("ACT covariance must match the observation-vector length"))
    covariance_cholesky = cholesky(Symmetric(covariance, :L))

    rows = _read_full_metadata(joinpath(data_directory, "spectrum_metadata.tsv"))
    sum(row.nbin for row in rows) == length(observed) ||
        throw(DimensionMismatch("ACT spectrum metadata bin counts do not match observations"))

    channels = String[]
    for row in rows
        for channel in (row.temperature_leg, row.polarization_leg)
            channel in channels || push!(channels, channel)
        end
    end
    channel_indices = Dict(channel => index for (index, channel) in enumerate(channels))
    windows_directory = joinpath(data_directory, "windows")
    spectra = ACTFullSpectrum[]
    output_start = 1
    n_ell = nothing
    for row in rows
        window = Float64.(readdlm(_full_window_path(windows_directory, row)))
        size(window, 2) == row.nbin ||
            throw(DimensionMismatch("ACT window bin count does not match metadata"))
        if isnothing(n_ell)
            n_ell = size(window, 1)
        elseif size(window, 1) != n_ell
            throw(DimensionMismatch("ACT windows must have a common theory multipole grid"))
        end
        output_indices = output_start:(output_start + row.nbin - 1)
        band_values, band_starts, band_offsets = _pack_window_bands(window)
        push!(spectra, ACTFullSpectrum(
            row.polarization,
            channel_indices[row.temperature_leg],
            channel_indices[row.polarization_leg],
            output_indices,
            band_values,
            band_starts,
            band_offsets,
        ))
        output_start += row.nbin
    end
    isnothing(n_ell) && throw(ArgumentError("ACT data has no spectra"))
    ells = collect(2:(n_ell + 1))
    log_normalization = _gaussian_log_normalization(covariance_cholesky, length(observed))
    return ACTDR6FullLikelihood(observed, covariance_cholesky, channels, spectra, ells,
                                String(data_directory), log_normalization)
end

"""
    _gaussian_log_normalization(factorization, n)

The parameter-independent Gaussian constant `-n/2·log(2π) - ½·logdet C`, derived
from the Cholesky factor rather than copied from upstream.
"""
function _gaussian_log_normalization(factorization::Cholesky, n::Integer)
    log_determinant = 2 * sum(log, diag(factorization.L))
    return -n / 2 * log(2 * pi) - log_determinant / 2
end

"""
    gaussian_normalization(like::ACTDR6FullLikelihood)

Return the fixed Gaussian normalization `-N/2·log(2π) - ½·logdet C`
(`-2145.1713776754923` for the released covariance).

This package keeps it **out** of [`loglikelihood`](@ref), which is data only.
The upstream `act_dr6_mflike` `loglike` equals
`loglikelihood(like, prediction) + gaussian_normalization(like)`.
"""
@inline gaussian_normalization(like::ACTDR6FullLikelihood) = like.log_normalization

function Base.show(io::IO, like::ACTDR6FullLikelihood)
    print(io, "ACTDR6FullLikelihood(", length(like.observed), " bandpowers, ",
          length(like.spectra), " ordered spectra, ", length(like.channels),
          " channels, ell ", first(like.ells), ":", last(like.ells), ")")
    return nothing
end

"""
    _polarization_block(TT, TE, EE, polarization)

Select the block of a TT/TE/EE triple by polarization. Taking the three blocks
as separate arguments rather than the containing struct is what lets
[`_act_projection`](@ref) be a single reverse-mode primitive over plain arrays.
"""
@inline function _polarization_block(TT, TE, EE, polarization::Symbol)
    polarization === :TT && return TT
    polarization === :TE && return TE
    polarization === :EE && return EE
    throw(ArgumentError("unsupported ACT polarization $polarization"))
end

function _act_cmb_indices(cmb::ACTCMBTheory, ells::AbstractVector{<:Integer})
    first_index = findfirst(==(first(ells)), cmb.ell)
    isnothing(first_index) &&
        throw(ArgumentError("CMB spectrum does not include ell=$(first(ells))"))
    last_index = first_index + length(ells) - 1
    last_index <= length(cmb.ell) ||
        throw(ArgumentError("CMB spectrum does not cover the ACT window support"))
    # The grid check is written as an explicit loop on purpose. A vectorised
    # `cmb.ell[first_index:last_index] == ells` on two integer vectors lowers
    # to a `memcmp` foreigncall that Mooncake cannot trace, and this function
    # sits directly on the public `predict(like, cmb, fg, p)` path. The loop
    # also avoids allocating the slice.
    @inbounds for offset in eachindex(ells)
        cmb.ell[first_index + offset - 1] == ells[offset] ||
            throw(ArgumentError("CMB multipoles do not match the ACT window grid"))
    end
    return first_index:last_index
end

function _validate_foregrounds(like::ACTDR6FullLikelihood, foregrounds::ACTForegrounds)
    expected = (length(like.channels), length(like.channels), length(like.ells))
    for field in (foregrounds.TT, foregrounds.TE, foregrounds.EE)
        size(field) == expected ||
            throw(DimensionMismatch("ACT foreground arrays must have shape $expected"))
    end
    return nothing
end

"""
    _map_calibration(params, field, channel)

The calibration factor applied to one map leg. Every calibration the model
actually varies is **required**: silently defaulting a missing calibration to
unity would let `predict` return a wrong model for an incomplete parameter
container, while foreground assembly rejects the same container.

`calT` is the one genuinely implicit quantity: ACT DR6 fixes the temperature
efficiency at `1` and carries no `calT_*` parameter at all.
"""
@inline function _map_calibration(params, field::Symbol, channel::String)
    field in (:T, :E) || throw(ArgumentError("ACT calibration field must be :T or :E"))
    calibration = inv(_required_act_parameter(params, :calG_all))
    calibration /= _required_act_parameter(params, Symbol("cal_$channel"))
    # `calT ≡ 1` by construction; only the polarization efficiency is a parameter.
    field === :T && return calibration
    return calibration / _required_act_parameter(params, Symbol("calE_$channel"))
end

@inline _fixed_lower_solve(factor::LowerTriangular, residual::AbstractVector) =
    factor \ residual

"""
    predict(like::ACTDR6FullLikelihood, cmb, foregrounds, nuisance)

Return the final ACT observation-space prediction. `cmb` contains lensed D_ell
spectra in microkelvin squared on an explicit integer multipole grid.
`foregrounds` contains pre-assembled `(channel, channel, ell)` total D_ell
arrays for TT, TE, and EE. This boundary keeps the released data mapping and
Gaussian likelihood independently testable while foreground assembly gains its
own original-code component fixtures.
"""
function predict(like::ACTDR6FullLikelihood, cmb::ACTCMBTheory,
                 foregrounds::ACTForegrounds, nuisance)
    cmb_indices = _act_cmb_indices(cmb, like.ells)
    return predict(like, cmb, foregrounds, nuisance, cmb_indices)
end

"""
    predict(like, cmb, foregrounds, nuisance, cmb_indices)

Evaluate the ACT prediction using prevalidated `cmb_indices`. This is the hot
path for repeated likelihood and automatic-differentiation calls.
"""
function predict(like::ACTDR6FullLikelihood, cmb::ACTCMBTheory,
                 foregrounds::ACTForegrounds, nuisance,
                 cmb_indices::AbstractUnitRange{<:Integer})
    _validate_foregrounds(like, foregrounds)
    calibrations = _spectrum_calibrations(like, nuisance)
    return _act_projection(like, cmb.TT, cmb.TE, cmb.EE,
                           foregrounds.TT, foregrounds.TE, foregrounds.EE,
                           calibrations, Int(first(cmb_indices)))
end

"""
    _spectrum_calibrations(like, nuisance) -> Vector

The single calibration factor each ordered spectrum applies, in spectrum order:
the product of the two map-leg factors. Kept outside [`_act_projection`](@ref)
so the parameter-dependent part stays ordinary differentiable scalar code and
the array kernel stays a single reverse-mode primitive.
"""
function _spectrum_calibrations(like::ACTDR6FullLikelihood, nuisance)
    return map(like.spectra) do spectrum
        first_field = spectrum.polarization === :EE ? :E : :T
        second_field = spectrum.polarization === :TT ? :T : :E
        _map_calibration(
            nuisance, first_field, like.channels[spectrum.temperature_leg],
        ) * _map_calibration(
            nuisance, second_field, like.channels[spectrum.polarization_leg],
        )
    end
end

"""
    _act_projection(like, cmb_TT, cmb_TE, cmb_EE, fg_TT, fg_TE, fg_EE,
                    calibrations, cmb_offset) -> model

Map theory to the 1651-element ACT observation vector: add the foreground block
to the CMB spectrum on each ordered map-leg pair, apply that spectrum's
calibration, and contract with the bandpower windows.

This is deliberately **one** function rather than a per-spectrum pipeline. It is
the whole array kernel of `predict`, and it is registered as a single
reverse-mode primitive, so the tape carries one entry instead of three per
spectrum and no per-spectrum length-`n_ell` temporary is ever exposed to AD.
The scratch buffer and the `model` write below are invisible to reverse mode for
that reason; ForwardDiff runs straight through this code with `Dual` elements,
which is why `T` is promoted rather than assumed.

Only the multipoles each window actually covers are touched. The released
windows are exactly zero outside one contiguous run per bandpower, so this is
exact, not an approximation, and it skips both the mostly-zero matrix and the
multipoles (the first 574 and last 576 of the 8500) that no window reaches.

The reverse-mode rule needs the projection *before* calibration to form the
calibration cotangents, so the work is split: [`_act_projection_parts`](@ref)
returns both vectors and this function, the primitive, keeps only the model. The
rule calls the former, which is why nothing has to divide by a calibration that
could be zero.
"""
function _act_projection(like::ACTDR6FullLikelihood,
                         cmb_TT::AbstractVector, cmb_TE::AbstractVector,
                         cmb_EE::AbstractVector,
                         fg_TT::AbstractArray{<:Real, 3},
                         fg_TE::AbstractArray{<:Real, 3},
                         fg_EE::AbstractArray{<:Real, 3},
                         calibrations::AbstractVector, cmb_offset::Int)
    model, _ = _act_projection_parts(like, cmb_TT, cmb_TE, cmb_EE,
                                     fg_TT, fg_TE, fg_EE, calibrations, cmb_offset)
    return model
end

"""
    _check_projection_inputs(like, …)

Validate everything [`_act_projection_parts`](@ref) then indexes with
`@inbounds`. The contraction reads the CMB at an explicit offset and the
foreground blocks by map leg, so a caller that reaches the prevalidated
`predict(like, cmb, fg, p, cmb_indices)` hot path with a range that does not
cover the window grid must get a `DimensionMismatch` here, not an out-of-bounds
read.
"""
function _check_projection_inputs(like::ACTDR6FullLikelihood,
                                  cmb_TT, cmb_TE, cmb_EE,
                                  fg_TT, fg_TE, fg_EE, calibrations,
                                  cmb_offset::Int)
    n_ell = length(like.ells)
    last_index = cmb_offset + n_ell - 1
    for cmb_block in (cmb_TT, cmb_TE, cmb_EE)
        Base.require_one_based_indexing(cmb_block)
        (cmb_offset >= 1 && last_index <= length(cmb_block)) || throw(DimensionMismatch(
            "CMB spectra must cover the $(n_ell)-multipole ACT window grid; " *
            "offset $cmb_offset needs indices $cmb_offset:$last_index of $(length(cmb_block))",
        ))
    end
    n_channel = length(like.channels)
    for block in (fg_TT, fg_TE, fg_EE)
        Base.require_one_based_indexing(block)
        size(block) == (n_channel, n_channel, n_ell) || throw(DimensionMismatch(
            "ACT foreground arrays must have shape $((n_channel, n_channel, n_ell))",
        ))
    end
    length(calibrations) == length(like.spectra) || throw(DimensionMismatch(
        "expected one calibration per ordered spectrum, got $(length(calibrations))",
    ))
    return nothing
end

"""
    _act_projection_parts(like, …) -> (model, uncalibrated)

[`_act_projection`](@ref) together with the projection before the per-spectrum
calibration factor, which is what the reverse-mode rule differentiates the
calibrations through.
"""
function _act_projection_parts(like::ACTDR6FullLikelihood,
                               cmb_TT::AbstractVector, cmb_TE::AbstractVector,
                               cmb_EE::AbstractVector,
                               fg_TT::AbstractArray{<:Real, 3},
                               fg_TE::AbstractArray{<:Real, 3},
                               fg_EE::AbstractArray{<:Real, 3},
                               calibrations::AbstractVector, cmb_offset::Int)
    _check_projection_inputs(like, cmb_TT, cmb_TE, cmb_EE,
                             fg_TT, fg_TE, fg_EE, calibrations, cmb_offset)
    T = promote_type(eltype(cmb_TT), eltype(cmb_TE), eltype(cmb_EE),
                     eltype(fg_TT), eltype(fg_TE), eltype(fg_EE),
                     eltype(calibrations))
    n_out = length(like.observed)
    model = Vector{T}(undef, n_out)
    uncalibrated = Vector{T}(undef, n_out)
    buffer = Vector{T}(undef, length(like.ells))
    for (index, spectrum) in enumerate(like.spectra)
        cmb_block = _polarization_block(cmb_TT, cmb_TE, cmb_EE, spectrum.polarization)
        foreground = _polarization_block(fg_TT, fg_TE, fg_EE, spectrum.polarization)
        calibration = calibrations[index]
        legs = (spectrum.temperature_leg, spectrum.polarization_leg)
        values = spectrum.band_values
        starts = spectrum.band_starts
        offsets = spectrum.band_offsets
        # One pass to add the CMB and the foreground leg over the span this
        # spectrum's windows reach, then a dot product per bandpower.
        span = _band_span(starts, offsets)
        @inbounds for ell_index in span
            buffer[ell_index] = cmb_block[cmb_offset + ell_index - 1] +
                foreground[legs[1], legs[2], ell_index]
        end
        @inbounds for (bin, output) in enumerate(spectrum.output_indices)
            offset = offsets[bin]
            width = offsets[bin + 1] - offset
            start = starts[bin]
            total = zero(T)
            @simd for step in 0:(width - 1)
                total += values[offset + step] * buffer[start + step]
            end
            uncalibrated[output] = total
            model[output] = total * calibration
        end
    end
    return model, uncalibrated
end

"""
    _band_span(starts, offsets) -> UnitRange

The multipole indices covered by any bandpower of one spectrum.
"""
@inline function _band_span(starts::Vector{Int}, offsets::Vector{Int})
    lowest = typemax(Int)
    highest = 0
    @inbounds for bin in eachindex(starts)
        width = offsets[bin + 1] - offsets[bin]
        width == 0 && continue
        lowest = min(lowest, starts[bin])
        highest = max(highest, starts[bin] + width - 1)
    end
    return lowest > highest ? (1:0) : (lowest:highest)
end



"""
    chi2(like::ACTDR6FullLikelihood, prediction)

Evaluate the fixed-covariance ACT data chi-square directly from the stored lower
Cholesky factor. This intentionally excludes priors and the Gaussian
normalization constant.
"""
function chi2(like::ACTDR6FullLikelihood, prediction::AbstractVector)
    length(prediction) == length(like.observed) ||
        throw(DimensionMismatch("ACT prediction must match the observation-vector length"))
    residual = like.observed .- prediction
    whitened = _fixed_lower_solve(like.covariance_cholesky.L, residual)
    return dot(whitened, whitened)
end

"""
    loglikelihood(like::ACTDR6FullLikelihood, prediction)

Return the **data-only** ACT log likelihood, `-chi2/2`.

Priors and the parameter-independent Gaussian normalization are intentionally
external, so that this value depends on nothing but the data, the covariance and
the model vector:

| quantity | value at the reference point |
|---|---|
| `chi2(like, prediction)` | `1592.0584129991541` |
| `loglikelihood(like, prediction)` | `-796.0292064995771` |
| `gaussian_normalization(like)` | `-2145.1713776754923` |
| upstream `act_dr6_mflike` `loglike` | `-2941.2005841750693` |

Use [`gaussian_normalization`](@ref) to recover the upstream convention, and
[`logposterior`](@ref) to add the official ACT DR6 nuisance priors.
"""
@inline loglikelihood(like::ACTDR6FullLikelihood, prediction::AbstractVector) =
    -chi2(like, prediction) / 2
