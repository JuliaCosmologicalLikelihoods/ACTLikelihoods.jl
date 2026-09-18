"""
    ACT_full_foregrounds.jl

ACT DR6 full-multifrequency foreground assembly, composed from the reviewed
CMBForegrounds numerical layer. This follows BandpowerForeground/fgspectra:
the total TT, TE, and EE arrays are constructed before survey-specific map
calibration and window convolution.
"""

struct ACTDR6FullForegroundModel{R, B, K, T, C, X}
    temperature_bands::Vector{R}
    polarization_bands::Vector{R}
    temperature_beams::Vector{B}
    polarization_beams::Vector{B}
    ksz::K
    tsz::T
    cibc::C
    szxcib::X
end

# The chromatic beams are measured instrument calibration products, never
# inference parameters, so foreground assembly uses the public fixed-beam
# bandpass API of CMBForegrounds. It is numerically identical to
# `prepare_chromatic_bandpass` / `eval_chromatic_sed_bands` and differs only in
# reverse mode, where the beam receives `NoTangent()` instead of a dense
# cotangent the size of the `(8500, ~600)` beam matrix. Derivatives with respect
# to the bandpass shifts are preserved, because the bands are shifted and
# renormalized inside the differentiated call.

@inline function _fixed_chromatic_sed_weight(sed, prepared, args...)
    return eval_fixed_chromatic_sed_bands(
        frequency -> sed_weight(sed, frequency, args...), prepared,
    )
end

function _load_act_template(path::AbstractString)
    raw = readdlm(path)
    size(raw, 2) == 2 || throw(ArgumentError("ACT foreground template must have two columns: $path"))
    ell = Int.(raw[:, 1])
    values = Float64.(raw[:, 2])
    minimum(ell) >= 0 || throw(ArgumentError("ACT foreground template has negative multipoles: $path"))
    template = zeros(Float64, maximum(ell) + 1)
    template[ell .+ 1] = values
    return template
end

function _load_act_raw_band(path::AbstractString)
    raw = readdlm(path)
    size(raw, 2) == 2 || throw(ArgumentError("ACT bandpass must have two columns: $path"))
    return RawBand(Float64.(raw[:, 1]), Float64.(raw[:, 2]))
end

"""
    ACTDR6FullForegroundModel(like::ACTDR6FullLikelihood)
    ACTDR6FullForegroundModel(asset_directory, like::ACTDR6FullLikelihood)

Build the ACT DR6 multifrequency foreground model: the ten array bandpasses
(temperature `_s0` and polarization `_s2`), the ten chromatic beam tables, and
the four original foreground templates (kSZ, tSZ, clustered CIB, tSZ×CIB).

The one-argument form reads them from the published Zenodo runtime artifact and
needs no local data tree. The directory form is retained for development and
validation; the channel order is taken from `like.channels`.

```julia
like = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
```
"""
function ACTDR6FullForegroundModel(like::ACTDR6FullLikelihood)
    return ACTDR6FullForegroundModel(act_dr6_tttee_artifact_path(), like)
end

function ACTDR6FullForegroundModel(asset_directory::AbstractString,
                                   like::ACTDR6FullLikelihood)
    bandpass_directory = joinpath(asset_directory, "bandpasses")
    beam_directory = joinpath(asset_directory, "beams")
    template_directory = joinpath(asset_directory, "templates")
    temperature_bands = [
        _load_act_raw_band(joinpath(bandpass_directory, "$(channel)_s0.txt"))
        for channel in like.channels
    ]
    polarization_bands = [
        _load_act_raw_band(joinpath(bandpass_directory, "$(channel)_s2.txt"))
        for channel in like.channels
    ]
    temperature_beams = [
        ChromaticBeam(like.ells, permutedims(Float64.(
            readdlm(joinpath(beam_directory, "$(channel)_s0.txt")),
        )))
        for channel in like.channels
    ]
    polarization_beams = [
        ChromaticBeam(like.ells, permutedims(Float64.(
            readdlm(joinpath(beam_directory, "$(channel)_s2.txt")),
        )))
        for channel in like.channels
    ]
    ksz = TemplateShape(_load_act_template(joinpath(template_directory, "cl_ksz_bat.dat"));
                        ell_0=3000, ell_min=0)
    tsz_base = TemplateShape(_load_act_template(joinpath(template_directory, "cl_tsz_150_bat.dat"));
                             ell_0=3000, ell_min=0)
    tsz = TiltedTemplateShape(tsz_base, 3000.0)
    cibc = TemplateShape(_load_act_template(joinpath(template_directory, "cl_cib_Choi2020.dat"));
                         ell_0=3000, ell_min=0)
    szxcib = TemplateShape(_load_act_template(joinpath(template_directory, "cl_sz_x_cib.dat"));
                           ell_0=3000, ell_min=0)
    return ACTDR6FullForegroundModel(
        temperature_bands, polarization_bands, temperature_beams, polarization_beams,
        ksz, tsz, cibc, szxcib,
    )
end

"""
    _required_act_parameter(params, key)

Fetch `key` from a nuisance container or throw. Uses the one lookup contract of
[`_act_parameter_lookup`](@ref), so a `String`-keyed dictionary resolves exactly
as a `Symbol`-keyed one does.
"""
@inline function _required_act_parameter(params, key::Symbol)
    value = _act_parameter_lookup(params, key)
    value === nothing && throw(ArgumentError("missing ACT nuisance parameter $key"))
    return value
end

"""
    foregrounds(model, like, nuisance)

Construct the ACT DR6 total foreground arrays in the original channel order.
The result can be passed directly to `predict(like, cmb, foregrounds, nuisance)`.
"""
function foregrounds(model::ACTDR6FullForegroundModel, like::ACTDR6FullLikelihood, params)
    length(model.temperature_bands) == length(like.channels) ||
        throw(DimensionMismatch("ACT foreground model channels do not match the likelihood"))
    length(model.polarization_bands) == length(like.channels) ||
        throw(DimensionMismatch("ACT foreground model channels do not match the likelihood"))
    length(model.temperature_beams) == length(like.channels) ||
        throw(DimensionMismatch("ACT foreground model channels do not match the likelihood"))
    length(model.polarization_beams) == length(like.channels) ||
        throw(DimensionMismatch("ACT foreground model channels do not match the likelihood"))

    shifts = [
        _required_act_parameter(params, Symbol("bandint_shift_$channel"))
        for channel in like.channels
    ]
    bands_T = [shift_and_normalize(band, shift)
               for (band, shift) in zip(model.temperature_bands, shifts)]
    bands_P = [shift_and_normalize(band, shift)
               for (band, shift) in zip(model.polarization_bands, shifts)]
    prepared_T = [prepare_fixed_chromatic_bandpass(band, beam)
                  for (band, beam) in zip(bands_T, model.temperature_beams)]
    prepared_P = [prepare_fixed_chromatic_bandpass(band, beam)
                  for (band, beam) in zip(bands_P, model.polarization_beams)]
    ells = like.ells

    a_tSZ = _required_act_parameter(params, :a_tSZ)
    alpha_tSZ = _required_act_parameter(params, :alpha_tSZ)
    a_kSZ = _required_act_parameter(params, :a_kSZ)
    a_p = _required_act_parameter(params, :a_p)
    beta_p = _required_act_parameter(params, :beta_p)
    a_c = _required_act_parameter(params, :a_c)
    beta_c = _required_act_parameter(params, :beta_c)
    a_s = _required_act_parameter(params, :a_s)
    beta_s = _required_act_parameter(params, :beta_s)
    a_gtt = _required_act_parameter(params, :a_gtt)
    a_gte = _required_act_parameter(params, :a_gte)
    a_gee = _required_act_parameter(params, :a_gee)
    a_psee = _required_act_parameter(params, :a_psee)
    a_pste = _required_act_parameter(params, :a_pste)
    xi = _required_act_parameter(params, :xi)
    alpha_s = _required_act_parameter(params, :alpha_s)
    T_effd = _required_act_parameter(params, :T_effd)
    beta_d = _required_act_parameter(params, :beta_d)
    alpha_dT = _required_act_parameter(params, :alpha_dT)
    alpha_dE = _required_act_parameter(params, :alpha_dE)
    alpha_p = _required_act_parameter(params, :alpha_p)
    T_d = _required_act_parameter(params, :T_d)

    tsz_sed = ThermalSZSED(150.0)
    cibp_sed = ModifiedBlackbodySED(150.0, T_d)
    cibc_sed = ModifiedBlackbodySED(150.0, T_d)
    dust_sed = ModifiedBlackbodySED(150.0, T_effd)
    radio_sed = RadioSED(150.0; convention=:rj)

    f_tsz_T = _fixed_chromatic_sed_weight(tsz_sed, prepared_T)
    f_cibp_T = _fixed_chromatic_sed_weight(cibp_sed, prepared_T, beta_p)
    f_cibc_T = _fixed_chromatic_sed_weight(cibc_sed, prepared_T, beta_c)
    f_dust_T = _fixed_chromatic_sed_weight(dust_sed, prepared_T, beta_d)
    f_dust_P = _fixed_chromatic_sed_weight(dust_sed, prepared_P, beta_d)
    f_radio_T = _fixed_chromatic_sed_weight(radio_sed, prepared_T, beta_s)
    f_radio_P = _fixed_chromatic_sed_weight(radio_sed, prepared_P, beta_s)
    f_ksz = _fixed_chromatic_sed_weight(ConstantSED(), prepared_T)

    ell_clp = ells .* (ells .+ 1)
    ell0_clp = 3000.0 * 3001.0
    cl_ksz = angular_power(model.ksz, ells; amp=a_kSZ)
    cl_tsz = angular_power(model.tsz, ells, alpha_tSZ; amp=a_tSZ)
    cl_cibc = angular_power(model.cibc, ells; amp=a_c)
    cl_szxcib = angular_power(model.szxcib, ells; amp=-xi * sqrt(a_tSZ * a_c))
    cl_cibp = (ell_clp ./ ell0_clp) .^ alpha_p
    cl_radio = (ell_clp ./ ell0_clp) .^ alpha_s
    cl_dust_T = angular_power(PowerLawShape(500.0), ells, alpha_dT)
    cl_dust_E = angular_power(PowerLawShape(500.0), ells, alpha_dE)

    TT = assemble_TT(a_p, a_gtt, a_s,
                     f_ksz, f_cibp_T, f_dust_T, f_radio_T, f_tsz_T, f_cibc_T,
                     cl_ksz, cl_cibp, cl_dust_T, cl_radio,
                     cl_tsz, cl_cibc, cl_szxcib)
    TE = assemble_TE(a_pste, a_gte,
                     f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                     cl_radio, cl_dust_E)
    EE = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dust_E)
    return ACTForegrounds(TT, TE, EE)
end
