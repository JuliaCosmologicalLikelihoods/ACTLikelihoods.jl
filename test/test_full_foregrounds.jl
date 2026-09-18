"""
    test/test_full_foregrounds.jl

Foreground assembly against independently generated original-code checkpoints:
per-component, totals, whole-array moments, SED weights, and the bandpass and
chromatic-beam responses.
"""

using CMBForegrounds: RawBand, shift_and_normalize, trapz,
                     prepare_fixed_chromatic_bandpass, eval_fixed_chromatic_sed_bands

const TOTAL_FOREGROUNDS = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)

"""
    check_checkpoints(rows, lookup; rtol, atol) -> Int

Compare every exported `(block, i, j, ell, value)` checkpoint against the Julia
array selected by `lookup(block)`. Returns the number of comparisons.
"""
function check_checkpoints(rows, lookup, block_columns::Int; rtol=1e-10, atol=1e-12)
    compared = 0
    for row in axes(rows, 1)
        block = ntuple(k -> String(rows[row, k]), block_columns)
        array = lookup(block)
        i = Int(rows[row, block_columns + 1])
        j = Int(rows[row, block_columns + 2])
        ell = Int(rows[row, block_columns + 3])
        expected = Float64(rows[row, block_columns + 4])
        @test array[i, j, ell_index(ell)] ≈ expected rtol=rtol atol=atol
        compared += 1
    end
    return compared
end

@testset "ACT DR6 foregrounds — only the public CMBForegrounds API is used" begin
    # Foreground assembly uses the exported fixed-beam bandpass API. Assert the
    # interface this package relies on is public, and that no private symbol
    # crept back into the source, the tests or the documentation.
    for name in (:prepare_fixed_chromatic_bandpass, :eval_fixed_chromatic_sed_bands)
        @test isdefined(CMBForegrounds, name)
        @test Base.isexported(CMBForegrounds, name)
    end

    root = normpath(joinpath(@__DIR__, ".."))
    sources = String[]
    for directory in ("src", "test")
        for (path, _, files) in walkdir(joinpath(root, directory))
            append!(sources, joinpath(path, f) for f in files if endswith(f, ".jl"))
        end
    end
    push!(sources, joinpath(root, "README.md"))
    # Assembled at run time so this assertion does not match its own source.
    needle = "CMBForegrounds" * "._"
    for path in sources
        @test !occursin(needle, read(path, String))
    end
end

@testset "ACT DR6 foregrounds — model layout" begin
    @test length(FOREGROUND_MODEL.temperature_bands) == 5
    @test length(FOREGROUND_MODEL.polarization_bands) == 5
    @test length(FOREGROUND_MODEL.temperature_beams) == 5
    @test length(FOREGROUND_MODEL.polarization_beams) == 5
    for beam in FOREGROUND_MODEL.temperature_beams
        @test size(beam.beam, 1) == length(LIKE.ells)
        @test beam.ells == LIKE.ells
    end
    expected = (5, 5, length(LIKE.ells))
    @test size(TOTAL_FOREGROUNDS.TT) == expected
    @test size(TOTAL_FOREGROUNDS.TE) == expected
    @test size(TOTAL_FOREGROUNDS.EE) == expected
end

@testset "ACT DR6 foregrounds — total checkpoints" begin
    _, rows = read_reference_table(REFERENCE_DIR, "foreground_totals_checkpoints.tsv")
    compared = check_checkpoints(
        rows, block -> foreground_block(TOTAL_FOREGROUNDS, block[1]), 1,
    )
    # 3 polarizations x 25 ordered channel pairs x 11 multipoles.
    @test compared == 3 * 25 * 11
end

@testset "ACT DR6 foregrounds — whole-array moments" begin
    # The checkpoints sample 11 multipoles; these reductions run over all 8500,
    # and the ell-weighted sum additionally detects a reversed or rolled axis.
    header, rows = read_reference_table(REFERENCE_DIR, "foreground_totals_moments.tsv")
    @test header == ["polarization", "i", "j", "sum", "abs_sum", "ell_weighted_sum"]
    weights = Float64.(LIKE.ells)
    for row in axes(rows, 1)
        array = foreground_block(TOTAL_FOREGROUNDS, String(rows[row, 1]))
        spectrum = @view array[Int(rows[row, 2]), Int(rows[row, 3]), :]
        @test sum(spectrum) ≈ Float64(rows[row, 4]) rtol=1e-10 atol=1e-12
        @test sum(abs, spectrum) ≈ Float64(rows[row, 5]) rtol=1e-10 atol=1e-12
        @test dot(spectrum, weights) ≈ Float64(rows[row, 6]) rtol=1e-10 atol=1e-9
    end
    @test size(rows, 1) == 3 * 25
end

@testset "ACT DR6 foregrounds — individual components" begin
    reference = free_parameters(REFERENCE_NUISANCE)

    # Each component is linear in its own amplitude, so switching the others off
    # reproduces exactly the array the upstream exporter wrote under that name.
    ksz = isolated_foregrounds(a_kSZ = reference.a_kSZ)
    tsz = isolated_foregrounds(a_tSZ = reference.a_tSZ)
    cibc = isolated_foregrounds(a_c = reference.a_c)
    cibp = isolated_foregrounds(a_p = reference.a_p)
    radio_tt = isolated_foregrounds(a_s = reference.a_s)
    dust_tt = isolated_foregrounds(a_gtt = reference.a_gtt)
    radio_te = isolated_foregrounds(a_pste = reference.a_pste)
    dust_te = isolated_foregrounds(a_gte = reference.a_gte)
    radio_ee = isolated_foregrounds(a_psee = reference.a_psee)
    dust_ee = isolated_foregrounds(a_gee = reference.a_gee)

    # tSZ x CIB cannot be switched on alone: it only exists inside the
    # correlated tSZ+CIB block, so it is recovered as a difference.
    correlated = isolated_foregrounds(a_tSZ = reference.a_tSZ, a_c = reference.a_c,
                                      xi = reference.xi)
    uncorrelated = isolated_foregrounds(a_tSZ = reference.a_tSZ, a_c = reference.a_c)
    szxcib = correlated.TT .- uncorrelated.TT
    tsz_and_cib = correlated.TT

    blocks = Dict(
        ("kSZ", "tt") => ksz.TT,
        ("tSZ", "tt") => tsz.TT,
        ("cibc", "tt") => cibc.TT,
        ("cibp", "tt") => cibp.TT,
        ("radio", "tt") => radio_tt.TT,
        ("dust", "tt") => dust_tt.TT,
        ("szxcib", "tt") => szxcib,
        ("tSZ_and_CIB", "tt") => tsz_and_cib,
        ("radio", "te") => radio_te.TE,
        ("dust", "te") => dust_te.TE,
        ("radio", "ee") => radio_ee.EE,
        ("dust", "ee") => dust_ee.EE,
    )

    _, rows = read_reference_table(REFERENCE_DIR, "foreground_components_checkpoints.tsv")
    compared = check_checkpoints(rows, block -> blocks[(block[1], block[2])], 2)
    @test compared == length(blocks) * 25 * 11

    header, moments = read_reference_table(REFERENCE_DIR,
                                           "foreground_components_moments.tsv")
    @test header == ["component", "polarization", "i", "j",
                     "sum", "abs_sum", "ell_weighted_sum"]
    weights = Float64.(LIKE.ells)
    for row in axes(moments, 1)
        array = blocks[(String(moments[row, 1]), String(moments[row, 2]))]
        spectrum = @view array[Int(moments[row, 3]), Int(moments[row, 4]), :]
        @test sum(spectrum) ≈ Float64(moments[row, 5]) rtol=1e-9 atol=1e-11
        @test sum(abs, spectrum) ≈ Float64(moments[row, 6]) rtol=1e-9 atol=1e-11
        @test dot(spectrum, weights) ≈ Float64(moments[row, 7]) rtol=1e-9 atol=1e-8
    end

    # Components must add up to the released total.
    reconstructed_tt = ksz.TT .+ cibp.TT .+ radio_tt.TT .+ dust_tt.TT .+ tsz_and_cib
    @test reconstructed_tt ≈ TOTAL_FOREGROUNDS.TT rtol=1e-10 atol=1e-12
    @test radio_te.TE .+ dust_te.TE ≈ TOTAL_FOREGROUNDS.TE rtol=1e-10 atol=1e-12
    @test radio_ee.EE .+ dust_ee.EE ≈ TOTAL_FOREGROUNDS.EE rtol=1e-10 atol=1e-12
end

@testset "ACT DR6 foregrounds — TE is not symmetric in its map legs" begin
    # The TE foreground block carries T on the first index and E on the second.
    # If it came out symmetric, the ordered ET legs would be indistinguishable.
    asymmetry = maximum(abs, TOTAL_FOREGROUNDS.TE .- permutedims(TOTAL_FOREGROUNDS.TE, (2, 1, 3)))
    @test asymmetry > 0
    for pol in (TOTAL_FOREGROUNDS.TT, TOTAL_FOREGROUNDS.EE)
        @test pol ≈ permutedims(pol, (2, 1, 3)) rtol=1e-12
    end
end

@testset "ACT DR6 foregrounds — SED weights" begin
    header, rows = read_reference_table(REFERENCE_DIR, "sed_checkpoints.tsv")
    @test header == ["sed", "field", "channel_index", "ell", "value"]

    reference = free_parameters(REFERENCE_NUISANCE)
    shifts = [getfield(reference, Symbol("bandint_shift_", channel))
              for channel in LIKE.channels]
    bands_T = [shift_and_normalize(band, shift)
               for (band, shift) in zip(FOREGROUND_MODEL.temperature_bands, shifts)]
    bands_P = [shift_and_normalize(band, shift)
               for (band, shift) in zip(FOREGROUND_MODEL.polarization_bands, shifts)]
    prepared_T = [prepare_fixed_chromatic_bandpass(band, beam)
                  for (band, beam) in zip(bands_T, FOREGROUND_MODEL.temperature_beams)]
    prepared_P = [prepare_fixed_chromatic_bandpass(band, beam)
                  for (band, beam) in zip(bands_P, FOREGROUND_MODEL.polarization_beams)]

    fixed = ACT_DR6_FIXED_PARAMETERS
    weights = Dict(
        ("ksz", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(ConstantSED(), prepared_T),
        ("tsz", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(ThermalSZSED(150.0), prepared_T),
        ("cibp", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(
            ModifiedBlackbodySED(150.0, fixed.T_d), prepared_T, reference.beta_p),
        ("cibc", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(
            ModifiedBlackbodySED(150.0, fixed.T_d), prepared_T, reference.beta_p),
        ("dust", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(
            ModifiedBlackbodySED(150.0, fixed.T_effd), prepared_T, fixed.beta_d),
        ("dust", "P") => ACTLikelihoods._fixed_chromatic_sed_weight(
            ModifiedBlackbodySED(150.0, fixed.T_effd), prepared_P, fixed.beta_d),
        ("radio", "T") => ACTLikelihoods._fixed_chromatic_sed_weight(
            RadioSED(150.0; convention=:rj), prepared_T, reference.beta_s),
        ("radio", "P") => ACTLikelihoods._fixed_chromatic_sed_weight(
            RadioSED(150.0; convention=:rj), prepared_P, reference.beta_s),
    )

    for row in axes(rows, 1)
        key = (String(rows[row, 1]), String(rows[row, 2]))
        array = weights[key]
        channel = Int(rows[row, 3])
        ell = Int(rows[row, 4])
        @test array[channel, ell_index(ell)] ≈ Float64(rows[row, 5]) rtol=1e-11 atol=1e-13
    end
    @test size(rows, 1) == length(weights) * 5 * 11

    # kSZ has no frequency dependence once the band is normalized.
    @test all(≈(1.0), weights[("ksz", "T")])
end

@testset "ACT DR6 foregrounds — bandpass and chromatic beam responses" begin
    header, rows = read_reference_table(REFERENCE_DIR, "bandpass_checkpoints.tsv")
    @test header == ["channel", "field", "channel_index", "nu_index", "nu",
                     "nu_shifted", "transmission", "normalized_transmission"]

    reference = free_parameters(REFERENCE_NUISANCE)
    raw_band(field, index) = field == "s0" ? FOREGROUND_MODEL.temperature_bands[index] :
                                             FOREGROUND_MODEL.polarization_bands[index]
    beam_of(field, index) = field == "s0" ? FOREGROUND_MODEL.temperature_beams[index] :
                                            FOREGROUND_MODEL.polarization_beams[index]

    for row in axes(rows, 1)
        field = String(rows[row, 2])
        index = Int(rows[row, 3])
        nu_index = Int(rows[row, 4])
        channel = LIKE.channels[index]
        @test String(rows[row, 1]) == channel

        raw = raw_band(field, index)
        shift = getfield(reference, Symbol("bandint_shift_", channel))
        @test raw.nu[nu_index] ≈ Float64(rows[row, 5]) rtol=1e-12
        @test raw.nu[nu_index] + shift ≈ Float64(rows[row, 6]) rtol=1e-12
        @test raw.bp[nu_index] ≈ Float64(rows[row, 7]) rtol=1e-12

        band = shift_and_normalize(raw, shift)
        @test band.norm_bp[nu_index] ≈ Float64(rows[row, 8]) rtol=1e-11 atol=1e-14
    end

    header, rows = read_reference_table(REFERENCE_DIR, "chromatic_beam_checkpoints.tsv")
    @test header == ["channel", "field", "channel_index", "nu_index", "ell",
                     "beam", "chromatic_response"]
    for row in axes(rows, 1)
        field = String(rows[row, 2])
        index = Int(rows[row, 3])
        nu_index = Int(rows[row, 4])
        ell = Int(rows[row, 5])
        channel = LIKE.channels[index]

        beam = beam_of(field, index)
        @test beam.beam[ell_index(ell), nu_index] ≈ Float64(rows[row, 6]) rtol=1e-12

        raw = raw_band(field, index)
        shift = getfield(reference, Symbol("bandint_shift_", channel))
        band = shift_and_normalize(raw, shift)
        profile = band.norm_bp .* @view beam.beam[ell_index(ell), :]
        response = profile[nu_index] / trapz(band.nu, profile)
        @test response ≈ Float64(rows[row, 7]) rtol=1e-10 atol=1e-13
    end
end
