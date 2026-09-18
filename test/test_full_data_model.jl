"""
    test/test_full_data_model.jl

The released data mapping: ordered spectra, map legs, windows, covariance, and
the calibrated zero-foreground prediction.
"""

@testset "ACT DR6 data model — released dimensions" begin
    @test length(LIKE.observed) == 1651
    @test length(LIKE.spectra) == 41
    @test LIKE.channels == collect(ACTLikelihoods.ACT_DR6_CHANNELS)
    @test length(LIKE.channels) == 5
    @test LIKE.ells == collect(2:8501)
    @test occursin("1651 bandpowers", sprint(show, LIKE))
end

@testset "ACT DR6 data model — observation vector matches the original" begin
    reference = read_reference_vector(REFERENCE_DIR, "data_vector.txt")
    @test length(reference) == 1651
    @test LIKE.observed == reference
end

@testset "ACT DR6 data model — ordered spectrum metadata" begin
    header, rows = read_reference_table(REFERENCE_DIR, "spectrum_metadata.tsv")
    @test header == ["index", "polarization", "temperature_leg", "polarization_leg",
                     "source_t1", "source_t2", "reversed_cross_spectrum",
                     "number_of_bins", "first_output_index", "last_output_index",
                     "window_ell_min", "window_ell_max"]
    @test size(rows, 1) == 41

    covered = falses(length(LIKE.observed))
    for row in axes(rows, 1)
        spectrum = LIKE.spectra[row]
        @test rows[row, 1] == row
        @test Symbol(uppercase(String(rows[row, 2]))) === spectrum.polarization
        @test LIKE.channels[spectrum.temperature_leg] == String(rows[row, 3])
        @test LIKE.channels[spectrum.polarization_leg] == String(rows[row, 4])
        @test length(spectrum.output_indices) == rows[row, 8]
        @test first(spectrum.output_indices) == rows[row, 9]
        @test last(spectrum.output_indices) == rows[row, 10]
        @test rows[row, 11] == first(LIKE.ells)
        @test rows[row, 12] == last(LIKE.ells)
        @test size(window_matrix(spectrum, length(LIKE.ells))) ==
            (length(LIKE.ells), length(spectrum.output_indices))
        covered[spectrum.output_indices] .= true
    end

    # Complete, non-overlapping coverage of all 1651 outputs.
    @test all(covered)
    @test sum(length(s.output_indices) for s in LIKE.spectra) == length(LIKE.observed)
    @test vcat((collect(s.output_indices) for s in LIKE.spectra)...) ==
        collect(1:length(LIKE.observed))
end

@testset "ACT DR6 data model — every channel and polarization is exercised" begin
    polarizations = unique(s.polarization for s in LIKE.spectra)
    @test sort(String.(polarizations)) == ["EE", "TE", "TT"]

    for channel in 1:5
        @test any(s -> s.temperature_leg == channel || s.polarization_leg == channel,
                  LIKE.spectra)
    end

    # pa4 f220 enters temperature only; it has no polarization leg anywhere.
    pa4 = findfirst(==("dr6_pa4_f220"), LIKE.channels)
    @test all(s -> s.polarization === :TT,
              filter(s -> s.temperature_leg == pa4 || s.polarization_leg == pa4,
                     LIKE.spectra))
end

@testset "ACT DR6 data model — ordered TE/ET map legs are preserved" begin
    te = filter(s -> s.polarization === :TE, LIKE.spectra)
    @test !isempty(te)

    # A reversed cross-spectrum carries T on the second released tracer. Both
    # orderings of the same channel pair must be present and distinct: this is
    # exactly what symmetrization would destroy.
    ordered = Set((s.temperature_leg, s.polarization_leg) for s in te)
    reversed_pairs = [(a, b) for (a, b) in ordered if a != b && (b, a) in ordered]
    @test !isempty(reversed_pairs)

    for (a, b) in reversed_pairs
        forward = only(filter(s -> s.temperature_leg == a && s.polarization_leg == b, te))
        backward = only(filter(s -> s.temperature_leg == b && s.polarization_leg == a, te))
        @test forward.output_indices != backward.output_indices
        @test isdisjoint(forward.output_indices, backward.output_indices)
    end

    # The pa5_f150 (T) x pa5_f090 (E) leg is the reversed partner referenced by
    # the original spec_meta.
    reversed_te = only(filter(s -> s.temperature_leg == 3 && s.polarization_leg == 2, te))
    @test length(reversed_te.output_indices) == 39
end

@testset "ACT DR6 data model — covariance and Cholesky solve" begin
    factorization = LIKE.covariance_cholesky
    @test size(factorization) == (1651, 1651)
    @test all(>(0), diag(factorization.L))

    residual = LIKE.observed .- read_reference_vector(REFERENCE_DIR, "model_vector.txt")
    whitened = factorization.L \ residual
    @test dot(whitened, whitened) ≈ dot(residual, factorization \ residual) rtol=1e-12

    # The stored factor really is a factorization of a symmetric matrix.
    reconstructed = factorization.L * factorization.L'
    @test reconstructed ≈ Matrix(Symmetric(reconstructed)) rtol=1e-12

    # And the Gaussian constant is derived, not copied.
    @test gaussian_normalization(LIKE) ≈
        -length(LIKE.observed) / 2 * log(2 * pi) - sum(log, diag(factorization.L))
end

@testset "ACT DR6 data model — zero-foreground parity with the original" begin
    zero_block = zeros(5, 5, length(LIKE.ells))
    empty_foregrounds = ACTForegrounds(zero_block, zero_block, zero_block)
    prediction = predict(LIKE, REFERENCE_CMB, empty_foregrounds, REFERENCE_NUISANCE)
    reference = read_reference_vector(REFERENCE_DIR, "model_vector_no_foregrounds.txt")

    @test length(prediction) == 1651
    @test prediction ≈ reference rtol=1e-12 atol=1e-11
end

@testset "ACT DR6 data model — malformed inputs are rejected" begin
    zero_block = zeros(5, 5, length(LIKE.ells))
    empty_foregrounds = ACTForegrounds(zero_block, zero_block, zero_block)

    # CMB grid that does not cover the window support.
    truncated = ACTCMBTheory(REFERENCE_CMB.ell[4:end], REFERENCE_CMB.TT[4:end],
                             REFERENCE_CMB.TE[4:end], REFERENCE_CMB.EE[4:end])
    @test_throws ArgumentError predict(LIKE, truncated, empty_foregrounds, REFERENCE_NUISANCE)

    short = ACTCMBTheory(collect(0:100), zeros(101), zeros(101), zeros(101))
    @test_throws ArgumentError predict(LIKE, short, empty_foregrounds, REFERENCE_NUISANCE)

    # Non-contiguous multipole grids are refused at construction.
    @test_throws ArgumentError ACTCMBTheory([2, 4, 6], zeros(3), zeros(3), zeros(3))
    @test_throws DimensionMismatch ACTCMBTheory(collect(2:5), zeros(4), zeros(3), zeros(4))

    # Wrong foreground shapes.
    wrong_channels = zeros(4, 4, length(LIKE.ells))
    @test_throws DimensionMismatch predict(
        LIKE, REFERENCE_CMB,
        ACTForegrounds(wrong_channels, wrong_channels, wrong_channels), REFERENCE_NUISANCE)

    wrong_ells = zeros(5, 5, length(LIKE.ells) - 1)
    @test_throws DimensionMismatch predict(
        LIKE, REFERENCE_CMB,
        ACTForegrounds(wrong_ells, wrong_ells, wrong_ells), REFERENCE_NUISANCE)

    # Wrong prediction length into chi2.
    @test_throws DimensionMismatch chi2(LIKE, zeros(1650))

    # A missing nuisance parameter must raise, not silently default.
    incomplete = (a_tSZ = 3.5,)
    @test_throws ArgumentError foregrounds(FOREGROUND_MODEL, LIKE, incomplete)
end

@testset "ACT DR6 data model — packed windows reproduce the released files" begin
    # The hot path and reverse mode both read the packed bands rather than the
    # released `(n_ell, n_bin)` matrices. That is only legitimate if the packing
    # loses nothing, so compare against the artifact files themselves.
    windows_directory = joinpath(ARTIFACT_DIR, "windows")
    # The released metadata names the window files, including for the reversed
    # cross spectra whose file name does not follow the map-leg order.
    metadata = ACTLikelihoods._read_full_metadata(
        joinpath(ARTIFACT_DIR, "spectrum_metadata.tsv"))
    @test length(metadata) == length(LIKE.spectra)
    n_ell = length(LIKE.ells)
    compared = 0
    nonzero = 0
    for (spectrum, row) in zip(LIKE.spectra, metadata)
        released = Float64.(readdlm(
            ACTLikelihoods._full_window_path(windows_directory, row)))
        @test size(released) == (n_ell, length(spectrum.output_indices))
        # Exact equality, not a tolerance: packing only drops structural zeros.
        @test window_matrix(spectrum, n_ell) == released
        compared += length(released)
        nonzero += count(!=(0), released)

        # Every released bandpower window is one contiguous run of nonzeros.
        # The packing stays exact either way, but this is the property that
        # makes it worth doing, so assert it rather than assume it.
        for bin in axes(released, 2)
            column = @view released[:, bin]
            indices = findall(!=(0), column)
            @test !isempty(indices)
            @test length(indices) == last(indices) - first(indices) + 1
            @test first(indices) == spectrum.band_starts[bin]
            @test length(indices) ==
                spectrum.band_offsets[bin + 1] - spectrum.band_offsets[bin]
        end
    end
    # Every bandpower of every spectrum was compared, against the sparsity the
    # packing relies on.
    @test compared == n_ell * sum(length(s.output_indices) for s in LIKE.spectra)
    @test nonzero / compared < 0.03

    packed_bytes = sum(sizeof(s.band_values) + sizeof(s.band_starts) +
                       sizeof(s.band_offsets) for s in LIKE.spectra)
    @test packed_bytes < compared * sizeof(Float64) / 20

    # The span a spectrum's windows cover must contain every one of its bands.
    for spectrum in LIKE.spectra
        span = ACTLikelihoods._band_span(spectrum.band_starts, spectrum.band_offsets)
        @test first(span) >= 1
        @test last(span) <= n_ell
        for bin in eachindex(spectrum.band_starts)
            width = spectrum.band_offsets[bin + 1] - spectrum.band_offsets[bin]
            @test spectrum.band_starts[bin] >= first(span)
            @test spectrum.band_starts[bin] + width - 1 <= last(span)
        end
    end
end

@testset "ACT DR6 data model — the projection validates what it indexes" begin
    # The contraction reads with `@inbounds`, so a CMB grid that does not cover
    # the windows must be rejected rather than read out of bounds.
    baseline = foregrounds(FOREGROUND_MODEL, LIKE, REFERENCE_NUISANCE)
    calibrations = ACTLikelihoods._spectrum_calibrations(LIKE, NamedTuple(REFERENCE_NUISANCE))
    arguments = (REFERENCE_CMB.TT, REFERENCE_CMB.TE, REFERENCE_CMB.EE,
                 baseline.TT, baseline.TE, baseline.EE, calibrations)
    offset = Int(first(CMB_INDICES))
    @test ACTLikelihoods._act_projection(LIKE, arguments..., offset) ==
        predict(LIKE, REFERENCE_CMB, baseline, REFERENCE_NUISANCE)

    too_far = length(REFERENCE_CMB.TT) - length(LIKE.ells) + 2
    @test_throws DimensionMismatch ACTLikelihoods._act_projection(LIKE, arguments..., too_far)
    @test_throws DimensionMismatch ACTLikelihoods._act_projection(LIKE, arguments..., 0)
    @test_throws DimensionMismatch ACTLikelihoods._act_projection(
        LIKE, REFERENCE_CMB.TT[1:100], arguments[2:end]..., offset)
    @test_throws DimensionMismatch ACTLikelihoods._act_projection(
        LIKE, arguments[1:3]..., baseline.TT[:, :, 1:100], arguments[5:end]..., offset)
    @test_throws DimensionMismatch ACTLikelihoods._act_projection(
        LIKE, arguments[1:6]..., calibrations[1:10], offset)

    # And the documented public path still rejects a CMB grid that misses the
    # window support before it ever reaches the kernel.
    short = ACTCMBTheory(collect(0:4000), zeros(4001), zeros(4001), zeros(4001))
    @test_throws ArgumentError predict(LIKE, short, baseline, REFERENCE_NUISANCE)
end
