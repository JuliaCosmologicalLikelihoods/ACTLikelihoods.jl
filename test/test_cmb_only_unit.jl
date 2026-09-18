"""
    test/test_cmb_only_unit.jl

Unit tests for the secondary, **foreground-marginalized** ACT DR6 CMB-only
variant.

These use small synthetic fixtures only. The CMB-only variant ships loose local
data files that are not part of the published runtime artifact, and no test in
this suite is allowed to depend on them: full-likelihood validation must never
be able to degrade into a skip. The local CMB-only parity script lives in
`test/manual/`.
"""

@testset "CMB-only (marginalized) — theory binning" begin
    n_ell = 100
    n_spectra = 3
    bins_per_spectrum = [10, 12, 8]

    windows = [abs.(randn(n_ell, bins)) for bins in bins_per_spectrum]
    for window in windows
        for column in axes(window, 2)
            window[:, column] ./= sum(window[:, column])
        end
    end

    binned = ACTLikelihoods.bin_theory_vector(ones(n_spectra * n_ell), windows, n_ell, n_spectra)
    @test length(binned) == sum(bins_per_spectrum)
    @test all(binned .≈ 1.0)
end

@testset "CMB-only (marginalized) — calibration" begin
    n_ell = 50
    spec_types = ["TT", "TE", "EE"]
    A_act = 1.1
    P_act = 0.9

    out = ACTLikelihoods.apply_calibration(ones(3 * n_ell), A_act, P_act, spec_types, n_ell)
    @test length(out) == 3 * n_ell
    @test all(out[1:n_ell] .≈ 1.0 / A_act^2)
    @test all(out[(n_ell + 1):(2n_ell)] .≈ 1.0 / (A_act^2 * P_act))
    @test all(out[(2n_ell + 1):(3n_ell)] .≈ 1.0 / (A_act^2 * P_act^2))

    identity_input = rand(3 * n_ell) .+ 1.0
    @test ACTLikelihoods.apply_calibration(identity_input, 1.0, 1.0, spec_types, n_ell) ≈ identity_input
end

@testset "CMB-only (marginalized) — foregrounds are marginalized away" begin
    # The CMB-only data product has the foregrounds already marginalized out,
    # which is exactly why it must not be used as evidence about the full
    # multifrequency foreground model.
    n_ell = 100
    fg = ACTLikelihoods.compute_foreground_Dls(ACTCMBOnlyFG(), (A_act = 1.0, P_act = 1.0),
                                collect(2.0:101.0), ["TT", "TE", "EE"], n_ell)
    @test length(fg) == 3 * n_ell
    @test all(iszero, fg)
end

@testset "CMB-only (marginalized) — synthetic dataset round trip" begin
    # Build a tiny, self-contained dataset so the loader is exercised without
    # any loose local data.
    mktempdir() do directory
        n_ell = 40
        n_bins = 4
        ell_min = 2
        windows_directory = joinpath(directory, "windows")
        mkpath(windows_directory)

        for name in ("TT_lxl", "TE_lxl", "EE_lxl")
            window = zeros(n_ell, n_bins)
            for bin in 1:n_bins
                window[((bin - 1) * 10 + 1):(bin * 10), bin] .= 0.1
            end
            writedlm(joinpath(windows_directory, "$(name)_window_functions.txt"),
                     hcat(collect(ell_min:(ell_min + n_ell - 1)), window))
        end

        bandpowers = collect(1.0:(3 * n_bins))
        writedlm(joinpath(directory, "synthetic_bdp.txt"), bandpowers)
        covariance = Matrix{Float64}(I, 3 * n_bins, 3 * n_bins)
        writedlm(joinpath(directory, "synthetic_cov.txt"), covariance)

        loaded = ACTLikelihoods.load_bandpowers(joinpath(directory, "synthetic_bdp.txt"))
        @test loaded == bandpowers
        loaded_covariance = ACTLikelihoods.load_covariance(joinpath(directory,
                                                                   "synthetic_cov.txt"))
        @test size(loaded_covariance) == (3 * n_bins, 3 * n_bins)
        @test issymmetric(Matrix(loaded_covariance))

        window = ACTLikelihoods.load_window_function(
            joinpath(windows_directory, "TT_lxl_window_functions.txt"); ell_min=ell_min)
        @test size(window, 2) == n_bins
        @test all(window .>= 0)
        @test all(sum(window; dims=1) .> 0)
    end
end

@testset "Gaussian log-likelihood — analytic reference" begin
    covariance = Matrix{Float64}(I, 2, 2)
    factorization = cholesky(Symmetric(covariance))
    data = [3.0, 5.0]

    exact = factorization.L \ (data .- [3.0, 5.0])
    @test -0.5 * dot(exact, exact) ≈ 0.0

    offset = factorization.L \ (data .- [4.0, 5.0])
    @test -0.5 * dot(offset, offset) ≈ -0.5
end
