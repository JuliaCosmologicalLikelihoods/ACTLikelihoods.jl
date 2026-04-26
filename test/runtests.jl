using ACTLikelihoods
using Test
using DelimitedFiles: readdlm

# ------------------------------------------------------------------ #
# 1. SED functions                                                      #
# ------------------------------------------------------------------ #

@testset "frequency — SED functions" begin
    nu_0 = 150.0

    # tSZ: must be 1.0 at nu_0
    # Both 90 and 150 GHz are below the ~217 GHz null → f(90)/f(150) is positive
    # (ratio of two negatives). Cross-frequency spectra with one band above the null
    # will be negative.
    @test tsz_sed(nu_0, nu_0) ≈ 1.0
    @test tsz_sed(217.0, nu_0) ≈ 0.0  atol=0.05   # null crossing ~217 GHz
    @test tsz_sed(90.0,  nu_0) > 0    # both below null → positive ratio
    @test tsz_sed(350.0, nu_0) < 0    # 350 above null, nu_0 below → negative

    # MBB: must be 1.0 at nu_0 for any beta/temp
    @test mbb_sed(nu_0, nu_0, 2.2, 9.7)  ≈ 1.0
    @test mbb_sed(nu_0, nu_0, 1.5, 19.6) ≈ 1.0
    # MBB must increase with frequency (in the relevant range)
    @test mbb_sed(220.0, nu_0, 1.5, 19.6) > mbb_sed(nu_0, nu_0, 1.5, 19.6)

    # Radio: must be 1.0 at nu_0
    @test radio_sed(nu_0, nu_0, -2.5) ≈ 1.0
    # Radio decreases with frequency for beta < 0
    @test radio_sed(220.0, nu_0, -2.5) < 1.0

    # kSZ: constant SED = 1
    @test constant_sed(90.0)  == 1.0
    @test constant_sed(150.0) == 1.0
    @test constant_sed(220.0) == 1.0

    # Vector input
    nus = [90.0, 150.0, 220.0]
    @test length(tsz_sed(nus, nu_0))       == 3
    @test length(mbb_sed(nus, nu_0, 2.2, 9.7)) == 3
    @test length(radio_sed(nus, nu_0, -2.5))    == 3

    # cmb2bb: positive and increasing in relevant range
    @test cmb2bb(90.0)  > 0
    @test cmb2bb(150.0) > cmb2bb(90.0)

    # rj2cmb: monotonically increasing from 1 in the RJ limit (low ν)
    @test rj2cmb(10.0)  ≈ 1.0 atol=0.01    # near RJ limit
    @test rj2cmb(150.0) > 1.0               # well above RJ limit at 150 GHz
    @test rj2cmb(150.0) > rj2cmb(90.0)     # increasing with frequency
end

# ------------------------------------------------------------------ #
# 2. Bandpass integration                                              #
# ------------------------------------------------------------------ #

@testset "bandpass — normalization and integration" begin
    nu_0 = 150.0

    # Top-hat band around 150 GHz
    nu   = collect(range(130.0, 170.0, length=50))
    bp   = ones(50)
    band = make_band(nu, bp)

    @test !band.monofreq
    @test trapz(band.nu, band.norm_bp) ≈ 1.0  atol=1e-10

    # Constant SED integrates to 1 over normalized band
    @test integrate_sed(ν -> 1.0, band) ≈ 1.0  atol=1e-6

    # tSZ at band center ≈ tSZ at effective frequency (narrow band)
    tsz_int = integrate_sed(ν -> tsz_sed(ν, nu_0), band)
    tsz_eff = tsz_sed(sum(nu .* band.norm_bp .* diff([nu[1]; nu])), nu_0)
    @test tsz_int ≈ tsz_sed(nu_0, nu_0)  atol=0.05   # ≈1 for band centered at nu_0

    # Monochromatic band
    nu_mono = [150.0]; bp_mono = [1.0]
    band_mono = make_band(nu_mono, bp_mono)
    @test band_mono.monofreq
    @test integrate_sed(ν -> tsz_sed(ν, nu_0), band_mono) ≈ 1.0
end

# ------------------------------------------------------------------ #
# 3. Power spectrum templates                                           #
# ------------------------------------------------------------------ #

@testset "power — template loading and eval" begin
    T_tsz  = load_tsz_template()
    T_ksz  = load_ksz_template()
    T_cibc = load_cibc_template()
    T_szx  = load_szxcib_template()

    # All loaded and non-trivial
    @test length(T_tsz)  > 3000
    @test length(T_ksz)  > 3000
    @test length(T_cibc) > 3000
    @test length(T_szx)  > 3000

    # Values are positive
    @test T_tsz[3001]  > 0   # D_ℓ at ℓ=3000
    @test T_ksz[3001]  > 0
    @test T_cibc[3001] > 0
    @test T_szx[3001]  > 0

    ell   = collect(500:100:4000)
    ell_0 = 3000

    # eval_template normalizes to 1 at ell_0
    cl_tsz = eval_template(T_tsz, ell, ell_0)
    @test cl_tsz[findfirst(==(3000), ell)] ≈ 1.0

    # eval_template_tilt with alpha=0 ≡ eval_template
    cl_tilt0 = eval_template_tilt(T_tsz, ell, ell_0, 0.0)
    @test cl_tilt0 ≈ cl_tsz  atol=1e-12

    # eval_powerlaw
    cl_pl = eval_powerlaw(Float64.(ell), Float64(ell_0), -0.6)
    @test cl_pl[findfirst(==(3000), ell)] ≈ 1.0
    @test cl_pl[findfirst(==(500),  ell)] > 1.0   # negative slope → larger at low ℓ
end

# ------------------------------------------------------------------ #
# 4. Cross-spectrum assembly                                            #
# ------------------------------------------------------------------ #

@testset "cross — factorized and correlated" begin
    n_freq = 3; n_ell = 100
    f  = [1.0, 2.0, 3.0]
    cl = ones(n_ell)

    # Factorized: shape and symmetry
    D = factorized_cross(f, cl)
    @test size(D) == (n_freq, n_freq, n_ell)
    @test D[1, 2, 1] ≈ D[2, 1, 1]    # symmetric
    @test D[1, 1, 1] ≈ 1.0
    @test D[2, 2, 1] ≈ 4.0
    @test D[3, 3, 1] ≈ 9.0

    # TE: asymmetric
    fT = [1.0, 2.0, 3.0]; fE = [2.0, 1.0, 0.5]
    DTE = factorized_cross_te(fT, fE, cl)
    @test DTE[1, 2, 1] ≈ fT[1] * fE[2]   # 1×1 = 1
    @test DTE[2, 1, 1] ≈ fT[2] * fE[1]   # 2×2 = 4
    @test !(DTE[1,2,1] ≈ DTE[2,1,1])     # NOT symmetric for TE

    # Correlated: reduces to factorized when off-diagonal = 0
    f2 = [f'; f']     # (2, n_freq) — same SED for both components
    cl_diag = zeros(2, 2, n_ell)
    cl_diag[1, 1, :] .= 1.0
    cl_diag[2, 2, :] .= 0.0
    D_corr = correlated_cross(f2, cl_diag)
    D_fact = factorized_cross(f, cl)
    @test D_corr ≈ D_fact  atol=1e-12

    # build_szxcib_cl: symmetric
    cl_tsz  = 2.0 .* ones(n_ell)
    cl_cibc = 3.0 .* ones(n_ell)
    cl_cross = -0.5 .* ones(n_ell)
    cl_joint = build_szxcib_cl(cl_tsz, cl_cibc, cl_cross)
    @test size(cl_joint) == (2, 2, n_ell)
    @test cl_joint[1, 2, 1] ≈ cl_joint[2, 1, 1]
    @test cl_joint[1, 1, 1] ≈ 2.0
    @test cl_joint[2, 2, 1] ≈ 3.0
    @test cl_joint[1, 2, 1] ≈ -0.5
end

# ------------------------------------------------------------------ #
# 5. Foreground model — smoke test (no data file needed)               #
# ------------------------------------------------------------------ #

@testset "foreground — smoke test" begin
    # Build a minimal ForegroundModel with top-hat RawBands
    nu_T = [90.0, 150.0, 220.0]
    experiments_smoke = ["exp090", "exp150", "exp220"]
    raw_T = [RawBand([ν-10, ν, ν+10.0], [1.0, 1.0, 1.0]) for ν in nu_T]
    # Distinct polarization bands (slightly narrower top-hats with non-flat
    # transmission) so a regression that swaps T↔P paths cannot pass silently.
    raw_P = [RawBand([ν-8, ν, ν+8.0], [0.8, 1.0, 0.7]) for ν in nu_T]

    ell = collect(2:4000)
    model = ForegroundModel(experiments_smoke, raw_T, raw_P, ell)

    # ACT DR6 approximate best-fit parameters
    fg = (
        a_tSZ   = 3.35,  alpha_tSZ = -0.53,
        a_kSZ   = 1.48,
        a_p     = 6.0,   beta_p = 2.2,  T_d  = 9.7,
        a_c     = 4.0,   beta_c = 2.2,
        xi      = 0.12,
        a_s     = 3.0,   beta_s = -2.5,
        a_gtt   = 8.0,
        a_gte   = 0.42,
        a_gee   = 0.168,
        a_pste  = 0.0,
        a_psee  = 0.05,
    )

    fg_TT, fg_TE, fg_EE = compute_fg_totals(fg, model)

    # Shape checks
    n_exp = length(nu_T)
    n_ell = length(ell)
    @test size(fg_TT) == (n_exp, n_exp, n_ell)
    @test size(fg_TE) == (n_exp, n_exp, n_ell)
    @test size(fg_EE) == (n_exp, n_exp, n_ell)

    # All auto-spectra positive (at ℓ=3000)
    idx_3000 = findfirst(==(3000), ell)
    for i in 1:n_exp
        @test fg_TT[i, i, idx_3000] > 0
        @test fg_EE[i, i, idx_3000] > 0
    end

    # TT amplitude order-of-magnitude check at ℓ=3000 (auto, 150 GHz)
    # should be O(10) μK²
    D_TT_150 = fg_TT[2, 2, idx_3000]
    @test 1.0 < D_TT_150 < 100.0

    # EE amplitude order-of-magnitude at ℓ=3000 (should be << TT)
    D_EE_150 = fg_EE[2, 2, idx_3000]
    @test D_EE_150 < D_TT_150
end

# ------------------------------------------------------------------ #
# 6. Data loading — integration test (requires extracted data)         #
# ------------------------------------------------------------------ #

const FILTERED_DIR  = act_dr6_filtered_dir()
const BANDPASS_DIR  = act_dr6_bandpass_dir()

if isdir(FILTERED_DIR) && isdir(BANDPASS_DIR)
    @testset "data loading" begin
        data = load_data(FILTERED_DIR)

        # Basic shape checks
        @test length(data.data_vec) == 1651
        @test size(data.inv_cov)   == (1651, 1651)
        @test length(data.spec_meta) == 41
        @test length(data.l_bpws)  > 0
        @test data.l_bpws[1]       == 2     # ell grid starts at ℓ=2
        @test data.ell_0           == 3000
        @test data.nu_0            ≈ 150.0

        # logp_const must be finite and negative
        @test isfinite(data.logp_const)
        @test data.logp_const < 0

        # Check that ids partition [1, 1651] contiguously
        all_ids = vcat([m.ids for m in data.spec_meta]...)
        @test sort(all_ids) == collect(1:1651)

        # ET detection: at least one hasYX_xsp == true
        @test any(m.hasYX_xsp for m in data.spec_meta)

        # Window matrices have correct shape
        for m in data.spec_meta
            n_bins_m = length(m.ids)
            @test size(m.W, 2) == n_bins_m     # columns = bins for this spec
            @test size(m.W, 1) == length(data.l_bpws)  # rows = ell grid
        end

        # Bandpass loading — returns RawBand (un-normalized)
        raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
        @test length(raw_T) == length(data.experiments)
        @test length(raw_P) == length(data.experiments)
        for r in raw_T
            @test length(r.nu) > 1
            @test length(r.nu) == length(r.bp)
            # shift_and_normalize(r, 0) must give a properly normalized Band
            b = shift_and_normalize(r, 0.0)
            @test !b.monofreq
            @test trapz(b.nu, b.norm_bp) ≈ 1.0  atol=1e-8
        end
    end
else
    @warn "Skipping data loading test — extracted data directory not found."
end

# ------------------------------------------------------------------ #
# 7. End-to-end loglike (uses reference CMB theory + fake FG params)  #
# ------------------------------------------------------------------ #

if isdir(FILTERED_DIR) && isdir(BANDPASS_DIR)
    @testset "end-to-end loglike" begin
        data = load_data(FILTERED_DIR)

        # Load the reference CMB theory D_ℓ dumped from the Python run.
        # Files: rows correspond to ℓ=0,1,2,...,9050 (9051 rows total).
        # The window grid covers ℓ=2..8501 → Julia rows 3:8502.
        n_ell = length(data.l_bpws)   # 8500
        ell_slice = 3:(2 + n_ell)     # rows 3..8502 → ℓ=2..8501

        tt_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))
        te_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))
        ee_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))

        cmb_dls = Dict(
            "tt" => tt_all[ell_slice],
            "te" => te_all[ell_slice],
            "ee" => ee_all[ell_slice],
        )

        # Build foreground model with actual ACT bandpasses and ell grid
        raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
        model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)

        # ACT DR6 approximate best-fit foreground parameters
        fg = (
            a_tSZ    = 3.35,  alpha_tSZ = -0.53,
            a_kSZ    = 1.48,
            a_p      = 6.91,  beta_p = 2.07,  T_d  = 9.6,
            a_c      = 4.88,  beta_c = 2.20,
            xi       = 0.12,
            a_s      = 3.09,  beta_s = -2.76,
            a_gtt    = 8.83,
            a_gte    = 0.42,
            a_gee    = 0.168,
            a_pste   = -0.023,
            a_psee   = 0.040,
        )

        # No calibration (all set to 1)
        cal = (calG_all = 1.0,)

        ll = loglike(cmb_dls, fg, cal, data, model)

        chi2 = -2.0 * (ll - data.logp_const)

        println("  loglike  = $ll")
        println("  chi²     = $chi2")
        println("  chi²/dof = $(chi2 / length(data.data_vec))")

        # Reference values computed by scripts/reference_chi2.py
        # (same CMB theory, same FG params, same data, pure-Python computation)
        ref_ll   = -3148.2570439824
        ref_chi2 =  2006.1713326138

        @test ll   ≈ ref_ll   rtol=1e-6
        @test chi2 ≈ ref_chi2 rtol=1e-6
    end
else
    @warn "Skipping end-to-end test — data directory not found."
end

# ------------------------------------------------------------------ #
# 8. Automatic differentiation — gradient of loglike w.r.t. FG params #
# ------------------------------------------------------------------ #

if isdir(FILTERED_DIR) && isdir(BANDPASS_DIR)
    @testset "AD — ForwardDiff vs finite differences vs Mooncake" begin
        using DifferentiationInterface
        import ForwardDiff, Mooncake
        using ADTypes: AutoForwardDiff, AutoMooncake

        data    = load_data(FILTERED_DIR)
        n_ell   = length(data.l_bpws)
        ell_slice = 3:(2 + n_ell)

        tt_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))
        te_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))
        ee_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))
        cmb_dls = Dict("tt" => tt_all[ell_slice],
                       "te" => te_all[ell_slice],
                       "ee" => ee_all[ell_slice])

        raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
        model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)
        cal   = (calG_all = 1.0,)

        # Canonical FG parameter vector (15 free parameters)
        FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p,
                         :a_c, :beta_c, :xi, :a_s, :beta_s,
                         :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)

        fg_vals = Float64[3.35, -0.53, 1.48, 6.91, 2.07,
                          4.88,  2.20, 0.12, 3.09, -2.76,
                          8.83,  0.42, 0.168, -0.023, 0.040]

        fg_from_vec(v) = NamedTuple{FG_KEYS}(Tuple(v))

        ll_fn(v) = loglike(cmb_dls, fg_from_vec(v), cal, data, model)

        # ---- ForwardDiff ---- #
        grad_fd = DifferentiationInterface.gradient(ll_fn, AutoForwardDiff(), fg_vals)

        @test all(isfinite, grad_fd)
        @test length(grad_fd) == length(fg_vals)

        # ---- Central finite differences (reference) ---- #
        ε = 1e-4
        grad_fin = similar(fg_vals)
        for i in eachindex(fg_vals)
            vp = copy(fg_vals); vp[i] += ε
            vm = copy(fg_vals); vm[i] -= ε
            grad_fin[i] = (ll_fn(vp) - ll_fn(vm)) / (2ε)
        end

        @test all(isfinite, grad_fin)

        # ForwardDiff must match finite differences to 0.1%
        for i in eachindex(grad_fd)
            @test grad_fd[i] ≈ grad_fin[i]  rtol=1e-3  atol=1e-6
        end

        println("  ForwardDiff gradient (first 5): ", round.(grad_fd[1:5], sigdigits=5))
        println("  Finite-diff gradient (first 5): ", round.(grad_fin[1:5], sigdigits=5))

        # ---- Mooncake ---- #
        grad_mk = DifferentiationInterface.gradient(
            ll_fn, AutoMooncake(; config=nothing), fg_vals)

        @test all(isfinite, grad_mk)

        # Mooncake must agree with ForwardDiff to ~30× machine epsilon
        # (both compute the same graph; only IEEE-754 reduction-order rounding differs)
        for i in eachindex(grad_mk)
            @test grad_mk[i] ≈ grad_fd[i]  rtol=1e-10  atol=1e-12
        end

        println("  Mooncake gradient  (first 5): ", round.(grad_mk[1:5], sigdigits=5))
        println("  Mooncake vs ForwardDiff: rtol=1e-10 ✓")

        # ---- Bandpass shift gradients ---- #
        # Include 5 bandint_shift params (default 0.0) in the parameter vector.
        # The gradient through shift → band renormalization must be finite and
        # agree with finite differences.
        SHIFT_KEYS = Tuple(Symbol("bandint_shift_" * e) for e in data.experiments)

        all_keys = (FG_KEYS..., SHIFT_KEYS...)
        all_vals = vcat(fg_vals, zeros(Float64, length(SHIFT_KEYS)))

        all_from_vec(v) = NamedTuple{all_keys}(Tuple(v))
        ll_all(v) = loglike(cmb_dls, all_from_vec(v), cal, data, model)

        grad_all_fd = DifferentiationInterface.gradient(ll_all, AutoForwardDiff(), all_vals)
        @test all(isfinite, grad_all_fd)

        # Shift gradients (last 5 elements) — compare to finite diff
        grad_all_fin = similar(all_vals)
        for i in eachindex(all_vals)
            ε_i = i <= length(fg_vals) ? 1e-4 : 1e-3   # larger ε for shifts
            vp = copy(all_vals); vp[i] += ε_i
            vm = copy(all_vals); vm[i] -= ε_i
            grad_all_fin[i] = (ll_all(vp) - ll_all(vm)) / (2ε_i)
        end

        n_fg = length(fg_vals)
        for i in (n_fg+1):length(all_vals)
            @test grad_all_fd[i] ≈ grad_all_fin[i]  rtol=1e-3  atol=1e-6
        end

        println("  Bandpass shift grads (ForwardDiff): ",
                round.(grad_all_fd[n_fg+1:end], sigdigits=4))
        println("  Bandpass shift grads (finite diff): ",
                round.(grad_all_fin[n_fg+1:end], sigdigits=4))

        # Mooncake also for the combined vector
        grad_all_mk = DifferentiationInterface.gradient(
            ll_all, AutoMooncake(; config=nothing), all_vals)
        @test all(isfinite, grad_all_mk)
        # Mooncake vs ForwardDiff on bandpass shifts: stringent (same graph, IEEE rounding only)
        for i in (n_fg+1):length(all_vals)
            @test grad_all_mk[i] ≈ grad_all_fd[i]  rtol=1e-10  atol=1e-12
        end
        println("  Bandpass shift grads (Mooncake):    ",
                round.(grad_all_mk[n_fg+1:end], sigdigits=4))

        # ---- Calibration gradient (calG_all) ---- #
        # calG_all multiplies all spectra — easy to verify with finite diff.
        function ll_cal(v)
            cal_nt = (calG_all = v[1],)
            loglike(cmb_dls, fg_from_vec(fg_vals), cal_nt, data, model)
        end
        cal_vals = [1.0]
        grad_cal_fd = DifferentiationInterface.gradient(ll_cal, AutoForwardDiff(), cal_vals)
        grad_cal_fin = [(ll_cal([1.0 + 1e-5]) - ll_cal([1.0 - 1e-5])) / 2e-5]
        @test isfinite(grad_cal_fd[1])
        @test grad_cal_fd[1] ≈ grad_cal_fin[1]  rtol=1e-3
        println("  calG_all gradient: FD=$(round(grad_cal_fd[1], sigdigits=5))",
                "  fin=$(round(grad_cal_fin[1], sigdigits=5))")
    end
else
    @warn "Skipping AD test — data directory not found."
end

# ------------------------------------------------------------------ #
# 7. Type-stability (JET.jl @test_opt)                                 #
# ------------------------------------------------------------------ #
# Verifies all hot-path functions are type-stable on the Float64 forward
# call.  Layer 9 documents a known instability under ForwardDiff.Dual
# values caused by runtime Symbol lookup in `fg_param`.

println("\nType-stability suite (JET.jl) …")
let t0 = time()
    include("test_type_stability.jl")
    println("  done in $(round(time() - t0, digits=2))s")
end

println("\nAll tests passed.")
