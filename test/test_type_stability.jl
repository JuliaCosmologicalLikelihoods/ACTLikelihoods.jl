# JET.jl @test_opt type-stability suite for ACTLikelihoods.jl hot paths.
#
# Probed with Float64 inputs (steady-state forward call) using
# `target_modules=(ACTLikelihoods,)` so reports are scoped to our package.
#
# Layers 1–8 (Float64 path) are fully clean.
#
# Layer 9 documents a real instability in `compute_fg_totals` when called
# with ForwardDiff.Dual-valued parameters (the forward-mode AD path).
# See the long comment there for the root cause.
#
# This file is included by runtests.jl.

using JET
import ForwardDiff
using ForwardDiff: Dual

# ------------------------------------------------------------------ #
# Layer 1 — pointwise SEDs                                             #
# ------------------------------------------------------------------ #

@testset "type-stability — frequency SEDs" begin
    @test_opt target_modules=(ACTLikelihoods,) cmb2bb(150.0)
    @test_opt target_modules=(ACTLikelihoods,) rj2cmb(150.0)
    @test_opt target_modules=(ACTLikelihoods,) tsz_sed(90.0, 150.0)
    @test_opt target_modules=(ACTLikelihoods,) mbb_sed(150.0, 150.0, 2.2, 9.7)
    @test_opt target_modules=(ACTLikelihoods,) radio_sed(150.0, 150.0, -2.5)
    @test_opt target_modules=(ACTLikelihoods,) constant_sed(150.0)
end

# ------------------------------------------------------------------ #
# Layer 2 — bandpass                                                    #
# ------------------------------------------------------------------ #

@testset "type-stability — bandpass" begin
    raw  = RawBand([140.0, 150.0, 160.0], [1.0, 1.0, 1.0])
    nu   = collect(range(130.0, 170.0, length=50))
    bp   = ones(50)
    band = make_band(nu, bp)

    @test_opt target_modules=(ACTLikelihoods,) trapz(nu, bp)
    @test_opt target_modules=(ACTLikelihoods,) shift_and_normalize(raw, 0.0)
    @test_opt target_modules=(ACTLikelihoods,) make_band(nu, bp)
    @test_opt target_modules=(ACTLikelihoods,) integrate_sed(ν -> tsz_sed(ν, 150.0), band)

    bands = [shift_and_normalize(raw, 0.0)]
    @test_opt target_modules=(ACTLikelihoods,) eval_sed_bands(ν -> tsz_sed(ν, 150.0), bands)
end

# ------------------------------------------------------------------ #
# Layer 3 — power-spectrum templates                                    #
# ------------------------------------------------------------------ #

@testset "type-stability — power" begin
    T_tsz = load_tsz_template()
    ell   = collect(500:100:4000)

    @test_opt target_modules=(ACTLikelihoods,) eval_template(T_tsz, ell, 3000)
    @test_opt target_modules=(ACTLikelihoods,) eval_template_tilt(T_tsz, ell, 3000, -0.5)
    @test_opt target_modules=(ACTLikelihoods,) eval_powerlaw(Float64.(ell), 3000.0, -0.6)
end

# ------------------------------------------------------------------ #
# Layer 4 — cross-spectrum kernels                                      #
# ------------------------------------------------------------------ #

@testset "type-stability — cross kernels" begin
    f      = [1.0, 2.0, 3.0]
    fE     = [2.0, 1.0, 0.5]
    cl     = ones(100)
    f2     = vcat(reshape(f, 1, :), reshape(f, 1, :))
    cl3d   = ones(2, 2, 100)
    cltsz  = 2.0  .* ones(100)
    clcibc = 3.0  .* ones(100)
    clcrs  = -0.5 .* ones(100)

    @test_opt target_modules=(ACTLikelihoods,) factorized_cross(f, cl)
    @test_opt target_modules=(ACTLikelihoods,) factorized_cross_te(f, fE, cl)
    @test_opt target_modules=(ACTLikelihoods,) correlated_cross(f2, cl3d)
    @test_opt target_modules=(ACTLikelihoods,) build_szxcib_cl(cltsz, clcibc, clcrs)
end

# ------------------------------------------------------------------ #
# Layer 5 — fused TT/EE/TE assemblers                                   #
# ------------------------------------------------------------------ #

@testset "type-stability — fused assemblers" begin
    f  = [1.0, 2.0, 3.0]
    cl = ones(100)
    @test_opt target_modules=(ACTLikelihoods,) assemble_TT(
        1.5, 6.0, 8.0, 3.0,
        f, f, f, f, f, f,
        cl, cl, cl, cl, cl, cl, cl)
    @test_opt target_modules=(ACTLikelihoods,) assemble_EE(0.05, 0.168, f, f, cl, cl)
    @test_opt target_modules=(ACTLikelihoods,) assemble_TE(0.0, 0.42, f, f, f, f, cl, cl)
end

# ------------------------------------------------------------------ #
# Layer 6 — compute_fg_totals (forward, Float64 inputs)                 #
# ------------------------------------------------------------------ #

@testset "type-stability — compute_fg_totals (Float64)" begin
    nu_T  = [90.0, 150.0, 220.0]
    exps  = ["exp090", "exp150", "exp220"]
    raw_T = [RawBand([ν - 10.0, ν, ν + 10.0], [1.0, 1.0, 1.0]) for ν in nu_T]
    raw_P = raw_T
    ell   = collect(2:4000)
    model = ForegroundModel(exps, raw_T, raw_P, ell)

    fg = (
        a_tSZ = 3.35, alpha_tSZ = -0.53, a_kSZ = 1.48,
        a_p = 6.0, beta_p = 2.2, T_d = 9.7,
        a_c = 4.0, beta_c = 2.2, xi = 0.12,
        a_s = 3.0, beta_s = -2.5,
        a_gtt = 8.0, a_gte = 0.42, a_gee = 0.168,
        a_pste = 0.0, a_psee = 0.05,
    )

    @test_opt target_modules=(ACTLikelihoods,) compute_fg_totals(fg, model)

    # With bandint_shift parameters present (still Float64-valued)
    shift_keys = Tuple(Symbol("bandint_shift_" * e) for e in exps)
    fg_with_shifts = merge(fg, NamedTuple{shift_keys}(ntuple(_ -> 0.0, length(shift_keys))))
    @test_opt target_modules=(ACTLikelihoods,) compute_fg_totals(fg_with_shifts, model)
end

# ------------------------------------------------------------------ #
# Layers 7–8 — theory_vector and loglike (data-dependent)               #
# ------------------------------------------------------------------ #

if isdir(FILTERED_DIR) && isdir(BANDPASS_DIR)
    @testset "type-stability — theory_vector and loglike" begin
        local data    = load_data(FILTERED_DIR)
        local raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
        local model   = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)

        local n_ell     = length(data.l_bpws)
        local ell_slice = 3:(2 + n_ell)
        local cmb_dls = Dict(
            "tt" => vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))[ell_slice],
            "te" => vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))[ell_slice],
            "ee" => vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))[ell_slice],
        )

        local fg = (
            a_tSZ = 3.35, alpha_tSZ = -0.53, a_kSZ = 1.48,
            a_p = 6.91, beta_p = 2.07, T_d = 9.6,
            a_c = 4.88, beta_c = 2.20, xi = 0.12,
            a_s = 3.09, beta_s = -2.76,
            a_gtt = 8.83, a_gte = 0.42, a_gee = 0.168,
            a_pste = -0.023, a_psee = 0.040,
        )
        local cal = (calG_all = 1.0,)
        local fg_totals = compute_fg_totals(fg, model)
        local cal_t_vec = ones(length(data.experiments))
        local cal_e_vec = ones(length(data.experiments))

        @test_opt target_modules=(ACTLikelihoods,) theory_vector_core(
            cmb_dls["tt"], cmb_dls["te"], cmb_dls["ee"],
            fg_totals[1], fg_totals[2], fg_totals[3],
            cal_t_vec, cal_e_vec, data)

        @test_opt target_modules=(ACTLikelihoods,) theory_vector(cmb_dls, fg_totals, cal, data)

        @test_opt target_modules=(ACTLikelihoods,) loglike(cmb_dls, fg, cal, data, model)
    end
end

# ------------------------------------------------------------------ #
# Layer 9 — compute_fg_totals under Dual values (ForwardDiff path)      #
# ------------------------------------------------------------------ #
#
# Regression test for the B1 fix in foreground.jl:
#
#   shift0 = zero(fg_param(p, :a_kSZ, 0.0))
#   fg_param(p, Symbol("bandint_shift_" * exp), shift0)
#
# The typed fallback keeps bandpass shifts concrete on Dual inputs,
# preventing the previous Union{Dual,Float64} cascade into SED closures.
@testset "type-stability — compute_fg_totals (Dual values)" begin
    nu_T  = [90.0, 150.0, 220.0]
    exps  = ["exp090", "exp150", "exp220"]
    raw_T = [RawBand([ν - 10.0, ν, ν + 10.0], [1.0, 1.0, 1.0]) for ν in nu_T]
    raw_P = raw_T
    ell   = collect(2:4000)
    model = ForegroundModel(exps, raw_T, raw_P, ell)

    T_tag = ForwardDiff.Tag{typeof(identity), Float64}
    d(x)  = Dual{T_tag}(x, (1.0,))

    fg_dual = (
        a_tSZ = d(3.35), alpha_tSZ = d(-0.53), a_kSZ = d(1.48),
        a_p = d(6.0), beta_p = d(2.2), T_d = d(9.7),
        a_c = d(4.0), beta_c = d(2.2), xi = d(0.12),
        a_s = d(3.0), beta_s = d(-2.5),
        a_gtt = d(8.0), a_gte = d(0.42), a_gee = d(0.168),
        a_pste = d(0.0), a_psee = d(0.05),
    )

    # Regression guard for B1 fix: this dropped from 4 reports to 0.
    local n_reports = length(JET.get_reports(
        JET.report_opt((() -> compute_fg_totals(fg_dual, model)), ();
                       target_modules=(ACTLikelihoods,))))
    @test n_reports == 0
end

# ------------------------------------------------------------------ #
# calibration vectors — runtime-symbol lookup on Dual calibration      #
# ------------------------------------------------------------------ #
#
# `_calibration_vectors` builds cal_t/cal_e from runtime keys
# `Symbol("cal_" * exp)` and `Symbol("calE_" * exp)`.
#
# Regression objective:
#   - keep this path JET-clean;
#   - keep element type concrete under Dual inputs, including missing
#     keys that fall back to defaults.
@testset "type-stability — calibration vectors (Dual cal)" begin
    T_tag = ForwardDiff.Tag{typeof(identity), Float64}
    d(x)  = Dual{T_tag}(x, (1.0,))

    cal_dual = (
        calG_all    = d(1.0),
        cal_exp090  = d(1.0), cal_exp150  = d(1.0), cal_exp220  = d(1.0),
        calE_exp090 = d(1.0), calE_exp150 = d(1.0), calE_exp220 = d(1.0),
    )
    exps = ["exp090", "exp150", "exp220"]

    local n_reports = length(JET.get_reports(
        JET.report_opt((() -> ACTLikelihoods._calibration_vectors(cal_dual, exps)), ();
                       target_modules=(ACTLikelihoods,))))
    @test n_reports == 0

    cal_t, cal_e = ACTLikelihoods._calibration_vectors(cal_dual, exps)
    @test eltype(cal_t) == typeof(d(1.0))
    @test eltype(cal_e) == typeof(d(1.0))

    # Sparse calibration: missing per-experiment keys must still keep
    # Dual element type (defaults are typed off calG).
    cal_sparse = (calG_all = d(1.0), cal_exp090 = d(1.0))
    cal_t_sparse, cal_e_sparse = ACTLikelihoods._calibration_vectors(cal_sparse, exps)
    @test eltype(cal_t_sparse) == typeof(d(1.0))
    @test eltype(cal_e_sparse) == typeof(d(1.0))

    cal_t_dict, cal_e_dict = ACTLikelihoods.calibration_factors(cal_sparse, exps)
    @test all(v -> v isa typeof(d(1.0)), values(cal_t_dict))
    @test all(v -> v isa typeof(d(1.0)), values(cal_e_dict))
end
