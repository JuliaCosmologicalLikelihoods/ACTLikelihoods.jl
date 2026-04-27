using Test
using ACTLikelihoods
using Turing
using DelimitedFiles: readdlm

# The Turing extension is loaded as soon as `using Turing` runs alongside
# `using ACTLikelihoods` (PDMats is pulled in transitively via Distributions).
const ATE = Base.get_extension(ACTLikelihoods, :ACTLikelihoodsTuringExt)

const FILTERED_DIR = act_dr6_filtered_dir()
const BANDPASS_DIR = act_dr6_bandpass_dir()

if isdir(FILTERED_DIR) && isdir(BANDPASS_DIR)

@testset "ACTLikelihoodsTuringExt — manual vs Turing log-likelihood" begin
    @test ATE !== nothing

    data = load_data(FILTERED_DIR)
    raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
    fgmodel = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws;
                              ell_0=data.ell_0, nu_0=data.nu_0)

    # Reference CMB theory D_ℓ on data.l_bpws (rows 3..2+n_ell map to ℓ=2..)
    n_ell     = length(data.l_bpws)
    ell_slice = 3:(2 + n_ell)
    tt_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))
    te_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))
    ee_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))
    cmb_dls = Dict(
        "tt" => tt_all[ell_slice],
        "te" => te_all[ell_slice],
        "ee" => ee_all[ell_slice],
    )

    # Fiducial nuisance values — every entry must lie within the support of
    # the corresponding prior in `_act_dr6_priors()` so that
    # `DynamicPPL.fix(model, nt)` succeeds.
    nt = (
        a_tSZ     = 3.35,
        alpha_tSZ = -0.53,
        a_kSZ     = 1.48,
        a_p       = 6.91,
        beta_p    = 2.07,
        a_c       = 4.88,
        xi        = 0.12,
        a_s       = 3.09,
        beta_s    = -2.76,
        a_gtt     = 8.0,    # within Gaussian prior support
        a_gte     = 0.42,
        a_gee     = 0.168,
        a_pste    = -0.023,
        a_psee    = 0.040,
        calG_all  = 1.0,
    )

    fg  = (a_tSZ=nt.a_tSZ, alpha_tSZ=nt.alpha_tSZ, a_kSZ=nt.a_kSZ,
           a_p=nt.a_p, beta_p=nt.beta_p, a_c=nt.a_c, xi=nt.xi,
           a_s=nt.a_s, beta_s=nt.beta_s, a_gtt=nt.a_gtt, a_gte=nt.a_gte,
           a_gee=nt.a_gee, a_pste=nt.a_pste, a_psee=nt.a_psee)
    cal = (calG_all = nt.calG_all,)

    logL_manual = loglike(cmb_dls, fg, cal, data, fgmodel)
    @test isfinite(logL_manual)

    @testset "Variant A — @addlogprob! exactness" begin
        r = ATE.act_dr6_loglike_check(nt, cmb_dls; data=data, model=fgmodel)
        @test r.direct ≈ logL_manual
        @test r.turing ≈ logL_manual
        @test r.diff   == 0.0          # bit-for-bit identical
    end

    @testset "Variant C — data ~ MvNormalCanon" begin
        r = ATE.act_dr6_loglike_check_canon(nt, cmb_dls; data=data, model=fgmodel)
        @test r.direct ≈ logL_manual
        # ACT's `loglike` already includes the full Gaussian normalization
        # (`data.logp_const`), so Variant C must match `loglike` to floating-
        # point relative precision (no additive offset).
        @test isapprox(r.turing, r.direct; rtol = 1e-10)
    end

    @testset "Cross-check: A and C agree" begin
        rA = ATE.act_dr6_loglike_check(nt, cmb_dls; data=data, model=fgmodel)
        rC = ATE.act_dr6_loglike_check_canon(nt, cmb_dls; data=data, model=fgmodel)
        @test isapprox(rA.turing, rC.turing; rtol = 1e-10)
    end
end

else
    @warn "Skipping Turing test — ACT DR6 data directory not found."
end
