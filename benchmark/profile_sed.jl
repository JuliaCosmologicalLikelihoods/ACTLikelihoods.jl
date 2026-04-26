"""
profile_sed.jl — fine-grained Mooncake profile of compute_fg_totals.

Breaks compute_fg_totals into:
  C1: 8 eval_sed_bands calls only (varies beta_p, beta_s)
  C2: ℓ-templates only (varies alpha_tSZ, a_tSZ, a_c, xi)
  C3: factorized_cross + correlated_cross combinators (varies amplitude params)
  C0: full compute_fg_totals (14 fg params, ACT DR6 baseline; beta_c tied to beta_p)

This pinpoints which sub-stage the remaining 317 ms Mooncake gradient
sits in, after the Phase-3 rrules.
"""

using ACTLikelihoods
using DifferentiationInterface
import Mooncake
using ADTypes: AutoMooncake
using DelimitedFiles: readdlm
using Printf
using Statistics: median

const FILTERED_DIR = ACTLikelihoods.act_dr6_filtered_dir()
const BANDPASS_DIR = ACTLikelihoods.act_dr6_bandpass_dir()

println("Loading data...")
data = load_data(FILTERED_DIR)
raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)

# Use the same default-shifted bands the model would see at shift=0
const ell  = model.ell
const ell0 = model.ell_0
const nu0  = model.nu_0
bands_T = [ACTLikelihoods.shift_and_normalize(r, 0.0) for r in raw_T]
bands_P = [ACTLikelihoods.shift_and_normalize(r, 0.0) for r in raw_P]

const T_d_def    = 9.6
const T_effd_def = 19.6
const beta_d_def = 1.5

function fmt(t)
    t < 1e-6 ? @sprintf("%.1f ns", t*1e9) :
    t < 1e-3 ? @sprintf("%.2f μs", t*1e6) :
    t < 1.0  ? @sprintf("%.2f ms", t*1e3) :
               @sprintf("%.2f s",  t)
end

function timeit(f, x, mk; n=3)
    DifferentiationInterface.gradient(f, mk, x)        # warmup
    fwd = minimum(@elapsed(f(x)) for _ in 1:n)
    grd = minimum(@elapsed(DifferentiationInterface.gradient(f, mk, x)) for _ in 1:n)
    return fwd, grd
end

println("\n", "="^66)
println("compute_fg_totals — sub-stage Mooncake profile")
println("="^66)

mk = AutoMooncake(; config=nothing)

# ---- Stage SED-MBB: only mbb_sed evaluations (2 distinct betas) ----
# vary 2 mbb spectral indices to exercise the AD path through eval_sed_bands
function ll_mbb(v)
    bp, bc = v[1], v[2]
    f1 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, bp, T_d_def), bands_T)
    f2 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, bc, T_d_def), bands_T)
    return sum(f1) + sum(f2)
end
fwd, grd = timeit(ll_mbb, [2.07, 2.20], mk)
@printf("  S1: 2× eval_sed_bands(mbb)        fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))

# ---- Stage SED-MBB-T: same but T_d differentiable ----
function ll_mbb_T(v)
    bp, bc, td = v[1], v[2], v[3]
    f1 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, bp, td), bands_T)
    f2 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, bc, td), bands_T)
    return sum(f1) + sum(f2)
end
fwd, grd = timeit(ll_mbb_T, [2.07, 2.20, 9.6], mk)
@printf("  S2: 2× eval_sed_bands(mbb,β,T)    fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))

# ---- Stage SED-RADIO: only radio_sed evaluations ----
function ll_radio(v)
    bs = v[1]
    f1 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.radio_sed(ν, nu0, bs), bands_T)
    f2 = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.radio_sed(ν, nu0, bs), bands_P)
    return sum(f1) + sum(f2)
end
fwd, grd = timeit(ll_radio, [-2.76], mk)
@printf("  S3: 2× eval_sed_bands(radio)      fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))

# ---- Stage TEMPLATE: ℓ-template stage (uses alpha_tSZ, a_*) ----
function ll_template(v)
    α, a_tSZ, a_c, ξ = v[1], v[2], v[3], v[4]
    cl_tsz   = ACTLikelihoods.eval_template_tilt(model.T_tsz, ell, ell0, α; amp=a_tSZ)
    cl_cibc  = ACTLikelihoods.eval_template(model.T_cibc, ell, ell0; amp=a_c)
    cl_szxcib = ACTLikelihoods.eval_template(model.T_szxcib, ell, ell0;
                                             amp=-ξ * sqrt(a_tSZ * a_c))
    return sum(cl_tsz) + sum(cl_cibc) + sum(cl_szxcib)
end
fwd, grd = timeit(ll_template, [-0.53, 3.35, 4.88, 0.12], mk)
@printf("  S4: ℓ-template tilt+amp           fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))

# ---- Stage FULL: full compute_fg_totals (14 fg params, ACT DR6 baseline) ----
# beta_c tied to beta_p (act_dr6_example.yml: beta_c: lambda beta_p: beta_p)
const FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p,
                 :a_c, :xi, :a_s, :beta_s,
                 :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)
fg_vals = Float64[3.35, -0.53, 1.48, 6.91, 2.07,
                  4.88,  0.12, 3.09, -2.76,
                  8.83,  0.42, 0.168, -0.023, 0.040]

ll_full(v) = sum(sum.(compute_fg_totals(NamedTuple{FG_KEYS}(Tuple(v)), model)))
fwd, grd = timeit(ll_full, fg_vals, mk)
@printf("  S5: compute_fg_totals (14 params) fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))

# ---- Stage ASSEMBLY: just the .+ chain at the end of fg_TT ----
# Use precomputed SEDs so we isolate the assembly cost.
n_exp = length(bands_T)
ell_clp = ell .* (ell .+ 1)
ell_0clp = ell0 * (ell0 + 1)
const f_ksz_T0  = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.constant_sed(ν), bands_T)
const f_tsz_T0  = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.tsz_sed(ν, nu0), bands_T)
const f_cibp0   = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, 2.07, T_d_def), bands_T)
const f_cibc0   = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, 2.20, T_d_def), bands_T)
const f_radio0  = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.radio_sed(ν, nu0, -2.76), bands_T)
const f_dust0   = ACTLikelihoods.eval_sed_bands(ν -> ACTLikelihoods.mbb_sed(ν, nu0, beta_d_def, T_effd_def), bands_T)
const cl_ksz0   = ACTLikelihoods.eval_template(model.T_ksz,  ell, ell0)
const cl_tsz0   = ACTLikelihoods.eval_template_tilt(model.T_tsz, ell, ell0, -0.53; amp=3.35)
const cl_cibc0  = ACTLikelihoods.eval_template(model.T_cibc, ell, ell0; amp=4.88)
const cl_szxcib0 = ACTLikelihoods.eval_template(model.T_szxcib, ell, ell0; amp=-0.12 * sqrt(3.35*4.88))
const cl_szcibc0 = ACTLikelihoods.build_szxcib_cl(cl_tsz0, cl_cibc0, cl_szxcib0)
const cl_cibp0  = ACTLikelihoods.eval_powerlaw(Float64.(ell_clp), Float64(ell_0clp), 1.0)
const cl_radio0 = ACTLikelihoods.eval_powerlaw(Float64.(ell_clp), Float64(ell_0clp), 1.0)
const cl_dustT0 = ACTLikelihoods.eval_powerlaw(Float64.(ell), 500.0, -0.6)

function ll_assembly(v)
    a_ksz, a_p, a_gtt, a_s, a_pste = v[1], v[2], v[3], v[4], v[5]
    fg_TT = a_ksz .* ACTLikelihoods.factorized_cross(f_ksz_T0,  cl_ksz0)   .+
            ACTLikelihoods.correlated_cross(vcat(f_tsz_T0', f_cibc0'), cl_szcibc0) .+
            a_p   .* ACTLikelihoods.factorized_cross(f_cibp0,   cl_cibp0) .+
            a_gtt .* ACTLikelihoods.factorized_cross(f_dust0,   cl_dustT0) .+
            a_s   .* ACTLikelihoods.factorized_cross(f_radio0,  cl_radio0)
    return sum(fg_TT)
end
fwd, grd = timeit(ll_assembly, [1.48, 6.91, 8.83, 3.09, -0.023], mk)
@printf("  S6: TT assembly (.+ chain, 5 amp) fwd=%-10s mooncake=%-10s ratio=%4dx\n",
        fmt(fwd), fmt(grd), round(Int, grd/fwd))
