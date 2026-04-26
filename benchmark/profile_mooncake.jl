"""
profile_mooncake.jl — per-stage Mooncake gradient timings

Goal: identify which stage of the loglike pipeline is responsible for the
~12 s Mooncake gradient (vs 9.7 ms forward, 127 ms ForwardDiff).

Each stage is wrapped in a closure `Vector{Float64} → Float64` so that
DifferentiationInterface can take its gradient.  Forward and Mooncake
gradient times are reported, plus the slowdown ratio.

Run:
    julia --project=. benchmark/profile_mooncake.jl
"""

using ACTLikelihoods
using BenchmarkTools
using DifferentiationInterface
import Mooncake
using ADTypes: AutoMooncake
using DelimitedFiles: readdlm
using Printf

# ------------------------------------------------------------------ #
# Setup                                                                #
# ------------------------------------------------------------------ #

const FILTERED_DIR = ACTLikelihoods.act_dr6_filtered_dir()
const BANDPASS_DIR = ACTLikelihoods.act_dr6_bandpass_dir()

println("Loading data...")
data = load_data(FILTERED_DIR)

n_ell     = length(data.l_bpws)
ell_slice = 3:(2 + n_ell)
tt_all    = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))
te_all    = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))
ee_all    = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))
cmb_dls   = Dict("tt" => tt_all[ell_slice],
                  "te" => te_all[ell_slice],
                  "ee" => ee_all[ell_slice])

raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)
cal_default = (calG_all = 1.0,)

# ACT DR6 production: 14 free FG params (beta_c tied to beta_p in act_dr6_example.yml)
const FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p,
                 :a_c, :xi, :a_s, :beta_s,
                 :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)
fg_vals = Float64[3.35, -0.53, 1.48, 6.91, 2.07,
                  4.88,  0.12, 3.09, -2.76,
                  8.83,  0.42, 0.168, -0.023, 0.040]

fg_from_vec(v) = NamedTuple{FG_KEYS}(Tuple(v))

# Frozen FG totals (constant w.r.t. gradient targets in some closures)
fg_totals_frozen = compute_fg_totals(fg_from_vec(fg_vals), model)
ps_vec_frozen    = theory_vector(cmb_dls, fg_totals_frozen, cal_default, data)

const MK = AutoMooncake(; config=nothing)

# ------------------------------------------------------------------ #
# Helper: time a closure forward + Mooncake gradient                  #
# ------------------------------------------------------------------ #

mutable struct StageReport
    name      :: String
    fwd_time  :: Float64   # seconds (minimum)
    mk_time   :: Float64   # seconds (minimum)
    n_inputs  :: Int
end

function bench_stage(name::String, f, x::Vector{Float64};
                     mk_max_seconds::Real = 60.0)
    println("\n── $name ─────────────────────────────────────────────")
    println("  n_inputs = ", length(x))

    # Forward — short benchmark
    bf_t = @belapsed $f($x) samples=50 seconds=2
    fwd_t = bf_t

    # Compile Mooncake gradient once (not timed)
    print("  compiling Mooncake gradient... ")
    t0 = time()
    DifferentiationInterface.gradient(f, MK, x)
    println(@sprintf("(%.1f s compile)", time() - t0))

    # Time Mooncake gradient — best of a few runs.  Use manual timing for
    # full control; @belapsed macro can't take dynamic seconds keyword.
    n_runs = 2
    times = Float64[]
    for _ in 1:n_runs
        t0 = time()
        DifferentiationInterface.gradient(f, MK, x)
        push!(times, time() - t0)
        if sum(times) > mk_max_seconds
            break
        end
    end
    mk_t = minimum(times)

    ratio = mk_t / fwd_t
    println(@sprintf("  forward  : %s", fmt_time(fwd_t)))
    println(@sprintf("  mooncake : %s   (%.0f× forward)", fmt_time(mk_t), ratio))

    return StageReport(name, fwd_t, mk_t, length(x))
end

function fmt_time(t_sec::Float64)
    if t_sec < 1e-6
        return @sprintf("%.1f ns", t_sec * 1e9)
    elseif t_sec < 1e-3
        return @sprintf("%.2f μs", t_sec * 1e6)
    elseif t_sec < 1.0
        return @sprintf("%.2f ms", t_sec * 1e3)
    else
        return @sprintf("%.2f s",  t_sec)
    end
end

reports = StageReport[]

# ------------------------------------------------------------------ #
# Stage A: full loglike (15 FG params)                                #
# ------------------------------------------------------------------ #

ll_full(v) = loglike(cmb_dls, fg_from_vec(v), cal_default, data, model)
push!(reports, bench_stage("A: full loglike (15 FG params)", ll_full, fg_vals;
                            mk_max_seconds=120.0))

# ------------------------------------------------------------------ #
# Stage C: compute_fg_totals only (sum of all three returned arrays)  #
# ------------------------------------------------------------------ #

function fg_only(v)
    fT, fE, fEE = compute_fg_totals(fg_from_vec(v), model)
    return sum(fT) + sum(fE) + sum(fEE)
end
push!(reports, bench_stage("C: compute_fg_totals (sum of outputs)",
                            fg_only, fg_vals; mk_max_seconds=120.0))

# ------------------------------------------------------------------ #
# Stage D: theory_vector with frozen fg_totals                        #
# Differentiate w.r.t. a flat cal vector  [calG_all, cal_<exp1>, ...] #
# ------------------------------------------------------------------ #

const CAL_KEYS = (:calG_all,
                   (Symbol("cal_"  * e) for e in data.experiments)...,
                   (Symbol("calE_" * e) for e in data.experiments)...)
cal_vals = vcat(1.0, ones(length(data.experiments)), ones(length(data.experiments)))

cal_from_vec(v) = NamedTuple{CAL_KEYS}(Tuple(v))

# We freeze fg_totals.  But we want theory_vector to be differentiable
# w.r.t. cal — it is.  Use sum of theory output as scalar.
function tv_only(v)
    cal_nt = cal_from_vec(v)
    ps     = theory_vector(cmb_dls, fg_totals_frozen, cal_nt, data)
    return sum(ps)
end
push!(reports, bench_stage("D: theory_vector (frozen FG, vary cal)",
                            tv_only, cal_vals; mk_max_seconds=120.0))

# ------------------------------------------------------------------ #
# Stage E: chi² with precomputed ps_vec                                #
# Differentiate w.r.t. ps_vec itself                                   #
# ------------------------------------------------------------------ #

function chi2_only(ps::Vector{Float64})
    δ = data.data_vec .- ps
    return -0.5 * dot(δ, data.inv_cov * δ) + data.logp_const
end
import LinearAlgebra: dot
push!(reports, bench_stage("E: chi² (vary ps_vec only)",
                            chi2_only, ps_vec_frozen; mk_max_seconds=60.0))

# ------------------------------------------------------------------ #
# Stage F: shift_and_normalize for one band, vary shift               #
# ------------------------------------------------------------------ #

raw_one = model.raw_T[1]
function shift_only(v::Vector{Float64})
    b = shift_and_normalize(raw_one, v[1])
    return sum(b.norm_bp)
end
push!(reports, bench_stage("F: shift_and_normalize (one band, scalar shift)",
                            shift_only, [0.0]; mk_max_seconds=60.0))

# ------------------------------------------------------------------ #
# Stage G: factorized_cross — one kernel call                          #
# Differentiate w.r.t. concatenation [f; cl]                           #
# ------------------------------------------------------------------ #

n_freq = length(data.experiments)   # 5
n_l    = n_ell                       # 8500
f_test  = randn(n_freq)
cl_test = randn(n_l)
fc_x = vcat(f_test, cl_test)

function fc_only(v::Vector{Float64})
    f  = v[1:n_freq]
    cl = v[n_freq+1:end]
    return sum(factorized_cross(f, cl))
end
push!(reports, bench_stage("G: factorized_cross",
                            fc_only, fc_x; mk_max_seconds=120.0))

# ------------------------------------------------------------------ #
# Stage H: correlated_cross — one kernel call                          #
# Differentiate w.r.t. concatenation [vec(f); vec(cl)]                 #
# ------------------------------------------------------------------ #

n_comp  = 2
f_corr  = randn(n_comp, n_freq)
cl_corr = randn(n_comp, n_comp, n_l)
cc_x = vcat(vec(f_corr), vec(cl_corr))

function cc_only(v::Vector{Float64})
    f  = reshape(v[1:n_comp*n_freq],                           n_comp, n_freq)
    cl = reshape(v[n_comp*n_freq+1:end],                       n_comp, n_comp, n_l)
    return sum(correlated_cross(f, cl))
end
push!(reports, bench_stage("H: correlated_cross",
                            cc_only, cc_x; mk_max_seconds=120.0))

# ------------------------------------------------------------------ #
# Summary                                                              #
# ------------------------------------------------------------------ #

println("\n" * "="^78)
println("Mooncake gradient profile — ACTLikelihoods.jl")
println("="^78)
println(@sprintf("%-50s %12s %12s %8s", "stage", "forward", "mooncake", "ratio"))
println("-"^78)
for r in reports
    println(@sprintf("%-50s %12s %12s %7.0fx",
                     r.name, fmt_time(r.fwd_time), fmt_time(r.mk_time),
                     r.mk_time / r.fwd_time))
end
println("="^78)

# Quick attribution: percent of stage A (full loglike) explained by each
A_time = reports[1].mk_time
println("\n%‐of‐full‐loglike attribution (Mooncake):")
for r in reports[2:end]
    pct = 100 * r.mk_time / A_time
    println(@sprintf("  %-50s %5.1f %% of A", r.name, pct))
end
