"""
benchmarks.jl — ACTLikelihoods.jl performance benchmarks

Measures (steady-state with DifferentiationInterface preparation):
  1. loglike forward pass (chi² evaluation)
  2. ForwardDiff gradient (15 FG params)
  3. ForwardDiff gradient (20 params: FG + bandpass shifts)
  4. Mooncake gradient    (15 FG params)
  5. Mooncake gradient    (20 params: FG + bandpass shifts)

All gradient benchmarks use `prepare_gradient` + `gradient!` so the
reported time reflects repeated-evaluation cost (relevant for MCMC,
optimization) rather than tape construction.

Run:
    julia --project=. benchmark/benchmarks.jl
"""

using ACTLikelihoods
using BenchmarkTools
using DifferentiationInterface
using DifferentiationInterface: Constant
import ForwardDiff, Mooncake
using ADTypes: AutoForwardDiff, AutoMooncake
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

cal = (calG_all = 1.0,)

## ACT DR6 production: 14 free FG params (beta_c tied to beta_p in act_dr6_example.yml)
const FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p,
                 :a_c, :xi, :a_s, :beta_s,
                 :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)
fg_vals = Float64[3.35, -0.53, 1.48, 6.91, 2.07,
                  4.88,  0.12, 3.09, -2.76,
                  8.83,  0.42, 0.168, -0.023, 0.040]

const SHIFT_KEYS = Tuple(Symbol("bandint_shift_" * e) for e in data.experiments)
const ALL_KEYS   = (FG_KEYS..., SHIFT_KEYS...)
all_vals = vcat(fg_vals, zeros(Float64, 5))

fg_from_vec(v)  = NamedTuple{FG_KEYS}(Tuple(v))
all_from_vec(v) = NamedTuple{ALL_KEYS}(Tuple(v))

# `data` is the only truly fixed argument (observation + covariance);
# `cmb_dls`, `cal`, and `model` may all become differentiable later
# (cosmology Cls, calibration nuisances, bandpass shifts wrt model internals).
# We close over them but keep them tangent-tracked.
ll_fg(v,  d) = loglike(cmb_dls, fg_from_vec(v),  cal, d, model)
ll_all(v, d) = loglike(cmb_dls, all_from_vec(v), cal, d, model)

# Warm up forward
ll_fg(fg_vals,  data)
ll_all(all_vals, data)

println("\n", "="^60)
println("ACTLikelihoods.jl — Benchmark results")
println("  n_bins    = ", length(data.data_vec))
println("  n_spectra = ", length(data.spec_meta))
println("  n_ell     = ", n_ell)
println("="^60, "\n")

# ------------------------------------------------------------------ #
# 1. Forward pass                                                      #
# ------------------------------------------------------------------ #

println("── 1. loglike forward pass ──────────────────────────────────")
b_fwd = @benchmark ll_fg($fg_vals, $data)
show(stdout, MIME("text/plain"), b_fwd)
println()

# ------------------------------------------------------------------ #
# Backends                                                              #
# ------------------------------------------------------------------ #

fd_backend = AutoForwardDiff()
mk_backend = AutoMooncake(; config=nothing)

# Pre-allocate gradient buffers and prepare DI extras (one-shot per backend+input).
grad_fg  = similar(fg_vals)
grad_all = similar(all_vals)

println("\nPreparing DifferentiationInterface contexts (data marked Constant)...")
prep_fd_fg  = prepare_gradient(ll_fg,  fd_backend, fg_vals,  Constant(data))
prep_fd_all = prepare_gradient(ll_all, fd_backend, all_vals, Constant(data))
prep_mk_fg  = prepare_gradient(ll_fg,  mk_backend, fg_vals,  Constant(data))
prep_mk_all = prepare_gradient(ll_all, mk_backend, all_vals, Constant(data))
println("  done.")

# Sanity warmup
DifferentiationInterface.gradient!(ll_fg,  grad_fg,  prep_fd_fg,  fd_backend, fg_vals,  Constant(data))
DifferentiationInterface.gradient!(ll_all, grad_all, prep_fd_all, fd_backend, all_vals, Constant(data))
DifferentiationInterface.gradient!(ll_fg,  grad_fg,  prep_mk_fg,  mk_backend, fg_vals,  Constant(data))
DifferentiationInterface.gradient!(ll_all, grad_all, prep_mk_all, mk_backend, all_vals, Constant(data))

# ------------------------------------------------------------------ #
# 2. ForwardDiff — 15 FG params                                       #
# ------------------------------------------------------------------ #

println("\n── 2. ForwardDiff ∇loglike (15 FG params) ──────────────────")
b_fd15 = @benchmark DifferentiationInterface.gradient!(
    $ll_fg, $grad_fg, $prep_fd_fg, $fd_backend, $fg_vals, $(Constant(data)))
show(stdout, MIME("text/plain"), b_fd15)
println()

# ------------------------------------------------------------------ #
# 3. ForwardDiff — 20 params (FG + bandpass shifts)                   #
# ------------------------------------------------------------------ #

println("\n── 3. ForwardDiff ∇loglike (20 params: FG + shifts) ────────")
b_fd20 = @benchmark DifferentiationInterface.gradient!(
    $ll_all, $grad_all, $prep_fd_all, $fd_backend, $all_vals, $(Constant(data)))
show(stdout, MIME("text/plain"), b_fd20)
println()

# ------------------------------------------------------------------ #
# 4. Mooncake — 15 FG params                                          #
# ------------------------------------------------------------------ #

println("\n── 4. Mooncake ∇loglike (15 FG params) ─────────────────────")
b_mk15 = @benchmark DifferentiationInterface.gradient!(
    $ll_fg, $grad_fg, $prep_mk_fg, $mk_backend, $fg_vals, $(Constant(data)))
show(stdout, MIME("text/plain"), b_mk15)
println()

# ------------------------------------------------------------------ #
# 5. Mooncake — 20 params (FG + bandpass shifts)                      #
# ------------------------------------------------------------------ #

println("\n── 5. Mooncake ∇loglike (20 params: FG + shifts) ───────────")
b_mk20 = @benchmark DifferentiationInterface.gradient!(
    $ll_all, $grad_all, $prep_mk_all, $mk_backend, $all_vals, $(Constant(data)))
show(stdout, MIME("text/plain"), b_mk20)
println()

# ------------------------------------------------------------------ #
# Summary table                                                        #
# ------------------------------------------------------------------ #

function fmt(b)
    t = minimum(b).time   # nanoseconds
    if t < 1e3
        return @sprintf("%.1f ns", t)
    elseif t < 1e6
        return @sprintf("%.2f μs", t / 1e3)
    elseif t < 1e9
        return @sprintf("%.2f ms", t / 1e6)
    else
        return @sprintf("%.2f s",  t / 1e9)
    end
end

println("\n", "="^60)
println("Summary (minimum time over all samples, with DI prep)")
println("="^60)
println(@sprintf("  %-45s %s", "loglike forward pass",          fmt(b_fwd)))
println(@sprintf("  %-45s %s", "ForwardDiff ∇ (15 FG params)",  fmt(b_fd15)))
println(@sprintf("  %-45s %s", "ForwardDiff ∇ (20 FG+shifts)",  fmt(b_fd20)))
println(@sprintf("  %-45s %s", "Mooncake    ∇ (15 FG params)",  fmt(b_mk15)))
println(@sprintf("  %-45s %s", "Mooncake    ∇ (20 FG+shifts)",  fmt(b_mk20)))
println("="^60)
