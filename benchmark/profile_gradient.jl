"""
profile_gradient.jl — profile Mooncake ∇loglike with prep

Reports both time-profile and allocation-profile breakdowns,
filtered to ACTLikelihoods internals.
"""

using ACTLikelihoods
using DifferentiationInterface
using DifferentiationInterface: Constant
import Mooncake, ForwardDiff
using ADTypes: AutoMooncake, AutoForwardDiff
using DelimitedFiles: readdlm
using Profile
import Profile.Allocs
using Printf

const FILTERED_DIR = ACTLikelihoods.act_dr6_filtered_dir()
const BANDPASS_DIR = ACTLikelihoods.act_dr6_bandpass_dir()

println("Loading data...")
data = load_data(FILTERED_DIR)
n_ell     = length(data.l_bpws)
ell_slice = 3:(2 + n_ell)
tt_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_tt.txt")))
te_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_te.txt")))
ee_all = vec(readdlm(joinpath(FILTERED_DIR, "cmb_theory_ee.txt")))
cmb_dls = Dict("tt" => tt_all[ell_slice], "te" => te_all[ell_slice], "ee" => ee_all[ell_slice])
raw_T, raw_P = load_bands(BANDPASS_DIR, data.experiments)
model = ForegroundModel(data.experiments, raw_T, raw_P, data.l_bpws)
cal = (calG_all = 1.0,)

# ACT DR6 production: 14 free FG params (beta_c tied to beta_p in act_dr6_example.yml)
const FG_KEYS = (:a_tSZ, :alpha_tSZ, :a_kSZ, :a_p, :beta_p, :a_c, :xi,
                 :a_s, :beta_s, :a_gtt, :a_gte, :a_gee, :a_pste, :a_psee)
fg_vals = Float64[3.35, -0.53, 1.48, 6.91, 2.07, 4.88, 0.12, 3.09, -2.76,
                  8.83, 0.42, 0.168, -0.023, 0.040]
fg_from_vec(v) = NamedTuple{FG_KEYS}(Tuple(v))
ll_fg(v, d) = loglike(cmb_dls, fg_from_vec(v), cal, d, model)

mk_backend = AutoMooncake(; config=nothing)
grad_fg = similar(fg_vals)
prep_mk = prepare_gradient(ll_fg, mk_backend, fg_vals, Constant(data))

# ------------------------------------------------------------------ #
# Warmup                                                                #
# ------------------------------------------------------------------ #

println("Warming up...")
for _ in 1:5
    DifferentiationInterface.gradient!(ll_fg, grad_fg, prep_mk, mk_backend, fg_vals, Constant(data))
end

# ------------------------------------------------------------------ #
# Time profile                                                          #
# ------------------------------------------------------------------ #

println("\n", "="^70)
println("TIME profile — Mooncake ∇loglike (15 FG params), 100 iterations")
println("="^70)

Profile.clear()
Profile.init(n=10^7, delay=0.0001)   # 100 µs sampling
@profile begin
    for _ in 1:100
        DifferentiationInterface.gradient!(ll_fg, grad_fg, prep_mk, mk_backend, fg_vals, Constant(data))
    end
end

# Flat profile filtered to ACTLikelihoods / Mooncake / DI hot frames.
println("\n— Top frames by self-count (flat, filtered to ACTLikelihoods):")
Profile.print(IOContext(stdout, :displaysize => (50, 200)),
              format=:flat, sortedby=:count, mincount=20,
              noisefloor=2.0)

println("\n— Hierarchical (top 50, mincount=50):")
Profile.print(IOContext(stdout, :displaysize => (50, 200)),
              format=:tree, mincount=50, maxdepth=20)

# ------------------------------------------------------------------ #
# Allocation profile                                                    #
# ------------------------------------------------------------------ #

println("\n", "="^70)
println("ALLOC profile — Mooncake ∇loglike (15 FG params), 50 iterations")
println("="^70)

Allocs.clear()
Allocs.@profile sample_rate=1.0 begin
    for _ in 1:50
        DifferentiationInterface.gradient!(ll_fg, grad_fg, prep_mk, mk_backend, fg_vals, Constant(data))
    end
end

results = Allocs.fetch()
allocs = results.allocs
println("\nTotal sampled allocations: ", length(allocs))
println("Total bytes:                ", sum(a.size for a in allocs) ÷ (1024^2), " MiB")

# Aggregate by call site (top stack frame).
agg = Dict{String, Tuple{Int, Int}}()
for a in allocs
    if !isempty(a.stacktrace)
        # find first ACTLikelihoods / Mooncake / Effort frame, fall back to top.
        f = a.stacktrace[1]
        for sf in a.stacktrace
            mod = string(sf.file)
            if occursin("ACTLikelihoods", mod) || occursin("Mooncake", mod) || occursin("rrules", mod)
                f = sf
                break
            end
        end
        key = string(f.func, " @ ", basename(string(f.file)), ":", f.line)
        old = get(agg, key, (0, 0))
        agg[key] = (old[1] + 1, old[2] + a.size)
    end
end

println("\n— Top 25 alloc sites by total size:")
sorted = sort(collect(agg), by=x->x[2][2], rev=true)
for (k, (n, sz)) in sorted[1:min(25, length(sorted))]
    @printf("  %8d KiB  (%5d allocs)  %s\n", sz ÷ 1024, n, k)
end
