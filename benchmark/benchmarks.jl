"""
    benchmark/benchmarks.jl

Hot-path benchmarks for the ACT DR6 full multifrequency likelihood.

Correctness is established by `Pkg.test()`; this file only measures.

Conventions:
* `BenchmarkTools` only, always with `\$` interpolation.
* `evals=1` wherever a stateful prepared AD cache is involved, so a prepared
  gradient is never measured against a warm inner loop it mutates.
* Cold artifact download, extraction and first construction are reported
  separately by `benchmark/cold_start.jl`; `@time`/`@elapsed` are never used as
  benchmark evidence.

Run with:

    julia --project=benchmark benchmark/benchmarks.jl
"""

using ACTLikelihoods
using ADTypes
using BenchmarkTools
using DelimitedFiles
using DifferentiationInterface
using ForwardDiff
using LinearAlgebra
using Mooncake
using Printf

const CMB_DIR = normpath(joinpath(@__DIR__, "..", "validation", "fixtures",
                                  "act_dr6_cmb_theory"))

function reference_cmb()
    load(name) = Float64.(vec(readdlm(joinpath(CMB_DIR, name))))
    TT = load("cmb_theory_tt.txt")
    return ACTCMBTheory(collect(0:(length(TT) - 1)), TT,
                        load("cmb_theory_te.txt"), load("cmb_theory_ee.txt"))
end

const ARTIFACT = act_dr6_tttee_artifact_path()
const LIKE = ACTDR6FullLikelihood()
const MODEL = ACTDR6FullForegroundModel(LIKE)
const CMB = reference_cmb()
const NUISANCE = ACTDR6Nuisance()
const PARAMETERS = NamedTuple(NUISANCE)
const X = parameter_vector(NUISANCE)
const CMB_INDICES = let
    first_index = findfirst(==(first(LIKE.ells)), CMB.ell)
    first_index:(first_index + length(LIKE.ells) - 1)
end
const FOREGROUNDS = foregrounds(MODEL, LIKE, NUISANCE)
const PREDICTION = predict(LIKE, CMB, FOREGROUNDS, PARAMETERS, CMB_INDICES)

function objective(x)
    p = ACTDR6Nuisance(x)
    fg = foregrounds(MODEL, LIKE, p)
    return loglikelihood(LIKE, predict(LIKE, CMB, fg, NamedTuple(p), CMB_INDICES))
end

const MOONCAKE = AutoMooncake(; config=nothing)
const FORWARD = AutoForwardDiff()

results = Pair{String, Any}[]
record(name, trial) = push!(results, name => trial)

@info "measuring construction"
record("artifact-backed likelihood construction",
       @benchmark ACTDR6FullLikelihood($ARTIFACT) samples=5 evals=1 seconds=120)
record("foreground-model construction",
       @benchmark ACTDR6FullForegroundModel($ARTIFACT, $LIKE) samples=5 evals=1 seconds=120)

@info "measuring the forward path"
record("foreground assembly", @benchmark foregrounds($MODEL, $LIKE, $NUISANCE))
record("predict", @benchmark predict($LIKE, $CMB, $FOREGROUNDS, $PARAMETERS, $CMB_INDICES))
record("chi2", @benchmark chi2($LIKE, $PREDICTION))
record("combined forward likelihood", @benchmark objective($X))

@info "measuring ForwardDiff"
record("ForwardDiff preparation",
       @benchmark prepare_gradient($objective, $FORWARD, $X) samples=10 evals=1 seconds=120)
forward_preparation = prepare_gradient(objective, FORWARD, X)
forward_gradient = similar(X)
record("ForwardDiff prepared gradient (hot)",
       @benchmark gradient!($objective, $forward_gradient, $forward_preparation,
                            $FORWARD, $X) evals=1 seconds=120)

@info "measuring Mooncake"
record("Mooncake preparation",
       @benchmark prepare_gradient($objective, $MOONCAKE, $X) samples=3 evals=1 seconds=300)
mooncake_preparation = prepare_gradient(objective, MOONCAKE, X)
mooncake_gradient = similar(X)
record("Mooncake prepared gradient (hot)",
       @benchmark gradient!($objective, $mooncake_gradient, $mooncake_preparation,
                            $MOONCAKE, $X) evals=1 seconds=120)

# Sanity: the benchmarked gradients must still be the right gradients.
@assert isapprox(mooncake_gradient, forward_gradient; rtol=1e-6, atol=1e-7)

function human(nanoseconds)
    nanoseconds < 1e3 && return @sprintf("%.0f ns", nanoseconds)
    nanoseconds < 1e6 && return @sprintf("%.1f µs", nanoseconds / 1e3)
    nanoseconds < 1e9 && return @sprintf("%.1f ms", nanoseconds / 1e6)
    return @sprintf("%.2f s", nanoseconds / 1e9)
end

println()
println("| operation | median | mean | allocations | memory |")
println("|---|---|---|---|---|")
for (name, trial) in results
    @printf("| %s | %s | %s | %d | %s |\n", name,
            human(median(trial).time), human(mean(trial).time),
            median(trial).allocs, BenchmarkTools.prettymemory(median(trial).memory))
end
println()
println("Julia $(VERSION) on $(Sys.CPU_NAME), $(Sys.CPU_THREADS) threads available")
