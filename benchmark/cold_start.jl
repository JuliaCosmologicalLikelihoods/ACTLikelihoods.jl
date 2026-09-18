"""
    benchmark/cold_start.jl

Cold-start cost of the published artifact, measured in a throwaway depot so the
download and extraction really happen.

The artifact is bound non-lazily, so `Pkg.instantiate()` is what fetches and
unpacks it. That step is therefore reported on its own, separately from package
loading and from first construction.

This is wall-clock reporting of a one-off, network- and I/O-bound step. It is
deliberately kept out of `benchmarks.jl` and is never presented as a
BenchmarkTools result.

Run with:

    julia --project=benchmark benchmark/cold_start.jl
"""

const PROJECT = normpath(joinpath(@__DIR__, ".."))

const INSTANTIATE = raw"""
using Printf
start = time_ns()
using Pkg
Pkg.instantiate()
@printf("instantiate + artifact download/unpack   %.1f s\n", (time_ns() - start) / 1e9)
"""

const CONSTRUCT = raw"""
using Printf
start = time_ns()
using ACTLikelihoods
loaded = time_ns()
path = act_dr6_tttee_artifact_path()
resolved = time_ns()
like = ACTDR6FullLikelihood()
constructed = time_ns()
model = ACTDR6FullForegroundModel(like)
finished = time_ns()
@printf("package load (cold precompile)           %.1f s\n", (loaded - start) / 1e9)
@printf("artifact path resolution                 %.2f s\n", (resolved - loaded) / 1e9)
@printf("first likelihood construction            %.1f s\n", (constructed - resolved) / 1e9)
@printf("first foreground-model construction      %.1f s\n", (finished - constructed) / 1e9)
@printf("total from cold depot to usable model    %.1f s\n", (finished - start) / 1e9)
println("artifact path: ", path)
println(like)
"""

mktempdir() do depot
    println("cold depot: $depot")
    println("project:    $PROJECT")
    for script in (INSTANTIATE, CONSTRUCT)
        run(setenv(`$(Base.julia_cmd()) --project=$PROJECT -e $script`,
                   "JULIA_DEPOT_PATH" => depot,
                   "HOME" => get(ENV, "HOME", depot),
                   "PATH" => get(ENV, "PATH", "")))
    end
end
