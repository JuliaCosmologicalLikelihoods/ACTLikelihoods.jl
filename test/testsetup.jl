"""
    test/testsetup.jl

Shared fixtures and the single artifact-backed likelihood instance used by the
whole suite.

Nothing here is conditional. A missing artifact, a malformed artifact or a
missing reference fixture must fail the suite, never skip it.
"""

using DelimitedFiles
using LinearAlgebra
using Test

const FIXTURE_ROOT = normpath(joinpath(@__DIR__, "..", "validation", "fixtures"))
const REFERENCE_DIR = joinpath(FIXTURE_ROOT, "act_dr6_full_reference")
const MULTIPOINT_DIR = joinpath(FIXTURE_ROOT, "act_dr6_full_multipoint")
const CMB_FIXTURE_DIR = joinpath(FIXTURE_ROOT, "act_dr6_cmb_theory")

"""
    fixture(parts...) -> String

Absolute path of a checked-in reference fixture. Throws if it is absent, so a
deleted fixture fails loudly instead of silently reducing coverage.
"""
function fixture(parts...)
    path = joinpath(parts...)
    isfile(path) || error("required reference fixture is missing: $path")
    return path
end

"Read a one-value-per-line text fixture."
read_reference_vector(parts...) = Float64.(vec(readdlm(fixture(parts...))))

"""
    read_reference_table(parts...) -> (header, rows)

Read a tab-separated fixture as a header vector and a `Matrix{Any}` of rows.
"""
function read_reference_table(parts...)
    raw, header = readdlm(fixture(parts...), '\t', Any; header=true)
    return (String.(vec(header)), raw)
end

"""
    reference_scalars() -> Dict{String, Float64}

The scalar reference values exported from the original ACT DR6 code.
"""
function reference_scalars()
    _, rows = read_reference_table(REFERENCE_DIR, "scalars.tsv")
    return Dict(String(rows[i, 1]) => Float64(rows[i, 2]) for i in axes(rows, 1))
end

"""
    reference_cmb() -> ACTCMBTheory

The frozen CAMB reference spectra at the ACT DR6 best-fit cosmology, on the
multipole grid `0:9050`. See `validation/fixtures/act_dr6_cmb_theory/README.md`.
"""
function reference_cmb()
    load(name) = read_reference_vector(CMB_FIXTURE_DIR, name)
    TT = load("cmb_theory_tt.txt")
    TE = load("cmb_theory_te.txt")
    EE = load("cmb_theory_ee.txt")
    return ACTCMBTheory(collect(0:(length(TT) - 1)), TT, TE, EE)
end

# ---------------------------------------------------------------------------
# The public path, built once. This is the artifact download and the metadata
# validation, exercised exactly as an ordinary user would.
# ---------------------------------------------------------------------------

const ARTIFACT_DIR = act_dr6_tttee_artifact_path()
const LIKE = ACTDR6FullLikelihood()
const FOREGROUND_MODEL = ACTDR6FullForegroundModel(LIKE)
const REFERENCE_CMB = reference_cmb()
const REFERENCE_NUISANCE = ACTDR6Nuisance()

"Prevalidated CMB index range for the ACT window support, for hot paths."
const CMB_INDICES = let
    first_index = findfirst(==(first(LIKE.ells)), REFERENCE_CMB.ell)
    first_index:(first_index + length(LIKE.ells) - 1)
end

"""
    isolated_foregrounds(; kwargs...) -> ACTForegrounds

Foreground arrays with every amplitude switched off except the ones named.

Each ACT DR6 foreground component is linear in its own amplitude, so switching
the others off isolates exactly the component the upstream exporter writes out
under the same name. The only exception is the tSZ x CIB cross term, which is
reachable only as a difference of the correlated block — see
`test_full_foregrounds.jl`.
"""
function isolated_foregrounds(; kwargs...)
    off = (a_tSZ = 0.0, a_kSZ = 0.0, a_p = 0.0, a_c = 0.0, a_s = 0.0,
           a_gtt = 0.0, a_gte = 0.0, a_gee = 0.0, a_psee = 0.0, a_pste = 0.0,
           xi = 0.0)
    nuisance = ACTDR6Nuisance(merge(free_parameters(REFERENCE_NUISANCE), off, values(kwargs)))
    return foregrounds(FOREGROUND_MODEL, LIKE, nuisance)
end

"Select the polarization block of an `ACTForegrounds` by lowercase name."
function foreground_block(fg::ACTForegrounds, polarization::AbstractString)
    polarization == "tt" && return fg.TT
    polarization == "te" && return fg.TE
    polarization == "ee" && return fg.EE
    error("unknown polarization block: $polarization")
end

"Index into `LIKE.ells` for a multipole value."
ell_index(ell::Integer) = Int(ell) - first(LIKE.ells) + 1
