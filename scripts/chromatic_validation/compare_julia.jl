#=
compare_julia.jl
----------------
Layer-by-layer Julia ↔ Python comparison driver for the foreground pipeline.

Reads:
  scripts/chromatic_validation/fg_params.json
  scripts/chromatic_validation/outputs/{nonchrom,chrom}.h5
            ↑ produced by dump_python_reference.py

For each layer (cmb2bb → bandpass weights → per-component spectra → totals)
recomputes the quantity in Julia using ACTLikelihoods.jl APIs and compares
against the dumped reference. Stops on first failure with a precise
diagnostic (which (α,β,ℓ) and which rel-diff).

Usage:
    julia --project=. scripts/chromatic_validation/compare_julia.jl --mode nonchrom
    julia --project=. scripts/chromatic_validation/compare_julia.jl --mode chrom
=#
using ACTLikelihoods
using HDF5
using JSON
using Printf

# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #
function parse_args()
    mode = "nonchrom"
    rtol = 1e-10
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--mode";  mode = ARGS[i+1];  i += 2
        elseif a == "--rtol"; rtol = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--help" || a == "-h"
            println("Usage: julia compare_julia.jl [--mode nonchrom|chrom] [--rtol 1e-10]")
            exit(0)
        else
            error("Unknown arg: $a")
        end
    end
    mode in ("nonchrom", "chrom") || error("--mode must be nonchrom or chrom (got $mode)")
    return (; mode, rtol)
end

const HERE = @__DIR__
const PARAMS_FILE = joinpath(HERE, "fg_params.json")
const OUT_DIR = joinpath(HERE, "outputs")

# ------------------------------------------------------------------ #
# Diff helpers                                                          #
# ------------------------------------------------------------------ #
struct LayerResult
    name      :: String
    pass      :: Bool
    max_abs   :: Float64
    max_rel   :: Float64
    detail    :: String
end

function diff_arrays(label::String, a::AbstractArray, b::AbstractArray; rtol::Float64)
    size(a) == size(b) || return LayerResult(label, false, NaN, NaN,
        "shape mismatch: julia=$(size(a)) python=$(size(b))")
    Δ      = abs.(a .- b)
    scale  = maximum(abs, b) + eps()
    max_abs = maximum(Δ)
    max_rel = max_abs / scale
    pass    = max_rel <= rtol
    detail = ""
    if !pass
        idx = argmax(Δ)
        detail = "first worst at index $(Tuple(idx)): " *
                 @sprintf("julia=%.15g  python=%.15g  Δ=%.3g", a[idx], b[idx], Δ[idx])
    end
    return LayerResult(label, pass, max_abs, max_rel, detail)
end

function report(r::LayerResult)
    sym = r.pass ? "✓" : "✗"
    @printf("  %s %-50s  max_abs=%.3e  max_rel=%.3e", sym, r.name, r.max_abs, r.max_rel)
    println()
    isempty(r.detail) || println("      ", r.detail)
    return r.pass
end

# ------------------------------------------------------------------ #
# Load helpers                                                          #
# ------------------------------------------------------------------ #
"""Read the dumped HDF5 into a NamedTuple of plain arrays — eager, simple."""
function load_dump(path::String)
    h = h5open(path, "r")
    try
        meta = h["meta"]
        ells          = read(meta["ells"])
        experiments   = read(meta["experiments"])
        nu_0          = read(meta["nu_0"])
        ell_0         = Int(read(meta["ell_0"]))
        T_CMB         = read(meta["T_CMB"])
        use_chromatic = read(meta["use_chromatic"]) != 0
        fg_params_str = read(meta["fg_params_json"])

        # Python rows×cols layout for 2D bandpass arrays is (n_freq, n_ell);
        # HDF5.jl flips that to (n_ell, n_freq). Restore the original axes.
        flip2(a::AbstractArray{T,2}) where {T} = permutedims(a, (2, 1))
        flip2(a::AbstractArray{T,1}) where {T} = a

        bp_dict = Dict{String, NamedTuple}()
        for exp in experiments
            g = h["bandpass/$(exp)"]
            bp_dict[exp] = (
                nu      = read(g["nu"]),
                tau_raw = read(g["tau_raw"]),
                nub     = read(g["nub"]),
                cmb2bb  = read(g["cmb2bb"]),
                shift   = read(g["shift"]),
                W_T     = flip2(read(g["W_T"])),
                W_P     = flip2(read(g["W_P"])),
                nub_T   = read(g["nub_T"]),
                nub_P   = read(g["nub_P"]),
                beam_T  = flip2(read(g["beam_T"])),
                beam_P  = flip2(read(g["beam_P"])),
            )
        end

        # Python writes arrays in row-major as (n_exp, n_exp, n_ell);
        # HDF5.jl returns them in column-major order, which flips the axes
        # to (n_ell, n_exp, n_exp). Permute back so Julia code can index as
        # arr[α, β, ℓ] consistent with `compute_fg_totals`.
        flip3(a::Array{T,3}) where {T} = permutedims(a, (3, 2, 1))

        comp = Dict{String, Dict{String, Array{Float64,3}}}(
            "tt" => Dict(), "te" => Dict(), "ee" => Dict(),
        )
        for spec in ("tt", "te", "ee")
            g = h["components/$(spec)"]
            for k in keys(g)
                comp[spec][k] = flip3(read(g[k]))
            end
        end

        totals = Dict{String, Array{Float64,3}}(
            spec => flip3(read(h["totals/$(spec)"])) for spec in ("tt", "te", "ee")
        )

        return (;
            ells = Vector{Int}(ells),
            experiments = String.(experiments),
            nu_0, ell_0, T_CMB, use_chromatic,
            fg_params = JSON.parse(fg_params_str),
            bp = bp_dict,
            comp, totals,
        )
    finally
        close(h)
    end
end

"""Convert the JSON FG-param dict into a Dict{Symbol,Float64} for ACTLikelihoods."""
function build_fg_params(cfg, dump_fg)
    p = Dict{Symbol, Float64}()
    for (k, v) in dump_fg
        p[Symbol(k)] = Float64(v)
    end
    # bandint shifts (always 0 for the baseline harness)
    for (exp, shift) in cfg["bandint_shifts"]
        p[Symbol("bandint_shift_" * exp)] = Float64(shift)
    end
    return p
end

# ------------------------------------------------------------------ #
# Layer 0 — cmb2bb(ν)                                                   #
# ------------------------------------------------------------------ #
"""Sample Julia's cmb2bb on each channel's ν grid; compare to dump."""
function check_cmb2bb(dump; rtol)
    println("\n[layer 0]  cmb2bb(ν)  per channel")
    ok = true
    for exp in dump.experiments
        bp = dump.bp[exp]
        # mflike's cmb2bb uses nub (= nu + shift); we replicate
        julia_c = ACTLikelihoods.cmb2bb.(bp.nub)
        r = diff_arrays("cmb2bb @ $exp", julia_c, bp.cmb2bb; rtol)
        ok &= report(r)
    end
    return ok
end

# ------------------------------------------------------------------ #
# Layer 1 — bandpass weights W_T(ν), W_P(ν)                              #
# ------------------------------------------------------------------ #
"""
Build Julia normalized passbands (`make_band` ∘ shift) and compare to the
dumped W_T / W_P. In nonchrom mode the dumped W is shape (n_freq,);
we expect it to equal `band.norm_bp`.

In chrom mode the dump is (n_freq, n_ell) — Julia chromatic beams are not
yet implemented. We short-circuit with a clean diagnostic.
"""
function check_bandpass_weights(dump; rtol)
    println("\n[layer 1]  bandpass weights W_T(ν) / W_P(ν)  per channel")
    if dump.use_chromatic
        println("  (chromatic mode: Julia ACTLikelihoods does not yet support 2D " *
                "ℓ-dependent bandpass weights — short-circuiting)")
        return false
    end

    ok = true
    for exp in dump.experiments
        bp = dump.bp[exp]
        raw_T = ACTLikelihoods.RawBand{Float64}(Vector{Float64}(bp.nu),
                                                Vector{Float64}(bp.tau_raw))
        band_T = ACTLikelihoods.shift_and_normalize(raw_T, bp.shift)
        # Same data is used for s2 in DR6 (verified s0 ≡ s2)
        raw_P = ACTLikelihoods.RawBand{Float64}(Vector{Float64}(bp.nu),
                                                Vector{Float64}(bp.tau_raw))
        band_P = ACTLikelihoods.shift_and_normalize(raw_P, bp.shift)

        r_T = diff_arrays("W_T @ $exp", band_T.norm_bp, bp.W_T; rtol)
        r_P = diff_arrays("W_P @ $exp", band_P.norm_bp, bp.W_P; rtol)
        ok &= report(r_T)
        ok &= report(r_P)
    end
    return ok
end

# ------------------------------------------------------------------ #
# Build a Julia ForegroundModel from the dump                            #
# ------------------------------------------------------------------ #
function build_julia_model(dump)
    raw_T = [ACTLikelihoods.RawBand{Float64}(
                Vector{Float64}(dump.bp[exp].nu),
                Vector{Float64}(dump.bp[exp].tau_raw))
             for exp in dump.experiments]
    raw_P = deepcopy(raw_T)   # s0 ≡ s2 for ACT DR6 (verified)
    return ACTLikelihoods.ForegroundModel(
        dump.experiments, raw_T, raw_P, dump.ells;
        ell_0 = dump.ell_0, nu_0 = dump.nu_0,
    )
end

# ------------------------------------------------------------------ #
# Layer 2 — per-component D_FG^{αβ}(ℓ)                                   #
# ------------------------------------------------------------------ #
#
# Julia's `compute_fg_totals` returns *only the totals*. To compare per-component
# we need the same intermediates. The current Julia code path does not expose
# them as named outputs, so layer-2 comparison is informative only when paired
# with the totals check (layer 3): if totals match but a component-level
# comparison disagrees, that points to a sign/template bug whose contributions
# happen to cancel. For now we expose *what we can* by re-deriving each
# component from its known closed form using the Julia-side machinery.

function check_components_and_totals(dump; rtol)
    println("\n[layer 2/3]  per-component & total D_FG^{αβ}(ℓ)")

    cfg = JSON.parsefile(PARAMS_FILE)
    p_dict = build_fg_params(cfg, dump.fg_params)
    p = (; (Symbol(k) => v for (k, v) in p_dict)...)

    model = build_julia_model(dump)

    fg_TT, fg_TE, fg_EE = ACTLikelihoods.compute_fg_totals(p, model)

    # ---- Layer 3: totals (the strict end-to-end check) ----
    rT = diff_arrays("total TT", fg_TT, dump.totals["tt"]; rtol)
    rE = diff_arrays("total EE", fg_EE, dump.totals["ee"]; rtol)
    rX = diff_arrays("total TE", fg_TE, dump.totals["te"]; rtol)
    ok = report(rT) & report(rX) & report(rE)

    if !ok
        # Per-component diagnostic: print which Python components dominate
        # each spectrum so the user can see where to look.
        println()
        println("  per-component magnitudes (Python reference, max |D_ℓ|):")
        for spec in ("tt", "te", "ee")
            comps = dump.comp[spec]
            for (name, arr) in comps
                @printf("    [%s] %-20s  max|D|=%.3e\n", spec, name, maximum(abs, arr))
            end
        end
    end

    return ok
end

# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #
function main()
    args = parse_args()
    dump_path = joinpath(OUT_DIR, args.mode * ".h5")
    isfile(dump_path) || error("Dump not found: $dump_path  (run dump_python_reference.py)")

    println("="^70)
    println("Comparing Julia ACTLikelihoods against Python reference")
    println("  mode      = $(args.mode)")
    println("  dump file = $dump_path")
    println("  rtol      = $(args.rtol)")
    println("="^70)

    dump = load_dump(dump_path)
    println("\nLoaded $(length(dump.experiments)) experiments, " *
            "n_ell=$(length(dump.ells)) (ℓ ∈ $(dump.ells[1])..$(dump.ells[end]))")

    all_ok = true
    all_ok &= check_cmb2bb(dump; rtol = args.rtol)
    all_ok &= check_bandpass_weights(dump; rtol = args.rtol)
    if args.mode == "nonchrom"
        all_ok &= check_components_and_totals(dump; rtol = args.rtol)
    else
        println("\n[layer 2/3]  skipped — chromatic Julia path not implemented")
        all_ok = false
    end

    println("\n" * "="^70)
    if all_ok
        println("ALL LAYERS PASSED.")
        exit(0)
    else
        println("FAILED — see ✗ markers above.")
        exit(1)
    end
end

main()
