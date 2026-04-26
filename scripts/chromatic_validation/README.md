# Chromatic-beam validation harness

Standalone Python ↔ Julia conformance harness for the ACT DR6 foreground
pipeline. Designed to (1) verify our existing non-chromatic Julia code matches
the production `mflike` + `fgspectra` reference bit-for-bit, and (2) act as
the conformance test for the future chromatic-beam port.

## Files

| File | Role |
|------|------|
| `fg_params.json` | Single fixed FG parameter set (ACT DR6 best-fit) — used by both Python and Julia. |
| `dump_python_reference.py` | Runs production `mflike.BandpowerForeground` for the fixed params in **both** chromatic and non-chromatic modes. Writes per-layer HDF5 dumps. |
| `compare_julia.jl` | Loads a dump and recomputes each layer using ACTLikelihoods.jl. Reports max-abs and max-rel diffs per layer. Exits 0 on full pass, 1 on first failure. |
| `outputs/` | Generated artifacts (gitignored). `nonchrom.h5` and `chrom.h5`. |

## Layer index

Both scripts agree on this order. The Julia comparator stops on the first
failure to make debugging easier:

| Layer | What is checked | Shape per channel |
|------:|------------------|-------------------|
| 0 | `cmb2bb(ν)` — Planck → RJ conversion factor at the bandpass grid | `(n_freq,)` |
| 1 | Bandpass weights `W_T(ν)`, `W_P(ν)` — normalized passband × ∂B/∂T | `(n_freq,)` non-chrom · `(n_freq, n_ell)` chrom |
| 2 | (informative) per-component `D_FG^{αβ}(ℓ)` — printed if layer 3 fails | `(n_exp, n_exp, n_ell)` |
| 3 | Total `D_FG^{αβ}(ℓ)` for TT, TE, EE | `(n_exp, n_exp, n_ell)` |

Layer 1 in chromatic mode is intentionally a short-circuit: the Julia
ACTLikelihoods pipeline does not yet implement ℓ-dependent bandpass weights,
so the comparator reports the unimplemented branch and exits.

## Run order

```bash
# 1. Dump Python reference (uses in-tree mflike + fgspectra, NOT site-packages)
PYTHONPATH=/home/marcobonici/Desktop/work/actgame/LAT_MFLike:/home/marcobonici/Desktop/work/actgame/fgspectra \
    python3 scripts/chromatic_validation/dump_python_reference.py

# 2. Validate non-chromatic Julia matches the reference (all layers must pass)
julia --project=. scripts/chromatic_validation/compare_julia.jl --mode nonchrom

# 3. Run chromatic mode — currently expected to short-circuit at layer 1
julia --project=. scripts/chromatic_validation/compare_julia.jl --mode chrom
```

Default tolerance is `rtol = 1e-10`. Override with `--rtol 1e-8` (etc.).

## HDF5 layout

```
/meta/
    ells, experiments, n_exp, n_ell,
    nu_0, ell_0, T_CMB, use_chromatic,
    fg_params_json   # JSON-serialized parameter dict

/bandpass/{exp}/
    nu, tau_raw, nub, cmb2bb,        # raw + normalized inputs
    shift,                           # the bandint shift used (= 0 in baseline)
    beam_T, beam_P,                  # SACC-loaded chromatic beams (2D)
    W_T, W_P,                        # normalized weights — 1D non-chrom, 2D chrom
    nub_T, nub_P                     # frequency grids the weights sit on

/components/{tt,te,ee}/{kSZ,tSZ,cibp,cibc,radio,dust,tSZ_and_CIB}/
    D_ℓ^{αβ}(ℓ)                      # (n_exp, n_exp, n_ell)

/totals/{tt,te,ee}/
    D_ℓ^{αβ}(ℓ)                      # (n_exp, n_exp, n_ell)
```

Note: per-component groups list only the components that exist for that
spectrum (TT has all seven; TE and EE only have radio and dust).

## When to refresh the dump

The dump is committed-static for fast iteration: regenerate it only when
- the FG parameter set changes (`fg_params.json`),
- the production `mflike` or `fgspectra` code changes,
- the SACC file (`dr6_data.fits`) changes.

## Adding a new comparison layer

1. Add the array(s) to `dump_python_reference.py` (one new `create_dataset`).
2. Add a `check_*` function in `compare_julia.jl` that recomputes the same
   thing using ACTLikelihoods APIs and calls `diff_arrays`.
3. Slot the new check into `main()` between the existing layer calls.

Keep new layers small and focused — the value of the harness is in pinpointing
*which* layer disagrees, not in dumping everything.
