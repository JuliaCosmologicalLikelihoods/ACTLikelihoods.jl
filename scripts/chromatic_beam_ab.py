"""
chromatic_beam_ab.py
--------------------
A/B test: production ACT DR6 model uses a chromatic beam factor b^α_ℓ(ν) inside
the band-integrated bandpass (eq. 2.7, Beringue+2025). Our Julia ACTLikelihood.jl
treats the bandpass as ν-only (1D), missing the ℓ-dependence.

This script quantifies the bias by running mflike's BandpowerForeground twice on
the same SACC bandpasses + 2D beams + same FG parameters:

  Mode A — non-chromatic  (use_beam_profile = False)  [our current Julia model]
  Mode B —     chromatic  (use_beam_profile = True )  [production B25/L25]

It then compares:
  * D_FG component-wise at ℓ ∈ {1500, 3000, 5000, 8000} per spectrum
  * Total D_FG difference relative to baseline
  * Δχ² when one model is substituted for the other against ACT DR6 data

Run with the in-tree LAT_MFLike + fgspectra (not site-packages):

  PYTHONPATH=$WORK/actgame/LAT_MFLike:$WORK/actgame/fgspectra \\
      python3 scripts/chromatic_beam_ab.py
"""
import os
import sys
import warnings
import numpy as np

import sacc
from mflike.foreground import BandpowerForeground

WORK = "/home/marcobonici/Desktop/work"
SACC_FILE = os.path.join(
    WORK, "CosmologicalLikelihoods/ACTLikelihood.jl/data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits"
)
FILTERED_DIR = os.path.join(
    WORK, "CosmologicalLikelihoods/ACTLikelihood.jl/data/ACT_DR6_TTTEEE_filtered"
)

EXPERIMENTS = [
    "dr6_pa4_f220",
    "dr6_pa5_f090",
    "dr6_pa5_f150",
    "dr6_pa6_f090",
    "dr6_pa6_f150",
]

LMIN, LMAX = 2, 8500
ELLS = np.arange(LMIN, LMAX + 1)


def load_sacc_bands_and_beams(sacc_file):
    """Extract bandpasses and 2D beams (n_freq, n_ell+2) from SACC file."""
    s = sacc.Sacc.load_fits(sacc_file)
    bands = {}
    beams = {}
    for exp in EXPERIMENTS:
        for spin in ("s0", "s2"):
            tname = f"{exp}_{spin}"
            t = s.tracers[tname]
            nu = np.asarray(t.nu, dtype=float)
            bp = np.asarray(t.bandpass, dtype=float)
            # SACC beam is shape (n_ell, n_freq); production wants (n_freq, n_ell)
            beam = np.asarray(t.beam, dtype=float).T  # → (n_freq, n_ell)
            n_freq = nu.size
            assert beam.shape[0] == n_freq, f"{tname} beam mismatch: {beam.shape} vs nu={n_freq}"
            bands[tname] = {"nu": nu, "bandpass": bp}
            beams[tname] = {"nu": nu, "beams": beam}
    return bands, beams


def build_fg_object(bands, beams, use_chromatic):
    """Construct a BandpowerForeground without going through Cobaya init."""
    fg = BandpowerForeground.__new__(BandpowerForeground)

    # ---- Foreground (parent) attributes ----
    fg.normalisation = {"nu_0": 150.0, "ell_0": 3000, "T_CMB": 2.725}
    fg.components = {
        "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
        "te": ["radio", "dust"],
        "ee": ["radio", "dust"],
    }
    fg.experiments = list(EXPERIMENTS)
    fg.lmin, fg.lmax = LMIN, LMAX
    fg.requested_cls = ["tt", "te", "ee"]
    fg.bandint_freqs = None
    fg.ells = ELLS

    # ---- BandpowerForeground attributes ----
    fg.bands = bands
    fg.beams = beams
    fg.top_hat_band = None
    if use_chromatic:
        fg.beam_profile = {"beam_from_file": None}
    else:
        fg.beam_profile = None
    fg._initialized = bool(use_chromatic)
    # Cobaya gives every Theory a `log`; provide a stub
    import logging
    fg.log = logging.getLogger("BPFG_AB")

    # Stub super().initialize() (Cobaya book-keeping); we only need the SED+template setup
    # in Foreground.initialize → call its body manually
    from fgspectra import cross as fgc
    from fgspectra import frequency as fgf
    from fgspectra import power as fgp
    fg.fg_component_list = {s: fg.components[s] for s in fg.requested_cls}
    fg.bandint_freqs_T = fg.bandint_freqs
    fg.bandint_freqs_P = fg.bandint_freqs
    template_path = os.path.join(os.path.dirname(os.path.abspath(fgp.__file__)), "data")
    fg.fg_nu_0 = fg.normalisation["nu_0"]
    fg.fg_ell_0 = fg.normalisation["ell_0"]

    tsz_file = os.path.join(template_path, "cl_tsz_150_bat.dat")
    cibc_file = os.path.join(template_path, "cl_cib_Choi2020.dat")
    cibxtsz_file = os.path.join(template_path, "cl_sz_x_cib.dat")
    fg.ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    fg.cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    fg.tsz = fgc.FactorizedCrossSpectrum(
        fgf.ThermalSZ(), fgp.PowerLawRescaledTemplate(tsz_file)
    )
    fg.cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
    tsz_cib_sed = fgf.Join(fgf.ThermalSZ(), fgf.CIB())
    tsz_cib_power_spectra = [
        fgp.PowerLawRescaledTemplate(tsz_file),
        fgp.PowerSpectrumFromFile(cibc_file),
        fgp.PowerSpectrumFromFile(cibxtsz_file),
    ]
    tsz_cib_cl = fgp.PowerSpectraAndCovariance(*tsz_cib_power_spectra)
    fg.tSZ_and_CIB = fgc.CorrelatedFactorizedCrossSpectrum(tsz_cib_sed, tsz_cib_cl)
    fg.radioTE = fgc.FactorizedCrossSpectrumTE(
        fgf.PowerLaw(), fgf.PowerLaw(), fgp.PowerLaw()
    )
    fg.dustTE = fgc.FactorizedCrossSpectrumTE(
        fgf.ModifiedBlackBody(), fgf.ModifiedBlackBody(), fgp.PowerLaw()
    )
    fg.radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    fg.dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())

    # Now run init_bandpowers (which builds bandint_freqs_T/P with or without chromatic beams)
    fg.init_bandpowers()
    return fg


def fg_params_default():
    # ACT DR6 best-fit (matches scripts/reference_chi2.py / runtests.jl)
    return dict(
        a_tSZ=3.35, alpha_tSZ=-0.53,
        a_kSZ=1.48,
        a_p=6.91, beta_p=2.07, T_d=9.6, alpha_p=2.0,
        a_c=4.88, beta_c=2.20,                                 # extension model: untied
        xi=0.12,
        a_s=3.09, beta_s=-2.76, alpha_s=1.0,
        a_gtt=8.83, alpha_dT=-0.6, beta_d=1.5, T_effd=19.6,
        a_gte=0.42, alpha_dE=-0.4,
        a_gee=0.168,
        a_pste=-0.023,
        a_psee=0.040,
    )


def compute_fg_dict(fg, fg_params):
    return fg._get_foreground_model_arrays(fg_params, ell=ELLS)


def total_TT_per_pair(fg_dict, n_exp):
    """Sum FG components for TT, returning dict[(i,j)] → array[n_ell].
    Each component already has shape either (n_exp, n_exp, n_ell) or
    (n_exp, n_exp, ...) depending on chromatic mode — fgspectra handles both.
    """
    keys = [k for k in fg_dict if k[0] == "tt"]
    out = {}
    for i in range(n_exp):
        for j in range(n_exp):
            tot = 0.0
            for k in keys:
                v = fg_dict[k]
                tot = tot + v[i, j]
            out[(i, j)] = np.asarray(tot)
    return out


def main():
    print("Loading SACC bandpasses + beams ...")
    bands, beams = load_sacc_bands_and_beams(SACC_FILE)
    n_exp = len(EXPERIMENTS)
    print(f"  {n_exp} experiments, n_ell beam template = {beams[EXPERIMENTS[0]+'_s0']['beams'].shape[1]}")

    print("Building NON-chromatic foreground object ...")
    fg_A = build_fg_object(bands, beams, use_chromatic=False)
    print(f"  bandint_freqs_T[0] type = {type(fg_A.bandint_freqs_T[0]).__name__}")
    if isinstance(fg_A.bandint_freqs_T[0], list):
        nu_A, bp_A = fg_A.bandint_freqs_T[0]
        print(f"  bp shape (non-chrom)  = {np.asarray(bp_A).shape}")

    print("Building   CHROMATIC foreground object ...")
    fg_B = build_fg_object(bands, beams, use_chromatic=True)
    if isinstance(fg_B.bandint_freqs_T[0], list):
        nu_B, bp_B = fg_B.bandint_freqs_T[0]
        print(f"  bp shape (chrom)      = {np.asarray(bp_B).shape}")

    p = fg_params_default()
    print("Computing D_FG (TT) for both modes ...")
    fg_A_dict = compute_fg_dict(fg_A, p)
    fg_B_dict = compute_fg_dict(fg_B, p)

    tot_A = total_TT_per_pair(fg_A_dict, n_exp)
    tot_B = total_TT_per_pair(fg_B_dict, n_exp)

    # ---- Per-pair, per-ell summary ----
    probe_ells = [1500, 3000, 5000, 8000]
    print("\nTT FG total D_ℓ at probe ells (μK² ; * means |chrom-nonchrom|/nonchrom > 1%) :")
    print(f"  pair                ℓ    nonchrom     chrom        ratio       Δ%")
    bias_summary = []
    for i, eA in enumerate(EXPERIMENTS):
        for j, eB in enumerate(EXPERIMENTS):
            if i > j:
                continue
            row_A = tot_A[(i, j)]
            row_B = tot_B[(i, j)]
            for ell in probe_ells:
                idx = ell - LMIN
                a, b = row_A[idx], row_B[idx]
                if a == 0:
                    continue
                ratio = b / a
                pct = 100.0 * (b - a) / a
                marker = "*" if abs(pct) > 1.0 else " "
                bias_summary.append((eA, eB, ell, a, b, ratio, pct))
                print(f"  {eA[4:]}×{eB[4:]:<14} {ell:5d}  {a:10.4f}  {b:10.4f}  {ratio:8.5f}  {pct:+7.3f}%{marker}")

    # ---- Aggregate metric: relative bias on ∑_ℓ FG over the analysis range ----
    print("\nAggregate TT FG bias over ℓ ∈ [500, 7000] (analysis range proxy):")
    ell_lo, ell_hi = 500, 7000
    sl = slice(ell_lo - LMIN, ell_hi - LMIN + 1)
    print(f"  pair                  ∫A           ∫B           Δ%")
    for i, eA in enumerate(EXPERIMENTS):
        for j, eB in enumerate(EXPERIMENTS):
            if i > j:
                continue
            sA = tot_A[(i, j)][sl].sum()
            sB = tot_B[(i, j)][sl].sum()
            if sA != 0:
                print(f"  {eA[4:]}×{eB[4:]:<14} {sA:12.2f}  {sB:12.2f}  {100*(sB-sA)/sA:+7.3f}%")

    # ---- Save spectra to disk for downstream chi² substitution ----
    out_dir = os.path.join(os.path.dirname(__file__), "ab_chrom_out")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "fg_tt_A_nonchrom.npz"),
        ells=ELLS, **{f"{i}_{j}": tot_A[(i, j)] for i in range(n_exp) for j in range(n_exp)},
    )
    np.savez(
        os.path.join(out_dir, "fg_tt_B_chrom.npz"),
        ells=ELLS, **{f"{i}_{j}": tot_B[(i, j)] for i in range(n_exp) for j in range(n_exp)},
    )
    print(f"\nWrote FG D_ℓ arrays to {out_dir}/ for downstream Δχ² computation.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
