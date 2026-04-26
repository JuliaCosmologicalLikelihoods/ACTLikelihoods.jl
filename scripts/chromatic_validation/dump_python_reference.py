"""
dump_python_reference.py
------------------------
Run the production foreground pipeline (mflike.BandpowerForeground + fgspectra)
for the fixed parameter set in fg_params.json, in BOTH chromatic and
non-chromatic modes, and dump every intermediate layer to HDF5.

Outputs:
  outputs/nonchrom.h5     use_beam_profile = False
  outputs/chrom.h5        use_beam_profile = True

The companion script `compare_julia.jl` reads these dumps and validates the
Julia ACTLikelihood.jl pipeline against them layer by layer.

Run from the repo root with the in-tree mflike + fgspectra (NOT site-packages):

  PYTHONPATH=$WORK/actgame/LAT_MFLike:$WORK/actgame/fgspectra \\
      python3 scripts/chromatic_validation/dump_python_reference.py

See README.md for the layer index.
"""
import json
import logging
import os
import warnings

import numpy as np
import sacc
import h5py

from mflike.foreground import BandpowerForeground

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "fg_params.json")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WORK = "/home/marcobonici/Desktop/work"
SACC_FILE = os.path.join(
    WORK, "CosmologicalLikelihoods/ACTLikelihood.jl/data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits"
)


# ------------------------------------------------------------------ #
# Component lists                                                      #
# ------------------------------------------------------------------ #
# Components dumped per spectrum. Note `tSZ` and `cibc` are dumped for
# diagnostics but are NOT summed into the total — production's TT total uses
# the *combined* `tSZ_and_CIB` term (which already contains tSZ + cibc + the
# cross), see mflike.foreground.Foreground.components["tt"].
TT_COMPONENTS_DUMP = ["kSZ", "tSZ", "cibp", "cibc", "radio", "dust", "tSZ_and_CIB"]
TT_COMPONENTS_SUM  = ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"]
TE_COMPONENTS = ["radio", "dust"]
EE_COMPONENTS = ["radio", "dust"]


# ------------------------------------------------------------------ #
# SACC loader                                                          #
# ------------------------------------------------------------------ #
def load_sacc_bands_and_beams(sacc_file, experiments):
    """Pull bandpasses and 2D beams from a SACC file.

    Returns
    -------
    bands : dict {f"{exp}_{spin}": {"nu": (n_freq,), "bandpass": (n_freq,)}}
    beams : dict {f"{exp}_{spin}": {"nu": (n_freq,), "beams": (n_freq, n_ell+2)}}
            Note SACC stores beams as (n_ell, n_freq); we transpose to
            (n_freq, n_ell) to match production expectations.
    """
    s = sacc.Sacc.load_fits(sacc_file)
    bands, beams = {}, {}
    for exp in experiments:
        for spin in ("s0", "s2"):
            tname = f"{exp}_{spin}"
            t = s.tracers[tname]
            nu = np.asarray(t.nu, dtype=float)
            bp = np.asarray(t.bandpass, dtype=float)
            beam = np.asarray(t.beam, dtype=float).T   # → (n_freq, n_ell)
            assert beam.shape[0] == nu.size, f"{tname} beam mismatch"
            bands[tname] = {"nu": nu, "bandpass": bp}
            beams[tname] = {"nu": nu, "beams": beam}
    return bands, beams


# ------------------------------------------------------------------ #
# BandpowerForeground bootstrap                                        #
# ------------------------------------------------------------------ #
def build_fg_object(bands, beams, experiments, ells, nu_0, ell_0, T_CMB,
                    use_chromatic):
    """Build a BandpowerForeground without going through Cobaya init.

    Mirrors the body of Foreground.initialize() inline so we get the same SED
    objects and template handles, then sets BandpowerForeground state and runs
    init_bandpowers() to populate self.bandint_freqs_T/P.
    """
    fg = BandpowerForeground.__new__(BandpowerForeground)

    # ---- Foreground (parent) attributes ----
    fg.normalisation = {"nu_0": nu_0, "ell_0": ell_0, "T_CMB": T_CMB}
    fg.components = {
        "tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
        "te": ["radio", "dust"],
        "ee": ["radio", "dust"],
    }
    fg.experiments = list(experiments)
    fg.lmin, fg.lmax = int(ells[0]), int(ells[-1])
    fg.requested_cls = ["tt", "te", "ee"]
    fg.bandint_freqs = None
    fg.ells = np.asarray(ells)

    # ---- BandpowerForeground attributes ----
    fg.bands = bands
    fg.beams = beams
    fg.top_hat_band = None
    fg.beam_profile = {"beam_from_file": None} if use_chromatic else None
    fg._initialized = bool(use_chromatic)
    fg.log = logging.getLogger("dump_python_reference")

    # ---- inline body of Foreground.initialize() ----
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

    # build bandint_freqs_T/P (with or without chromatic beams)
    fg.init_bandpowers()
    return fg


# ------------------------------------------------------------------ #
# Per-component evaluator                                              #
# ------------------------------------------------------------------ #
def eval_components(fg, fg_params, ells):
    """Compute every FG component separately.

    The mflike top-level dict from `_get_foreground_model_arrays` already
    contains all components keyed (spec, name). We just split by spec and
    sanity-check the required keys.
    """
    model = fg._get_foreground_model_arrays(fg_params, ell=ells)

    out = {"tt": {}, "te": {}, "ee": {}}
    for c in TT_COMPONENTS_DUMP:
        out["tt"][c] = np.asarray(model[("tt", c)])
    for c in TE_COMPONENTS:
        out["te"][c] = np.asarray(model[("te", c)])
    for c in EE_COMPONENTS:
        out["ee"][c] = np.asarray(model[("ee", c)])
    return out


# ------------------------------------------------------------------ #
# Bandpass-weight extractor                                            #
# ------------------------------------------------------------------ #
def extract_bandpass_layer(fg, exp_idx):
    """Pull the per-experiment bandpass-weight tensor produced by
    init_bandpowers().

    Returns a dict with:
      nub        : (n_freq,)
      W_T        : (n_freq,)              [non-chromatic]
                or (n_freq, n_ell)        [chromatic]
      W_P        : same shape as W_T
    """
    e_T = fg.bandint_freqs_T[exp_idx]
    e_P = fg.bandint_freqs_P[exp_idx]
    if isinstance(e_T, list):
        nub_T, W_T = e_T
        nub_P, W_P = e_P
        return {
            "nub_T": np.asarray(nub_T, dtype=float),
            "nub_P": np.asarray(nub_P, dtype=float),
            "W_T": np.asarray(W_T, dtype=float),
            "W_P": np.asarray(W_P, dtype=float),
        }
    # delta-function bandpass case (not used in DR6 baseline, but handle it)
    return {
        "nub_T": np.atleast_1d(np.asarray(e_T, dtype=float)),
        "nub_P": np.atleast_1d(np.asarray(e_P, dtype=float)),
        "W_T": np.array([1.0]),
        "W_P": np.array([1.0]),
    }


# ------------------------------------------------------------------ #
# Main dump per mode                                                   #
# ------------------------------------------------------------------ #
def dump_mode(out_path, *, use_chromatic, cfg, bands, beams):
    print(f"\n=== dumping mode: {'chrom' if use_chromatic else 'nonchrom'}  →  {out_path} ===")

    experiments = cfg["experiments"]
    ells = np.arange(cfg["ell_grid"]["lmin"], cfg["ell_grid"]["lmax"] + 1, dtype=int)
    n_exp = len(experiments)
    n_ell = ells.size

    fg = build_fg_object(
        bands, beams, experiments, ells,
        nu_0=cfg["normalisation"]["nu_0"],
        ell_0=cfg["normalisation"]["ell_0"],
        T_CMB=cfg["normalisation"]["T_CMB"],
        use_chromatic=use_chromatic,
    )

    # cmb2bb is private — pull via the same import path mflike uses
    from mflike.foreground import _cmb2bb

    # Patch in nonzero shifts if requested (DR6 baseline = 0)
    fg_p = dict(cfg["fg_params"])

    print("Evaluating per-component arrays …")
    comps = eval_components(fg, fg_p, ells)

    print("Computing totals …")
    totals = {
        "tt": sum(comps["tt"][c] for c in TT_COMPONENTS_SUM),
        "te": sum(comps["te"][c] for c in TE_COMPONENTS),
        "ee": sum(comps["ee"][c] for c in EE_COMPONENTS),
    }

    print("Writing HDF5 …")
    with h5py.File(out_path, "w") as h:
        # ---- meta ----
        m = h.create_group("meta")
        m.create_dataset("ells", data=ells)
        m.create_dataset(
            "experiments",
            data=np.array(experiments, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        m.create_dataset("n_exp", data=n_exp)
        m.create_dataset("n_ell", data=n_ell)
        m.create_dataset("nu_0", data=float(cfg["normalisation"]["nu_0"]))
        m.create_dataset("ell_0", data=float(cfg["normalisation"]["ell_0"]))
        m.create_dataset("T_CMB", data=float(cfg["normalisation"]["T_CMB"]))
        m.create_dataset("use_chromatic", data=bool(use_chromatic))
        m.create_dataset("fg_params_json", data=json.dumps(fg_p))

        # ---- bandpass / per experiment ----
        bp_g = h.create_group("bandpass")
        for i, exp in enumerate(experiments):
            g = bp_g.create_group(exp)
            nu = bands[f"{exp}_s0"]["nu"]
            tau_raw = bands[f"{exp}_s0"]["bandpass"]
            shift = float(cfg["bandint_shifts"][exp])
            nub = nu + shift
            cmb2bb_arr = _cmb2bb(nub)

            g.create_dataset("nu", data=nu)
            g.create_dataset("tau_raw", data=tau_raw)
            g.create_dataset("nub", data=nub)
            g.create_dataset("cmb2bb", data=cmb2bb_arr)
            g.create_dataset("shift", data=shift)

            # raw beams (same in both modes; exists in SACC regardless)
            g.create_dataset("beam_T", data=beams[f"{exp}_s0"]["beams"])
            g.create_dataset("beam_P", data=beams[f"{exp}_s2"]["beams"])

            # post-init normalized weights
            layer = extract_bandpass_layer(fg, i)
            g.create_dataset("W_T", data=layer["W_T"])
            g.create_dataset("W_P", data=layer["W_P"])
            g.create_dataset("nub_T", data=layer["nub_T"])
            g.create_dataset("nub_P", data=layer["nub_P"])

        # ---- per-component spectra ----
        c_g = h.create_group("components")
        for spec, comps_for_spec in comps.items():
            sg = c_g.create_group(spec)
            for name, arr in comps_for_spec.items():
                sg.create_dataset(name, data=arr)

        # ---- totals ----
        t_g = h.create_group("totals")
        for spec, arr in totals.items():
            t_g.create_dataset(spec, data=arr)

    print(f"  done. shapes: tt {totals['tt'].shape}  te {totals['te'].shape}  ee {totals['ee'].shape}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(PARAMS_FILE) as f:
        cfg = json.load(f)

    print(f"Loading SACC bandpasses + beams from {SACC_FILE}")
    bands, beams = load_sacc_bands_and_beams(SACC_FILE, cfg["experiments"])

    n_ell = beams[cfg["experiments"][0] + "_s0"]["beams"].shape[1]
    print(f"  {len(cfg['experiments'])} experiments,  beam template n_ell = {n_ell}")

    dump_mode(os.path.join(OUT_DIR, "nonchrom.h5"),
              use_chromatic=False, cfg=cfg, bands=bands, beams=beams)
    dump_mode(os.path.join(OUT_DIR, "chrom.h5"),
              use_chromatic=True, cfg=cfg, bands=bands, beams=beams)

    print("\nAll dumps complete.")


if __name__ == "__main__":
    main()
