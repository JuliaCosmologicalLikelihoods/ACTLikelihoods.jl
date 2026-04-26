"""
extract_data.py
---------------
One-time extraction of ACT DR6 likelihood data from the SACC file into HDF5,
for loading in Julia without needing sacc/Python at runtime.

Usage:
    python extract_data.py [--packages-path /path/to/cobaya/packages]

Produces: ../data/dr6_data.h5
"""

import argparse
import os
import sys
import numpy as np
import h5py

def get_packages_path(args_path=None):
    """Find cobaya packages path."""
    if args_path:
        return args_path
    # Try common locations
    for p in [
        os.path.expanduser("~/cobaya_packages"),
        os.path.expanduser("~/packages"),
        os.environ.get("COBAYA_PACKAGES_PATH", ""),
    ]:
        if p and os.path.exists(p):
            return p
    raise RuntimeError(
        "Could not find cobaya packages path. "
        "Pass --packages-path or set COBAYA_PACKAGES_PATH."
    )


def main():
    parser = argparse.ArgumentParser(description="Extract ACT DR6 likelihood data to HDF5")
    parser.add_argument("--packages-path", default=None, help="Path to cobaya packages directory")
    parser.add_argument("--output", default=None, help="Output HDF5 path (default: ../data/dr6_data.h5)")
    args = parser.parse_args()

    packages_path = get_packages_path(args.packages_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, "../data/dr6_data.h5")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Packages path: {packages_path}")
    print(f"Output: {output_path}")

    # ------------------------------------------------------------------ #
    # 1. Initialize the ACT DR6 likelihood (loads SACC, builds metadata)  #
    # ------------------------------------------------------------------ #
    from act_dr6_mflike import ACTDR6MFLike
    from cobaya.conventions import packages_path_input

    like = ACTDR6MFLike({
        packages_path_input: packages_path,
        "path": None,
    })
    like.initialize()

    # ------------------------------------------------------------------ #
    # 2. Core data quantities                                              #
    # ------------------------------------------------------------------ #
    data_vec   = like.data_vec                    # shape (n_bins,)
    cov        = like.cov                         # shape (n_bins, n_bins)
    inv_cov    = like.inv_cov                     # shape (n_bins, n_bins)
    logp_const = like.logp_const                  # scalar
    l_bpws     = like.l_bpws                      # shape (n_ell,)  — ell grid

    print(f"  data_vec shape:  {data_vec.shape}")
    print(f"  covariance shape: {cov.shape}")
    print(f"  ell grid length:  {len(l_bpws)}, range [{l_bpws.min()}, {l_bpws.max()}]")

    # ------------------------------------------------------------------ #
    # 3. Bandpower windows — per spectrum                                  #
    # ------------------------------------------------------------------ #
    # spec_meta is a list of dicts; each has a 'bpw' field (BandpowerWindows object)
    # We store the window weight matrix W[n_bins_for_spec, n_ell_full] per spectrum,
    # along with the ids (indices into data_vec), polarization, and experiment names.
    spec_meta_list = []
    for m in like.spec_meta:
        bpw = m["bpw"]
        # bpw.weight: shape (n_ell_full, n_bins_for_spec) — note transposed convention
        # bpw.values: the ell values of the window columns
        spec_meta_list.append({
            "ids":        m["ids"],                    # indices into data_vec
            "pol":        m["pol"],                    # "tt", "te", "ee"
            "t1":         m["t1"],                     # experiment 1 name
            "t2":         m["t2"],                     # experiment 2 name
            "hasYX_xsp":  m["hasYX_xsp"],              # bool: ET-type spectrum
            "W":          bpw.weight.T.astype(np.float64),  # (n_bins, n_ell_full)
            "W_ells":     bpw.values.astype(np.int32),      # ell array for W columns
            # Optimized window representation (nonzeros + sliced weights)
            "nonzero_starts":  np.array([s.start for s in bpw.nonzeros], dtype=np.int32),
            "nonzero_stops":   np.array([s.stop  for s in bpw.nonzeros], dtype=np.int32),
            "sliced_weights":  [w.astype(np.float64) for w in bpw.sliced_weights],
        })

    print(f"  n_spectra: {len(spec_meta_list)}")

    # ------------------------------------------------------------------ #
    # 4. Bandpass info (nu + bp per experiment)                            #
    # ------------------------------------------------------------------ #
    experiments = like.experiments
    bands = {}
    for exp in experiments:
        for spin in ["s0", "s2"]:
            key = f"{exp}_{spin}"
            b = like.bands[key]
            bands[key] = {
                "nu":       np.asarray(b["nu"],       dtype=np.float64),
                "bandpass": np.asarray(b["bandpass"], dtype=np.float64),
            }

    print(f"  experiments: {experiments}")

    # ------------------------------------------------------------------ #
    # 5. Write HDF5                                                        #
    # ------------------------------------------------------------------ #
    print(f"\nWriting {output_path} ...")
    with h5py.File(output_path, "w") as f:
        # Core likelihood data
        f.create_dataset("data_vec",   data=data_vec)
        f.create_dataset("covariance", data=cov)
        f.create_dataset("inv_cov",    data=inv_cov)
        f.create_dataset("logp_const", data=np.array([logp_const]))
        f.create_dataset("l_bpws",     data=l_bpws.astype(np.int32))

        # Experiments list (stored as variable-length strings)
        dt = h5py.special_dtype(vlen=str)
        exp_ds = f.create_dataset("experiments", (len(experiments),), dtype=dt)
        exp_ds[:] = experiments

        # Bandpasses
        bp_grp = f.create_group("bands")
        for key, b in bands.items():
            g = bp_grp.create_group(key)
            g.create_dataset("nu",       data=b["nu"])
            g.create_dataset("bandpass", data=b["bandpass"])

        # Spec meta
        sm_grp = f.create_group("spec_meta")
        for i, m in enumerate(spec_meta_list):
            g = sm_grp.create_group(str(i))
            g.create_dataset("ids",           data=m["ids"])
            g.attrs["pol"]       = m["pol"]
            g.attrs["t1"]        = m["t1"]
            g.attrs["t2"]        = m["t2"]
            g.attrs["hasYX_xsp"] = int(m["hasYX_xsp"])
            g.create_dataset("W",             data=m["W"])
            g.create_dataset("W_ells",        data=m["W_ells"])
            g.create_dataset("nonzero_starts",data=m["nonzero_starts"])
            g.create_dataset("nonzero_stops", data=m["nonzero_stops"])
            sw_grp = g.create_group("sliced_weights")
            for j, w in enumerate(m["sliced_weights"]):
                sw_grp.create_dataset(str(j), data=w)

        # Metadata
        f.attrs["n_bins"]    = int(data_vec.size)
        f.attrs["n_spectra"] = int(len(spec_meta_list))
        f.attrs["ell_min"]   = int(l_bpws.min())
        f.attrs["ell_max"]   = int(l_bpws.max())
        f.attrs["nu_0"]      = 150.0
        f.attrs["ell_0"]     = 3000

    print("Done.")
    print(f"\nSummary:")
    print(f"  {data_vec.size} bandpower bins")
    print(f"  {len(spec_meta_list)} spectra")
    print(f"  ell range: [{l_bpws.min()}, {l_bpws.max()}]")
    print(f"  logp_const = {logp_const:.6f}")


if __name__ == "__main__":
    main()
