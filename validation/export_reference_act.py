#!/usr/bin/env python3
"""Export the ACT DR6 full-likelihood baseline reference as deterministic text.

Run from the repository root:

    python3 validation/export_reference_act.py \
        --sacc-file data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits \
        --cmb-directory validation/fixtures/act_dr6_cmb_theory \
        --output validation/fixtures/act_dr6_full_reference \
        --act-source ../act_dr6_mflike

Add ``--check`` to verify that a fresh run reproduces the checked-in fixtures
byte-for-byte instead of overwriting them.

Everything written here is plain text. No NumPy binaries, and no bulk copies of
data that the published Zenodo artifact already carries.
"""

from __future__ import annotations

import argparse
import difflib
import filecmp
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act_dr6_reference_common import (  # noqa: E402
    CHECKPOINT_ELLS,
    CHECKPOINT_NU_INDICES,
    FOREGROUNDS,
    SYSTEMATICS,
    build_foreground,
    build_likelihood,
    checkpoint_rows,
    fmt,
    git_revision,
    load_cmb,
    moment_rows,
    sha256,
    write_table,
    write_vector,
)

# Component keys produced by ``Foreground._get_foreground_model_arrays``.
# ``tSZ_and_CIB`` is the correlated block; ``tSZ`` and ``cibc`` are the same
# templates evaluated alone, so their difference isolates the tSZ x CIB cross
# term, which cannot be switched on by itself.
COMPONENT_KEYS = (
    ("tt", "kSZ"),
    ("tt", "tSZ"),
    ("tt", "cibc"),
    ("tt", "cibp"),
    ("tt", "radio"),
    ("tt", "dust"),
    ("tt", "tSZ_and_CIB"),
    ("te", "radio"),
    ("te", "dust"),
    ("ee", "radio"),
    ("ee", "dust"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sacc-file", type=Path, required=True,
                        help="the original ACT DR6 SACC release (dr6_data.fits)")
    parser.add_argument("--cmb-directory", type=Path, required=True,
                        help="directory holding the frozen cmb_theory_{tt,te,ee}.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--act-source", type=Path, help="act_dr6_mflike checkout")
    parser.add_argument("--mflike-source", type=Path, help="LAT_MFLike checkout")
    parser.add_argument("--fgspectra-source", type=Path, help="fgspectra checkout")
    parser.add_argument("--check", action="store_true",
                        help="verify the checked-in fixtures instead of rewriting them")
    return parser.parse_args()


def export(output: Path, args: argparse.Namespace, stack: list) -> None:
    like = build_likelihood(args.sacc_file, stack)
    foreground = build_foreground(like)
    ells = np.asarray(foreground.ells)
    parameters = FOREGROUNDS | SYSTEMATICS

    cmb = load_cmb(args.cmb_directory)

    totals = foreground.get_foreground_model_totals(**parameters)
    components = foreground._get_foreground_model_arrays(parameters)

    model = like._get_power_spectra(cmb, totals, **SYSTEMATICS)
    zero_totals = [np.zeros_like(total) for total in totals]
    model_no_foregrounds = like._get_power_spectra(cmb, zero_totals, **SYSTEMATICS)
    residual = like.data_vec - model
    chi2 = float(like._fast_chi_squared(like.inv_cov, residual))
    loglike = float(like.loglike(cmb, totals, **SYSTEMATICS))

    # --- 1. full observation-space vectors -------------------------------
    write_vector(output / "data_vector.txt", like.data_vec)
    write_vector(output / "model_vector.txt", model)
    write_vector(output / "model_vector_no_foregrounds.txt", model_no_foregrounds)
    write_vector(output / "residual.txt", residual)

    # --- 2. scalars -------------------------------------------------------
    write_table(
        output / "scalars.tsv",
        ("name", "value"),
        (
            ("number_of_bandpowers", str(int(like.data_vec.size))),
            ("number_of_spectra", str(len(like.spec_meta))),
            ("ell_min", str(int(ells[0]))),
            ("ell_max", str(int(ells[-1]))),
            ("chi2", fmt(chi2)),
            ("data_only_loglikelihood", fmt(-0.5 * chi2)),
            ("gaussian_log_normalization", fmt(like.logp_const)),
            ("upstream_loglike", fmt(loglike)),
        ),
    )

    # --- 3. ordered spectrum metadata ------------------------------------
    metadata_rows = []
    for index, item in enumerate(like.spec_meta, start=1):
        reversed_cross = bool(item["hasYX_xsp"])
        metadata_rows.append((
            index,
            item["pol"],
            item["t2"] if reversed_cross else item["t1"],
            item["t1"] if reversed_cross else item["t2"],
            item["t1"],
            item["t2"],
            str(reversed_cross),
            int(len(item["ids"])),
            int(item["ids"][0]) + 1,
            int(item["ids"][-1]) + 1,
            int(item["bpw"].values[0]),
            int(item["bpw"].values[-1]),
        ))
    write_table(
        output / "spectrum_metadata.tsv",
        ("index", "polarization", "temperature_leg", "polarization_leg",
         "source_t1", "source_t2", "reversed_cross_spectrum", "number_of_bins",
         "first_output_index", "last_output_index", "window_ell_min", "window_ell_max"),
        metadata_rows,
    )

    # --- 4. foreground totals --------------------------------------------
    total_arrays = dict(zip(foreground.requested_cls, totals))
    checkpoints = []
    moments = []
    for polarization, array in total_arrays.items():
        checkpoints.extend(checkpoint_rows(array, ells, (polarization,)))
        moments.extend(moment_rows(array, ells, (polarization,)))
    write_table(output / "foreground_totals_checkpoints.tsv",
                ("polarization", "i", "j", "ell", "value"), checkpoints)
    write_table(output / "foreground_totals_moments.tsv",
                ("polarization", "i", "j", "sum", "abs_sum", "ell_weighted_sum"), moments)

    # --- 5. individual components ----------------------------------------
    component_arrays = {key: np.asarray(components[key]) for key in COMPONENT_KEYS}
    # tSZ x CIB is only reachable as a difference of the correlated block.
    component_arrays[("tt", "szxcib")] = (
        component_arrays[("tt", "tSZ_and_CIB")]
        - component_arrays[("tt", "tSZ")]
        - component_arrays[("tt", "cibc")]
    )

    checkpoints = []
    moments = []
    for (polarization, component) in sorted(component_arrays):
        array = component_arrays[(polarization, component)]
        checkpoints.extend(checkpoint_rows(array, ells, (component, polarization)))
        moments.extend(moment_rows(array, ells, (component, polarization)))
    write_table(output / "foreground_components_checkpoints.tsv",
                ("component", "polarization", "i", "j", "ell", "value"), checkpoints)
    write_table(output / "foreground_components_moments.tsv",
                ("component", "polarization", "i", "j", "sum", "abs_sum", "ell_weighted_sum"),
                moments)

    # --- 6. bandpass and chromatic-beam evaluations -----------------------
    export_bandpasses(output, foreground, ells)
    export_seds(output, foreground, ells, parameters)

    # --- 7. provenance -----------------------------------------------------
    provenance = {
        "sacc_file": args.sacc_file.name,
        "sacc_sha256": sha256(args.sacc_file),
        "number_of_bandpowers": int(like.data_vec.size),
        "checkpoint_ells": list(CHECKPOINT_ELLS),
        "checkpoint_nu_indices": list(CHECKPOINT_NU_INDICES),
        "foreground_parameters": FOREGROUNDS,
        "systematics_parameters": SYSTEMATICS,
        "source_revisions": {
            name: git_revision(path)
            for name, path in (
                ("act_dr6_mflike", args.act_source),
                ("LAT_MFLike", args.mflike_source),
                ("fgspectra", args.fgspectra_source),
            )
            if path is not None
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def export_bandpasses(output: Path, foreground, ells) -> None:
    """Normalized transmissions and beam-weighted chromatic responses.

    ``bandpass_checkpoints.tsv`` holds the achromatic normalized transmission
    at the reference bandpass shift, i.e. what ``make_band`` produces in Julia.
    ``chromatic_beam_checkpoints.tsv`` holds the beam-weighted normalized
    transmission at selected multipoles, which is the quantity the chromatic
    band integral actually contracts the SED against.
    """
    from mflike.foreground import _cmb2bb

    index_of = {int(ell): index for index, ell in enumerate(ells)}
    bandpass_rows = []
    chromatic_rows = []
    for channel_index, experiment in enumerate(foreground.experiments, start=1):
        for field in ("s0", "s2"):
            band = foreground.bands[f"{experiment}_{field}"]
            nu = np.asarray(band["nu"], dtype=float)
            transmission = np.asarray(band["bandpass"], dtype=float)
            shift = float(SYSTEMATICS[f"bandint_shift_{experiment}"])
            shifted = nu + shift
            weight = transmission * _cmb2bb(shifted)
            normalized = weight / np.trapezoid(weight, shifted)

            beam_t, beam_p = foreground.return_beams(experiment, nu, 0.0)
            beam = beam_t if field == "s0" else beam_p

            for nu_index in CHECKPOINT_NU_INDICES:
                bandpass_rows.append((
                    experiment, field, channel_index, nu_index + 1,
                    fmt(nu[nu_index]), fmt(shifted[nu_index]),
                    fmt(transmission[nu_index]), fmt(normalized[nu_index]),
                ))
                for ell in CHECKPOINT_ELLS:
                    ell_index = index_of[ell]
                    profile = normalized * beam[:, ell_index]
                    response = profile[nu_index] / np.trapezoid(profile, shifted)
                    chromatic_rows.append((
                        experiment, field, channel_index, nu_index + 1, ell,
                        fmt(beam[nu_index, ell_index]), fmt(response),
                    ))

    write_table(
        output / "bandpass_checkpoints.tsv",
        ("channel", "field", "channel_index", "nu_index", "nu", "nu_shifted",
         "transmission", "normalized_transmission"),
        bandpass_rows,
    )
    write_table(
        output / "chromatic_beam_checkpoints.tsv",
        ("channel", "field", "channel_index", "nu_index", "ell", "beam",
         "chromatic_response"),
        chromatic_rows,
    )


def export_seds(output: Path, foreground, ells, parameters) -> None:
    """Chromatic SED weights f_ell per channel, straight from ``fgspectra``.

    These are the arrays the Julia layer calls ``f_tsz_T``, ``f_cibp_T`` and so
    on. Checking them separately localizes an SED error to the frequency
    response rather than the angular shape.
    """
    from fgspectra import frequency as fgf

    # Make the bandpass state explicit rather than relying on the last caller.
    foreground._bandpass_construction(**parameters)

    index_of = {int(ell): index for index, ell in enumerate(ells)}
    nu_0 = foreground.fg_nu_0
    definitions = (
        ("ksz", "T", fgf.ConstantSED(), {"nu": foreground.bandint_freqs_T}),
        ("tsz", "T", fgf.ThermalSZ(), {"nu": foreground.bandint_freqs_T, "nu_0": nu_0}),
        ("cibp", "T", fgf.ModifiedBlackBody(),
         {"nu": foreground.bandint_freqs_T, "nu_0": nu_0,
          "temp": parameters["T_d"], "beta": parameters["beta_p"]}),
        ("cibc", "T", fgf.CIB(),
         {"nu": foreground.bandint_freqs_T, "nu_0": nu_0,
          "temp": parameters["T_d"], "beta": parameters["beta_c"]}),
        ("dust", "T", fgf.ModifiedBlackBody(),
         {"nu": foreground.bandint_freqs_T, "nu_0": nu_0,
          "temp": parameters["T_effd"], "beta": parameters["beta_d"]}),
        ("dust", "P", fgf.ModifiedBlackBody(),
         {"nu": foreground.bandint_freqs_P, "nu_0": nu_0,
          "temp": parameters["T_effd"], "beta": parameters["beta_d"]}),
        ("radio", "T", fgf.PowerLaw(),
         {"nu": foreground.bandint_freqs_T, "nu_0": nu_0, "beta": parameters["beta_s"]}),
        ("radio", "P", fgf.PowerLaw(),
         {"nu": foreground.bandint_freqs_P, "nu_0": nu_0, "beta": parameters["beta_s"]}),
    )

    rows = []
    for name, field, sed, kwargs in definitions:
        # fgspectra returns the chromatic weights as (n_ell, n_channel).
        weights = np.atleast_2d(np.asarray(sed(**kwargs), dtype=float))
        n_ell, n_channels = weights.shape
        if n_channels != len(foreground.experiments):
            raise SystemExit(
                f"unexpected SED shape {weights.shape} for {name}/{field}; "
                f"expected (n_ell, {len(foreground.experiments)})"
            )
        for channel_index in range(n_channels):
            for ell in CHECKPOINT_ELLS:
                row = index_of[ell] if n_ell > 1 else 0
                rows.append((name, field, channel_index + 1, ell,
                             fmt(weights[row, channel_index])))
    write_table(output / "sed_checkpoints.tsv",
                ("sed", "field", "channel_index", "ell", "value"), rows)


def compare(reference: Path, produced: Path) -> int:
    """Report any fixture that a fresh run failed to reproduce byte-for-byte."""
    expected = sorted(path.relative_to(produced) for path in produced.rglob("*")
                      if path.is_file())
    failures = 0
    for relative in expected:
        left = reference / relative
        right = produced / relative
        if not left.is_file():
            print(f"MISSING  {relative}")
            failures += 1
            continue
        if filecmp.cmp(left, right, shallow=False):
            print(f"ok       {relative}")
            continue
        failures += 1
        print(f"DIFFERS  {relative}")
        diff = difflib.unified_diff(
            left.read_text().splitlines(), right.read_text().splitlines(),
            fromfile=f"checked-in/{relative}", tofile=f"regenerated/{relative}", lineterm="",
        )
        for line in list(diff)[:12]:
            print("    " + line)
    stale = sorted(path.relative_to(reference) for path in reference.rglob("*")
                   if path.is_file() and not (produced / path.relative_to(reference)).is_file())
    for relative in stale:
        print(f"STALE    {relative}")
        failures += 1
    return failures


def main() -> None:
    args = parse_args()
    stack: list = []
    try:
        if args.check:
            with tempfile.TemporaryDirectory(prefix="act_dr6_check_") as scratch:
                produced = Path(scratch) / "fixtures"
                produced.mkdir()
                export(produced, args, stack)
                failures = compare(args.output, produced)
            if failures:
                raise SystemExit(f"{failures} fixture(s) did not reproduce")
            print("all fixtures reproduce byte-for-byte")
            return
        args.output.mkdir(parents=True, exist_ok=True)
        export(args.output, args, stack)
        print(f"wrote deterministic text fixtures to {args.output}")
    finally:
        for entry in stack:
            entry.cleanup()


if __name__ == "__main__":
    main()
