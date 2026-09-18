#!/usr/bin/env python3
"""Export several ACT DR6 nuisance points from the original code, as text.

Baseline-only agreement is not sufficient evidence: a swapped parameter or a
dead branch can reproduce a single point by accident. These three points move
the foregrounds, the systematics, and both together.

    python3 validation/export_reference_act_multipoint.py \
        --sacc-file data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits \
        --cmb-directory validation/fixtures/act_dr6_cmb_theory \
        --output validation/fixtures/act_dr6_full_multipoint

Add ``--check`` to verify the checked-in fixtures reproduce byte-for-byte.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act_dr6_reference_common import (  # noqa: E402
    FOREGROUNDS,
    SYSTEMATICS,
    build_foreground,
    build_likelihood,
    checkpoint_rows,
    fmt,
    load_cmb,
    write_table,
    write_vector,
)
from export_reference_act import compare  # noqa: E402

CASES = {
    "foregrounds": {
        "a_tSZ": 4.25,
        "alpha_tSZ": -0.15,
        "a_kSZ": 1.70,
        "a_p": 9.20,
        "beta_p": 2.05,
        "beta_c": 2.05,
        "a_c": 2.40,
        "a_s": 3.70,
        "beta_s": -2.45,
        "a_gtt": 8.25,
        "a_gte": 0.36,
        "a_gee": 0.20,
        "a_psee": 0.08,
        "a_pste": 0.12,
        "xi": 0.11,
    },
    "systematics": {
        "calG_all": 0.998,
        "cal_dr6_pa4_f220": 1.012,
        "cal_dr6_pa5_f090": 0.994,
        "cal_dr6_pa5_f150": 1.006,
        "cal_dr6_pa6_f090": 1.009,
        "cal_dr6_pa6_f150": 0.996,
        "calE_dr6_pa4_f220": 1.015,
        "calE_dr6_pa5_f090": 0.975,
        "calE_dr6_pa5_f150": 1.008,
        "calE_dr6_pa6_f090": 0.985,
        "calE_dr6_pa6_f150": 1.012,
        "bandint_shift_dr6_pa4_f220": 2.4,
        "bandint_shift_dr6_pa5_f090": -1.3,
        "bandint_shift_dr6_pa5_f150": 0.8,
        "bandint_shift_dr6_pa6_f090": 1.1,
        "bandint_shift_dr6_pa6_f150": -1.7,
    },
    "combined": {
        "a_tSZ": 2.1,
        "alpha_tSZ": -0.8,
        "a_kSZ": 2.2,
        "a_p": 5.4,
        "beta_p": 1.55,
        "beta_c": 1.55,
        "a_c": 5.1,
        "a_s": 1.8,
        "beta_s": -3.05,
        "a_gtt": 7.6,
        "a_gte": 0.47,
        "a_gee": 0.145,
        "a_psee": 0.22,
        "a_pste": -0.18,
        "xi": 0.035,
        "calG_all": 1.004,
        "cal_dr6_pa4_f220": 0.991,
        "cal_dr6_pa5_f090": 1.007,
        "cal_dr6_pa5_f150": 0.995,
        "cal_dr6_pa6_f090": 1.003,
        "cal_dr6_pa6_f150": 1.011,
        "calE_dr6_pa4_f220": 0.982,
        "calE_dr6_pa5_f090": 1.014,
        "calE_dr6_pa5_f150": 0.979,
        "calE_dr6_pa6_f090": 1.006,
        "calE_dr6_pa6_f150": 0.988,
        "bandint_shift_dr6_pa4_f220": -2.0,
        "bandint_shift_dr6_pa5_f090": 1.6,
        "bandint_shift_dr6_pa5_f150": -0.4,
        "bandint_shift_dr6_pa6_f090": -1.2,
        "bandint_shift_dr6_pa6_f150": 2.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sacc-file", type=Path, required=True)
    parser.add_argument("--cmb-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def export(output: Path, args: argparse.Namespace, stack: list) -> None:
    like = build_likelihood(args.sacc_file, stack)
    foreground = build_foreground(like)
    ells = np.asarray(foreground.ells)
    cmb = load_cmb(args.cmb_directory)

    case_rows = []
    checkpoints = []
    for name, updates in CASES.items():
        parameters = FOREGROUNDS | SYSTEMATICS | updates
        totals = foreground.get_foreground_model_totals(**parameters)
        model = like._get_power_spectra(cmb, totals, **parameters)
        residual = like.data_vec - model
        chi2 = float(like._fast_chi_squared(like.inv_cov, residual))

        write_vector(output / f"model_vector_{name}.txt", model)
        case_rows.append((name, fmt(chi2), fmt(-0.5 * chi2)))

        for polarization, array in zip(foreground.requested_cls, totals):
            checkpoints.extend(checkpoint_rows(array, ells, (name, polarization)))

    write_table(output / "cases.tsv",
                ("name", "chi2", "data_only_loglikelihood"), case_rows)
    write_table(output / "foreground_totals_checkpoints.tsv",
                ("case", "polarization", "i", "j", "ell", "value"), checkpoints)

    parameter_rows = []
    for name, updates in CASES.items():
        for key in sorted(updates):
            parameter_rows.append((name, key, fmt(updates[key])))
    write_table(output / "case_parameters.tsv", ("case", "parameter", "value"),
                parameter_rows)


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
            print("all multipoint fixtures reproduce byte-for-byte")
            return
        args.output.mkdir(parents=True, exist_ok=True)
        export(args.output, args, stack)
        print(f"wrote deterministic text fixtures to {args.output}")
    finally:
        for entry in stack:
            entry.cleanup()


if __name__ == "__main__":
    main()
