#!/usr/bin/env python3
"""Shared configuration for the ACT DR6 full-likelihood reference exporters.

Every value here is taken from the official ACT DR6 configuration shipped with
``act_dr6_mflike`` and ``LAT_MFLike``; nothing is invented. The exporters build
the likelihood from the *unmodified* ``act_dr6.yaml`` defaults.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reference nuisance point (identical to ACT_DR6_REFERENCE_NUISANCE in Julia)
# ---------------------------------------------------------------------------

FOREGROUNDS = {
    "a_tSZ": 3.50114277,
    "alpha_tSZ": -0.4597721879,
    "a_kSZ": 0.986604682,
    "a_p": 7.647742104,
    "beta_p": 1.86490755,
    "a_c": 3.805341822,
    "beta_c": 1.86490755,
    "a_s": 2.886594272,
    "beta_s": -2.7567784,
    "a_gtt": 7.974213801,
    "a_gte": 0.4184588365,
    "a_gee": 0.1676466062,
    "a_psee": 0.003755819497,
    "a_pste": -0.02500092711,
    "xi": 0.06424293336,
    # fixed model constants
    "alpha_s": 1.0,
    "T_effd": 19.6,
    "beta_d": 1.5,
    "alpha_dT": -0.6,
    "alpha_dE": -0.4,
    "alpha_p": 1.0,
    "T_d": 9.60,
}

SYSTEMATICS = {
    "calG_all": 1.001567048,
    "cal_dr6_pa4_f220": 0.9808084654,
    "cal_dr6_pa5_f090": 1.000098497,
    "cal_dr6_pa5_f150": 0.9991342522,
    "cal_dr6_pa6_f090": 0.9998031382,
    "cal_dr6_pa6_f150": 1.001407626,
    "calE_dr6_pa4_f220": 1.0,
    "calE_dr6_pa5_f090": 0.9874026803,
    "calE_dr6_pa5_f150": 0.9975776488,
    "calE_dr6_pa6_f090": 0.9975750142,
    "calE_dr6_pa6_f150": 0.9968551529,
    "bandint_shift_dr6_pa4_f220": 6.399328024,
    "bandint_shift_dr6_pa5_f090": -0.2911716302,
    "bandint_shift_dr6_pa5_f150": -1.056426408,
    "bandint_shift_dr6_pa6_f090": 0.3121747872,
    "bandint_shift_dr6_pa6_f150": -0.4252785128,
}

# Checkpoint grid. Fixed, deterministic, and chosen to span the full multipole
# range including both endpoints, so that an off-by-one or reversed ell axis
# cannot pass.
CHECKPOINT_ELLS = (2, 3, 101, 503, 1009, 2003, 3001, 4001, 5003, 6007, 8501)

# Frequency-grid samples used for the bandpass/beam checkpoints.
CHECKPOINT_NU_INDICES = (0, 7, 19, 31, 49)

FLOAT_FORMAT = "%.18e"


def fmt(value) -> str:
    """Deterministic float formatting used by every fixture column."""
    return FLOAT_FORMAT % float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_likelihood(sacc_file: Path, stack: list):
    """Instantiate ``ACTDR6MFLike`` from the unmodified official configuration.

    ``act_dr6_mflike`` resolves its data through the Cobaya packages path, so a
    throwaway tree containing a symlink to ``sacc_file`` is created and
    ``COBAYA_PACKAGES_PATH`` is pointed at it. No defaults are overridden: the
    scale cuts, polarizations and ``symmetrize: false`` all come from
    ``act_dr6_mflike/act_dr6.yaml``.
    """
    sacc_file = Path(sacc_file).resolve()
    if not sacc_file.is_file():
        raise SystemExit(f"ACT DR6 SACC release not found: {sacc_file}")

    staging = tempfile.TemporaryDirectory(prefix="act_dr6_packages_")
    stack.append(staging)
    target = Path(staging.name) / "data" / "ACTDR6MFLike" / "v1.0"
    target.mkdir(parents=True)
    (target / "dr6_data.fits").symlink_to(sacc_file)
    os.environ["COBAYA_PACKAGES_PATH"] = staging.name

    from act_dr6_mflike import ACTDR6MFLike

    return ACTDR6MFLike({"data_folder": "ACTDR6MFLike/v1.0", "input_file": "dr6_data.fits"})


def build_foreground(like):
    """Chromatic-beam ``BandpowerForeground`` exactly as ACT DR6 configures it."""
    from mflike import BandpowerForeground

    foreground = BandpowerForeground(
        like.get_fg_requirements() | {"beam_profile": {"beam_from_file": None}}
    )
    foreground.init_bandpowers()
    return foreground


def load_cmb(directory: Path) -> dict:
    """Load the frozen CAMB reference spectra (see ``scripts/dump_theory.py``)."""
    directory = Path(directory)
    return {
        polarization: np.loadtxt(directory / f"cmb_theory_{polarization}.txt")
        for polarization in ("tt", "te", "ee")
    }


def write_table(path: Path, header, rows) -> None:
    """Write a deterministic tab-separated table with a trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as stream:
        stream.write("\t".join(header) + "\n")
        for row in rows:
            stream.write("\t".join(str(field) for field in row) + "\n")


def write_vector(path: Path, values) -> None:
    """Write one value per line in the shared deterministic float format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as stream:
        for value in np.asarray(values).ravel():
            stream.write(fmt(value) + "\n")


def checkpoint_rows(array, ells, label_fields):
    """Rows `(… , i, j, ell, value)` over every channel pair and checkpoint ell.

    Covering all 25 ordered channel pairs is what makes a swapped map leg, a
    transposed TE block or a permuted channel order detectable.
    """
    index_of = {int(ell): index for index, ell in enumerate(ells)}
    n_channels = array.shape[0]
    for i in range(n_channels):
        for j in range(n_channels):
            for ell in CHECKPOINT_ELLS:
                yield (*label_fields, i + 1, j + 1, ell, fmt(array[i, j, index_of[ell]]))


def moment_rows(array, ells, label_fields):
    """Whole-array reductions per channel pair.

    ``sum`` and ``abs_sum`` catch amplitude errors anywhere in the 8500-element
    multipole axis; ``ell_weighted_sum`` additionally catches a reversed or
    rolled ell axis, which a symmetric reduction alone would miss.
    """
    weights = np.asarray(ells, dtype=float)
    n_channels = array.shape[0]
    for i in range(n_channels):
        for j in range(n_channels):
            spectrum = np.asarray(array[i, j, :], dtype=float)
            yield (
                *label_fields,
                i + 1,
                j + 1,
                fmt(spectrum.sum()),
                fmt(np.abs(spectrum).sum()),
                fmt(spectrum @ weights),
            )
