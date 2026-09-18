# Frozen CMB reference spectra

`cmb_theory_{tt,te,ee}.txt` are lensed `D_ell` in μK², one value per line,
starting at `ell = 0` (9051 lines, `ell = 0:9050`). The first two entries are
zero.

They were produced once by `scripts/dump_theory.py` with CAMB at the ACT DR6
best-fit cosmology (the same parameters used by the `act_dr6_mflike` test):

    cosmomc_theta = 1.040547237e-02
    As            = 2.127445742e-09
    ombh2         = 2.261650205e-02
    omch2         = 1.240404189e-01
    ns            = 9.663813976e-01
    tau           = 5.655092745e-02

They are checked in as a **frozen input**, not as a generated fixture: CAMB is
not bit-reproducible across versions, so regenerating them would change every
downstream reference. The exporters read these files; they never rewrite them.
