"""
reference_chi2.py
-----------------
Compute the exact reference loglike and chi² from the extracted text data,
using the same CMB theory that was dumped by dump_theory.py.

Mirrors exactly what our Julia ACTLikelihood.jl does, so the outputs
can be used as numerical targets for the end-to-end test.

Run:
    python scripts/reference_chi2.py
"""

import os
import sys
import numpy as np

FILTERED_DIR = (
    "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/"
    "ACTLikelihood.jl/data/ACT_DR6_TTTEEE_filtered"
)
BANDPASS_DIR = (
    "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/"
    "ACTLikelihood.jl/data/ACT_DR6_TTTEEE/bandpasses"
)
FGSPECTRA_DATA = (
    "/home/marcobonici/Desktop/work/actgame/fgspectra/fgspectra/data"
)

# ------------------------------------------------------------------ #
# 1. Load data                                                          #
# ------------------------------------------------------------------ #

data_vec = np.loadtxt(os.path.join(FILTERED_DIR, "data_vec.txt"))
inv_cov  = np.loadtxt(os.path.join(FILTERED_DIR, "inv_cov.txt"))
cov      = np.loadtxt(os.path.join(FILTERED_DIR, "cov.txt"))

N = len(data_vec)
sign, logdet = np.linalg.slogdet(cov)
logp_const = -0.5 * N * np.log(2 * np.pi) - 0.5 * logdet

print(f"n_bins     = {N}")
print(f"logp_const = {logp_const:.10f}")

# ------------------------------------------------------------------ #
# 2. Parse spec_meta and load windows                                   #
# ------------------------------------------------------------------ #

sm_raw = []
with open(os.path.join(FILTERED_DIR, "spec_meta.txt")) as f:
    for line in f:
        parts = line.strip().split()
        sm_raw.append((parts[0], int(parts[1])))

windows  = []
spec_ids = []
spec_pol = []
spec_t1  = []
spec_t2  = []
spec_hasYX = []

seen  = {}
id_offset = 0

for name, n_bins in sm_raw:
    # parse "dr6_pa4_f220_x_dr6_pa5_f090_TT"
    left, right = name.split("_x_", 1)
    t1 = left
    pol = right.rsplit("_", 1)[1].lower()
    t2  = right.rsplit("_", 1)[0]

    key   = (t1, t2, pol)
    count = seen.get(key, 0) + 1
    seen[key] = count
    hasYX = (pol == "te") and (count == 2)

    W = np.loadtxt(os.path.join(FILTERED_DIR, "windows", name + ".txt"))  # (n_ell, n_bins)

    ids = np.arange(id_offset, id_offset + n_bins)
    id_offset += n_bins

    windows.append(W)
    spec_ids.append(ids)
    spec_pol.append(pol)
    spec_t1.append(t1)
    spec_t2.append(t2)
    spec_hasYX.append(hasYX)

n_ell = windows[0].shape[0]   # 8500
ell   = np.arange(2, 2 + n_ell)   # 2, 3, ..., 8501

print(f"n_spectra  = {len(sm_raw)}")
print(f"n_ell      = {n_ell}  (ell {ell[0]}..{ell[-1]})")

# ------------------------------------------------------------------ #
# 3. Load CMB theory (CAMB, ell=0..9050)                               #
# ------------------------------------------------------------------ #

tt_all = np.loadtxt(os.path.join(FILTERED_DIR, "cmb_theory_tt.txt"))
te_all = np.loadtxt(os.path.join(FILTERED_DIR, "cmb_theory_te.txt"))
ee_all = np.loadtxt(os.path.join(FILTERED_DIR, "cmb_theory_ee.txt"))

# Slice to ell=2..8501 (0-indexed rows 2..8501)
sl = slice(2, 2 + n_ell)
cmb = {
    "tt": tt_all[sl],
    "te": te_all[sl],
    "ee": ee_all[sl],
}

# ------------------------------------------------------------------ #
# 4. Bandpass loading and SED integration                               #
# ------------------------------------------------------------------ #

T_CMB = 2.72548
H_OVER_KT = 6.62607015e-34 * 1e9 / (1.380649e-23 * T_CMB)   # 1/GHz

def x_cmb(nu):
    return H_OVER_KT * nu

def cmb2bb(nu):
    x = x_cmb(nu)
    return np.exp(x) * (nu * x / np.expm1(x))**2

def rj2cmb(nu):
    x = x_cmb(nu)
    return (np.expm1(x) / x)**2 / np.exp(x)

def tsz_f(nu):
    x = x_cmb(nu)
    return x / np.tanh(x / 2) - 4

def mbb(nu, nu0, beta, temp):
    H_K = 6.62607015e-34 * 1e9 / 1.380649e-23
    x   = H_K * nu  / temp
    x0  = H_K * nu0 / temp
    return (nu/nu0)**(beta+1) * np.expm1(x0)/np.expm1(x) * rj2cmb(nu)/rj2cmb(nu0)

def radio(nu, nu0, beta):
    return (nu/nu0)**beta * rj2cmb(nu)/rj2cmb(nu0)

def trapz(x, y):
    return np.trapz(y, x)

def make_band(nu, bp):
    w = bp * cmb2bb(nu)
    return w / trapz(nu, w)

def integrate_sed(sed_vals, nu, norm_bp):
    return trapz(nu, sed_vals * norm_bp)

# Load all bandpasses
experiments = ["dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",
               "dr6_pa6_f090", "dr6_pa6_f150"]

bands = {}   # key: "exp_s0" or "exp_s2"
for exp in experiments:
    for spin in ("s0", "s2"):
        path = os.path.join(BANDPASS_DIR, f"{exp}_{spin}.txt")
        bp_data = np.loadtxt(path)
        nu = bp_data[:, 0]
        bp = bp_data[:, 1]
        norm = make_band(nu, bp)
        bands[f"{exp}_{spin}"] = (nu, norm)

nu_0 = 150.0

def band_integrate(exp, spin, sed_fn):
    nu, norm = bands[f"{exp}_{spin}"]
    return integrate_sed(sed_fn(nu), nu, norm)

# ------------------------------------------------------------------ #
# 5. Foreground templates                                               #
# ------------------------------------------------------------------ #

def load_template(fname):
    raw = np.loadtxt(fname)
    ells = raw[:, 0].astype(int)
    dls  = raw[:, 1]
    arr  = np.zeros(ells.max() + 2)
    arr[ells] = dls
    return arr

T_tsz   = load_template(os.path.join(FGSPECTRA_DATA, "cl_tsz_150_bat.dat"))
T_ksz   = load_template(os.path.join(FGSPECTRA_DATA, "cl_ksz_bat.dat"))
T_cibc  = load_template(os.path.join(FGSPECTRA_DATA, "cl_cib_Choi2020.dat"))
T_szxcib = load_template(os.path.join(FGSPECTRA_DATA, "cl_sz_x_cib.dat"))

# ------------------------------------------------------------------ #
# 6. Foreground model                                                   #
# ------------------------------------------------------------------ #

def compute_fg_totals(p):
    """
    Returns fg_TT, fg_TE, fg_EE each (n_exp, n_exp, n_ell).
    Mirrors our Julia compute_fg_totals exactly.
    """
    ell0  = 3000
    T_d   = p.get("T_d",   9.6)
    T_eff = p.get("T_effd", 19.6)
    beta_d = p.get("beta_d", 1.5)
    alpha_dT = p.get("alpha_dT", -0.6)
    alpha_dE = p.get("alpha_dE", -0.4)
    alpha_p  = p.get("alpha_p",  1.0)
    alpha_s  = p.get("alpha_s",  1.0)

    n_exp = len(experiments)

    # SED integrals
    f_ksz  = [band_integrate(e, "s0", lambda nu: np.ones_like(nu)) for e in experiments]
    f_tsz  = [band_integrate(e, "s0", lambda nu: tsz_f(nu) / tsz_f(nu_0)) for e in experiments]
    f_cibp = [band_integrate(e, "s0", lambda nu: mbb(nu, nu_0, p["beta_p"], T_d)) for e in experiments]
    f_cibc = [band_integrate(e, "s0", lambda nu: mbb(nu, nu_0, p["beta_c"], T_d)) for e in experiments]
    f_dust_T = [band_integrate(e, "s0", lambda nu: mbb(nu, nu_0, beta_d, T_eff)) for e in experiments]
    f_radio_T = [band_integrate(e, "s0", lambda nu: radio(nu, nu_0, p["beta_s"])) for e in experiments]
    f_dust_P = [band_integrate(e, "s2", lambda nu: mbb(nu, nu_0, beta_d, T_eff)) for e in experiments]
    f_radio_P = [band_integrate(e, "s2", lambda nu: radio(nu, nu_0, p["beta_s"])) for e in experiments]

    f_ksz  = np.array(f_ksz)
    f_tsz  = np.array(f_tsz)
    f_cibp = np.array(f_cibp)
    f_cibc = np.array(f_cibc)
    f_dust_T = np.array(f_dust_T)
    f_radio_T = np.array(f_radio_T)
    f_dust_P = np.array(f_dust_P)
    f_radio_P = np.array(f_radio_P)

    # Templates (already amplitude-scaled)
    cl_ksz  = T_ksz[ell]  / T_ksz[ell0]
    cl_tsz  = p["a_tSZ"] * T_tsz[ell] / T_tsz[ell0] * (ell / ell0)**p["alpha_tSZ"]
    cl_cibc = p["a_c"]   * T_cibc[ell] / T_cibc[ell0]
    cl_szx  = -p["xi"] * np.sqrt(p["a_tSZ"] * p["a_c"]) * T_szxcib[ell] / T_szxcib[ell0]
    ell_clp  = ell * (ell + 1)
    ell0_clp = ell0 * (ell0 + 1)
    cl_cibp = (ell_clp / ell0_clp)**alpha_p
    cl_radio = (ell_clp / ell0_clp)**alpha_s
    cl_dustT = (ell / 500.0)**alpha_dT
    cl_dustE = (ell / 500.0)**alpha_dE

    def factorized(f, cl):
        outer = f[:, None] * f[None, :]    # (n, n)
        return outer[:, :, None] * cl[None, None, :]   # (n, n, nell)

    def factorized_te(fT, fE, cl):
        outer = fT[:, None] * fE[None, :]
        return outer[:, :, None] * cl[None, None, :]

    def correlated(fk, fn, cl_kn):
        outer = fk[:, None] * fn[None, :]
        return outer[:, :, None] * cl_kn[None, None, :]

    # tSZ+CIBc+cross (combined)
    fg_TT  = p["a_kSZ"] * factorized(f_ksz,  cl_ksz)
    fg_TT += correlated(f_tsz,  f_tsz,  cl_tsz)
    fg_TT += correlated(f_tsz,  f_cibc, cl_szx)
    fg_TT += correlated(f_cibc, f_tsz,  cl_szx)
    fg_TT += correlated(f_cibc, f_cibc, cl_cibc)
    fg_TT += p["a_p"]   * factorized(f_cibp,  cl_cibp)
    fg_TT += p["a_gtt"] * factorized(f_dust_T, cl_dustT)
    fg_TT += p["a_s"]   * factorized(f_radio_T, cl_radio)

    fg_EE  = p["a_psee"] * factorized(f_radio_P, cl_radio)
    fg_EE += p["a_gee"]  * factorized(f_dust_P,  cl_dustE)

    fg_TE  = p["a_pste"] * factorized_te(f_radio_T, f_radio_P, cl_radio)
    fg_TE += p["a_gte"]  * factorized_te(f_dust_T,  f_dust_P,  cl_dustE)

    return fg_TT, fg_TE, fg_EE

# ------------------------------------------------------------------ #
# 7. Theory vector and chi²                                             #
# ------------------------------------------------------------------ #

def theory_vector(cmb_dls, fg_TT, fg_TE, fg_EE, cal):
    exp_idx = {e: i for i, e in enumerate(experiments)}
    calG = 1.0 / cal.get("calG_all", 1.0)

    def ct(e):
        return calG / cal.get(f"cal_{e}", 1.0)

    def ce(e):
        return calG / cal.get(f"cal_{e}", 1.0) / cal.get(f"calE_{e}", 1.0)

    ps_vec = np.zeros(N)
    for k, (name, n_bins) in enumerate(sm_raw):
        W   = windows[k]          # (n_ell, n_bins)
        ids = spec_ids[k]
        pol = spec_pol[k]
        t1  = spec_t1[k]
        t2  = spec_t2[k]
        hasYX = spec_hasYX[k]
        i = exp_idx[t1]
        j = exp_idx[t2]

        if hasYX:
            i, j = j, i   # ET: swap to look up fg[j,i,:]

        if pol == "tt":
            dl = cmb["tt"] + fg_TT[i, j, :]
            cf = ct(t1) * ct(t2)
        elif pol == "te":
            dl = cmb["te"] + fg_TE[i, j, :]
            cf = ct(t1) * ce(t2) if not hasYX else ce(t1) * ct(t2)
        else:
            dl = cmb["ee"] + fg_EE[i, j, :]
            cf = ce(t1) * ce(t2)

        ps_vec[ids] = W.T @ (dl * cf)

    return ps_vec

# ------------------------------------------------------------------ #
# 8. Evaluate at best-fit + default systematics                         #
# ------------------------------------------------------------------ #

fg_params = dict(
    a_tSZ    = 3.35,  alpha_tSZ = -0.53,
    a_kSZ    = 1.48,
    a_p      = 6.91,  beta_p = 2.07,  T_d  = 9.6,
    a_c      = 4.88,  beta_c = 2.20,
    xi       = 0.12,
    a_s      = 3.09,  beta_s = -2.76,
    a_gtt    = 8.83,
    a_gte    = 0.42,
    a_gee    = 0.168,
    a_pste   = -0.023,
    a_psee   = 0.040,
)

# No calibration corrections (all 1.0)
cal_params = dict(calG_all=1.0)

fg_TT, fg_TE, fg_EE = compute_fg_totals(fg_params)
ps_vec = theory_vector(cmb, fg_TT, fg_TE, fg_EE, cal_params)

delta  = data_vec - ps_vec
chi2   = delta @ inv_cov @ delta
ll     = -0.5 * chi2 + logp_const

print()
print("=== REFERENCE VALUES (Python) ===")
print(f"loglike   = {ll:.10f}")
print(f"chi2      = {chi2:.10f}")
print(f"chi2/dof  = {chi2/N:.10f}")
