"""
    ACT_foreground_model.jl

Foreground placeholder for the ACT DR6 CMB-only likelihood.

The full multifrequency foreground model is implemented separately in
`ACT_full_foregrounds.jl`, where it is validated against the original ACT code.
"""

# ---------------------------------------------------------------------------
# CMB-only foreground struct (no foreground components; calibration handled
# in the main likelihood; keep as a placeholder for the API)
# ---------------------------------------------------------------------------

"""
    ACTCMBOnlyFG

Empty foreground model for the ACT DR6 CMB-only likelihood variant.
No foreground terms are added; only overall and polarization calibration are applied.

### Nuisance parameters
| Symbol  | Meaning                              | Prior (candl default)  |
|---------|--------------------------------------|------------------------|
| `A_act` | Overall temperature calibration      | N(1.0, 0.003²)        |
| `P_act` | Polarization calibration             | N(1.0, 0.1²) [loose]  |
"""
struct ACTCMBOnlyFG end

"""
    compute_foreground_Dls(::ACTCMBOnlyFG, params, ells, spec_order, N_ell) -> zeros

Returns a zero vector (no foreground terms in CMB-only variant).
"""
function compute_foreground_Dls(::ACTCMBOnlyFG, params, ells::AbstractVector,
                                 spec_order::Vector{String}, N_ell::Int)
    T = typeof(get(params, :A_act, 1.0))
    return zeros(T, length(spec_order) * N_ell)
end


# The multi-frequency foreground model is implemented in ACT_mf_foreground.jl
