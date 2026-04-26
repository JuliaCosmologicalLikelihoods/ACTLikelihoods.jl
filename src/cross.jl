"""
    cross.jl

Cross-spectrum assembly: outer-product of frequency SEDs and ℓ-templates.
Mirrors fgspectra/cross.py.

All functions return 3D arrays of shape (n_freq, n_freq, n_ell).
All are pure functions — no mutation — compatible with ForwardDiff and Mooncake.
"""

# ------------------------------------------------------------------ #
# Factorized cross-spectrum                                            #
# D[i,j,ℓ] = f[i] · f[j] · Cl[ℓ]                                     #
# ------------------------------------------------------------------ #

"""
    factorized_cross(f, cl)

Factorized cross-spectrum: outer product of SED vector with itself,
times a scalar template:

    D[i, j, ℓ] = f[i] · f[j] · Cl[ℓ]

Arguments:
- `f`:  SED values, shape (n_freq,)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross(f::AbstractVector, cl::AbstractVector)
    n_freq = length(f)
    n_ell  = length(cl)
    # outer[i,j] = f[i]*f[j], then broadcast with cl
    outer = f .* f'                                        # (n_freq, n_freq)
    return reshape(outer, n_freq, n_freq, 1) .* reshape(cl, 1, 1, n_ell)
end

# ------------------------------------------------------------------ #
# TE factorized cross-spectrum                                         #
# D[i,j,ℓ] = fT[i] · fE[j] · Cl[ℓ]                                   #
# (T and E can have different SEDs, e.g. different beam normalizations) #
# ------------------------------------------------------------------ #

"""
    factorized_cross_te(fT, fE, cl)

TE cross-spectrum: product of separate T and E SED vectors:

    D[i, j, ℓ] = fT[i] · fE[j] · Cl[ℓ]

Arguments:
- `fT`: temperature SED values, shape (n_freq,)
- `fE`: E-mode SED values, shape (n_freq,)
- `cl`: D_ℓ template, shape (n_ell,)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function factorized_cross_te(fT::AbstractVector, fE::AbstractVector, cl::AbstractVector)
    n_freq = length(fT)
    n_ell  = length(cl)
    outer  = fT .* fE'                                     # (n_freq, n_freq)
    return reshape(outer, n_freq, n_freq, 1) .* reshape(cl, 1, 1, n_ell)
end

# ------------------------------------------------------------------ #
# Correlated cross-spectrum (tSZ + CIB + cross)                        #
# D[i,j,ℓ] = Σ_{k,n} f[k,i] · f[n,j] · C[k,n,ℓ]                     #
# ------------------------------------------------------------------ #

"""
    correlated_cross(f, cl)

Correlated cross-spectrum for multiple components:

    D[i, j, ℓ] = Σ_{k,n} f[k, i] · f[n, j] · C[k, n, ℓ]

This is the ACT DR6 model for tSZ + CIB clustered + tSZ×CIB:
- k=1: ThermalSZ;  k=2: CIB (MBB)
- C[1,1]: tSZ template × a_tSZ  (with tilt)
- C[2,2]: CIB clustered template × a_c
- C[1,2] = C[2,1]: cross template × (−ξ√(a_tSZ · a_c))

Arguments:
- `f`:  SED matrix, shape (n_comp, n_freq)
- `cl`: covariance spectrum tensor, shape (n_comp, n_comp, n_ell)

Returns array of shape (n_freq, n_freq, n_ell).
Matches `CorrelatedFactorizedCrossSpectrum` in fgspectra/cross.py.
"""
function correlated_cross(f::AbstractMatrix, cl::AbstractArray{<:Any,3})
    n_comp, n_freq = size(f)
    n_ell          = size(cl, 3)
    # Pure sum — no mutation — compatible with ForwardDiff and Mooncake
    return sum(
        reshape(f[k, :] .* f[n, :]', n_freq, n_freq, 1) .*
        reshape(cl[k, n, :], 1, 1, n_ell)
        for k in 1:n_comp, n in 1:n_comp
    )
end

# ------------------------------------------------------------------ #
# Helper: build the 2×2×n_ell covariance spectrum for tSZ+CIB          #
# ------------------------------------------------------------------ #

"""
    build_szxcib_cl(cl_tsz, cl_cibc, cl_cross)

Assemble the 2×2 component-component D_ℓ covariance matrix for the
tSZ+CIB correlated model.

Layout:
    C[1,1,ℓ] = cl_tsz[ℓ]    (tSZ auto, with amplitude a_tSZ and tilt)
    C[2,2,ℓ] = cl_cibc[ℓ]   (CIB clustered auto, amplitude a_c)
    C[1,2,ℓ] = C[2,1,ℓ] = cl_cross[ℓ]  (−ξ√(a_tSZ·a_c) × template)

All inputs are Vectors of length n_ell (already amplitude-multiplied).
"""
function build_szxcib_cl(cl_tsz::AbstractVector, cl_cibc::AbstractVector,
                          cl_cross::AbstractVector)
    n_ell = length(cl_tsz)
    # Pure construction — no mutation — compatible with ForwardDiff and Mooncake
    zeros_n = zero(cl_tsz)
    layer11 = reshape(cl_tsz,  1, 1, n_ell)
    layer12 = reshape(cl_cross, 1, 1, n_ell)
    layer21 = reshape(cl_cross, 1, 1, n_ell)
    layer22 = reshape(cl_cibc,  1, 1, n_ell)
    row1 = cat(layer11, layer12; dims=2)   # (1, 2, n_ell)
    row2 = cat(layer21, layer22; dims=2)   # (1, 2, n_ell)
    return cat(row1, row2; dims=1)          # (2, 2, n_ell)
end

# ------------------------------------------------------------------ #
# Fused TT/EE/TE foreground assemblers                                #
# Each fuses the α .* factorized .+ correlated .+ ... chain into a    #
# single primitive so the Mooncake tape collapses to one entry per    #
# spectrum (replaces ~15k tape entries from per-element broadcasts).  #
# ------------------------------------------------------------------ #

"""
    assemble_TT(a_kSZ, a_p, a_gtt, a_s,
                f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                cl_ksz, cl_cibp, cl_dustT, cl_radio,
                cl_tsz, cl_cibc, cl_szxcib)

Fused TT foreground assembly — equivalent to:

    a_kSZ .* factorized_cross(f_ksz,  cl_ksz)   .+
    correlated_cross(vcat(f_tsz', f_cibc'), build_szxcib_cl(cl_tsz, cl_cibc, cl_szxcib)) .+
    a_p   .* factorized_cross(f_cibp, cl_cibp)  .+
    a_gtt .* factorized_cross(f_dust, cl_dustT) .+
    a_s   .* factorized_cross(f_radio, cl_radio)

Returns array of shape (n_freq, n_freq, n_ell).
"""
function assemble_TT(a_kSZ::Real, a_p::Real, a_gtt::Real, a_s::Real,
                     f_ksz::AbstractVector,   f_cibp::AbstractVector,
                     f_dust::AbstractVector,  f_radio::AbstractVector,
                     f_tsz::AbstractVector,   f_cibc::AbstractVector,
                     cl_ksz::AbstractVector,  cl_cibp::AbstractVector,
                     cl_dustT::AbstractVector, cl_radio::AbstractVector,
                     cl_tsz::AbstractVector,  cl_cibc::AbstractVector,
                     cl_szxcib::AbstractVector)
    n_freq = length(f_ksz)
    n_ell  = length(cl_ksz)
    T = promote_type(typeof(a_kSZ), typeof(a_p), typeof(a_gtt), typeof(a_s),
                     eltype(f_ksz), eltype(f_cibp), eltype(f_dust), eltype(f_radio),
                     eltype(f_tsz), eltype(f_cibc),
                     eltype(cl_ksz), eltype(cl_cibp), eltype(cl_dustT), eltype(cl_radio),
                     eltype(cl_tsz), eltype(cl_cibc), eltype(cl_szxcib))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        cksz, ccp, cdt, crd = cl_ksz[ℓ], cl_cibp[ℓ], cl_dustT[ℓ], cl_radio[ℓ]
        ctsz, ccc, csxc     = cl_tsz[ℓ], cl_cibc[ℓ], cl_szxcib[ℓ]
        for j in 1:n_freq
            ftj, fcj, fkj, fpj, fdj, frj =
                f_tsz[j], f_cibc[j], f_ksz[j], f_cibp[j], f_dust[j], f_radio[j]
            for i in 1:n_freq
                fti, fci, fki, fpi, fdi, fri =
                    f_tsz[i], f_cibc[i], f_ksz[i], f_cibp[i], f_dust[i], f_radio[i]
                out[i, j, ℓ] = a_kSZ * fki * fkj * cksz +
                               fti  * ftj  * ctsz +
                               (fti * fcj + fci * ftj) * csxc +
                               fci  * fcj  * ccc +
                               a_p   * fpi * fpj * ccp +
                               a_gtt * fdi * fdj * cdt +
                               a_s   * fri * frj * crd
            end
        end
    end
    return out
end

"""
    assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)

Fused EE foreground assembly — equivalent to:

    a_psee .* factorized_cross(f_radio_P, cl_radio) .+
    a_gee  .* factorized_cross(f_dust_P,  cl_dustE)
"""
function assemble_EE(a_psee::Real, a_gee::Real,
                     f_radio_P::AbstractVector, f_dust_P::AbstractVector,
                     cl_radio::AbstractVector, cl_dustE::AbstractVector)
    n_freq = length(f_radio_P)
    n_ell  = length(cl_radio)
    T = promote_type(typeof(a_psee), typeof(a_gee),
                     eltype(f_radio_P), eltype(f_dust_P),
                     eltype(cl_radio), eltype(cl_dustE))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        crd, cdE = cl_radio[ℓ], cl_dustE[ℓ]
        for j in 1:n_freq
            frj, fdj = f_radio_P[j], f_dust_P[j]
            for i in 1:n_freq
                out[i, j, ℓ] = a_psee * f_radio_P[i] * frj * crd +
                               a_gee  * f_dust_P[i]  * fdj * cdE
            end
        end
    end
    return out
end

"""
    assemble_TE(a_pste, a_gte,
                f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                cl_radio, cl_dustE)

Fused TE foreground assembly — equivalent to:

    a_pste .* factorized_cross_te(f_radio_T, f_radio_P, cl_radio) .+
    a_gte  .* factorized_cross_te(f_dust_T,  f_dust_P,  cl_dustE)
"""
function assemble_TE(a_pste::Real, a_gte::Real,
                     f_radio_T::AbstractVector, f_radio_P::AbstractVector,
                     f_dust_T::AbstractVector,  f_dust_P::AbstractVector,
                     cl_radio::AbstractVector,  cl_dustE::AbstractVector)
    n_freq = length(f_radio_T)
    n_ell  = length(cl_radio)
    T = promote_type(typeof(a_pste), typeof(a_gte),
                     eltype(f_radio_T), eltype(f_radio_P),
                     eltype(f_dust_T),  eltype(f_dust_P),
                     eltype(cl_radio),  eltype(cl_dustE))
    out = Array{T}(undef, n_freq, n_freq, n_ell)
    @inbounds for ℓ in 1:n_ell
        crd, cdE = cl_radio[ℓ], cl_dustE[ℓ]
        for j in 1:n_freq
            frPj, fdPj = f_radio_P[j], f_dust_P[j]
            for i in 1:n_freq
                out[i, j, ℓ] = a_pste * f_radio_T[i] * frPj * crd +
                               a_gte  * f_dust_T[i]  * fdPj * cdE
            end
        end
    end
    return out
end
