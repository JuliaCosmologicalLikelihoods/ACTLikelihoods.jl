"""
    rrules.jl

ChainRulesCore.rrule definitions for the cross-spectrum kernels in cross.jl.

These exist purely to speed up reverse-mode differentiation (Mooncake).
ForwardDiff does not use rrules — it goes through the original forward
implementations in cross.jl unchanged.

Each rrule:
  1. Computes the forward result by calling the existing kernel.
  2. Returns a pullback closure that maps the output cotangent back to
     input cotangents using BLAS calls instead of element-wise scalar ops.

All incoming cotangents are `unthunk`ed defensively because Mooncake passes
`InplaceableThunk`s into pullbacks.
"""

using ChainRulesCore: rrule, NoTangent, unthunk

# ------------------------------------------------------------------ #
# factorized_cross(f, cl):  D[i,j,ℓ] = f[i] f[j] cl[ℓ]                  #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross),
                              f::AbstractVector, cl::AbstractVector)
    n_freq = length(f)
    n_ell  = length(cl)
    D      = factorized_cross(f, cl)

    function factorized_cross_pullback(D̄_thunked)
        D̄      = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # M[i,j] = Σ_ℓ D̄[i,j,ℓ] cl[ℓ]
        M = reshape(D̄_flat * cl, n_freq, n_freq)

        # df̄[i] = Σ_j (M[i,j] + M[j,i]) f[j]
        df̄ = (M + transpose(M)) * f

        # dcl̄[ℓ] = Σ_{i,j} D̄[i,j,ℓ] f[i] f[j] = vec(f f')' * D̄[:,:,ℓ]
        dcl̄ = transpose(D̄_flat) * vec(f * transpose(f))

        return NoTangent(), df̄, dcl̄
    end

    return D, factorized_cross_pullback
end

# ------------------------------------------------------------------ #
# factorized_cross_te(fT, fE, cl):  D[i,j,ℓ] = fT[i] fE[j] cl[ℓ]        #
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(factorized_cross_te),
                              fT::AbstractVector,
                              fE::AbstractVector,
                              cl::AbstractVector)
    n_freq = length(fT)
    @assert length(fE) == n_freq
    n_ell  = length(cl)
    D      = factorized_cross_te(fT, fE, cl)

    function factorized_cross_te_pullback(D̄_thunked)
        D̄      = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # M[i,j] = Σ_ℓ D̄[i,j,ℓ] cl[ℓ]
        M = reshape(D̄_flat * cl, n_freq, n_freq)

        # dfT̄[i] = Σ_j M[i,j] fE[j]
        dfT̄ = M * fE

        # dfĒ[j] = Σ_i M[i,j] fT[i]
        dfĒ = transpose(M) * fT

        # dcl̄[ℓ] = fT' * D̄[:,:,ℓ] * fE = vec(fT fE')' * D̄[:,:,ℓ]
        dcl̄ = transpose(D̄_flat) * vec(fT * transpose(fE))

        return NoTangent(), dfT̄, dfĒ, dcl̄
    end

    return D, factorized_cross_te_pullback
end

# ------------------------------------------------------------------ #
# correlated_cross(f, cl):  D[:,:,ℓ] = f' cl[:,:,ℓ] f                   #
# ------------------------------------------------------------------ #
# Forward (per ℓ):  D_ℓ = (f') · C_ℓ · f
# Reverse (per ℓ):
#   dC̄[:,:,ℓ] = f · D̄_ℓ · f'
#   df̄ contribution = C_ℓ · (f · D̄_ℓ') + C_ℓ' · (f · D̄_ℓ)
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(theory_vector_core),
                              cmb_tt::AbstractVector{<:Number},
                              cmb_te::AbstractVector{<:Number},
                              cmb_ee::AbstractVector{<:Number},
                              fg_TT::AbstractArray{<:Number,3},
                              fg_TE::AbstractArray{<:Number,3},
                              fg_EE::AbstractArray{<:Number,3},
                              cal_t_vec::AbstractVector{<:Number},
                              cal_e_vec::AbstractVector{<:Number},
                              data::ACTData)
    ps_vec      = theory_vector_core(cmb_tt, cmb_te, cmb_ee,
                                     fg_TT, fg_TE, fg_EE,
                                     cal_t_vec, cal_e_vec, data)
    exp_idx     = data.exp_idx
    spec_meta   = data.spec_meta

    function theory_vector_core_pullback(dps_thunked)
        dps = unthunk(dps_thunked)
        T   = promote_type(eltype(dps), eltype(cmb_tt), eltype(cal_t_vec))

        d_cmb_tt = zeros(T, length(cmb_tt))
        d_cmb_te = zeros(T, length(cmb_te))
        d_cmb_ee = zeros(T, length(cmb_ee))
        d_fg_TT  = zeros(T, size(fg_TT))
        d_fg_TE  = zeros(T, size(fg_TE))
        d_fg_EE  = zeros(T, size(fg_EE))
        d_cal_t  = zeros(T, length(cal_t_vec))
        d_cal_e  = zeros(T, length(cal_e_vec))

        @inbounds for m in spec_meta
            i  = exp_idx[m.t1]
            j  = exp_idx[m.t2]
            fi = m.hasYX_xsp ? j : i
            fj = m.hasYX_xsp ? i : j
            ids = m.ids

            # forward (k-th): dl_k, cf_k → ps_part = W' (dl .* cf)
            # reverse:
            #   d_dlcf  = W * dps_part            (n_ell)
            #   d_dl    = d_dlcf .* cf            (scatter into cmb + fg)
            #   d_cf    = dot(d_dlcf, dl)         (scatter into cal vectors)

            dpsk    = view(dps, ids)
            d_dlcf  = m.W * dpsk                  # (n_ell,)

            if m.pol == "tt"
                cf      = cal_t_vec[i] * cal_t_vec[j]
                fg_view = view(fg_TT, fi, fj, :)
                dl      = cmb_tt .+ fg_view
                d_dl    = d_dlcf .* cf
                d_cf    = dot(d_dlcf, dl)

                d_cmb_tt        .+= d_dl
                @views d_fg_TT[fi, fj, :] .+= d_dl

                d_cal_t[i] += d_cf * cal_t_vec[j]
                d_cal_t[j] += d_cf * cal_t_vec[i]

            elseif m.pol == "te"
                if m.hasYX_xsp
                    cf = cal_e_vec[i] * cal_t_vec[j]
                else
                    cf = cal_t_vec[i] * cal_e_vec[j]
                end
                fg_view = view(fg_TE, fi, fj, :)
                dl      = cmb_te .+ fg_view
                d_dl    = d_dlcf .* cf
                d_cf    = dot(d_dlcf, dl)

                d_cmb_te        .+= d_dl
                @views d_fg_TE[fi, fj, :] .+= d_dl

                if m.hasYX_xsp
                    d_cal_e[i] += d_cf * cal_t_vec[j]
                    d_cal_t[j] += d_cf * cal_e_vec[i]
                else
                    d_cal_t[i] += d_cf * cal_e_vec[j]
                    d_cal_e[j] += d_cf * cal_t_vec[i]
                end

            else  # ee
                cf      = cal_e_vec[i] * cal_e_vec[j]
                fg_view = view(fg_EE, fi, fj, :)
                dl      = cmb_ee .+ fg_view
                d_dl    = d_dlcf .* cf
                d_cf    = dot(d_dlcf, dl)

                d_cmb_ee        .+= d_dl
                @views d_fg_EE[fi, fj, :] .+= d_dl

                d_cal_e[i] += d_cf * cal_e_vec[j]
                d_cal_e[j] += d_cf * cal_e_vec[i]
            end
        end

        return (NoTangent(),
                d_cmb_tt, d_cmb_te, d_cmb_ee,
                d_fg_TT, d_fg_TE, d_fg_EE,
                d_cal_t, d_cal_e,
                NoTangent())
    end

    return ps_vec, theory_vector_core_pullback
end

# ------------------------------------------------------------------ #
# assemble_TT — fused TT-spectrum rrule                                #
# ------------------------------------------------------------------ #
# Forward (per ℓ):
#   D[i,j,ℓ] = a_kSZ f_ksz[i] f_ksz[j] cl_ksz[ℓ]
#            + f_tsz[i] f_tsz[j] cl_tsz[ℓ]
#            + (f_tsz[i] f_cibc[j] + f_cibc[i] f_tsz[j]) cl_szxcib[ℓ]
#            + f_cibc[i] f_cibc[j] cl_cibc[ℓ]
#            + a_p   f_cibp[i]  f_cibp[j]  cl_cibp[ℓ]
#            + a_gtt f_dust[i]  f_dust[j]  cl_dustT[ℓ]
#            + a_s   f_radio[i] f_radio[j] cl_radio[ℓ]
# ------------------------------------------------------------------ #

function _factorized_term_grads!(D̄_flat::AbstractMatrix, cl::AbstractVector,
                                  f::AbstractVector, α::Real)
    # Returns (M, d_α, d_f, d_cl) for the term: α · f[i] · f[j] · cl[ℓ]
    n = length(f)
    M = reshape(D̄_flat * cl, n, n)
    d_α = dot(f, M, f)
    d_f = α * ((M + transpose(M)) * f)
    d_cl = α * (transpose(D̄_flat) * vec(f * transpose(f)))
    return M, d_α, d_f, d_cl
end

function ChainRulesCore.rrule(::typeof(assemble_TT),
                              a_kSZ::Real, a_p::Real, a_gtt::Real, a_s::Real,
                              f_ksz::AbstractVector,   f_cibp::AbstractVector,
                              f_dust::AbstractVector,  f_radio::AbstractVector,
                              f_tsz::AbstractVector,   f_cibc::AbstractVector,
                              cl_ksz::AbstractVector,  cl_cibp::AbstractVector,
                              cl_dustT::AbstractVector, cl_radio::AbstractVector,
                              cl_tsz::AbstractVector,  cl_cibc::AbstractVector,
                              cl_szxcib::AbstractVector)
    n_freq = length(f_ksz)
    n_ell  = length(cl_ksz)
    D = assemble_TT(a_kSZ, a_p, a_gtt, a_s,
                    f_ksz, f_cibp, f_dust, f_radio, f_tsz, f_cibc,
                    cl_ksz, cl_cibp, cl_dustT, cl_radio,
                    cl_tsz, cl_cibc, cl_szxcib)

    function assemble_TT_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        # Standard factorized terms (α explicit)
        _, d_a_kSZ, d_f_ksz,   d_cl_ksz   = _factorized_term_grads!(D̄_flat, cl_ksz,   f_ksz,   a_kSZ)
        _, d_a_p,   d_f_cibp,  d_cl_cibp  = _factorized_term_grads!(D̄_flat, cl_cibp,  f_cibp,  a_p)
        _, d_a_gtt, d_f_dust,  d_cl_dustT = _factorized_term_grads!(D̄_flat, cl_dustT, f_dust,  a_gtt)
        _, d_a_s,   d_f_radio, d_cl_radio = _factorized_term_grads!(D̄_flat, cl_radio, f_radio, a_s)

        # tSZ auto: f_tsz f_tsz' cl_tsz   (α = 1)
        M_tsz = reshape(D̄_flat * cl_tsz, n_freq, n_freq)
        d_f_tsz_auto = (M_tsz + transpose(M_tsz)) * f_tsz
        d_cl_tsz     = transpose(D̄_flat) * vec(f_tsz * transpose(f_tsz))

        # CIB-clustered auto: f_cibc f_cibc' cl_cibc   (α = 1)
        M_cibc = reshape(D̄_flat * cl_cibc, n_freq, n_freq)
        d_f_cibc_auto = (M_cibc + transpose(M_cibc)) * f_cibc
        d_cl_cibc     = transpose(D̄_flat) * vec(f_cibc * transpose(f_cibc))

        # tSZ × CIB cross: (f_tsz f_cibc' + f_cibc f_tsz') cl_szxcib   (α = 1)
        M_sxc = reshape(D̄_flat * cl_szxcib, n_freq, n_freq)
        d_f_tsz_cross  = M_sxc * f_cibc + transpose(M_sxc) * f_cibc
        d_f_cibc_cross = M_sxc * f_tsz  + transpose(M_sxc) * f_tsz
        d_cl_szxcib    = transpose(D̄_flat) *
                         vec(f_tsz * transpose(f_cibc) + f_cibc * transpose(f_tsz))

        d_f_tsz  = d_f_tsz_auto  + d_f_tsz_cross
        d_f_cibc = d_f_cibc_auto + d_f_cibc_cross

        return (NoTangent(),
                d_a_kSZ, d_a_p, d_a_gtt, d_a_s,
                d_f_ksz, d_f_cibp, d_f_dust, d_f_radio,
                d_f_tsz, d_f_cibc,
                d_cl_ksz, d_cl_cibp, d_cl_dustT, d_cl_radio,
                d_cl_tsz, d_cl_cibc, d_cl_szxcib)
    end

    return D, assemble_TT_pullback
end

# ------------------------------------------------------------------ #
# assemble_EE — fused EE-spectrum rrule                                #
# ------------------------------------------------------------------ #
# Forward (per ℓ):
#   D[i,j,ℓ] = a_psee f_radio_P[i] f_radio_P[j] cl_radio[ℓ]
#            + a_gee  f_dust_P[i]  f_dust_P[j]  cl_dustE[ℓ]
# ------------------------------------------------------------------ #

function ChainRulesCore.rrule(::typeof(assemble_EE),
                              a_psee::Real, a_gee::Real,
                              f_radio_P::AbstractVector, f_dust_P::AbstractVector,
                              cl_radio::AbstractVector,  cl_dustE::AbstractVector)
    n_freq = length(f_radio_P)
    n_ell  = length(cl_radio)
    D = assemble_EE(a_psee, a_gee, f_radio_P, f_dust_P, cl_radio, cl_dustE)

    function assemble_EE_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        _, d_a_psee, d_f_radio_P, d_cl_radio =
            _factorized_term_grads!(D̄_flat, cl_radio, f_radio_P, a_psee)
        _, d_a_gee,  d_f_dust_P,  d_cl_dustE =
            _factorized_term_grads!(D̄_flat, cl_dustE, f_dust_P,  a_gee)

        return (NoTangent(),
                d_a_psee, d_a_gee,
                d_f_radio_P, d_f_dust_P,
                d_cl_radio,  d_cl_dustE)
    end

    return D, assemble_EE_pullback
end

# ------------------------------------------------------------------ #
# assemble_TE — fused TE-spectrum rrule                                #
# ------------------------------------------------------------------ #
# Forward (per ℓ):
#   D[i,j,ℓ] = a_pste f_radio_T[i] f_radio_P[j] cl_radio[ℓ]
#            + a_gte  f_dust_T[i]  f_dust_P[j]  cl_dustE[ℓ]
# ------------------------------------------------------------------ #

function _te_term_grads(D̄_flat::AbstractMatrix, cl::AbstractVector,
                        fT::AbstractVector, fE::AbstractVector, α::Real)
    n = length(fT)
    M = reshape(D̄_flat * cl, n, n)
    d_α  = dot(fT, M, fE)
    d_fT = α * (M           * fE)
    d_fE = α * (transpose(M) * fT)
    d_cl = α * (transpose(D̄_flat) * vec(fT * transpose(fE)))
    return d_α, d_fT, d_fE, d_cl
end

function ChainRulesCore.rrule(::typeof(assemble_TE),
                              a_pste::Real, a_gte::Real,
                              f_radio_T::AbstractVector, f_radio_P::AbstractVector,
                              f_dust_T::AbstractVector,  f_dust_P::AbstractVector,
                              cl_radio::AbstractVector,  cl_dustE::AbstractVector)
    n_freq = length(f_radio_T)
    n_ell  = length(cl_radio)
    D = assemble_TE(a_pste, a_gte,
                    f_radio_T, f_radio_P, f_dust_T, f_dust_P,
                    cl_radio, cl_dustE)

    function assemble_TE_pullback(D̄_thunked)
        D̄ = unthunk(D̄_thunked)
        D̄_flat = reshape(D̄, n_freq * n_freq, n_ell)

        d_a_pste, d_f_radio_T, d_f_radio_P, d_cl_radio =
            _te_term_grads(D̄_flat, cl_radio, f_radio_T, f_radio_P, a_pste)
        d_a_gte,  d_f_dust_T,  d_f_dust_P,  d_cl_dustE =
            _te_term_grads(D̄_flat, cl_dustE, f_dust_T,  f_dust_P,  a_gte)

        return (NoTangent(),
                d_a_pste, d_a_gte,
                d_f_radio_T, d_f_radio_P,
                d_f_dust_T,  d_f_dust_P,
                d_cl_radio,  d_cl_dustE)
    end

    return D, assemble_TE_pullback
end

function ChainRulesCore.rrule(::typeof(correlated_cross),
                              f::AbstractMatrix,
                              cl::AbstractArray{<:Any,3})
    n_comp, n_freq = size(f)
    @assert size(cl, 1) == n_comp && size(cl, 2) == n_comp
    n_ell = size(cl, 3)
    D     = correlated_cross(f, cl)

    function correlated_cross_pullback(D̄_thunked)
        D̄  = unthunk(D̄_thunked)
        T  = promote_type(eltype(D̄), eltype(f), eltype(cl))
        df̄  = zeros(T, n_comp, n_freq)
        dcl̄ = zeros(T, n_comp, n_comp, n_ell)

        @inbounds for ℓ in 1:n_ell
            D̄ℓ = @view D̄[:, :, ℓ]
            Cℓ = @view cl[:, :, ℓ]

            # df̄ contributions (two terms — one per appearance of f)
            df̄ .+= Cℓ           * (f * transpose(D̄ℓ))
            df̄ .+= transpose(Cℓ) * (f * D̄ℓ)

            # dcl̄[:,:,ℓ] = f * D̄_ℓ * f'
            dcl̄[:, :, ℓ] = f * D̄ℓ * transpose(f)
        end

        return NoTangent(), df̄, dcl̄
    end

    return D, correlated_cross_pullback
end
