"""
    rrules.jl

ChainRulesCore.rrule for `theory_vector_core` — the only rrule that
remains in ACT (the cross-spectrum kernel rrules now live in
CMBForegrounds.jl/src/rrules.jl).

This pullback depends on `ACTData` (spec metadata, exp_idx, window
matrices) so it is intrinsically ACT-specific.
"""

using ChainRulesCore: rrule, NoTangent, unthunk

# ------------------------------------------------------------------ #
# theory_vector_core: full ACT theory vector assembly                 #
# ------------------------------------------------------------------ #
# Forward: walks `data.spec_meta`, computes per-spectrum
#   ps_part = W_k' * ((cmb + fg_view) .* cf)
# where cf is the relevant calibration product. Reverse propagates
# back to cmb_*, fg_*, cal_t/e via the chain rule.
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
