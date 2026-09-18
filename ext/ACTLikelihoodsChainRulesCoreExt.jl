module ACTLikelihoodsChainRulesCoreExt

using ACTLikelihoods
import ChainRulesCore
using ChainRulesCore: NoTangent, ProjectTo, rrule, unthunk
using LinearAlgebra

function ChainRulesCore.rrule(::typeof(ACTLikelihoods._fixed_lower_solve),
                              factor::LinearAlgebra.LowerTriangular,
                              residual::AbstractVector)
    whitened = ACTLikelihoods._fixed_lower_solve(factor, residual)
    project_residual = ProjectTo(residual)

    function fixed_lower_solve_pullback(whitened̄_thunked)
        whitened̄ = unthunk(whitened̄_thunked)
        residual̄ = project_residual(transpose(factor) \ whitened̄)
        return NoTangent(), NoTangent(), residual̄
    end

    return whitened, fixed_lower_solve_pullback
end

"""
    rrule(::typeof(_act_projection), like, cmb_TT, …, calibrations, cmb_offset)

Reverse-mode rule for the whole theory-to-observation kernel.

`_act_projection` is one primitive on purpose: without this rule the tape would
carry, per ordered spectrum, a length-`n_ell` sum, a broadcast and a window
contraction, and reverse mode would allocate and zero a cotangent for every one
of those 41 intermediates. Here it allocates cotangents only for what the caller
can actually differentiate — the three CMB spectra, the three foreground blocks
and the 41 calibration scalars.

The kernel is linear in the CMB and foreground blocks, so their cotangents are
the same scattered window column, weighted by the calibration; both legs receive
it. The calibration cotangents come from the uncalibrated projection that the
forward pass computes anyway, so nothing is divided by a calibration.

Only the multipoles each window covers are written, exactly as in the forward
pass, and the windows are the released data rather than a parameter, so they
take `NoTangent()`.
"""
function ChainRulesCore.rrule(::typeof(ACTLikelihoods._act_projection),
                              like::ACTLikelihoods.ACTDR6FullLikelihood,
                              cmb_TT::AbstractVector, cmb_TE::AbstractVector,
                              cmb_EE::AbstractVector,
                              fg_TT::AbstractArray{<:Real, 3},
                              fg_TE::AbstractArray{<:Real, 3},
                              fg_EE::AbstractArray{<:Real, 3},
                              calibrations::AbstractVector, cmb_offset::Int)
    model, uncalibrated = ACTLikelihoods._act_projection_parts(
        like, cmb_TT, cmb_TE, cmb_EE, fg_TT, fg_TE, fg_EE, calibrations, cmb_offset,
    )

    function act_projection_pullback(cotangent_thunked)
        model̄ = unthunk(cotangent_thunked)

        cmb_TT̄ = zero(cmb_TT)
        cmb_TĒ = zero(cmb_TE)
        cmb_EĒ = zero(cmb_EE)
        fg_TT̄ = zero(fg_TT)
        fg_TĒ = zero(fg_TE)
        fg_EĒ = zero(fg_EE)
        calibrations̄ = zero(calibrations)
        scratch = zeros(eltype(model̄), length(like.ells))

        for (index, spectrum) in enumerate(like.spectra)
            calibration = calibrations[index]
            values = spectrum.band_values
            starts = spectrum.band_starts
            offsets = spectrum.band_offsets
            span = ACTLikelihoods._band_span(starts, offsets)

            # d(model_b)/d(calibration) is the uncalibrated projection.
            derivative = zero(eltype(calibrations̄))
            @inbounds for (bin, output) in enumerate(spectrum.output_indices)
                derivative += model̄[output] * uncalibrated[output]
            end
            calibrations̄[index] += derivative

            # Scatter the window columns once, then pay the strided writes once.
            @inbounds for ell_index in span
                scratch[ell_index] = zero(eltype(scratch))
            end
            @inbounds for (bin, output) in enumerate(spectrum.output_indices)
                weight = model̄[output] * calibration
                iszero(weight) && continue
                offset = offsets[bin]
                width = offsets[bin + 1] - offset
                start = starts[bin]
                @simd for step in 0:(width - 1)
                    scratch[start + step] += weight * values[offset + step]
                end
            end

            cmb̄ = ACTLikelihoods._polarization_block(cmb_TT̄, cmb_TĒ, cmb_EĒ,
                                                    spectrum.polarization)
            fḡ = ACTLikelihoods._polarization_block(fg_TT̄, fg_TĒ, fg_EĒ,
                                                   spectrum.polarization)
            first_leg = spectrum.temperature_leg
            second_leg = spectrum.polarization_leg
            @inbounds for ell_index in span
                contribution = scratch[ell_index]
                cmb̄[cmb_offset + ell_index - 1] += contribution
                fḡ[first_leg, second_leg, ell_index] += contribution
            end
        end

        return (NoTangent(), NoTangent(), cmb_TT̄, cmb_TĒ, cmb_EĒ,
                fg_TT̄, fg_TĒ, fg_EĒ, calibrations̄, NoTangent())
    end

    return model, act_projection_pullback
end

end
