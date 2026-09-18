module ACTLikelihoodsMooncakeExt

using ACTLikelihoods
using LinearAlgebra: LowerTriangular
using Mooncake: @from_chainrules, MinimalCtx

@from_chainrules MinimalCtx Tuple{
    typeof(ACTLikelihoods._fixed_lower_solve),
    LowerTriangular{Float64, Matrix{Float64}}, Vector{Float64},
}

# The whole theory-to-observation kernel as one tape entry. Without this the
# reverse pass allocates and zeroes a cotangent for each of the 41 per-spectrum
# intermediates; with it, only the CMB spectra, the foreground blocks and the
# calibration scalars ever carry one.
@from_chainrules MinimalCtx Tuple{
    typeof(ACTLikelihoods._act_projection),
    ACTLikelihoods.ACTDR6FullLikelihood,
    Vector{Float64}, Vector{Float64}, Vector{Float64},
    Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3},
    Vector{Float64}, Int,
}

end
