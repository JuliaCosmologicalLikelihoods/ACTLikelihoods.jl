"""
    test/runtests.jl — ACTLikelihoods.jl test suite

Everything here is unconditional.

The full multifrequency ACT DR6 likelihood is the authoritative product of this
package, and it is validated through the **published Zenodo runtime artifact**,
not through loose local directories. A missing artifact, a malformed artifact, a
missing reference fixture or a failed package extension must fail this suite.
There are no `isdir(...)` guards and no environment-variable gates: a skipped
scientific test is a failed scientific test.

Run with:

    julia -e 'using Pkg; Pkg.test("ACTLikelihoods")'
"""

using Test
using LinearAlgebra
using DelimitedFiles
using SHA
using CMBForegrounds
using ACTLikelihoods

include("testsetup.jl")

@info """ACT DR6 full multifrequency likelihood — test configuration
  artifact          $(ACTLikelihoods.ACT_DR6_TTTEEE_ARTIFACT)
  DOI               $(ACTLikelihoods.ACT_DR6_TTTEEE_DOI)
  tree hash         $(ACTLikelihoods.ACT_DR6_TTTEEE_TREE_SHA1)
  artifact path     $(ARTIFACT_DIR)
  bandpowers        $(length(LIKE.observed))
  ordered spectra   $(length(LIKE.spectra))
  channels          $(join(LIKE.channels, ", "))
  free parameters   $(length(ACT_DR6_FREE_PARAMETERS))
"""

@testset verbose = true "ACTLikelihoods.jl" begin
    @testset "published runtime artifact" begin
        include("test_artifact.jl")
    end
    @testset "full likelihood — data model" begin
        include("test_full_data_model.jl")
    end
    @testset "full likelihood — foregrounds" begin
        include("test_full_foregrounds.jl")
    end
    @testset "full likelihood — baseline parity" begin
        include("test_full_likelihood.jl")
    end
    @testset "full likelihood — nuisance multipoints" begin
        include("test_full_multipoint.jl")
    end
    @testset "nuisance parameters and priors" begin
        include("test_nuisance.jl")
    end
    @testset "automatic differentiation — extensions" begin
        include("test_ad_extensions.jl")
    end
    @testset "automatic differentiation — full path" begin
        include("test_full_ad.jl")
    end
    @testset "README examples" begin
        include("test_readme.jl")
    end
    @testset "CMB-only marginalized variant — units" begin
        include("test_cmb_only_unit.jl")
    end
end
