"""
    test/test_readme.jl

Execute the README examples.

Every fenced `julia` block in `README.md` that is preceded by a
`<!-- runnable -->` marker is evaluated here, in a fresh module. Documentation
that stops working therefore fails the suite.

The sandbox supplies `example_cmb_spectra()`, which stands in for whatever
Boltzmann solver or emulator a user brings; the package itself never loads a
default cosmology.
"""

"""
    runnable_readme_blocks(path) -> Vector{String}

Extract the fenced `julia` blocks marked `<!-- runnable -->` in the README.
"""
function runnable_readme_blocks(path::AbstractString)
    lines = readlines(path)
    blocks = String[]
    index = 1
    while index <= length(lines)
        if strip(lines[index]) == "<!-- runnable -->"
            opening = index + 1
            while opening <= length(lines) && isempty(strip(lines[opening]))
                opening += 1
            end
            opening <= length(lines) && startswith(strip(lines[opening]), "```julia") ||
                error("README: a <!-- runnable --> marker on line $index is not followed " *
                      "by a ```julia block")
            closing = findnext(line -> strip(line) == "```", lines, opening + 1)
            isnothing(closing) && error("README: unterminated runnable block at line $opening")
            push!(blocks, join(lines[(opening + 1):(closing - 1)], "\n"))
            index = closing + 1
        else
            index += 1
        end
    end
    return blocks
end

@testset "README — runnable examples" begin
    readme = normpath(joinpath(@__DIR__, "..", "README.md"))
    @test isfile(readme)

    blocks = runnable_readme_blocks(readme)
    @test !isempty(blocks)

    sandbox = Module(:ACTLikelihoodsREADMESandbox)
    Core.eval(sandbox, :(using ACTLikelihoods))
    Core.eval(sandbox, :(using LinearAlgebra))
    Core.eval(sandbox, :(using Test))
    # The README examples are honest about needing an external CMB source; the
    # package deliberately loads no default theory. Provide the frozen fixture.
    Core.eval(sandbox, :(const __reference_cmb = $REFERENCE_CMB))
    Core.eval(sandbox, :(example_cmb_spectra() = __reference_cmb))
    Core.eval(sandbox, :(const __expected_chi2 = 1592.0584129991541))

    # A block that throws fails the suite; the explicit `@test true` records
    # that it ran to completion.
    for (index, block) in enumerate(blocks)
        @testset "README block $index" begin
            include_string(sandbox, block, "README.md:block$index")
            @test true
        end
    end
end

@testset "README — documents the conventions it must" begin
    readme = read(normpath(joinpath(@__DIR__, "..", "README.md")), String)
    for required in (
        "10.5281/zenodo.22821597",
        "904418cec753af4ecb197861f0eebbccfe41ca3bce6c68ea53b6e3e05a6eae6a",
        "a91a5b3cbb9be682b1ee442b60cae5701f7e84a9",
        "eca996ddb1fc57750299bf5757a40f5d178c1a8b3446d3341a99ca5ba7378e8b",
        "fc63c8c40533cea0bbec91677208de285c50fc10",
        "666b5580c6567d415de31caf51c1e16cca5133c8",
        "4c4d29c448aea0b8bb0162a328b89be24b9ae729",
        "1592.0584129991541",
        "-2145.171",
        "-2941.200",
        "foreground-marginalized",
        "AutoMooncake",
    )
        @test occursin(required, readme)
    end

    # Every free parameter must appear in the README parameter table.
    for name in ACT_DR6_FREE_PARAMETERS
        @test occursin("`$(name)`", readme)
    end
end
