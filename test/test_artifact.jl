"""
    test/test_artifact.jl

The published runtime artifact: download, identity, metadata and the complete
checksum ledger.
"""

using JSON3

@testset "ACT DR6 artifact — download and layout" begin
    @test isdir(ARTIFACT_DIR)
    for name in ("data_vec.txt", "cov.txt", "spectrum_metadata.tsv",
                 "metadata.json", "SHA256SUMS")
        @test isfile(joinpath(ARTIFACT_DIR, name))
    end
    for directory in ("windows", "bandpasses", "beams", "templates")
        @test isdir(joinpath(ARTIFACT_DIR, directory))
    end
    @test length(readdir(joinpath(ARTIFACT_DIR, "windows"))) == 35
    @test length(readdir(joinpath(ARTIFACT_DIR, "bandpasses"))) == 10
    @test length(readdir(joinpath(ARTIFACT_DIR, "beams"))) == 10
    @test sort(readdir(joinpath(ARTIFACT_DIR, "templates"))) == [
        "cl_cib_Choi2020.dat", "cl_ksz_bat.dat", "cl_sz_x_cib.dat", "cl_tsz_150_bat.dat",
    ]
end

@testset "ACT DR6 artifact — provenance constants" begin
    provenance = act_dr6_provenance()
    @test provenance.artifact == "ACT_DR6_TTTEEE_v1"
    @test provenance.doi == "10.5281/zenodo.22821597"
    @test provenance.archive == "ACT_DR6_TTTEEE_v1_20260917.tar.xz"
    @test provenance.archive_sha256 ==
        "904418cec753af4ecb197861f0eebbccfe41ca3bce6c68ea53b6e3e05a6eae6a"
    @test provenance.tree_sha1 == "a91a5b3cbb9be682b1ee442b60cae5701f7e84a9"
    @test provenance.source_sacc_sha256 ==
        "eca996ddb1fc57750299bf5757a40f5d178c1a8b3446d3341a99ca5ba7378e8b"
    @test provenance.marginalized === false
    @test provenance.source_revisions.act_dr6_mflike ==
        "fc63c8c40533cea0bbec91677208de285c50fc10"
    @test provenance.source_revisions.LAT_MFLike ==
        "666b5580c6567d415de31caf51c1e16cca5133c8"
    @test provenance.source_revisions.fgspectra ==
        "4c4d29c448aea0b8bb0162a328b89be24b9ae729"

    # Artifacts.toml must bind exactly the tree the provenance advertises.
    artifacts_toml = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    @test isfile(artifacts_toml)
    declared = read(artifacts_toml, String)
    @test occursin(provenance.tree_sha1, declared)
    @test occursin(provenance.archive_sha256, declared)
    @test occursin(provenance.archive, declared)
end

@testset "ACT DR6 artifact — metadata validation" begin
    metadata = validate_act_dr6_metadata(ARTIFACT_DIR)
    @test metadata[:artifact_name] == "ACT_DR6_TTTEEE_v1"
    @test metadata[:marginalized] === false
    @test metadata[:dimensions][:number_of_bandpowers] == 1651
    @test metadata[:dimensions][:number_of_spectra] == 41
    @test metadata[:dimensions][:number_of_channels] == 5
    @test metadata[:dimensions][:number_of_window_files] == 35
    @test metadata[:dimensions][:ell_min] == 2
    @test metadata[:dimensions][:ell_max] == 8501
    @test String.(metadata[:channels]) == collect(ACTLikelihoods.ACT_DR6_CHANNELS)
end

@testset "ACT DR6 artifact — malformed metadata must fail, not skip" begin
    original = JSON3.read(read(joinpath(ARTIFACT_DIR, "metadata.json"), String), Dict)

    mktempdir() do scratch
        # Every one of these is a reason to refuse to build the likelihood.
        mutations = Dict(
            "marginalized" => ("marginalized" => true),
            "artifact name" => ("artifact_name" => "ACT_DR6_CMB_only"),
            "schema version" => ("schema_version" => 2),
            "channel order" => ("channels" => ["dr6_pa5_f090", "dr6_pa4_f220",
                                               "dr6_pa5_f150", "dr6_pa6_f090",
                                               "dr6_pa6_f150"]),
        )
        for (name, (key, value)) in mutations
            broken = deepcopy(original)
            broken[key] = value
            directory = mktempdir(scratch)
            write(joinpath(directory, "metadata.json"), JSON3.write(broken))
            @testset "rejects wrong $name" begin
                @test_throws ArgumentError validate_act_dr6_metadata(directory)
            end
        end

        # Nested dimension and provenance mismatches.
        for (name, path, value) in (
            ("bandpower count", ("dimensions", "number_of_bandpowers"), 1650),
            ("ell range", ("dimensions", "ell_max"), 8500),
            ("source SACC digest", ("source_input", "sha256"), repeat("0", 64)),
            ("act_dr6_mflike revision", ("source_revisions", "act_dr6_mflike"), "deadbeef"),
        )
            broken = deepcopy(original)
            broken[path[1]][path[2]] = value
            directory = mktempdir(scratch)
            write(joinpath(directory, "metadata.json"), JSON3.write(broken))
            @testset "rejects wrong $name" begin
                @test_throws ArgumentError validate_act_dr6_metadata(directory)
            end
        end

        # A missing metadata file is an error too.
        empty_directory = mktempdir(scratch)
        @test_throws ArgumentError validate_act_dr6_metadata(empty_directory)
        @test_throws ArgumentError ACTDR6FullLikelihood(empty_directory;
                                                        validate_metadata=true)
    end
end

@testset "ACT DR6 artifact — complete SHA256SUMS ledger" begin
    # The full 474 MB ledger is verified here, and deliberately not on every
    # likelihood construction.
    checked = verify_act_dr6_checksums(ARTIFACT_DIR)
    @test checked == 63
    @test checked == countlines(joinpath(ARTIFACT_DIR, "SHA256SUMS"))
end

@testset "ACT DR6 artifact — checksum ledger rejects corruption" begin
    mktempdir() do scratch
        write(joinpath(scratch, "payload.txt"), "reference payload\n")
        digest = bytes2hex(SHA.sha256("reference payload\n"))
        ledger = joinpath(scratch, "SHA256SUMS")

        write(ledger, "$digest  payload.txt\n")
        @test verify_act_dr6_checksums(scratch) == 1

        write(joinpath(scratch, "payload.txt"), "tampered payload\n")
        @test_throws ArgumentError verify_act_dr6_checksums(scratch)

        write(ledger, "$digest  absent.txt\n")
        @test_throws ArgumentError verify_act_dr6_checksums(scratch)

        write(ledger, "")
        @test_throws ArgumentError verify_act_dr6_checksums(scratch)

        rm(ledger)
        @test_throws ArgumentError verify_act_dr6_checksums(scratch)
    end
end
