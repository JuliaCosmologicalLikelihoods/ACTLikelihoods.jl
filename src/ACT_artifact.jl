"""
    ACT_artifact.jl

Binding and validation for the published ACT DR6 full-multifrequency runtime
artifact.

The artifact is an immutable Zenodo deposit. It carries the *non-marginalized*
multifrequency likelihood: the ordered observation vector, the fixed Gaussian
covariance, the released bandpower windows, the array bandpasses, the chromatic
beams, and the four foreground templates. It is the only runtime data source the
public constructors need; no local directory tree is required.
"""

# ---------------------------------------------------------------------------
# Provenance constants. These are declarations about the published artifact and
# must not be edited: a change here is a new artifact, not a new value.
# ---------------------------------------------------------------------------

"Name of the published runtime artifact, as bound in `Artifacts.toml`."
const ACT_DR6_TTTEEE_ARTIFACT = "ACT_DR6_TTTEEE_v1"

"Concept DOI of the published ACT DR6 full-likelihood runtime artifact."
const ACT_DR6_TTTEEE_DOI = "10.5281/zenodo.22821597"

"Zenodo record holding the published runtime artifact."
const ACT_DR6_TTTEEE_RECORD_URL = "https://zenodo.org/records/22821597"

"File name of the published archive."
const ACT_DR6_TTTEEE_ARCHIVE = "ACT_DR6_TTTEEE_v1_20260917.tar.xz"

"SHA-256 of the published archive."
const ACT_DR6_TTTEEE_ARCHIVE_SHA256 =
    "904418cec753af4ecb197861f0eebbccfe41ca3bce6c68ea53b6e3e05a6eae6a"

"Julia (git) tree hash of the extracted artifact."
const ACT_DR6_TTTEEE_TREE_SHA1 = "a91a5b3cbb9be682b1ee442b60cae5701f7e84a9"

"SHA-256 of the original ACT DR6 SACC release the artifact was derived from."
const ACT_DR6_SOURCE_SACC_SHA256 =
    "eca996ddb1fc57750299bf5757a40f5d178c1a8b3446d3341a99ca5ba7378e8b"

"""
Upstream revisions used to generate and validate the artifact.

* `act_dr6_mflike` — the ACT DR6 Cobaya likelihood
* `LAT_MFLike`     — the multifrequency foreground/systematics layer
* `fgspectra`      — the foreground SED and angular-shape library
"""
const ACT_DR6_SOURCE_REVISIONS = (
    act_dr6_mflike = "fc63c8c40533cea0bbec91677208de285c50fc10",
    LAT_MFLike = "666b5580c6567d415de31caf51c1e16cca5133c8",
    fgspectra = "4c4d29c448aea0b8bb0162a328b89be24b9ae729",
)

"Ordered array-frequency channels of the ACT DR6 multifrequency likelihood."
const ACT_DR6_CHANNELS = (
    "dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150",
)

"Fixed dimensions of the released full likelihood."
const ACT_DR6_DIMENSIONS = (
    number_of_bandpowers = 1651,
    number_of_spectra = 41,
    number_of_channels = 5,
    number_of_window_files = 35,
    ell_min = 2,
    ell_max = 8501,
)

"""
    act_dr6_tttee_artifact_path() -> String

Absolute path of the extracted `ACT_DR6_TTTEEE_v1` artifact, downloading and
unpacking it on first use. Requires no local data tree.
"""
act_dr6_tttee_artifact_path() = artifact"ACT_DR6_TTTEEE_v1"

# ---------------------------------------------------------------------------
# metadata.json validation
# ---------------------------------------------------------------------------

_metadata_path(directory::AbstractString) = joinpath(directory, "metadata.json")

"""
    act_dr6_metadata(directory) -> JSON3.Object

Read `metadata.json` from an ACT DR6 full-likelihood data directory.
"""
function act_dr6_metadata(directory::AbstractString)
    path = _metadata_path(directory)
    isfile(path) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing: $path"))
    return JSON3.read(read(path, String))
end

function _require_metadata(metadata, key::Symbol, expected, description::AbstractString)
    haskey(metadata, key) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing `$key` ($description)"))
    actual = metadata[key]
    actual == expected || throw(ArgumentError(
        "ACT DR6 artifact metadata mismatch for `$key` ($description): " *
        "expected $(repr(expected)), found $(repr(actual))",
    ))
    return actual
end

"""
    validate_act_dr6_metadata(directory) -> JSON3.Object

Check that `metadata.json` in `directory` describes the expected published
artifact, and return it.

Verified: artifact identity and schema version, `marginalized == false`, the
number of bandpowers, spectra, channels and window files, the multipole range,
the ordered channel list, the SHA-256 of the source SACC release, and the three
upstream source revisions.

Bulk file hashing is deliberately *not* performed here — the full `SHA256SUMS`
ledger is 474 MB of input and is verified in the test suite by
[`verify_act_dr6_checksums`](@ref) instead.
"""
function validate_act_dr6_metadata(directory::AbstractString)
    metadata = act_dr6_metadata(directory)

    _require_metadata(metadata, :artifact_name, ACT_DR6_TTTEEE_ARTIFACT,
                      "published artifact identity")
    _require_metadata(metadata, :schema_version, 1, "metadata schema version")
    _require_metadata(metadata, :marginalized, false,
                      "this package requires the non-marginalized multifrequency likelihood")

    haskey(metadata, :dimensions) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing `dimensions`"))
    dimensions = metadata[:dimensions]
    for key in keys(ACT_DR6_DIMENSIONS)
        _require_metadata(dimensions, key, ACT_DR6_DIMENSIONS[key], "released dimension")
    end

    haskey(metadata, :channels) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing `channels`"))
    collect(String.(metadata[:channels])) == collect(ACT_DR6_CHANNELS) ||
        throw(ArgumentError(
            "ACT DR6 artifact channel order mismatch: expected " *
            "$(collect(ACT_DR6_CHANNELS)), found $(collect(String.(metadata[:channels])))",
        ))

    haskey(metadata, :source_input) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing `source_input`"))
    _require_metadata(metadata[:source_input], :sha256, ACT_DR6_SOURCE_SACC_SHA256,
                      "SHA-256 of the original ACT DR6 SACC release")

    haskey(metadata, :source_revisions) ||
        throw(ArgumentError("ACT DR6 artifact metadata is missing `source_revisions`"))
    revisions = metadata[:source_revisions]
    for key in keys(ACT_DR6_SOURCE_REVISIONS)
        _require_metadata(revisions, key, ACT_DR6_SOURCE_REVISIONS[key],
                          "upstream source revision")
    end

    return metadata
end

# ---------------------------------------------------------------------------
# Full ledger verification (tests only — reads the whole 474 MB tree)
# ---------------------------------------------------------------------------

"""
    verify_act_dr6_checksums(directory) -> Int

Verify every entry of the artifact's `SHA256SUMS` ledger and return the number of
files checked. This reads the complete 474 MB tree and is intended for the test
suite, not for likelihood construction.

Throws if the ledger is missing, if a listed file is absent, or if any digest
disagrees.
"""
function verify_act_dr6_checksums(directory::AbstractString)
    ledger = joinpath(directory, "SHA256SUMS")
    isfile(ledger) ||
        throw(ArgumentError("ACT DR6 artifact checksum ledger is missing: $ledger"))
    checked = 0
    for line in eachline(ledger)
        entry = strip(line)
        isempty(entry) && continue
        fields = split(entry, limit=2)
        length(fields) == 2 ||
            throw(ArgumentError("malformed SHA256SUMS entry in $ledger: $entry"))
        expected = String(fields[1])
        relative = String(lstrip(fields[2], [' ', '*']))
        path = joinpath(directory, relative)
        isfile(path) ||
            throw(ArgumentError("ACT DR6 artifact file listed in SHA256SUMS is missing: $relative"))
        actual = open(path, "r") do stream
            bytes2hex(SHA.sha256(stream))
        end
        actual == expected || throw(ArgumentError(
            "ACT DR6 artifact checksum mismatch for $relative: expected $expected, found $actual",
        ))
        checked += 1
    end
    checked > 0 ||
        throw(ArgumentError("ACT DR6 artifact checksum ledger is empty: $ledger"))
    return checked
end

"""
    act_dr6_provenance() -> NamedTuple

Machine-readable provenance of the bound runtime artifact: DOI, record URL,
archive name and SHA-256, Julia tree hash, source SACC digest, and the upstream
`act_dr6_mflike` / `LAT_MFLike` / `fgspectra` revisions.
"""
act_dr6_provenance() = (
    artifact = ACT_DR6_TTTEEE_ARTIFACT,
    doi = ACT_DR6_TTTEEE_DOI,
    record_url = ACT_DR6_TTTEEE_RECORD_URL,
    archive = ACT_DR6_TTTEEE_ARCHIVE,
    archive_sha256 = ACT_DR6_TTTEEE_ARCHIVE_SHA256,
    tree_sha1 = ACT_DR6_TTTEEE_TREE_SHA1,
    source_sacc_sha256 = ACT_DR6_SOURCE_SACC_SHA256,
    source_revisions = ACT_DR6_SOURCE_REVISIONS,
    marginalized = false,
)
