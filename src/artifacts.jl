"""
    artifacts.jl

Resolve the ACT DR6 data artifact (data, covariance, bandpasses, foreground
templates) hosted on Zenodo via Julia's `Artifacts.toml`.

The artifact is declared `lazy=true` in `Artifacts.toml`, so it is downloaded
on first use rather than at package install time.

Layout inside the artifact:
    <artifact>/act_dr6_full_artifact/
      filtered/        # data_vec.txt, cov.txt, inv_cov.txt, spec_meta.txt, windows/, cmb_theory_*.txt
      bandpasses/      # <exp>_s0.txt, <exp>_s2.txt
      templates/       # cl_tsz_150_bat.dat, cl_ksz_bat.dat, cl_cib_Choi2020.dat, cl_sz_x_cib.dat
"""

using LazyArtifacts

"""
    act_dr6_artifact_root() → String

Path to the unpacked artifact root (`<artifact>/act_dr6_full_artifact`).
"""
act_dr6_artifact_root() = joinpath(artifact"act_dr6_data", "act_dr6_full_artifact")

"""
    act_dr6_filtered_dir() → String

Path to the filtered likelihood data directory (input to [`load_data`](@ref)).
"""
act_dr6_filtered_dir() = joinpath(act_dr6_artifact_root(), "filtered")

"""
    act_dr6_bandpass_dir() → String

Path to the bandpasses directory (input to [`load_bands`](@ref)).
"""
act_dr6_bandpass_dir() = joinpath(act_dr6_artifact_root(), "bandpasses")

"""
    act_dr6_template_dir() → String

Path to the foreground templates directory (used by [`load_tsz_template`](@ref) etc.).
"""
act_dr6_template_dir() = joinpath(act_dr6_artifact_root(), "templates")
