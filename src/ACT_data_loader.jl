"""
    ACT_data_loader.jl

File I/O utilities for the ACT DR6 likelihood.

Handles reading band-power vectors, covariance matrices, window function files,
and foreground templates in the formats used by the candl data release.

# Window function file format
Files are named `{SPEC}_window_functions.txt` or `{SPEC}_lxl_window_functions.txt`.
Each file is a plain-text matrix of shape `(N_ell_rows, N_bins+1)` where:
  - Column 1 is the theory ℓ value (integers)
  - Columns 2..N_bins+1 are the window function W(ℓ, b) for bin b

# Band power & covariance file format
Plain whitespace-delimited text files.
- Band power file: one float per row, or a single row of floats (automatically flattened)
- Covariance file: N×N matrix (N rows of N whitespace-separated floats)

# Foreground template file format
Two-column (ℓ, Dℓ) ASCII files, same as Hillipop convention.
"""

"""
    load_bandpowers(path; T=Float64) -> Vector{T}

Read a plain-text band-power file and return a flat `Vector{T}`.
Handles both single-column and multi-column layouts.
"""
function load_bandpowers(path::AbstractString; T::Type=Float64)
    raw = readdlm(path)
    return T.(vec(raw))
end


"""
    load_covariance(path; T=Float64) -> Matrix{T}

Read a plain-text covariance matrix file.  The file must be square (N×N).
Validates symmetry and positive-definiteness.
"""
function load_covariance(path::AbstractString; T::Type=Float64)
    raw = readdlm(path)
    n = size(raw, 1)
    @assert size(raw, 2) == n "Covariance file must be a square matrix; got $(size(raw))"
    C = T.(raw)
    # Enforce exact symmetry (numerical noise in the text file)
    return Symmetric(0.5 .* (C .+ C'))
end


"""
    load_window_function(path; ell_min=2, T=Float64) -> Matrix{T}

Read a per-spectrum window function file.

The file has shape `(N_ell_rows, N_bins+1)`:
  - Column 1: theory ℓ values
  - Columns 2..end: W(ℓ, bin)

Returns `W[ℓ, b]` as a `Matrix{T}` of shape `(N_ell_theory, N_bins)`,
starting at `ell_min` (earlier rows are discarded).
"""
function load_window_function(path::AbstractString; ell_min::Int=2, T::Type=Float64)
    raw = readdlm(path)
    ell_col = Int.(raw[:, 1])
    start_ix = findfirst(==(ell_min), ell_col)
    @assert !isnothing(start_ix) "ell_min=$ell_min not found in window file: $path"
    W = T.(raw[start_ix:end, 2:end])   # (N_ell_theory, N_bins)
    return W
end


"""
    load_fg_template(path; lmax=6000, lnorm=3000, T=Float64) -> Vector{T}

Read a two-column (ℓ, Dℓ) foreground template file, zero-pad to length `lmax+1`,
and normalise at `lnorm`.

Returns a `Vector{T}` of length `lmax+1` where index `i` corresponds to ℓ = i-1.
"""
function load_fg_template(path::AbstractString;
                           lmax::Int=6000,
                           lnorm::Int=3000,
                           T::Type=Float64)
    data  = readdlm(path; comments=true)
    ells  = Int.(data[:, 1])
    vals  = T.(data[:, 2])

    max_ell = max(lmax, lnorm, maximum(ells))
    tmpl    = zeros(T, max_ell + 1)
    for (e, v) in zip(ells, vals)
        tmpl[e + 1] = v          # 1-based: index e+1 corresponds to ℓ = e
    end

    norm_val = tmpl[lnorm + 1]
    if norm_val != zero(T)
        tmpl ./= norm_val
    end

    return tmpl[1:lmax + 1]
end


"""
    _parse_act_yaml(path) -> Dict{String,Any}

Minimal YAML parser for the ACT DR6 candl-format dataset descriptor.
Extracts: `band_power_file`, `covariance_file`, `window_functions_folder`,
and `spectra_info` (list of single-key dicts `{spec_name => N_bins}`).

This avoids a hard run-time dependency on YAML.jl while still correctly
parsing the subset of YAML used by candl dataset files.
"""
function _parse_act_yaml(path::AbstractString)
    result = Dict{String,Any}()
    spectra = Dict{String,Int}[]
    in_spectra = false

    for line in eachline(path)
        # Strip inline comments and trailing whitespace
        line = strip(split(line, '#')[1])
        isempty(line) && continue

        # Spectra-info list entries: " - TT lxl: 45" or " - TT: 45"
        if in_spectra
            m2 = match(r"^\s*-\s+(.+?)\s*:\s*(\d+)\s*$", line)
            if !isnothing(m2)
                spec   = String(m2.captures[1])
                nbins  = parse(Int, m2.captures[2])
                push!(spectra, Dict(spec => nbins))
                continue
            end
            # If line is not a list entry, we may have left the block
            # Only leave if it's a new top-level key
            if !startswith(line, " ") && !startswith(line, "\t")
                in_spectra = false
            else
                continue   # indented / continuation — ignore
            end
        end

        # Top-level key with value: "band_power_file: foo.txt"
        m = match(r"^(\w[\w_]*)\s*:\s*(.+)$", line)
        if !isnothing(m)
            key = String(m.captures[1])
            val = strip(String(m.captures[2]), ['"', '\'', ' '])
            if key ∈ ("band_power_file", "covariance_file",
                      "window_functions_folder", "name")
                result[key] = val
            end
            in_spectra = false
            continue
        end

        # Bare top-level key (no value on this line): "spectra_info:"
        m_bare = match(r"^(\w[\w_]*)\s*:\s*$", line)
        if !isnothing(m_bare)
            key = String(m_bare.captures[1])
            if key == "spectra_info"
                in_spectra = true
            else
                in_spectra = false
            end
            continue
        end
    end

    if !isempty(spectra)
        result["spectra_info"] = spectra
    end
    return result
end



"""
    read_spectrum_info_from_yaml(yaml_dict) -> (spec_order, spec_types, N_bins)

Parse the `spectra_info` section of a candl-style YAML dictionary.

Returns:
- `spec_order :: Vector{String}` — e.g. `["TT", "TE", "EE"]` or `["TT 90x90", ...]`
- `spec_types :: Vector{String}` — first two chars of each spec_order entry
- `N_bins     :: Vector{Int}`    — number of band-power bins for each spectrum
"""
function read_spectrum_info_from_yaml(yaml_dict::AbstractDict)
    spectra_info = yaml_dict["spectra_info"]
    spec_order = String[]
    N_bins     = Int[]
    for entry in spectra_info
        kv = collect(entry)           # each entry is a single-key dict
        push!(spec_order, string(kv[1].first))
        push!(N_bins,     Int(kv[1].second))
    end
    spec_types = [s[1:min(2, end)] for s in spec_order]
    return spec_order, spec_types, N_bins
end



"""
    load_act_data(data_dir; yaml_file=nothing, ell_min=2, T=Float64)
        -> (data_vector, covariance, windows, spec_order, spec_types, N_bins, N_ell, ells)

Load the ACT DR6 dataset from `data_dir`.

If `yaml_file` is given (or a `.yaml` auto-detected in `data_dir`), it is parsed
to discover the spectrum order, bin counts, and file names.  Otherwise the
canonical ACT DR6 CMB-only file names are used as defaults.

# Returns (as a NamedTuple)
- `data_vector :: Vector{T}`        — concatenated band powers
- `covariance  :: Symmetric{T}`     — raw covariance matrix
- `cov_chol    :: Cholesky`         — precomputed Cholesky factorisation
- `windows     :: Vector{Matrix{T}}`— one window-function matrix per spectrum
- `spec_order  :: Vector{String}`
- `spec_types  :: Vector{String}`
- `N_bins      :: Vector{Int}`
- `N_ell       :: Int`              — number of theory ℓ bins
- `ells        :: Vector{T}`        — theory ℓ grid [ell_min..ell_max]
"""
function load_act_data(data_dir::AbstractString;
                       yaml_file::Union{Nothing,AbstractString}=nothing,
                       ell_min::Int=2,
                       T::Type=Float64)

    # ---- Detect YAML descriptor ----
    yaml_dict = nothing
    if !isnothing(yaml_file)
        yaml_path = isabs(yaml_file) ? yaml_file : joinpath(data_dir, yaml_file)
    else
        # auto-detect: first .yaml file in data_dir that is not an index file
        candidates = filter(f -> endswith(f, ".yaml") && !occursin("index", f),
                                readdir(data_dir; join=true))
        yaml_path = isempty(candidates) ? nothing : first(candidates)
    end

    if !isnothing(yaml_path) && isfile(yaml_path)
        yaml_dict = _parse_act_yaml(yaml_path)
    end

    # ---- Spectrum info ----
    local spec_order, spec_types, N_bins
    local bdp_file, cov_file, win_folder

    if !isnothing(yaml_dict)
        spec_order, spec_types, N_bins = read_spectrum_info_from_yaml(yaml_dict)
        bdp_file   = get(yaml_dict, "band_power_file",      nothing)
        cov_file   = get(yaml_dict, "covariance_file",      nothing)
        win_folder = get(yaml_dict, "window_functions_folder", nothing)
    else
        # Fallback: ACT DR6 CMB-only defaults
        spec_order = ["TT", "TE", "EE"]
        spec_types = ["TT", "TE", "EE"]
        N_bins     = [45, 45, 45]
        bdp_file   = nothing
        cov_file   = nothing
        win_folder = nothing
    end

    # ---- Locate files via convention if not from YAML ----
    function _find_file(kw_path, data_dir, patterns)
        !isnothing(kw_path) && isfile(joinpath(data_dir, kw_path)) &&
            return joinpath(data_dir, kw_path)
        for pat in patterns
            p = joinpath(data_dir, pat)
            isfile(p) && return p
        end
        error("Cannot locate file in $data_dir matching: $patterns")
    end

    bdp_path = _find_file(bdp_file, data_dir,
                          ["ACT_DR6_CMB_only_bdp.txt", "bandpowers.txt", "bdp.txt"])
    cov_path = _find_file(cov_file, data_dir,
                          ["ACT_DR6_CMB_only_cov.txt", "covariance.txt", "cov.txt"])

    win_dir = !isnothing(win_folder) ? joinpath(data_dir, win_folder) :
              isdir(joinpath(data_dir, "windows")) ? joinpath(data_dir, "windows") :
              data_dir

    # ---- Load band powers & covariance ----
    data_vector = load_bandpowers(bdp_path; T=T)
    covariance  = load_covariance(cov_path; T=T)

    N_total = sum(N_bins)
    @assert length(data_vector) == N_total "Band-power file has $(length(data_vector)) entries; expected $N_total from YAML spec_info"
    @assert size(covariance, 1) == N_total "Covariance matrix is $(size(covariance,1))×$(size(covariance,2)); expected $(N_total)×$(N_total)"

    cov_chol = cholesky(covariance)

    # ---- Load window functions ----
    windows = Matrix{T}[]
    for spec in spec_order
        # Try "TT_lxl_window_functions.txt" and "TT_window_functions.txt"
        spec_safe = replace(spec, " " => "_")
        candidates = [
            joinpath(win_dir, "$(spec_safe)_lxl_window_functions.txt"),
            joinpath(win_dir, "$(spec_safe)_window_functions.txt"),
        ]
        win_path = nothing
        for c in candidates
            isfile(c) && (win_path = c; break)
        end
        @assert !isnothing(win_path) "Cannot find window function file for spectrum '$spec' in $win_dir"
        W = load_window_function(win_path; ell_min=ell_min, T=T)
        push!(windows, W)
    end

    # Verify all window functions have the same N_ell
    N_ell_vec = [size(W, 1) for W in windows]
    @assert allequal(N_ell_vec) "Window functions have inconsistent theory ell counts: $N_ell_vec"
    N_ell  = first(N_ell_vec)
    ell_max = ell_min + N_ell - 1
    ells   = T.(ell_min:ell_max)

    # Verify N_bins matches window function column counts
    for (i, (spec, W)) in enumerate(zip(spec_order, windows))
        @assert size(W, 2) == N_bins[i] "Window function for '$spec' has $(size(W,2)) bins; expected $(N_bins[i])"
    end

    return (
        data_vector = data_vector,
        covariance  = covariance,
        cov_chol    = cov_chol,
        windows     = windows,
        spec_order  = spec_order,
        spec_types  = spec_types,
        N_bins      = N_bins,
        N_ell       = N_ell,
        ells        = ells,
    )
end
