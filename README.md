# ACTLikelihoods.jl

Native Julia implementation of the **ACT DR6 full multifrequency TT/TE/ET/EE
likelihood**, with an explicit foreground model, prepared reverse-mode automatic
differentiation, and runtime data distributed as an immutable published
artifact.

---

## Which likelihood is this?

ACT DR6 ships two different likelihood products. They are not interchangeable.

| | **full multifrequency** (this package's primary product) | CMB-only |
|---|---|---|
| type | non-marginalized | **foreground-marginalized** |
| data | 1651 bandpowers, 41 ordered TT/TE/ET/EE spectra, 5 array-frequency channels | 135 bandpowers, 3 spectra |
| foregrounds | modelled explicitly (tSZ, kSZ, CIB, radio, dust, tSZ×CIB) | already marginalized out |
| nuisance parameters | 29 free | 2 calibrations |
| Julia type | [`ACTDR6FullLikelihood`](#quickstart) | `ACTDR6Likelihood` |
| runtime data | published Zenodo artifact, downloaded automatically | loose local files |

`ACTDR6FullLikelihood` is the authoritative ACT DR6 likelihood in this package
and the one that is validated against the original code. The
foreground-marginalized `ACTDR6Likelihood` remains available for convenience,
but it is a compressed product: **a result obtained with the
foreground-marginalized likelihood is not evidence about the full
multifrequency likelihood**, and this package never uses it as such.

---

## Installation

The package is not in the General registry yet; install it by URL.

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/ACTLikelihoods.jl")
```

### Julia 1.10 versus 1.11+

`ACTLikelihoods.jl` depends on `CMBForegrounds.jl` **v0.4.1**, which is also
distributed by URL rather than through the General registry. The two supported
Julia versions handle that differently:

* **Julia 1.11 and 1.12** understand the `[sources]` section of `Project.toml`
  and resolve the `CMBForegrounds` URL and tag **recursively**. Nothing extra is
  needed:

  ```julia
  Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/ACTLikelihoods.jl")
  ```

* **Julia 1.10** ignores `[sources]`. Install the pinned dependency explicitly
  **before** `ACTLikelihoods.jl`:

  ```julia
  using Pkg
  Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl",
          rev="v0.4.1")
  Pkg.add(url="https://github.com/JuliaCosmologicalLikelihoods/ACTLikelihoods.jl")
  ```

Supported: Julia 1.10, 1.11 and 1.12.

---

## Quickstart

Everything the likelihood needs at runtime lives in the published artifact, so
the ordinary path touches no local directory. The artifact is downloaded and
unpacked on first use (38 MB compressed, 474 MB unpacked).

The package deliberately loads **no** default cosmology, Boltzmann solver or
emulator: you supply lensed `D_ℓ = ℓ(ℓ+1)C_ℓ/2π` in μK² on a contiguous integer
multipole grid that covers `2:8501`. Below, `example_cmb_spectra()` stands in
for whatever produces yours.

<!-- runnable -->
```julia
using ACTLikelihoods

like  = ACTDR6FullLikelihood()             # artifact-backed, metadata validated
model = ACTDR6FullForegroundModel(like)    # bandpasses, chromatic beams, templates
p     = ACTDR6Nuisance()                   # the validated reference point

cmb = example_cmb_spectra()                # ACTCMBTheory(ell, Dl_TT, Dl_TE, Dl_EE)

fg         = foregrounds(model, like, p)
prediction = predict(like, cmb, fg, p)

chi2_value = chi2(like, prediction)              # 1592.0584129991541
logl       = loglikelihood(like, prediction)     # data only: -chi2/2
logp       = logposterior(like, prediction, p)   # + official ACT DR6 priors

@assert isapprox(chi2_value, __expected_chi2; rtol=1e-10)
@assert isapprox(logl, -chi2_value / 2)
```

Building your own CMB input:

```julia
ell = collect(2:9000)
cmb = ACTCMBTheory(ell, Dl_TT, Dl_TE, Dl_EE)   # μK², same length as ell
```

Explicit directory constructors remain available for development and
validation:

```julia
like  = ACTDR6FullLikelihood("/path/to/extracted/artifact")
model = ACTDR6FullForegroundModel("/path/to/extracted/artifact", like)
```

---

## Likelihood conventions

### Bandpower windows

Every ACT DR6 bandpower window is exactly zero outside one contiguous run of
multipoles: 97.9% of the released `(8500, n_bin)` matrices are structural zeros,
and the nonzero run averages 174 multipoles. The likelihood therefore stores the
runs, not the matrices — 2.2 MiB in place of 107 MiB — and the projection reads
only the multipoles a window actually covers.

This is exact rather than a truncation: the run is taken from the first to the
last nonzero entry of each released column, the test suite compares the packed
form against the released window files for **equality**, and it asserts the
contiguity the packing relies on rather than assuming it. Reconstruct the
released dense form at any time:

```julia
window_matrix(like.spectra[1], length(like.ells))   # (8500, n_bin), as released
```

What does change is the order in which each bandpower sum is accumulated, so
values move in their last digits: `chi2` is `1592.0584129991541` where the dense
matrix product gave `1592.0584129991526`, a relative shift of 1e-15. Both
reproduce the frozen upstream export, `1592.0584129991614`, to 5e-15.

The bandpower sums are vectorized, so their last one or two digits also depend
on the vector width the compiler picks: the same code returns
`1592.0584129991553` on Julia 1.10 and `1592.0584129991541` on 1.11 and 1.12.
Quoted values in this README are from Julia 1.12. This is why the parity tests
assert `rtol=1e-10` and not bit equality — it was equally true of the BLAS matrix
product this replaced, whose kernel choice varies with the CPU.

### `loglikelihood` is data only

```
loglikelihood(like, prediction) = -chi2(like, prediction) / 2
```

No priors, no normalization. This is intentional: the value depends on nothing
but the data, the covariance and the model vector.

### The fixed Gaussian normalization is separate

```
gaussian_normalization(like) = -N/2 · log(2π) - ½ · logdet C
```

It is computed from the stored Cholesky factor, not copied from upstream. At the
reference point:

| quantity | value |
|---|---|
| `chi2(like, prediction)` | `1592.0584129991541` |
| `loglikelihood(like, prediction)` | `-796.0292064995771` |
| `gaussian_normalization(like)` | `-2145.1713776754927` |
| upstream `act_dr6_mflike` `loglike` | `-2941.2005841750693` |

So the upstream convention is recovered as

<!-- runnable -->
```julia
like  = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
p     = ACTDR6Nuisance()
pred  = predict(like, example_cmb_spectra(), foregrounds(model, like, p), p)

upstream = loglikelihood(like, pred) + gaussian_normalization(like)
@assert isapprox(upstream, -2941.2005841750693; rtol=1e-12)
```

### Ordered TE/ET map legs

The released likelihood is built with `symmetrize: false`. A `TE` spectrum with
`T` on channel *a* and `E` on channel *b* is a different observable from the one
with the legs exchanged, and both appear in the 1651-element vector. This
package preserves that ordering exactly and **never** symmetrizes.

---

## Nuisance parameters

`ACTDR6Nuisance` is a statically typed container for the **29** free baseline
parameters. `parameter_vector(p)` produces them in the documented order below
and `ACTDR6Nuisance(x)` inverts it exactly.

| # | parameter | meaning | prior |
|---|---|---|---|
| 1 | `a_tSZ` | thermal SZ amplitude at 150 GHz, ℓ=3000 | U(0, 10) |
| 2 | `alpha_tSZ` | tSZ frequency-scaling tilt | U(−5, 5) |
| 3 | `a_kSZ` | kinetic SZ amplitude | U(0, 10) |
| 4 | `a_p` | Poisson CIB amplitude | U(0, 50) |
| 5 | `beta_p` | Poisson CIB spectral index | U(0, 5) |
| 6 | `a_c` | clustered CIB amplitude | U(0, 50) |
| 7 | `a_s` | radio point-source amplitude | U(0, 50) |
| 8 | `beta_s` | radio spectral index | U(−3.5, −1.5) |
| 9 | `a_gtt` | galactic dust amplitude, TT | U(0, 50) **and** N(7.95, 0.32) |
| 10 | `a_gte` | galactic dust amplitude, TE | U(0, 1) **and** N(0.423, 0.03) |
| 11 | `a_gee` | galactic dust amplitude, EE | U(0, 1) **and** N(0.1681, 0.017) |
| 12 | `a_psee` | EE point-source amplitude | U(0, 1) |
| 13 | `a_pste` | TE point-source amplitude | U(−1, 1) |
| 14 | `xi` | tSZ×CIB correlation | U(0, 0.2) |
| 15 | `calG_all` | global calibration | N(1.0, 0.003) |
| 16 | `cal_dr6_pa4_f220` | pa4 f220 map calibration | N(1.0, 0.013) |
| 17 | `cal_dr6_pa5_f090` | pa5 f090 map calibration | N(1.0, 0.0016) |
| 18 | `cal_dr6_pa5_f150` | pa5 f150 map calibration | N(1.0, 0.0020) |
| 19 | `cal_dr6_pa6_f090` | pa6 f090 map calibration | N(1.0, 0.0018) |
| 20 | `cal_dr6_pa6_f150` | pa6 f150 map calibration | N(1.0, 0.0024) |
| 21 | `calE_dr6_pa5_f090` | pa5 f090 polarization efficiency | U(0.9, 1.1) |
| 22 | `calE_dr6_pa5_f150` | pa5 f150 polarization efficiency | U(0.9, 1.1) |
| 23 | `calE_dr6_pa6_f090` | pa6 f090 polarization efficiency | U(0.9, 1.1) |
| 24 | `calE_dr6_pa6_f150` | pa6 f150 polarization efficiency | U(0.9, 1.1) |
| 25 | `bandint_shift_dr6_pa4_f220` | bandpass shift [GHz] | N(0.0, 3.6) |
| 26 | `bandint_shift_dr6_pa5_f090` | bandpass shift [GHz] | N(0.0, 1.0) |
| 27 | `bandint_shift_dr6_pa5_f150` | bandpass shift [GHz] | N(0.0, 1.3) |
| 28 | `bandint_shift_dr6_pa6_f090` | bandpass shift [GHz] | N(0.0, 1.2) |
| 29 | `bandint_shift_dr6_pa6_f150` | bandpass shift [GHz] | N(0.0, 1.1) |

Three parameters carry both a uniform range and a normal prior, exactly as the
official run configuration combines its sampled ranges with the external
`TTdust_prior` / `TEdust_prior` / `EEdust_prior` entries.

### Derived and fixed values

`beta_c` is **derived**: the ACT DR6 baseline sets `beta_c = beta_p`, and
`NamedTuple(p)` supplies it. It is not a free parameter.

The following are **fixed model constants**, kept strictly apart from the free
set in `ACT_DR6_FIXED_PARAMETERS`:

| constant | value | note |
|---|---|---|
| `T_d` | 9.60 K | dust temperature for the CIB components |
| `T_effd` | 19.6 K | effective galactic-dust temperature |
| `beta_d` | 1.5 | galactic-dust spectral index |
| `alpha_dT` | −0.6 | galactic-dust TT multipole slope |
| `alpha_dE` | −0.4 | galactic-dust E-mode multipole slope |
| `alpha_p` | 1.0 | Poisson CIB multipole slope |
| `alpha_s` | 1.0 | radio multipole slope |
| `calE_dr6_pa4_f220` | 1.0 | ACT DR6 uses no pa4 f220 polarization channel |

> `calE_dr6_pa4_f220` is fixed in the official configuration and has **no** path
> into the model: pa4 f220 appears only in TT spectra. The test suite asserts
> that perturbing it leaves the 1651-element prediction bit-identical, rather
> than taking the YAML's word for it.

### Priors and the posterior

<!-- runnable -->
```julia
like  = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
p     = ACTDR6Nuisance()
pred  = predict(like, example_cmb_spectra(), foregrounds(model, like, p), p)

logprior(p)                     # normalized log prior density
prior_chi2(p)                   # Σ ((x-μ)/σ)² over the normal priors only
logposterior(like, pred, p)     # loglikelihood + logprior, still no constant

@assert isapprox(logposterior(like, pred, p),
                 loglikelihood(like, pred) + logprior(p))
```

`logprior` returns `-Inf` outside any uniform range; `prior_chi2` returns `Inf`
there. A non-finite parameter — `NaN`, `Inf` or `-Inf` — is outside the support
by the same rule, so a bad proposal is rejected at the prior rather than turning
into a `NaN` further down. Both also accept the 29-element parameter vector
directly.

### Building a nuisance point

<!-- runnable -->
```julia
p = ACTDR6Nuisance()                          # validated reference point
q = ACTDR6Nuisance(p; a_tSZ = 4.0, xi = 0.1)  # copy with overrides
r = ACTDR6Nuisance((a_tSZ = 4.0,))            # by name, rest from the reference
s = ACTDR6Nuisance(Dict("a_tSZ" => 4.0))      # Symbol or String keys

x = parameter_vector(p)                       # 29 values, documented order
@assert ACTDR6Nuisance(x) == p                # exact round trip
@assert ACT_DR6_FREE_PARAMETERS[1] === :a_tSZ
@assert length(ACT_DR6_FREE_PARAMETERS) == 29
@assert NamedTuple(q).beta_c == q.beta_p      # derived
```

Every constructor validates the names that carry no freedom rather than dropping
them: an unknown name, a fixed constant given a value other than its official
one, or a `beta_c` that contradicts the effective `beta_p` is an `ArgumentError`,
including in the copy-with-overrides form.

### What the model requires of a parameter container

`foregrounds` and `predict` also accept a plain `NamedTuple` or `AbstractDict`
(`Symbol` or `String` keys, resolved identically). Such a container must be
**complete**: every foreground parameter, `calG_all`, every `cal_<channel>`, and
the polarization efficiency `calE_<channel>` for each channel with a
polarization leg. A missing name is an `ArgumentError`, never a silent default,
so an incomplete container cannot produce a quietly wrong model vector. The one
implicit quantity is the temperature efficiency `calT ≡ 1`, which ACT DR6 fixes
and which has no parameter at all.

`NamedTuple(::ACTDR6Nuisance)` always produces a complete container, so the
ordinary path never has to think about this.

---

## Foreground model

Foregrounds are assembled as `(channel, channel, ℓ)` arrays of `D_ℓ` in μK²,
following `LAT_MFLike`/`fgspectra`, and are added to the CMB **before** map
calibration and window convolution. With the pivot frequency `ν₀ = 150 GHz` and
pivot multipole `ℓ₀ = 3000`:

**Temperature (TT)**

```
D^TT_ij(ℓ) = a_kSZ  f^kSZ_i  f^kSZ_j  T_kSZ(ℓ)
           + a_p    f^CIB_i(β_p) f^CIB_j(β_p) (ℓ(ℓ+1) / ℓ₀(ℓ₀+1))^α_p
           + a_s    f^rad_i(β_s) f^rad_j(β_s) (ℓ(ℓ+1) / ℓ₀(ℓ₀+1))^α_s
           + a_gtt  f^dust_i(β_d) f^dust_j(β_d) (ℓ / 500)^α_dT
           + a_tSZ  g_i(α_tSZ) g_j(α_tSZ) T_tSZ(ℓ)
           + a_c    f^CIB_i(β_c) f^CIB_j(β_c) T_CIB(ℓ)
           − ξ √(a_tSZ a_c) [ g_i f^CIB_j + g_j f^CIB_i ] T_tSZ×CIB(ℓ)
```

**Temperature–polarization (TE), ordered legs**

```
D^TE_ij(ℓ) = a_pste f^rad,T_i(β_s) f^rad,P_j(β_s) (ℓ(ℓ+1) / ℓ₀(ℓ₀+1))^α_s
           + a_gte  f^dust,T_i(β_d) f^dust,P_j(β_d) (ℓ / 500)^α_dE
```

**Polarization (EE)**

```
D^EE_ij(ℓ) = a_psee f^rad,P_i(β_s) f^rad,P_j(β_s) (ℓ(ℓ+1) / ℓ₀(ℓ₀+1))^α_s
           + a_gee  f^dust,P_i(β_d) f^dust,P_j(β_d) (ℓ / 500)^α_dE
```

The TE block is **not** symmetric in `(i, j)`: the first index is the
temperature leg and the second the polarization leg.

### Units and frequency response

* All `D_ℓ` are in μK² (CMB thermodynamic temperature).
* `T_kSZ`, `T_tSZ`, `T_CIB` and `T_tSZ×CIB` are the four released templates,
  normalized to 1 at `ℓ₀ = 3000`.
* Bandpass shifts `Δν` are in GHz and shift the transmission grid,
  `ν → ν + Δν`, with the normalization recomputed at the shifted frequencies.
* Each frequency response `f_i(ℓ)` is the **chromatic** band integral, so it
  depends on multipole through the frequency-dependent beam:

  ```
  f_i(ℓ) = ∫dν  b_ℓ(ν) τ_i(ν+Δν) (∂B/∂T)(ν+Δν) S(ν+Δν)
         / ∫dν  b_ℓ(ν) τ_i(ν+Δν) (∂B/∂T)(ν+Δν)
  ```

  Following upstream, the beam is evaluated on the **unshifted** frequency grid;
  the bandpass shift is not propagated into `b_ℓ(ν)`.

### Calibration

Applied at map level, after the foregrounds are added:

```
D^cal,XY_ij(ℓ) = D^XY_ij(ℓ) / ( cal_G² · cal_i · cal_j · cal^X_i · cal^Y_j )
```

with `cal^T ≡ 1` (ACT DR6 varies no `calT`) and `cal^E = calE`.

### Fixed-beam chromatic bandpass

The ACT DR6 chromatic beams are measured instrument calibration products, never
inference parameters. Foreground assembly therefore uses the fixed-beam pair
exported by `CMBForegrounds` v0.4.1:

```julia
prepare_fixed_chromatic_bandpass(band, beam)
eval_fixed_chromatic_sed_bands(sed_function, prepared)
```

These are numerically identical to `prepare_chromatic_bandpass` /
`eval_chromatic_sed_bands`. They differ only in reverse mode: the active-beam
route returns a dense cotangent the size of the beam matrix — `(8500 × ~600)`
per channel, allocated once per SED per channel in every reverse pass — while
the fixed route returns `NoTangent()` for the beam. Derivatives with respect to
the bandpass shifts are preserved either way, because the bands are shifted and
renormalized by `shift_and_normalize` *inside* the differentiated call.

This package uses no private `CMBForegrounds` symbols; a test scans `src/`,
`test/` and this README to keep it that way.

---

## Automatic differentiation

Reverse mode is a primary requirement. The complete public path — nuisance
container, bandpass shifts, chromatic beams, SEDs, angular templates,
calibration, window convolution and the Cholesky solve — is differentiable, and
the test suite checks prepared Mooncake against ForwardDiff and against finite
differences at several points.

<!-- runnable -->
```julia
using ADTypes, DifferentiationInterface, Mooncake

like  = ACTDR6FullLikelihood()
model = ACTDR6FullForegroundModel(like)
cmb   = example_cmb_spectra()

function objective(x)
    p  = ACTDR6Nuisance(x)
    fg = foregrounds(model, like, p)
    return loglikelihood(like, predict(like, cmb, fg, p))
end

x = parameter_vector(ACTDR6Nuisance())

backend = AutoMooncake(; config=nothing)
prep    = prepare_gradient(objective, backend, x)
g       = gradient(objective, prep, backend, x)

@assert length(g) == 29
@assert all(isfinite, g)
```

The prepared cache is reusable at a different parameter point — reuse it rather
than re-preparing inside a sampler loop.

### Custom reverse-mode rules

Two steps carry hand-written pullbacks, registered with Mooncake through
`@from_chainrules`, on top of the rules `CMBForegrounds` provides for its own
kernels:

| primitive | why it is one |
|---|---|
| the fixed covariance solve | the Cholesky factor is released data, so it takes `NoTangent()` and only the residual receives a cotangent |
| the theory-to-observation projection | the whole CMB + foreground + calibration + window step is **one** tape entry |

The second matters more than it looks. Written as ordinary code the projection
creates, per ordered spectrum, a length-8500 sum, a broadcast and a window
contraction — 41 intermediates whose cotangents reverse mode must allocate and
zero on every gradient. As a single primitive, cotangents exist only for what a
caller can actually vary: the three CMB spectra, the three foreground blocks and
the 41 calibration scalars. That, with the packed windows, takes the reverse pass
over `predict` from 31 ms to 6.6 ms.

Derivatives flow to the CMB spectra as well as to the nuisance parameters, which
is what an emulator-based cosmology gradient needs; the test suite checks the
pullback against the exact adjoint of the forward map with arbitrary cotangents.

---

## Benchmarks

Measured with `BenchmarkTools`, `$`-interpolated, and `evals=1` wherever a
stateful prepared cache is involved. Reproduce with:

```
julia --project=benchmark benchmark/benchmarks.jl
julia --project=benchmark benchmark/cold_start.jl
```

| operation | median | mean | allocations | memory |
|---|---|---|---|---|
| artifact-backed likelihood construction | 3.92 s | 3.87 s | 50,533,614 | 2.05 GiB |
| foreground-model construction | 1.37 s | 1.34 s | 15,150,400 | 669.90 MiB |
| foreground assembly | 9.6 ms | 12.9 ms | 860 | 16.58 MiB |
| `predict` | 509.1 µs | 558.0 µs | 246 | 137.13 KiB |
| `chi2` | 223.0 µs | 221.9 µs | 6 | 26.02 KiB |
| combined forward likelihood | 10.3 ms | 13.6 ms | 1,114 | 16.74 MiB |
| ForwardDiff preparation | 1.4 µs | 7.5 µs | 8 | 4.29 KiB |
| ForwardDiff prepared gradient (hot) | 356.3 ms | 316.6 ms | 4,734 | 542.64 MiB |
| Mooncake preparation | 101.2 ms | 370.4 ms | 17,178 | 143.39 MiB |
| **Mooncake prepared gradient (hot)** | **44.7 ms** | 64.5 ms | 6,303 | 56.33 MiB |

Julia 1.12.6, Intel Alder Lake, 20 threads available, `CMBForegrounds v0.4.1`.

Allocation counts and memory are the reproducible quantities here and are stable
to the byte across runs; the wall-clock column was measured on a shared machine
under concurrent load, so treat it as an upper bound rather than a best case.

A prepared reverse-mode gradient of all 29 free parameters costs roughly four
forward evaluations, which is the number that matters for sampling.

### Where the time goes

`predict` is dominated by the bandpower windows and foreground assembly by the
chromatic beams, and both are fixed data streamed from memory rather than
arithmetic. The windows are packed (see [Bandpower windows](#bandpower-windows)),
which is what takes `predict` from 4.9 ms to 509 µs and its reverse pass from
31 ms to 6.6 ms.

Foreground assembly is now the larger term. Its 8 SED-weight evaluations each
multiply the *same* per-channel chromatic beam matrix, so the 38.4 MiB of beam
tables is streamed eight times per call; batching them into one matrix product
per channel measures 2.3× faster in isolation, but it needs a batched entry
point in `CMBForegrounds` and is not done here.

Construction is a one-off: the two construction rows are dominated by parsing
the 287 MB of window text and the 67 MB covariance, and at about 5 s combined
they do not justify republishing the artifact in a binary format. If that ever
becomes a real constraint, the right move is a **v2 artifact** with the
covariance, windows and beams stored as typed binary — not a silent format
change to the published one.

### Cold start

Reported separately, because it is a one-off, network- and I/O-bound step and
not a BenchmarkTools result. Measured in a brand-new empty `JULIA_DEPOT_PATH`
with `benchmark/cold_start.jl`:

| step | wall clock |
|---|---|
| `Pkg.instantiate()`, incl. registry, dependencies and the 36.2 MiB artifact download and unpack | 113.0 s |
| package load | 0.1 s |
| artifact path resolution | < 0.01 s |
| first likelihood construction | 5.8 s |
| first foreground-model construction | 1.7 s |
| **cold depot → usable model, after instantiate** | **7.6 s** |

The artifact unpacks into `<depot>/artifacts/a91a5b3cbb9be682b1ee442b60cae5701f7e84a9`
— the directory name is the content tree hash, which is what makes a stale or
corrupt cache impossible to use by accident.

---

## Provenance

### Published runtime artifact

| | |
|---|---|
| artifact name | `ACT_DR6_TTTEEE_v1` |
| Zenodo record | <https://zenodo.org/records/22821597> |
| DOI | `10.5281/zenodo.22821597` |
| archive | `ACT_DR6_TTTEEE_v1_20260917.tar.xz` (37,929,552 bytes) |
| archive SHA-256 | `904418cec753af4ecb197861f0eebbccfe41ca3bce6c68ea53b6e3e05a6eae6a` |
| Julia tree hash | `a91a5b3cbb9be682b1ee442b60cae5701f7e84a9` |
| unpacked size | 474 MB |

Contents: `data_vec.txt`, `cov.txt`, `spectrum_metadata.tsv`, 35 unique window
files serving the 41 ordered spectra, 10 temperature/polarization bandpasses, 10
chromatic beam tables, four foreground templates, `metadata.json` and a
`SHA256SUMS` ledger of 63 entries.

The artifact is immutable. `metadata.json` is validated on every likelihood
construction — artifact identity, `marginalized == false`, dimensions, channel
order, multipole range, the source SACC digest and the upstream revisions. The
full `SHA256SUMS` ledger is verified by the test suite rather than on every
construction.

`act_dr6_provenance()` returns all of this at runtime.

### Upstream sources

| source | revision |
|---|---|
| ACT DR6 SACC release `dr6_data.fits` (SHA-256) | `eca996ddb1fc57750299bf5757a40f5d178c1a8b3446d3341a99ca5ba7378e8b` |
| [`act_dr6_mflike`](https://github.com/ACTCollaboration/act_dr6_mflike) | `fc63c8c40533cea0bbec91677208de285c50fc10` |
| [`LAT_MFLike`](https://github.com/simonsobs/LAT_MFLike) | `666b5580c6567d415de31caf51c1e16cca5133c8` |
| [`fgspectra`](https://github.com/simonsobs/fgspectra) | `4c4d29c448aea0b8bb0162a328b89be24b9ae729` |
| [`CMBForegrounds.jl`](https://github.com/JuliaCosmologicalLikelihoods/CMBForegrounds.jl) | `v0.4.1` |

The reference fixtures under `validation/fixtures/` are generated from those
exact revisions using the **unmodified** official `act_dr6.yaml` configuration —
no overridden scale cuts, no overridden polarization selection,
`symmetrize: false` as released. Regenerate and verify with:

```
python3 validation/export_reference_act.py \
    --sacc-file data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits \
    --cmb-directory validation/fixtures/act_dr6_cmb_theory \
    --output validation/fixtures/act_dr6_full_reference \
    --act-source ../act_dr6_mflike

python3 validation/export_reference_act_multipoint.py \
    --sacc-file data/ACT_DR6_TTTEEE/v1.0/dr6_data.fits \
    --cmb-directory validation/fixtures/act_dr6_cmb_theory \
    --output validation/fixtures/act_dr6_full_multipoint
```

Append `--check` to either command to verify that a fresh run reproduces the
checked-in text fixtures byte-for-byte instead of rewriting them. The fixtures
are plain text; no NumPy binaries and no bulk data are committed.

---

## Validation

`Pkg.test()` is unconditional and always exercises the published artifact. It
covers the artifact download and metadata, the full `SHA256SUMS` ledger, the
data model and ordered spectrum metadata, every foreground component against
independently generated original-code checkpoints, whole-array moments, SED and
bandpass/beam evaluations, baseline parity, three further nuisance points, the
nuisance container and priors, the custom pullbacks, and the complete AD battery.

There are no `isdir(...)` skips: a missing artifact, a malformed artifact, a
missing fixture or a failed extension fails the suite.

---

## Citation

If you use this likelihood, cite the ACT DR6 papers:

* T. Louis et al. (ACT Collaboration), *The Atacama Cosmology Telescope: DR6
  Power Spectra, Likelihoods and ΛCDM Parameters*, 2025,
  [arXiv:2503.14452](https://arxiv.org/abs/2503.14452) (accepted, JCAP).
* E. Calabrese et al. (ACT Collaboration), *The Atacama Cosmology Telescope: DR6
  Constraints on Extended Cosmological Models*, 2025,
  [arXiv:2503.14454](https://arxiv.org/abs/2503.14454) (accepted, JCAP).
* B. Beringue et al. (ACT Collaboration), *The Atacama Cosmology Telescope: DR6
  Power Spectrum Foreground Model and Validation*, JCAP **2025**, 10, 082,
  [doi:10.1088/1475-7516/2025/10/082](https://doi.org/10.1088/1475-7516/2025/10/082),
  [arXiv:2506.06274](https://arxiv.org/abs/2506.06274).

and the runtime artifact:

* M. Bonici, *ACT DR6 full multifrequency TT/TE/EE likelihood runtime data for
  ACTLikelihood.jl*, Zenodo, 2026,
  [doi:10.5281/zenodo.22821597](https://doi.org/10.5281/zenodo.22821597).

The implementation follows the official `act_dr6_mflike`, `LAT_MFLike` and
`fgspectra` codes at the revisions listed above.

---

## License

MIT, as in [`LICENSE`](LICENSE). The ACT DR6 data products distributed through
the Zenodo runtime artifact carry their own terms and citation requirements from
the ACT Collaboration; see [Provenance](#provenance).
