# Manually run checks

Nothing in this directory is part of `Pkg.test()`.

These scripts validate the **foreground-marginalized CMB-only** ACT DR6 variant
against `candl`, and they need loose local data under
`data/ACT_DR6_CMB_only/` that is not distributed with the package and is not
part of the published runtime artifact.

They are kept for auditing. They are deliberately excluded from the automated
suite, because the automated suite must never depend on assets whose absence
would turn a scientific test into a skip.

Run, from the repository root, with the local CMB-only data present:

    julia --project=. test/manual/cmb_only_local_parity.jl

The full multifrequency likelihood — the authoritative ACT DR6 likelihood in
this package — is validated unconditionally by `Pkg.test()` through the
published artifact, and never by these scripts.
