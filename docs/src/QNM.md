# Quasinormal modes

The QNM interface combines a Leaver continued-fraction root with an independent
GSN-ISEM radial evaluation. Units are $M=1$.

## Compact interface

```julia
using GeneralizedSasakiNakamura

ordinary_mode = qnm(0.68, -2, 2, 2, 0)
mirror_mode = qnm(0.68, -2, 2, 2, 0, mirror)
```

The five numerical inputs are `(a, s, l, m, n)`. `ordinary` is the default
branch. `mirror` polishes the negative-real root and evaluates its amplitudes
and frequency derivative. The radial solver may use the exact conjugation
symmetry; the QNM result is not copied from the ordinary branch.

A successful compact result exposes:

```julia
ordinary_mode.omega
ordinary_mode.lambda
ordinary_mode.incidence_amplitude
ordinary_mode.reflection_amplitude
ordinary_mode.incidence_derivative
ordinary_mode.excitation_factor
```

For a QNM frequency $\omega_q$, define

```math
\alpha^{{\rm GSN}}_{\ell m\omega_q}
=\left.\frac{\partial A^{{\rm GSN},{\rm inc}}_{\ell m\omega}}
{\partial\omega}\right|_{\omega=\omega_q}.
```

The GSN excitation factor returned by `qnm` is

```math
B^{{\rm GSN}}_{\ell m\omega_q}
=\frac{A^{{\rm GSN},{\rm ref}}_{\ell m\omega_q}}
{2\omega_q\alpha^{{\rm GSN}}_{\ell m\omega_q}}.
```

The derivative is obtained from Richardson-extrapolated frequency stencils.
The incidence zero and derivative convergence are checked before acceptance.
Wronskian checks apply only to routes that compute a Wronskian.
Richardson convergence does not measure systematic radial-amplitude error;
passing these checks does not guarantee a total `1e-8` error in the derivative
or excitation factor.

## Root-only interface

Use `qnm_frequency` when amplitudes are not needed:

```julia
mode = QNMMode(-2, 2, 2, 0)
root = qnm_frequency(mode, 0.68)

sequence = qnm_sequence(mode, [0.0, 0.3, 0.68])
```

The default `convention=:overtone` treats `n` as a physical overtone label.
The continued-fraction inversion convention is available separately and
requires both an inversion index and a root guess:

```julia
leaver_root = qnm_frequency(
    mode, 0.68;
    convention=:leaver,
    inversion_index=root.inversion_index,
    guess=root.omega,
)
```

`convention=:complete_spectrum` is provided for the Schwarzschild
algebraically-special neighborhood, where the complete-spectrum ordering and
the usual overtone labels differ.

For example, at `a=0.7`, `s=-2`, `l=m=2`, the default `n=10` gives
`0.4219528033 - 1.8892581520im`. The root
`0.4198385711 - 2.0935596107im` has default label `n=11` and Leaver inversion
index 10. These labels differ because the default sequence retains both
branches in the algebraically-special neighborhood. Match conventions before
comparing high-overtone tables.

## Detailed radial solutions

The compact call avoids constructing callable radial functions. Request the
full bridge only when it is needed:

```julia
detailed = qnm(0.68, -2, 2, 2, 0; detailed=true)

r = 10.0
rstar = rstar_from_r(0.68, r)
detailed.X(rstar)
detailed.Y(r)
detailed.R(r)
```

`X` uses the tortoise coordinate $r_*$; `Y` and `R` use Boyer-Lindquist $r$.
The detailed normalization check compares GSN and Teukolsky conventions;
passing it is not an independent accuracy bound for either solution.

`detailed.teukolsky.excitation_factor` uses the convention

```math
B^{\mathrm{T}}_{\ell m\omega_q}
=\frac{(B^{\mathrm{ref}}/B^{\mathrm{trans}})_{\ell m\omega_q}}
{2i\omega_q\left.\partial_\omega
(B^{\mathrm{inc}}/B^{\mathrm{trans}})_{\ell m\omega}\right|_{\omega_q}}.
```

A reference defined with `2 omega` instead of `2 i omega` must therefore be
divided by `i` before comparison. This phase convention is distinct from a
radial normalization error.

## Result types

- `QNMResult`: all required root, radial, normalization, and derivative gates
  pass.
- `QNMEstimate`: finite diagnostic result whose limiting gate remains recorded.
- `QNMFailure`: the first failed stage and reason are returned without
  promoting nonfinite amplitudes.
- `QNMEndpointResult`: an exact extremal branch endpoint that is not treated as
  an ordinary isolated simple pole.

At the Schwarzschild algebraically-special frequency and at a synchronous
extremal accumulation endpoint, the ordinary simple-zero excitation formula
is not applied.

## Lower-level functions

```@docs
QNMMode
QNMBranch
QNMResult
QNMEstimate
QNMFailure
QNMEndpointResult
QNMPairResult
LeaverResult
ISEMValidationResult
ExcitationFactorResult
angular_A_to_lambda
lambda_to_angular_A
qnm
qnm_pair
qnm_frequency
qnm_sequence
validate_qnm_with_isem
qnm_excitation_factor
```
