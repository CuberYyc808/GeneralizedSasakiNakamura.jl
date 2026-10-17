# Arbitrary-precision acceptance tests

Opt-in tests of the arbitrary-precision ISEM route (BigFloat spins and frequencies) against the Black
Hole Perturbation Toolkit Teukolsky paclet (MST method). They are not part of `test/runtests.jl`.

## References

`references/bhpt_<prec>_<group>.txt` hold the paclet output of `bhpt_acceptance_reference.wls`
(`wolframscript -file bhpt_acceptance_reference.wls <workingprecision> <group>`, Teukolsky paclet
1.1.1), one record per line, `tag|key|value`, with Mathematica's `InputForm` digits and precision
marks. For every case the normalization-independent quantities are stored: the separation constant,
the renormalized angular momentum, `Incidence/Transmission`, `Reflection/Transmission`,
`R(rprime)/Transmission` and `dR(rprime)/Transmission` of the In and Up solutions at
`rprime = 3.1126842514806725` (the binary double).

Each group exists at two working precisions (60 and 120 digits; 200 and 400 for `omega = 10`, where
the paclet needs the higher precision). The digits on which the two agree are the certified accuracy
of the reference; the tests require agreement with the higher-precision run to
`min(N log10(2) - 6, certified digits - 1)` at `N` working bits.

## Running

```sh
GSN_AP_BITS=113,256 julia --project=<env> test/arbitrary_precision/runtests.jl
GSN_AP_BITS=53,113,192,256 GSN_AP_CASES=all julia --project=<env> test/arbitrary_precision/runtests.jl
```

`GSN_AP_CASES` is `quick` (default), `real`, `complex`, `axis` or `all`; `GSN_SOURCE=<checkout>` loads the
package by `include` as in `test/extended`. The environment must provide a
`SpinWeightedSpheroidalHarmonics` that accepts BigFloat at complex `c` (the `ArbitraryPrecision`
branch, v1.4.1 or later); the registered 1.4.0 rejects it.

Frequencies with negative real part are not in the references on purpose: the principal-branch MST
formulas are not on the physical sheet there, and the package obtains that half plane from the
symmetry `(m, omega) -> (-m, -conj(omega))` with complex conjugation.
