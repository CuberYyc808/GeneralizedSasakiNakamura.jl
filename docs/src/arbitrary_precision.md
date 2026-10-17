# Arbitrary precision

The ISEM routes solve the homogeneous GSN/Teukolsky radial problem in the floating type of their
inputs. With `BigFloat` spin and frequency the IN and UP solutions, their asymptotic amplitudes and
the radial functions are computed to the working precision of `BigFloat`, for real and complex
frequencies; the Float64 inputs keep the Float64 routes unchanged.

```julia
using GeneralizedSasakiNakamura
setprecision(BigFloat, 256)
a = BigFloat("0.4"); omega = BigFloat("0.5")               # 4/10 and 1/2 to 256 bits
Rin, Rup = Teukolsky_radial(-2, 2, 2, a, omega)             # unit Teukolsky transmission
Rin.incidence_amplitude                                       # Complex{BigFloat}, ~74 digits
R, dRdr, d2Rdr2, err = Rin.Teukolsky_solution(BigFloat("3.5"))
Xin = GSN_radial(-2, 2, 2, a, omega, IN)                     # unit GSN transmission
Xin(BigFloat(10))                                            # (X, dX/dr*, 0) at r* = 10
Rc = Teukolsky_radial(-2, 2, 2, a, Complex{BigFloat}(BigFloat("0.5"), BigFloat("-0.3")), IN)
```

A binary64 value among the inputs is used exactly: `0.4` is the binary double closest to 4/10,
`BigFloat("0.4")` is 4/10 to the current precision. The separation constant is taken from
`SpinWeightedSpheroidalHarmonics` at the same precision, which requires a version of that package
that accepts `BigFloat` at complex `c` (v1.4.1 or later). `data_type=Complex{BigFloat}` with
`method="auto"` or `"ISEM"` promotes Float64 inputs to `BigFloat`. Quasinormal frequencies and the
frequency derivative of the incidence amplitude at arbitrary precision are provided by
`QNM.qnm_arbitrary_precision` (see [Quasinormal modes](QNM.md)).

## What is computed and how

The route is the ISEM construction in the compact coordinate ``x = (r - r_+)/(r - r_-)``: the
horizon Frobenius series, Taylor patches along the real axis, and a representation of the two
solutions at infinity, matched at an interior point. All series are the exact generated coefficient
tables of the Float64 routes, evaluated in the working type.

- **Tolerance and orders.** The relative truncation tolerance of every series is 64 ulps of the
  inputs' precision unless `tol` is given; the Taylor order of the patches is 2.2 per requested
  decimal digit (`N` overrides it), the horizon series uses twice and the asymptotic series three
  times that order. The patch step is at most half the empirical radius of the solution series
  (ratio of its last coefficients) and is shrunk until the last four terms are below the tolerance.
- **Working precision.** The BigFloat precision is raised internally by 24 guard bits above the bits
  needed for the tolerance, plus the continuation loss below when the asymptotic seed is used at a
  complex frequency. The returned amplitudes and the stored series keep that precision.
- **Real frequencies.** The infinity side is the asymptotic series ``e^{\sigma/y} y^{\rho} \sum d_n y^n``
  (``y = 1 - x``), optimally truncated at the largest ``y`` where a term falls below the tolerance,
  and continued inward by Taylor patches. At 256 bits (77 digits) the IN and UP amplitude ratios of
  ``a = 0.4, s = -2, l = m = 2, \omega = 1/2`` agree with the Black Hole Perturbation Toolkit
  Teukolsky paclet to 74 digits.
- **Complex frequencies.** The asymptotic series cannot define a single branch at infinity when
  ``\operatorname{Im}\omega \ne 0``: the truncated series leaves an admixture of the recessive
  solution of relative size ``e^{-2|\omega| r_0}`` at the seed radius ``r_0``, while the recessive
  solution is ``e^{-2|\operatorname{Im}\omega| r_0}`` smaller than the dominant one, so the branch
  coefficient is wrong by ``e^{-2(|\omega| - |\operatorname{Im}\omega|) r_0}``; continuing the
  inward-decaying branch from ``r_0`` to the matching radius ``r_m`` then loses a factor
  ``e^{2|\operatorname{Im}\omega|(r_0 - r_m)}``. With ``r_0 \sim \ln(1/\mathrm{tol})/(2|\omega|)``
  the result is only ``\mathrm{tol}^{1 - |\operatorname{Im}\omega|/|\omega|}`` accurate, and nothing
  helps on the negative imaginary axis. The route therefore uses the Mano-Suzuki-Takasugi (MST)
  series for the UP and DOWN solutions at complex frequency: the renormalized angular momentum is
  solved in the working type from the Float64 value, the series coefficients come from the
  continued fractions, the Tricomi functions are built from Kummer's series at a precision raised by
  ``|z|/\ln 2`` bits to absorb the cancellation, and the two Teukolsky solutions are converted to GSN
  states at the matching point, which is placed where ``|2\omega(r - r_-)| \approx 18``. At 256 bits
  the IN and UP amplitude ratios at ``\omega = 1/2 - 0.3i`` agree with the paclet to 67 and 76
  digits; at ``2 - 2i`` to 46-72 digits, where the paclet's own 120-digit run is the limit.
- **Negative real part.** The principal-branch MST formulas are not on the physical sheet for
  ``\operatorname{Re}\omega < 0``; those frequencies are solved for the mirror mode
  ``(m, \omega) \to (-m, -\bar\omega)`` and complex-conjugated, as the Float64 routes do for real
  negative frequencies.
- **Not covered.** The large-frequency and near-extremal regimes of the Float64 routes rest on
  binary64 tables and are not used; the Taylor patches cover ``|\omega| \le 10`` at the cost of more
  patches. The superradiant threshold (logarithmic horizon companion) is not built for the UP route.

## Precision bookkeeping

The acceptance tests in `test/arbitrary_precision` compare every stored paclet case at the requested
working bits and print, per quantity, the digits agreed, the digits expected (``N\log_{10}2 - 6``)
and the certified accuracy of the reference (the agreement between two paclet working precisions).
Reported precision is always the agreement with that independent reference, never a
self-consistency figure between tolerances.
