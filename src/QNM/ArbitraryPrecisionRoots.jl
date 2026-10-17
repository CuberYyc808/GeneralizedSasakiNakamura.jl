# Quasinormal frequencies at arbitrary precision.
#
# A quasinormal frequency is a zero of the incidence amplitude of the IN solution. Starting from the
# binary64 Leaver root of `qnm_frequency`, the root is polished by the secant method on the GSN
# incidence amplitude B_inc(omega) of the arbitrary-precision ISEM route (complex frequency, MST
# infinity side), in BigFloat at the requested number of bits. The frequency derivative of the
# incidence amplitude at the root, needed by the excitation factor, is taken by a fourth-order
# central difference evaluated at a raised working precision, so that its truncation and rounding
# errors are both below the requested precision; the Teukolsky-normalized derivative is
# Binc/Btrans(omega_n) times the GSN one (B_inc vanishes at the root, so the frequency dependence of
# the conversion factors does not enter).

export qnm_arbitrary_precision

const _AP_QNM_MAX_ITERATIONS = 60

# GSN incidence amplitude of the IN solution at omega (BigFloat parameters, MST infinity side), built
# as a root search: near the root the amplitude is small by nature, so the cancellation of its
# numerator is not counted as a loss (the route still covers the determinant's, exp(2 |Im omega| r*_m)
# at a strongly damped frequency), and the amplitude comes out accurate in absolute terms.
function _ap_incidence(s, l, m, a::BigFloat, omega::Complex{BigFloat}; tol=nothing)
    params = ISEM.Parameters.gsn_parameters(s, l, m, a, omega)
    route = ISEM.ArbitraryPrecisionRoute.arbitrary_precision_route(params, :IN; root_search=true, tol=tol)
    return route.incidence, route
end

const _AP_QNM_MAX_TIGHTENINGS = 2

"""
    qnm_arbitrary_precision(a, s, l, m, n, branch=ordinary; bits=256, tol=nothing, guess=nothing,
                            convention=:overtone, derivative=true)

Quasinormal frequency of the mode `(s, l, m, n)` at spin `a` to `bits` bits of precision, polished
from the binary64 root of [`qnm_frequency`](@ref) (or from `guess`) by the secant method on the GSN
incidence amplitude of the arbitrary-precision ISEM route. `tol` is the relative tolerance of the
root (default: 64 ulps at `bits`). With `derivative=true` the frequency derivative of the incidence
amplitude at the root is also computed (GSN normalization, and Teukolsky normalization through the
conversion factors).

Returns a named tuple `(omega, lambda, incidence_derivative, incidence_derivative_teukolsky,
reflection, residual, iterations, bits, seed, history)`; `residual` is `|B_inc(omega)|` at the
returned root, `reflection` the GSN reflection amplitude there and `history` the secant iterates.
"""
function qnm_arbitrary_precision(a, s::Integer, l::Integer, m::Integer, n::Integer,
        branch::QNMBranch=ordinary; bits::Integer=256, tol=nothing, guess=nothing,
        convention=:overtone, derivative::Bool=true)
    bits >= 64 || throw(ArgumentError("bits must be at least 64."))
    seed = if guess === nothing
        leaver = qnm_frequency(Float64(a), s, l, m, n, branch; convention)
        leaver isa LeaverResult || error("the binary64 Leaver root search did not return a root: $leaver")
        ComplexF64(leaver.omega)
    else
        ComplexF64(guess)
    end
    return setprecision(BigFloat, Int(bits)) do
        a_b = BigFloat(a)
        tolerance = tol === nothing ? 64 * eps(BigFloat) : BigFloat(tol)
        x0 = Complex{BigFloat}(seed)
        # A second starting point a few binary64 ulps away; the seed is accurate to about 1e-13.
        x1 = x0 * (1 + Complex{BigFloat}(1e-9, 0.5e-9))
        history = Complex{BigFloat}[x0, x1]
        iterations = 0
        converged = false
        # The route gives B_inc with an absolute error of about its series tolerance times the size of
        # the amplitudes (|B_ref| in GSN normalization at the root), which the secant turns into a root
        # error of that over |dB_inc/domega|. When the slope of the last secant step says that this
        # exceeds the requested tolerance of the root, the route tolerance is tightened by the ratio
        # (with a guard) and the iteration continues from the last two iterates.
        route_tol = nothing
        local f0, f1, route1
        for tightening in 0:_AP_QNM_MAX_TIGHTENINGS
            f0, _ = _ap_incidence(s, l, m, a_b, x0; tol=route_tol)
            f1, route1 = _ap_incidence(s, l, m, a_b, x1; tol=route_tol)
            converged = false
            slope = zero(BigFloat)
            for _ in 1:_AP_QNM_MAX_ITERATIONS
                iterations += 1
                if f1 == f0 || x1 == x0
                    converged = true        # the step fell below the resolution of the working precision
                    break
                end
                slope = abs(f1 - f0) / abs(x1 - x0)
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                f2, route2 = _ap_incidence(s, l, m, a_b, x2; tol=route_tol)
                push!(history, x2)
                x0, f0, x1, f1, route1 = x1, f1, x2, f2, route2
                if abs(x1 - x0) <= tolerance * abs(x1)
                    converged = true
                    break
                end
            end
            series_tol = route_tol === nothing ? BigFloat(route1.tolerance) : BigFloat(route_tol)
            (isfinite(slope) && slope > 0) || break
            root_error = series_tol * max(abs(route1.reflection), one(BigFloat)) / slope
            root_error <= tolerance * abs(x1) / 16 && break
            tightening == _AP_QNM_MAX_TIGHTENINGS && break
            route_tol = max(series_tol * (tolerance * abs(x1) / 16) / root_error / 256, ldexp(tolerance, -256))
            # The converged pass may have ended with coincident iterates; the next one starts from a
            # baseline a few estimated root errors away, well above the tightened noise.
            x0 = x1 * (1 + max(16 * root_error / abs(x1), 64 * eps(BigFloat)) * Complex{BigFloat}(1, 1 // 2))
        end
        converged || @warn "qnm_arbitrary_precision: the secant iteration did not reach the tolerance" residual=Float64(abs(f1)) iterations
        omega = x1
        lambda = route1.params.lambda
        dinc = missing
        dinc_teukolsky = missing
        if derivative
            dinc = _ap_incidence_derivative(s, l, m, a_b, omega, tolerance)
            factor = ConversionFactors.Binc(s, m, a_b, omega, lambda) /
                ConversionFactors.Btrans(s, m, a_b, omega, lambda)
            dinc_teukolsky = factor * dinc
        end
        (omega=omega, lambda=lambda, incidence_derivative=dinc,
            incidence_derivative_teukolsky=dinc_teukolsky, reflection=route1.reflection,
            residual=abs(f1), iterations=iterations, bits=Int(bits), seed=seed, history=history)
    end
end

# dB_inc/domega at omega by the fourth-order central difference
#   (f(w-2h) - 8 f(w-h) + 8 f(w+h) - f(w+2h)) / (12 h),
# with h = |omega| tol^(1/4) / 100 (the fifth derivative of B_inc is large on the scale |omega|: at
# h = |omega| tol^(1/4) the truncation error cost 4 digits of 34 against the independent reference) and
# the four evaluations at a working precision raised by a quarter of the bits plus a guard, so that both
# the truncation error (h^4) and the rounding error (eps/h) are below the requested tolerance. B_inc
# is analytic in omega away from the negative imaginary axis.
function _ap_incidence_derivative(s, l, m, a::BigFloat, omega::Complex{BigFloat}, tolerance::BigFloat)
    bits = precision(BigFloat)
    h = abs(omega) * BigFloat(tolerance)^(1 // 4) / 100
    raised = bits + ceil(Int, bits / 4) + 40
    value = setprecision(BigFloat, raised) do
        a_r = BigFloat(a)
        w = Complex{BigFloat}(omega)
        hh = BigFloat(h)
        f(z) = first(_ap_incidence(s, l, m, a_r, z))
        (f(w - 2hh) - 8 * f(w - hh) + 8 * f(w + hh) - f(w + 2hh)) / (12 * hh)
    end
    return Complex{BigFloat}(value)
end
