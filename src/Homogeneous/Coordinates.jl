module Coordinates
using ..Kerr

export rstar_from_r, r_from_rstar

_is_exact_extremal_spin(a) = isreal(a) && isone(abs(real(a)))

# Exact extremal spin (|a| = 1, r_+ = 1): closed-form r* as a function of h = r - r_+.
function _rstar_from_exact_extremal_rp(h)
    h < 0 && throw(DomainError(h, "r-r_+ must be nonnegative"))
    h == 0 && return -Inf
    return 1 + h - log(4) + 2*(log(h) - 1/h)
end

function rstar_from_rp(a, r_from_rp)
    if _is_exact_extremal_spin(a)
        return _rstar_from_exact_extremal_rp(r_from_rp)
    elseif abs(a) < 1
        rp = r_plus(a)
        rm = r_minus(a)
        return rp + r_from_rp + (2*1*rp)/(rp-rm) * log(r_from_rp/(2*1)) - (2*1*rm)/(rp-rm) * log((r_from_rp+rp-rm)/(2*1))
    else
        throw(ArgumentError("a must be in the range [-1, 1]"))
    end
end

@doc raw"""
    rstar_from_r(a, r)

Convert a Boyer-Lindquist coordinate `r` to the corresponding tortoise coordinate `rstar`.
With $M = 1$ and $r_\pm = 1 \pm \sqrt{1 - a^2}$,
```math
r_* = r + \frac{2r_+}{r_+-r_-}\ln\frac{r-r_+}{2} - \frac{2r_-}{r_+-r_-}\ln\frac{r-r_-}{2}
```
for $|a| < 1$, and $r_* = r + 2\ln(r-1) - 2/(r-1) - \ln 4$ for $|a| = 1$.
It requires $|a| \le 1$ and $r \ge r_+$; at $r = r_+$ it returns `-Inf`.
"""
function rstar_from_r(a, r)
    _is_exact_extremal_spin(a) && return _rstar_from_exact_extremal_rp(r - 1)

    rp = r_plus(a)
    return rstar_from_rp(a, r-rp)
end

"""
    horizon_distance_from_rstar(a, rstar)

The distance h = r - r_+ from the outer horizon at the tortoise coordinate `rstar` (|a| <= 1, exact extremal
included, any floating type), with its full relative precision also deep in the horizon region, where r = r_+ + h
cannot hold it (at r* = -200 and a = 0.9, h = 4e-27). Newton's method on u = log h, with drstar/du =
(r^2 + a^2) / (h + r_+ - r_-) (r^2 + 1 over h at |a| = 1), from the leading behaviour of r*(h): at the horizon
r* ~ r_+ + A log(h/2) - B log((r_+ - r_-)/2) (r* ~ -2/h at |a| = 1) for r* <= 0, and r* ~ r + 2 log(r/2) beyond.
r*(u) is increasing; a step longer than 10 (a factor e^10 in h) is damped.
"""
function horizon_distance_from_rstar(a, rstar)
    T = promote_type(typeof(float(rstar)), typeof(float(a)))
    rs = T(rstar)
    rs == -Inf && return zero(T)
    rp, rm = T(r_plus(a)), T(r_minus(a))
    u = if rs > 0
        log(max(rs - 2 * log(max(rs, T(2)) / 2) - rp, T(1) / 2))
    elseif _is_exact_extremal_spin(a)
        log(-2 / min(rs, -one(T)))
    else
        A, B = 2 * rp / (rp - rm), 2 * rm / (rp - rm)
        (rs - rp + B * log((rp - rm) / 2)) / A + log(T(2))
    end
    previous = T(Inf)
    for _ in 1:200
        h = exp(u)
        r = rp + h
        step = (rstar_from_rp(a, h) - rs) * (h + rp - rm) / (r^2 + T(a)^2)
        u -= clamp(step, -T(10), T(10))
        # converged, or at the rounding floor of r*(h) (the step no longer shrinks)
        (abs(step) <= 2 * eps(T) || (abs(step) >= previous && abs(step) <= 1e3 * eps(T))) && break
        previous = abs(step)
    end
    return exp(u)
end

@doc raw"""
    r_from_rstar(a, rstar)

Convert a tortoise coordinate `rstar` to the corresponding Boyer-Lindquist coordinate `r` = r_+ + h, with h from
[`horizon_distance_from_rstar`](@ref) (Newton's method on log h, exact extremal spin included).

The function assumes that $r \geq r_{+}$ where $r_{+}$ is the outer event horizon.
"""
r_from_rstar(a, rstar) = r_plus(a) + horizon_distance_from_rstar(a, rstar)

end
