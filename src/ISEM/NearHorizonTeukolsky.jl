module NearHorizonTeukolsky

export TeukolskyTaylorChain, teukolsky_taylor_chain, teukolsky_taylor_state

# Taylor series of the radial Teukolsky equation (M = 1) at real r > r_+,
#     Delta^2 R'' + (s + 1) Delta Delta' R' + [K^2 - 2 i s (r - 1) K + Delta (4 i s omega r - lambda)] R = 0,
# multiplied out so that the three coefficients are polynomials in r of degree <= 4. About r0, R(r0 + t) = sum c_n t^n
# follows from a recurrence of at most five terms; the series converges for |t| < r0 - r_+, the distance to the nearest
# singular point (r_- lies further away, infinity is irregular). Used by RadialFunctions.near_horizon_y for Y of
# s = -2 IN at real omega between the horizon series and the matching point, where the converted Y = C11 X + C12 X'
# cancels by 1e4 - 4e4 and passes the error of X on enlarged (G-005).
#
# Step control (the invariant of every patch [r0, r0 + h]): h <= STEP_FRACTION (r0 - r_+), so |t| / radius <= 1/2, and
# the last two terms, |c_{N-1}| h^{N-1} + |c_N| h^N, are at most TRUNCATION_TOL |R(r0 + h)|; h is halved until both hold,
# and the chain fails (returns `nothing`) if h would drop below MIN_STEP_FRACTION (r0 - r_+) or more than MAX_PATCHES
# patches are needed.

const ORDER = 32
const STEP_FRACTION = 0.5
const MIN_STEP_FRACTION = 1e-6
const TRUNCATION_TOL = 1e-15
const MAX_PATCHES = 400

const _P5 = NTuple{5,ComplexF64}

# Patches [starts[i], stops[i]] with coefficients coeffs[(i - 1) * (ORDER + 1) + 1 : i * (ORDER + 1)]; built once per
# solution and only read afterwards.
struct TeukolskyTaylorChain
    starts::Vector{Float64}
    stops::Vector{Float64}
    coeffs::Vector{ComplexF64}
end

_c(x...) = ntuple(i -> i <= length(x) ? ComplexF64(x[i]) : 0.0im, 5)
_padd(p::_P5, q::_P5) = ntuple(i -> p[i] + q[i], 5)
# Product truncated to degree 4. Exact for every product formed here: delta (2) * delta (2), delta (2) * delta' (1),
# (s + 1) (0) * that (3), K (2) * K (2), (2 i s - 2 i s r) (1) * K (2), delta (2) * (4 i s omega r - lambda) (1).
_pmul(p::_P5, q::_P5) = ntuple(k -> sum(p[i] * q[k - i + 1] for i in 1:k), 5)

# coefficients of p(r0 + t) in t, from those of p(r) (ascending), by repeated synthetic division (degree 4)
function _shift(p::_P5, r0::Float64)
    q0, q1, q2, q3, q4 = p
    q3 += r0 * q4; q2 += r0 * q3; q1 += r0 * q2; q0 += r0 * q1
    q3 += r0 * q4; q2 += r0 * q3; q1 += r0 * q2
    q3 += r0 * q4; q2 += r0 * q3
    q3 += r0 * q4
    return (q0, q1, q2, q3, q4)
end

function _teukolsky_polynomials(s, m, a, omega, lambda)
    delta = _c(a^2, -2, 1)
    ddelta = _c(-2, 2)
    K = _c(a^2 * omega - a * m, 0, omega)
    P2 = _pmul(delta, delta)
    P1 = _pmul(_c(s + 1), _pmul(delta, ddelta))
    P0 = _padd(_padd(_pmul(K, K), _pmul(_c(2im * s, -2im * s), K)), _pmul(delta, _c(-lambda, 4im * s * omega)))
    return P2, P1, P0
end

# Taylor coefficients c_0..c_ORDER of R about r0 (into buf), given R(r0) and R'(r0) and the shifted polynomials.
function _taylor!(buf, P2::_P5, P1::_P5, P0::_P5, R0, dR0)
    buf[1] = R0
    buf[2] = dR0
    inv0 = 1 / P2[1]
    @inbounds for n in 0:(ORDER - 2)
        acc = 0.0im
        for j in 1:min(4, n + 2)
            acc += P2[j + 1] * ((n + 2 - j) * (n + 1 - j)) * buf[n + 3 - j]
        end
        for j in 0:min(3, n + 1)
            acc += P1[j + 1] * (n + 1 - j) * buf[n + 2 - j]
        end
        for j in 0:min(4, n)
            acc += P0[j + 1] * buf[n + 1 - j]
        end
        buf[n + 3] = -acc * inv0 / ((n + 2) * (n + 1))
    end
    return buf
end

# (value, derivative) of sum_{k=0}^{ORDER} c[off + k + 1] t^k
@inline function _horner(c, off, t)
    v = c[off + ORDER + 1]
    d = 0.0im
    @inbounds for k in ORDER:-1:1
        d = d * t + v
        v = v * t + c[off + k]
    end
    return v, d
end

"""
    teukolsky_taylor_chain(s, m, a, omega, lambda, r_start, R0, dR0, r_stop)

Taylor patches of the Teukolsky solution with R(r_start) = R0, R'(r_start) = dR0, covering [r_start, r_stop]
(r_+ < r_start < r_stop), or `nothing` if a step cannot meet the step-control invariant (see the module header).
"""
function teukolsky_taylor_chain(s, m, a, omega, lambda, r_start::Float64, R0, dR0, r_stop::Float64)
    rplus = 1 + sqrt((1 - a) * (1 + a))
    (r_start > rplus && r_stop > r_start) || return nothing
    P2r, P1r, P0r = _teukolsky_polynomials(s, m, a, omega, lambda)
    starts = Float64[]; stops = Float64[]
    coeffs = ComplexF64[]
    sizehint!(coeffs, 16 * (ORDER + 1))
    buf = Vector{ComplexF64}(undef, ORDER + 1)
    r0 = r_start; R = ComplexF64(R0); dR = ComplexF64(dR0)
    while r0 < r_stop
        length(starts) < MAX_PATCHES || return nothing
        _taylor!(buf, _shift(P2r, r0), _shift(P1r, r0), _shift(P0r, r0), R, dR)
        h = min(STEP_FRACTION * (r0 - rplus), r_stop - r0)
        while true
            v, _ = _horner(buf, 0, h)
            tail = (abs(buf[ORDER]) * h^(ORDER - 1) + abs(buf[ORDER + 1]) * h^ORDER) / max(abs(v), floatmin(Float64))
            tail <= TRUNCATION_TOL && break
            h /= 2
            h >= MIN_STEP_FRACTION * (r0 - rplus) || return nothing
        end
        r1 = r_stop - (r0 + h) <= 4eps(r_stop) ? r_stop : r0 + h
        R, dR = _horner(buf, 0, r1 - r0)
        push!(starts, r0); push!(stops, r1); append!(coeffs, buf)
        r0 = r1
    end
    return TeukolskyTaylorChain(starts, stops, coeffs)
end

"""
    teukolsky_taylor_state(chain, r)

(R, dR/dr) at r in [first start, last stop] of the chain.
"""
function teukolsky_taylor_state(chain::TeukolskyTaylorChain, r::Real)
    rr = Float64(r)
    # callers stay inside [starts[1], stops[end]]; the clamp only absorbs rounding at the two ends
    i = clamp(searchsortedlast(chain.starts, rr), 1, length(chain.starts))
    return _horner(chain.coeffs, (i - 1) * (ORDER + 1), rr - chain.starts[i])
end

end
