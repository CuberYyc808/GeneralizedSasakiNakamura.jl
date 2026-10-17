# Type-generic Mano-Suzuki-Takasugi (MST) representation of the Teukolsky UP and DOWN solutions.
#
# The series of Sasaki & Tagoshi, Living Rev. Relativ. 6, 6 (2003), Sec. 4.4 (R_-^nu, the outgoing
# "up" solution, and R_+^nu, the ingoing "down" solution at infinity) are evaluated in the floating
# type R of the mode parameters, with the conventions of the Black Hole Perturbation Toolkit
# Teukolsky paclet (Kernel/MST/MST.m) so that the unit-transmission UP solution is directly comparable
# with TeukolskyRadial[..., "BoundaryConditions" -> "Up"]:
#
#   R_up(r)   = prefacUp(zhat) sum_n fUp_n (-2 i zhat)^n U(n + nu + s + 1 - i eps, 2n + 2nu + 2, -2 i zhat) / UpTrans,
#   fUp_n     = (-1)^n (nu + 1 + s - i eps)_n / (nu + 1 - s + i eps)_n f_n,
#   UpTrans   = (eps/2)^(-1-2s) exp(i eps (log eps - (1 - kappa)/2)) 2^(-s-1+i eps) exp(-pi eps/2 - i pi (nu+1+s)/2) sum_n fUp_n,
#
# with zhat = omega (r - r_-), eps = 2 omega, tau = (eps - m a)/kappa, f_n the MST series coefficients
# (f_0 = 1) and nu the renormalized angular momentum. The DOWN solution is R_+^nu of ST Eq. (154)
# normalized to a unit ingoing wave r^-1 e^{-i omega r*} (ST Eqs. 156-157). Both solutions converge
# at every finite r > r_+, so they are free of the Stokes ambiguity of the asymptotic series at
# complex frequencies (see ArbitraryPrecisionRoute.jl).
#
# The confluent hypergeometric U is built from Kummer's M by the connection formula at a working
# precision raised by the cancellation of the Kummer series (about |z| / ln 2 bits) and by the
# closeness of the second parameter to an integer; complex log-gamma comes from SpecialFunctions.
module ManoSuzukiTakasugiGeneric

using SpecialFunctions: loggamma
using ..Parameters: GSNParameters

export MSTGenericData, mst_generic_data
export mst_nu_generic, mst_up_state, mst_down_state, mst_up_transmission, mst_down_transmission

const NU_CF_DEPTH_INITIAL = 40
const NU_CF_DEPTH_MAX = 20000
const NU_MAX_ITERATIONS = 100
const COEFFICIENT_CAP = 2000
const SERIES_MIN_TERMS = 4
const U_GUARD_BITS = 64
const KUMMER_MAX_TERMS = 200000

_real_type(::GSNParameters{R}) where {R} = R
_complex_type(::GSNParameters{R}) where {R} = Complex{R}

"""
    MSTGenericData

MST data of one mode in the floating type of its parameters: the renormalized angular momentum `nu`,
the series coefficients `f_n` for `nmin <= n <= nmax` (`coefficient(data, n)`), the tolerance of the
sums, and the derived quantities eps = 2 omega and tau = (eps - m a)/kappa.
"""
struct MSTGenericData{P<:GSNParameters,R,C}
    params::P
    epsilon::C
    tau::C
    nu::C
    coefficients::Vector{C}
    nmin::Int
    nmax::Int
    tolerance::R
    nu_residual::R
    nu_depth::Int
end

coefficient(data::MSTGenericData, n::Integer) = data.coefficients[n - data.nmin + 1]

# ---------------------------------------------------------------------------------------------
# Three-term recurrence alpha_n f_(n+1) + beta_n f_n + gamma_n f_(n-1) = 0 (ST Eq. 124)
# ---------------------------------------------------------------------------------------------
@inline function _alpha(p::GSNParameters, eps, kappa, tau, nu, n)
    s = p.s
    npnu1 = n + nu + 1
    return im * eps * kappa * (npnu1 + s + im * eps) * (npnu1 + s - im * eps) * (npnu1 + im * tau) /
        (npnu1 * (2 * npnu1 + 1))
end
@inline function _beta(p::GSNParameters, eps, kappa, tau, nu, n)
    s = p.s
    product = (n + nu) * (n + nu + 1)
    linear = eps * (eps - p.m * p.a)
    return -p.lambda - s * (s + 1) + product + eps^2 + linear + linear * (s^2 + eps^2) / product
end
@inline function _gamma(p::GSNParameters, eps, kappa, tau, nu, n)
    s = p.s
    npnu = n + nu
    return -im * eps * kappa * (npnu - s + im * eps) * (npnu - s - im * eps) * (npnu - im * tau) /
        (npnu * (2 * npnu - 1))
end

# R_n = f_n / f_(n-1) for n = 1..N by backward recursion from depth N + extra (ST Eq. 125), and
# L_n = f_n / f_(n+1) for n = -1..-N by the mirrored recursion. Returned as vectors R[n], L[-n].
function _ratios(p::GSNParameters, eps, kappa, tau, nu, N::Int, depth::Int)
    C = typeof(nu)
    top = N + depth
    R = Vector{C}(undef, N)
    r = -_gamma(p, eps, kappa, tau, nu, top) / _beta(p, eps, kappa, tau, nu, top)
    for k in (top - 1):-1:1
        r = -_gamma(p, eps, kappa, tau, nu, k) / (_beta(p, eps, kappa, tau, nu, k) + _alpha(p, eps, kappa, tau, nu, k) * r)
        k <= N && (R[k] = r)
    end
    L = Vector{C}(undef, N)
    l = -_alpha(p, eps, kappa, tau, nu, -top) / _beta(p, eps, kappa, tau, nu, -top)
    for k in (-top + 1):-1
        l = -_alpha(p, eps, kappa, tau, nu, k) / (_beta(p, eps, kappa, tau, nu, k) + _gamma(p, eps, kappa, tau, nu, k) * l)
        k >= -N && (L[-k] = l)
    end
    return R, L
end

# Characteristic function whose zero is nu: beta_0 + alpha_0 R_1 + gamma_0 L_(-1) (continued fractions
# of depth `depth`).
function _characteristic(p::GSNParameters, eps, kappa, tau, nu, depth::Int)
    r = -_gamma(p, eps, kappa, tau, nu, depth) / _beta(p, eps, kappa, tau, nu, depth)
    for k in (depth - 1):-1:1
        r = -_gamma(p, eps, kappa, tau, nu, k) / (_beta(p, eps, kappa, tau, nu, k) + _alpha(p, eps, kappa, tau, nu, k) * r)
    end
    l = -_alpha(p, eps, kappa, tau, nu, -depth) / _beta(p, eps, kappa, tau, nu, -depth)
    for k in (-depth + 1):-1
        l = -_alpha(p, eps, kappa, tau, nu, k) / (_beta(p, eps, kappa, tau, nu, k) + _gamma(p, eps, kappa, tau, nu, k) * l)
    end
    return _beta(p, eps, kappa, tau, nu, 0) + _alpha(p, eps, kappa, tau, nu, 0) * r + _gamma(p, eps, kappa, tau, nu, 0) * l
end

"""
    mst_nu_generic(params, seed; tol)

Renormalized angular momentum in the floating type of `params`, by secant iteration on the
continued-fraction characteristic equation from `seed` (a binary64 value, e.g. from the Float64 MST
module). The continued-fraction depth is doubled until the characteristic function at the seed is
stable to `tol`. Returns `(nu, residual, depth)`.
"""
function mst_nu_generic(p::GSNParameters, seed; tol=nothing)
    R = _real_type(p)
    C = _complex_type(p)
    tolerance = tol === nothing ? 64 * eps(R) : R(tol)
    eps_ = C(2 * p.omega)
    kappa = p.kappa
    tau = (eps_ - p.m * p.a) / kappa
    nu0 = C(seed)
    depth = NU_CF_DEPTH_INITIAL
    g0 = _characteristic(p, eps_, kappa, tau, nu0, depth)
    while depth < NU_CF_DEPTH_MAX
        g1 = _characteristic(p, eps_, kappa, tau, nu0, 2depth)
        abs(g1 - g0) <= tolerance * max(one(R), abs(g1)) && break
        depth *= 2
        g0 = g1
    end
    depth = 2depth
    scale = max(one(R), abs(p.lambda), R(p.l * (p.l + 1)))
    x0 = nu0
    x1 = nu0 + C(1e-6)
    f0 = _characteristic(p, eps_, kappa, tau, x0, depth)
    f1 = _characteristic(p, eps_, kappa, tau, x1, depth)
    for _ in 1:NU_MAX_ITERATIONS
        f1 == f0 && break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = _characteristic(p, eps_, kappa, tau, x2, depth)
        x0, f0, x1, f1 = x1, f1, x2, f2
        if abs(x1 - x0) <= tolerance * max(one(R), abs(x1)) && abs(f1) <= sqrt(tolerance) * scale
            return x1, abs(f1), depth
        end
    end
    error("MST generic: the renormalized angular momentum iteration did not converge from the seed $seed (residual $(abs(f1))).")
end

"""
    mst_generic_data(params, nu; tol, N)

MST coefficients f_n for |n| <= N (default: until |f_n| falls below tol relative to f_0 by a wide
margin, at most $(COEFFICIENT_CAP)) from the continued-fraction ratios, for the given renormalized
angular momentum `nu` (from `mst_nu_generic`).
"""
function mst_generic_data(p::GSNParameters, nu; tol=nothing, N=nothing, nu_residual=zero(_real_type(p)), nu_depth=0)
    R = _real_type(p)
    C = _complex_type(p)
    tolerance = tol === nothing ? 64 * eps(R) : R(tol)
    eps_ = C(2 * p.omega)
    kappa = p.kappa
    tau = (eps_ - p.m * p.a) / kappa
    nu_c = C(nu)
    count = N === nothing ? COEFFICIENT_CAP : Int(N)
    ratios_R, ratios_L = _ratios(p, eps_, kappa, tau, nu_c, count, 60)
    # f_n by products of the ratios; stop each side once the coefficients are far below the tolerance
    # (the U factors and powers of 2 zhat can enhance a term by many orders of magnitude, hence the margin).
    margin = tolerance * ldexp(one(R), -40)
    positive = C[]
    f = one(C)
    for n in 1:count
        f *= ratios_R[n]
        push!(positive, f)
        N === nothing && abs(f) < margin && n >= SERIES_MIN_TERMS && break
    end
    negative = C[]
    f = one(C)
    for n in 1:count
        f *= ratios_L[n]
        push!(negative, f)
        N === nothing && abs(f) < margin && n >= SERIES_MIN_TERMS && break
    end
    nmin = -length(negative)
    nmax = length(positive)
    coefficients = vcat(reverse(negative), [one(C)], positive)
    return MSTGenericData{typeof(p),R,C}(p, eps_, tau, nu_c, coefficients, nmin, nmax, tolerance, R(nu_residual), nu_depth)
end

# ---------------------------------------------------------------------------------------------
# Confluent hypergeometric functions in the working type
# ---------------------------------------------------------------------------------------------
# Kummer M(a, b, z) by its power series (entire in z), summed at the current precision.
function _kummer_M(a::C, b::C, z::C) where {C<:Complex}
    T = real(C)
    term = one(C)
    total = one(C)
    threshold = eps(T) / 4
    for k in 0:KUMMER_MAX_TERMS
        term *= (a + k) / (b + k) * z / (k + 1)
        total += term
        k > 2 * abs(z) && abs(term) <= threshold * abs(total) && return total
    end
    error("MST generic: the Kummer series did not converge for |z| = $(Float64(abs(z))).")
end

# Distance of a complex number to the nearest integer, in the real type.
_integer_distance(b) = abs(b - round(real(b)))

# Tricomi U(a, b, z) for non-integer b by the connection formula
#   U = Gamma(1-b)/Gamma(a-b+1) M(a,b,z) + Gamma(b-1)/Gamma(a) z^(1-b) M(a-b+1,2-b,z),
# evaluated at a working precision raised by the cancellation of the Kummer series (|z| / ln 2 bits
# plus a half) and by the closeness of b to an integer, then rounded to the caller's precision.
function _tricomi_U(a::Complex{BigFloat}, b::Complex{BigFloat}, z::Complex{BigFloat})
    bits = precision(BigFloat)
    distance = Float64(_integer_distance(b))
    distance > 0 || error("MST generic: Tricomi U with an integer second parameter is not implemented.")
    extra = ceil(Int, 1.5 * Float64(abs(z))) + U_GUARD_BITS + ceil(Int, max(0.0, -log2(distance)))
    value = setprecision(BigFloat, bits + extra) do
        aa = Complex{BigFloat}(a); bb = Complex{BigFloat}(b); zz = Complex{BigFloat}(z)
        t1 = exp(loggamma(1 - bb) - loggamma(aa - bb + 1)) * _kummer_M(aa, bb, zz)
        t2 = exp(loggamma(bb - 1) - loggamma(aa) + (1 - bb) * log(zz)) * _kummer_M(aa - bb + 1, 2 - bb, zz)
        t1 + t2
    end
    return Complex{BigFloat}(value)
end
function _tricomi_U(a::C, b::C, z::C) where {C<:Complex}
    # Other floating types: evaluate through BigFloat at a precision matching the type plus the guard.
    T = real(C)
    bits = precision(T)
    value = setprecision(BigFloat, bits + U_GUARD_BITS) do
        _tricomi_U(Complex{BigFloat}(a), Complex{BigFloat}(b), Complex{BigFloat}(z))
    end
    return C(value)
end

# log of the Pochhammer symbol (x)_n = Gamma(x + n) / Gamma(x) for any integer n.
_log_pochhammer(x, n::Integer) = loggamma(x + n) - loggamma(x)

# ---------------------------------------------------------------------------------------------
# UP solution (ST Eq. 155 / BHPT prefacUp, fUp, UpTrans)
# ---------------------------------------------------------------------------------------------
_zhat(data::MSTGenericData, r) = data.params.omega * (r - (one(data.params.kappa) - data.params.kappa))

# fUp_n = (-1)^n (nu + 1 + s - i eps)_n / (nu + 1 - s + i eps)_n f_n
function _fup(data::MSTGenericData, n::Integer)
    p = data.params
    nu = data.nu
    e = data.epsilon
    ratio = exp(_log_pochhammer(nu + 1 + p.s - im * e, n) - _log_pochhammer(nu + 1 - p.s + im * e, n))
    return (isodd(n) ? -1 : 1) * ratio * coefficient(data, n)
end

"""
    mst_up_transmission(data)

The amplitude UpTrans of the raw UP series (BHPT prefacUpTrans * A_-), i.e. the Teukolsky
transmission amplitude of the raw series; dividing by it gives the unit-transmission UP solution
R_up -> r^(-2s-1) exp(i omega r*).
"""
function mst_up_transmission(data::MSTGenericData)
    p = data.params
    e = data.epsilon
    k = p.kappa
    nu = data.nu
    C = typeof(nu)
    total = zero(C)
    for n in data.nmin:data.nmax
        total += _fup(data, n)
    end
    aminus = 2^(-p.s - 1 + im * e) * exp(-pi * e / 2 - im * pi * (nu + 1 + p.s) / 2) * total
    prefactor = (e / 2)^(-1 - 2 * p.s) * exp(im * e * (log(e) - (1 - k) / 2))
    return prefactor * aminus
end

# prefacUp(zhat) and d log prefacUp / d zhat
function _up_prefactor(data::MSTGenericData, zhat)
    p = data.params
    e = data.epsilon
    k = p.kappa
    nu = data.nu
    eplus = (e + data.tau) / 2
    value = 2^nu * exp(-pi * e) * exp(-im * pi * (nu + 1 + p.s)) * exp(im * zhat) *
        zhat^(nu + im * eplus) * (zhat - e * k)^(-p.s - im * eplus)
    dlog = im + (nu + im * eplus) / zhat + (-p.s - im * eplus) / (zhat - e * k)
    return value, dlog
end

# Sum over n of weight_n (c)^n U(n + a, 2n + b, c) and its zhat-derivative, c = sign * 2 i zhat,
# adding terms on each side of n = 0 until they fall below the tolerance relative to the sum.
function _tricomi_sum(data::MSTGenericData, weight, a, b, c, dc_dzhat)
    C = typeof(c)
    R = real(C)
    total = zero(C)
    dtotal = zero(C)
    absolute = zero(R)
    nused = (0, 0)
    for direction in (1, -1)
        n = direction == 1 ? 0 : -1
        last = direction == 1 ? data.nmax : data.nmin
        streak = 0
        while direction == 1 ? n <= last : n >= last
            U0 = _tricomi_U(n + a, 2n + b, c)
            U1 = _tricomi_U(n + a + 1, 2n + b + 1, c)
            cn = c^n
            H = cn * U0
            dH = (n * cn / c * U0 - cn * (n + a) * U1) * dc_dzhat
            w = weight(n)
            term = w * H
            total += term
            dtotal += w * dH
            absolute += abs(term)
            if abs(n) >= SERIES_MIN_TERMS && abs(term) <= data.tolerance * abs(total)
                streak += 1
                streak >= 2 && break
            else
                streak = 0
            end
            n += direction
        end
        nused = direction == 1 ? (nused[1], n) : (n, nused[2])
    end
    return total, dtotal, absolute, nused
end

"""
    mst_up_state(data, r)

(R_up, dR_up/dr) at the Boyer-Lindquist radius r, unit transmission at infinity. Returns a named
tuple with the state, the relative cancellation of the sum and the terms used.
"""
function mst_up_state(data::MSTGenericData, r)
    p = data.params
    R = _real_type(p)
    zhat = _zhat(data, R(r))
    c = -2im * zhat
    a = data.nu + p.s + 1 - im * data.epsilon
    b = 2 * data.nu + 2
    prefactor, dlog = _up_prefactor(data, zhat)
    total, dtotal, absolute, nused = _tricomi_sum(data, n -> _fup(data, n), a, b, c, -2im)
    norm = mst_up_transmission(data)
    value = prefactor * total / norm
    derivative = prefactor * (dlog * total + dtotal) * p.omega / norm   # dzhat/dr = omega
    return (R=value, dR=derivative, cancellation=Float64(absolute / max(abs(total), floatmin(R))), terms=nused)
end

# ---------------------------------------------------------------------------------------------
# DOWN solution (ST Eq. 154, R_+^nu): the ingoing branch at infinity
#   R_+ = 2^nu e^{-pi eps} e^{i pi (nu+1-s)} Gamma(nu+1-s+i eps)/Gamma(nu+1+s-i eps) e^{-i zhat}
#         zhat^{nu + i eps_+} (zhat - eps kappa)^{-s - i eps_+}
#         sum_n i^n (nu+1+s-i eps)_n/(nu+1-s+i eps)_n f_n (2 zhat)^n U(n+nu+1+s-i eps, 2n+2nu+2, 2 i zhat),
# normalized to a unit ingoing wave r^-1 e^{-i omega r*} through A_+ (ST Eq. 157) and the same
# frequency factors as the "In incidence" normalization of the BHPT paclet.
# ---------------------------------------------------------------------------------------------
function _down_prefactor(data::MSTGenericData, zhat)
    p = data.params
    e = data.epsilon
    k = p.kappa
    nu = data.nu
    eplus = (e + data.tau) / 2
    gamma_ratio = exp(loggamma(nu + 1 - p.s + im * e) - loggamma(nu + 1 + p.s - im * e))
    value = 2^nu * exp(-pi * e) * exp(im * pi * (nu + 1 - p.s)) * gamma_ratio * exp(-im * zhat) *
        zhat^(nu + im * eplus) * (zhat - e * k)^(-p.s - im * eplus)
    dlog = -im + (nu + im * eplus) / zhat + (-p.s - im * eplus) / (zhat - e * k)
    return value, dlog
end

# i^n (nu+1+s-i eps)_n / (nu+1-s+i eps)_n f_n (2 zhat)^n = (-1)^n fUp_n (2 i zhat)^n / (2 i zhat)^n ... written
# directly: with c = 2 i zhat, i^n (2 zhat)^n = c^n, so the weight is the Pochhammer ratio times f_n.
function _fdown(data::MSTGenericData, n::Integer)
    p = data.params
    nu = data.nu
    e = data.epsilon
    ratio = exp(_log_pochhammer(nu + 1 + p.s - im * e, n) - _log_pochhammer(nu + 1 - p.s + im * e, n))
    return ratio * coefficient(data, n)
end

"""
    mst_down_transmission(data)

Normalization of the raw DOWN series to a unit ingoing wave r^-1 exp(-i omega r*): the amplitude
A_+ (ST Eq. 157, BHPT prefacAplus * sum f_n) times the frequency factors of the ingoing wave,
(eps/2)^-1 exp(-i eps (log eps - (1 - kappa)/2)).
"""
function mst_down_transmission(data::MSTGenericData)
    p = data.params
    e = data.epsilon
    k = p.kappa
    nu = data.nu
    C = typeof(nu)
    total = zero(C)
    for n in data.nmin:data.nmax
        total += coefficient(data, n)
    end
    aplus = 2^(-1 + p.s - im * e) * exp(-pi * e / 2 + im * pi * (1 - p.s + nu) / 2) *
        exp(loggamma(1 - p.s + im * e + nu) - loggamma(1 + p.s - im * e + nu)) * total
    prefactor = (e / 2)^(-1) * exp(-im * e * (log(e) - (1 - k) / 2))
    return prefactor * aplus
end

"""
    mst_down_state(data, r)

(R_down, dR_down/dr) at the Boyer-Lindquist radius r, unit ingoing amplitude at infinity.
"""
function mst_down_state(data::MSTGenericData, r)
    p = data.params
    R = _real_type(p)
    zhat = _zhat(data, R(r))
    c = 2im * zhat
    a = data.nu + p.s + 1 - im * data.epsilon
    b = 2 * data.nu + 2
    prefactor, dlog = _down_prefactor(data, zhat)
    total, dtotal, absolute, nused = _tricomi_sum(data, n -> _fdown(data, n), a, b, c, 2im)
    norm = mst_down_transmission(data)
    value = prefactor * total / norm
    derivative = prefactor * (dlog * total + dtotal) * p.omega / norm
    return (R=value, dR=derivative, cancellation=Float64(absolute / max(abs(total), floatmin(R))), terms=nused)
end

end
