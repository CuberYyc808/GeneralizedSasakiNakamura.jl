module RadialFunctions
using ...Errors: absorbable
using ...GeneralizedSasakiNakamura: IN, UP, Mode, GSNRadialFunction, TeukolskyRadialFunction, UNIT_GSN_TRANS, UNIT_TEUKOLSKY_TRANS

using ...Coordinates: r_from_rstar, rstar_from_r
using ...ConversionFactors
using ...Solutions: Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix
using ..AsymptoticAmplitudes: _try_unit_pair
using ..ComplexPathAmplitudes: _infinity_in_coefficients, _infinity_out_coefficients, _infinity_solutions
using ..Matching:
    Route,
    ConjugatedRoute,
    ScaledBasis,
    route_state
using ..Matching: Matching, route_plan_of, route_truncations
using ..Iteration: _endpoint_last_term_radius, _ENDPOINT_SEED_TARGET
using ..Matching: _mano_suzuki_takasugi_evaluation_region

include("NearHorizonTeukolsky.jl")
using .NearHorizonTeukolsky: TeukolskyTaylorChain, teukolsky_taylor_chain, teukolsky_taylor_state
using ..TeukolskyTransformation: _GSN_to_Y_coeffs_minus2_in

export compact_kappa, compact_r_plus, compact_r_minus
export r_from_x, x_from_r
export dx_drstar
export gsn_solution_rstar, gsn_radial_function
export route_teukolsky_solution, teukolsky_radial_function
export y_branch_supported, route_y_solution, y_radial_function, near_horizon_y
export RstarYSolution, y_at_rstar

# Teukolsky horizon solutions R_in -> Delta^{-s} e^{-i k r*} and R_out -> e^{+i k r*}
# (leading coefficient one, r* as in Coordinates) at real r near r_+, from the
# Frobenius series of z^2 p2 R'' + z p1 R' + p0 R = 0 in z = r - r_+ (the
# Teukolsky equation times Delta), exponents -s - i u and i u, u = r_+ k / kappa.
# Returns (R_in, dR_in/dr, R_out, dR_out/dr, relative truncation error, and the
# cancellation sum |terms| / |sum| of the R_in and R_out series); the series
# converges for |z| < 2 kappa (the next singular point is r_-).
function teukolsky_horizon_pair(s, m, a, omega, lambda, r::Real)
    h = _horizon_frobenius(s, m, a, omega, lambda)
    p2, p1, p0 = h.p2, h.p1, h.p0
    z = Float64(r) - h.rplus
    function series(rho)
        T(j, n) = p2[j + 1] * (n + rho) * (n + rho - 1) + p1[j + 1] * (n + rho) +
            p0[j + 1]
        coefficients = ComplexF64[1.0]
        sizehint!(coefficients, 128)   # up to 400 terms; grown once instead of at every doubling
        value = 1.0 + 0.0im
        derivative = rho / z
        zn = 1.0
        n = 0
        tail = Inf
        magnitude = 1.0
        while n < 400
            n += 1
            acc = 0.0im
            for j in 1:min(n, 4)
                acc += T(j, n - j) * coefficients[n - j + 1]
            end
            c = -acc / T(0, n)
            push!(coefficients, c)
            zn *= z
            term = c * zn
            value += term
            magnitude += abs(term)
            derivative += (n + rho) * term / z
            tail = abs(term) / max(abs(value), floatmin(Float64))
            n >= 4 && tail <= eps(Float64) && break
        end
        power = exp(rho * log(z))
        return power * value, power * derivative, tail,
            magnitude / max(abs(value), floatmin(Float64))
    end
    Rin, dRin, tin, cin = series(h.rho_in)
    Rout, dRout, tout, cout = series(h.rho_out)
    # Delta^{-s} e^{-i k r*} = norm_in z^{rho_in} (1 + O(z)), e^{i k r*} likewise.
    return (Rin * h.norm_in, dRin * h.norm_in, Rout * h.norm_out, dRout * h.norm_out,
        max(tin, tout), cin, cout)
end

# Frobenius data of teukolsky_horizon_pair: p2, p1, p0 as coefficients in z (ascending), the exponents and the
# normalisations of R_in and R_out. `kappa` defaults to sqrt(1 - a^2), which the pair has always taken; it loses the
# relative precision of kappa near a = 1 (7e-15 at a = 0.999), and RstarYSolution passes the kappa of its route.
@inline function _horizon_frobenius(s, m, a, omega, lambda, kappa=sqrt(1 - a^2))
    rplus = 1 + kappa
    rminus = 1 - kappa
    k = omega - m * a / (2 * rplus)
    u = rplus * k / kappa
    K0 = 2 * rplus * omega - a * m
    L0 = lambda - 4im * s * omega * rplus
    # p2 = (z + 2 kappa)^2, p1 = 2 (s + 1)(z + kappa)(z + 2 kappa),
    # p0 = K^2 - 2 i s (z + kappa) K - z (z + 2 kappa)(lambda - 4 i s omega r),
    # K = K0 + 2 r_+ omega z + omega z^2.
    Kc = (K0, 2 * rplus * omega, omega)
    K2 = (Kc[1]^2, 2 * Kc[1] * Kc[2], Kc[2]^2 + 2 * Kc[1] * Kc[3],
        2 * Kc[2] * Kc[3], Kc[3]^2)
    sK = (kappa * Kc[1], Kc[1] + kappa * Kc[2], Kc[2] + kappa * Kc[3], Kc[3])
    zz = (0.0, 2 * kappa, 1.0)                      # z (z + 2 kappa)
    L = (L0, -4im * s * omega)                       # lambda - 4 i s omega r
    zzL = (0.0im, zz[2] * L[1], zz[2] * L[2] + zz[3] * L[1], zz[3] * L[2])
    p2 = (complex(4 * kappa^2), complex(4 * kappa), 1.0 + 0.0im, 0.0im, 0.0im)
    p1 = (complex(4 * (s + 1) * kappa^2), complex(6 * (s + 1) * kappa),
        complex(2 * (s + 1)), 0.0im, 0.0im)
    p0 = ntuple(j -> K2[j] - 2im * s * (j <= 4 ? sK[j] : 0.0im) -
        (j <= 4 ? zzL[j] : 0.0im), 5)
    norm_in = (2 * kappa)^(-s) * exp(-im * k * rplus) * exp(im * u * log(2.0)) *
        exp(im * k * rminus / kappa * log(kappa))
    norm_out = exp(im * k * rplus) * exp(-im * u * log(2.0)) *
        exp(-im * k * rminus / kappa * log(kappa))
    return (rplus=rplus, p2=p2, p1=p1, p0=p0, rho_in=-s - im * u, rho_out=im * u,
        norm_in=norm_in, norm_out=norm_out)
end

# c_n of the Frobenius series with exponent rho, from c_0 .. c_{n-1} (coefficients[1:n]): the step of the recurrence in
# teukolsky_horizon_pair.
function _frobenius_coefficient(h, coefficients, rho, n)
    T(j, n) = h.p2[j + 1] * (n + rho) * (n + rho - 1) + h.p1[j + 1] * (n + rho) + h.p0[j + 1]
    acc = 0.0im
    for j in 1:min(n, 4)
        acc += T(j, n - j) * coefficients[n - j + 1]
    end
    return -acc / T(0, n)
end

# Teukolsky solution callables: r -> (R, R', R'', error), r -> (R, R') and r -> R.
struct TeukolskySolution{F,G,H}
    tuple_value::F
    pair_value::G
    radial_value::H
end

(solution::TeukolskySolution)(r) = solution.tuple_value(r)

# Asymptotic Teukolsky series at infinity in z = 1 / (omega r), valid for |z| <= radius.
struct TeukolskyInfinitySeries{P}
    params::P
    coefficients::Vector{ComplexF64}
    radius::Float64
    tolerance::Float64
end

function _teukolsky_infinity_coefficients(s, m, a, w, L, order)
    a2 = a*a
    a4 = a2*a2
    # Equation for R=exp(i*w*rstar)*r^(-2s-1)*f(1/r), multiplied by r^-4.
    L2 = ComplexF64.((0im,0im,0im,0im,1+0im,-4+0im,2(a2+2),-4a2,a4))
    L1 = ComplexF64.((0im,0im,-2im*w,2(s+2im*w+1),-2(2im*a2*w+5s+5),
        2(3a2*s+2im*a2*w+3a2+6s+6),-2a2*(im*a2*w+7s+7),4a4*(s+1),0im))
    L0 = ComplexF64.((0im,0im,-L-2a*m*w,
        -2(-L+2im*a2*s*w+im*a2*w-im*a*m*s+2s*s+3s+1),
        -L*a2-2a*a2*m*w+a2*m*m+4a2*s*s+8im*a2*s*w+6a2*s+
            4im*a2*w+2a2-2im*a*m*s+8s*s+12s+4,
        -2a2*(2s+1)*(im*a2*w+3s+3),2a4*(s+1)*(2s+1),0im,0im))
    coefficient = zeros(ComplexF64, order+1)
    coefficient[1] = 1
    at(v,n) = 0<=n<length(v) ? v[n+1] : 0im
    # Store coefficients in 1/(w*r); cancel w in the indicial denominator first.
    for n in 1:order
        acc = 0im
        for j in max(0,n-8):n-1
            acc += (at(L2,n-j+3)*j*(j-1)+at(L1,n-j+2)*j+at(L0,n-j+1))*
                w^(n-j-1)*coefficient[j+1]
        end
        coefficient[n+1] = acc/(2im*n)
    end
    return coefficient
end

function teukolsky_infinity_series(p, order, tolerance)
    coefficient = _teukolsky_infinity_coefficients(p.s, p.m, p.a, p.omega,
        p.lambda, order)
    radius = _endpoint_last_term_radius(coefficient,Float64(tolerance),Inf)
    return TeukolskyInfinitySeries(p,coefficient,radius,Float64(tolerance))
end

# (R, dR/dr, error estimate) from the series, or `nothing` if r is outside the
# radius or the truncation error exceeds the tolerance.
function (series::TeukolskyInfinitySeries)(r)
    p = series.params
    z = inv(p.omega*r)
    abs(z)<=series.radius || return nothing
    c = series.coefficients
    value = c[end]
    derivative = zero(value)
    for j in length(c)-1:-1:1
        derivative = derivative*z+value
        value = value*z+c[j]
    end
    power = -2p.s-1
    radial_factor = power+im*p.omega*(r*r+p.a*p.a)*r/((r-compact_r_plus(p))*(r-compact_r_minus(p)))
    slope = radial_factor*value-z*derivative
    term, norm_value, norm_slope = 1+0im, 0.0, 0.0
    for j in eachindex(c)
        contribution = c[j]*term
        norm_value += abs(contribution)
        norm_slope += abs((radial_factor-(j-1))*contribution)
        term *= z
    end
    ratio = abs(c[end]*z/c[end-1])
    ratio<1 || return nothing
    tail = abs(c[end]*z^(length(c)-1))/(1-ratio)
    error = max((tail+eps(Float64)*norm_value)/abs(value),
        (tail*abs(radial_factor-length(c))+eps(Float64)*norm_slope)/abs(slope))
    error<=series.tolerance || return nothing
    phase = p.omega*rstar_from_r(p.a,r)
    scale = exp(im*phase)*r^power
    return (scale*value,scale*slope/r,error+eps(Float64)*abs(phase))
end

# Ingoing Teukolsky solution at infinity, R_in = e^{-i omega r*} sum_n c_n r^{-1-n}
# (c_0 = 1), from the Teukolsky equation times Delta, Delta^2 R'' + Delta (2 (s+1)
# (r - 1)) R' + (K^2 - 2 i s (r - 1) K + Delta (4 i s omega r - lambda)) R = 0, with
# R = e^{-i omega r*} h: Delta^2 h'' + Delta (2 (s+1)(r - 1) - 2 i omega rho) h' +
# (Delta (4 i s omega - 2 i omega r - lambda) - 8 i s omega r (r - 1) -
# 2 a m omega rho + a^2 m^2 + 2 i s a m (r - 1)) h = 0, rho = r^2 + a^2. The
# series is asymptotic; it is summed to its smallest term.
struct TeukolskyInfinityInSeries
    s::Int
    a::Float64
    omega::Float64
    coefficients::Vector{ComplexF64}
end

function teukolsky_infinity_in_series(s, m, a, omega, lambda; order::Int=60)
    w = Float64(omega)
    pmul(p, q) = (out = zeros(ComplexF64, length(p) + length(q) - 1);
        for i in eachindex(p), j in eachindex(q); out[i+j-1] += p[i] * q[j]; end; out)
    padd(ps...) = (out = zeros(ComplexF64, maximum(length, ps));
        for p in ps; out[1:length(p)] .+= p; end; out)
    D = ComplexF64[a^2, -2, 1]
    rho = ComplexF64[a^2, 0, 1]
    P2 = pmul(D, D)
    P1 = pmul(D, padd(-2im * w .* rho, ComplexF64[-2(s + 1), 2(s + 1)]))
    P0 = padd(pmul(D, ComplexF64[4im * s * w - lambda, -2im * w]),
        ComplexF64[0, 8im * s * w, -8im * s * w], -2a * m * w .* rho,
        ComplexF64[a^2 * m^2 - 2im * s * a * m, 2im * s * a * m])
    at(v, k) = 0 <= k < length(v) ? v[k+1] : 0.0im
    c = zeros(ComplexF64, order + 1)
    c[1] = 1
    for n in 1:order
        acc = 0.0im
        for d in 1:5
            j = n - d
            j < 0 && break
            acc += (at(P2, 5 - d) * (j + 1) * (j + 2) - at(P1, 4 - d) * (j + 1) +
                at(P0, 3 - d)) * c[j+1]
        end
        c[n+1] = -acc / (2im * w * n)
    end
    return TeukolskyInfinityInSeries(Int(s), Float64(a), w, c)
end

# (R_in, dR_in/dr, relative truncation error) at r, with r* = rstar.
function (series::TeukolskyInfinityInSeries)(r, rstar)
    c = series.coefficients
    a = series.a
    inv_r = inv(r)
    power = inv_r
    h = 0.0im
    hp = 0.0im
    last = Inf
    for n in 0:(length(c) - 1)
        term = c[n+1] * power
        abs(term) > last && break
        h += term
        hp -= (n + 1) * term * inv_r
        last = abs(term)
        last <= eps(Float64) * abs(h) && break
        power *= inv_r
    end
    delta = r^2 - 2r + a^2
    e = exp(-im * series.omega * rstar)
    return e * h, e * (hp - im * series.omega * (r^2 + a^2) / delta * h),
        last / max(abs(h), floatmin(Float64))
end

struct TeukolskyPairEvaluator{R,C,T,E,W}
    route::R
    converter::C
    inv_scale::T
    infinity_series::E
    infinity_scale::ComplexF64
    two_wave::W
end

TeukolskyPairEvaluator(route, converter, inv_scale, infinity_series, infinity_scale) =
    TeukolskyPairEvaluator(route, converter, inv_scale, infinity_series, infinity_scale, nothing)

# R of a real-frequency IN route at large r from the two Teukolsky series at infinity,
# R = B_inc R_in + B_ref R_out (route normalization), where their truncation is below the
# conversion's own error eps (2 omega r)^4: the conversion from (X, X') amplifies the rounding of X
# by about that much (R_in of (s, l, m, a, omega) = (-2, 2, 2, 0, 15) was off by 2e-7 at r = 6 and
# 3e-5 at r = 20), the series not at all, given an accurate B_ref (Matching._complex_path_reflection).
struct TeukolskyTwoWave
    a::Float64
    omega::Float64
    s::Int
    incidence::ComplexF64
    reflection::ComplexF64
    cin::Vector{ComplexF64}
    cout::Vector{ComplexF64}
    r_two::Float64
end

function _teukolsky_two_wave(route::Route, factors)
    p = route.params
    (route.branch === :IN && isreal(p.omega) && real(p.omega) > 0 && 0 <= p.a < 1) || return nothing
    return teukolsky_two_wave(p.s, p.m, p.a, real(p.omega), real(p.lambda),
        route.incidence * factors.incidence, route.reflection * factors.reflection)
end

# The two-wave R of an IN solution with Teukolsky amplitudes (incidence, reflection) in its own normalization, or
# nothing if the series reach their tolerance at none of TWO_WAVE_RADII.
function teukolsky_two_wave(s, m, a, omega, lambda, incidence, reflection)
    a, w, lambda = Float64(a), Float64(omega), Float64(lambda)
    (w > 0 && 0 <= a < 1 && isfinite(incidence) && isfinite(reflection)) || return nothing
    cin = _infinity_in_coefficients(s, m, a, w, lambda, TWO_WAVE_SERIES_ORDER)
    cout = _infinity_out_coefficients(s, m, a, w, lambda, TWO_WAVE_SERIES_ORDER)
    kappa = sqrt(1 - a^2)
    for r in TWO_WAVE_RADII
        r > 1 + kappa || continue
        if _two_wave_series_ok(_infinity_solutions(s, a, w, ComplexF64(r), cin, cout)[5], w, r)
            return TeukolskyTwoWave(a, w, s, ComplexF64(incidence), ComplexF64(reflection), cin, cout, r)
        end
    end
    return nothing
end

function _two_wave_pair(t::TeukolskyTwoWave, r)
    (r isa Real && r >= t.r_two) || return nothing
    Rin, dRin, Rout, dRout, err = _infinity_solutions(t.s, t.a, t.omega, ComplexF64(r), t.cin, t.cout)
    _two_wave_series_ok(err, t.omega, r) || return nothing
    return t.incidence * Rin + t.reflection * Rout, t.incidence * dRin + t.reflection * dRout, err
end

# Use the infinity series where it applies (UP, real omega; IN at large r, real omega); otherwise
# convert the GSN state at r to (R, R').
function _teukolsky_pair_with_error(e::TeukolskyPairEvaluator,r)
    if e.infinity_series !== nothing
        endpoint = e.infinity_series(r)
        endpoint !== nothing && return e.infinity_scale*endpoint[1],e.infinity_scale*endpoint[2],endpoint[3]
    end
    if e.two_wave !== nothing
        pair = _two_wave_pair(e.two_wave, r)
        pair !== nothing && return e.inv_scale * pair[1], e.inv_scale * pair[2], pair[3]
    end
    x = x_from_r(e.route.params, r)
    state = route_state(e.route, x; radius=r)
    R, Rp = e.converter(r, state.X, state.dXdx)
    return e.inv_scale * R, e.inv_scale * Rp, 0.0
end

function (e::TeukolskyPairEvaluator)(r)
    R,Rp,_ = _teukolsky_pair_with_error(e,r)
    return R,Rp
end

struct TeukolskyRadialEvaluator{P}
    pair::P
end

function (e::TeukolskyRadialEvaluator)(r)
    R, _ = e.pair(r)
    return R
end

struct TeukolskyTupleEvaluator{P,E,F}
    params::P
    pair::E
    d2R::F
end

function (e::TeukolskyTupleEvaluator)(r)
    R, Rp, error = _teukolsky_pair_with_error(e.pair,r)
    p = e.params
    Rpp = e.d2R(r, R, Rp, p.s, p.a, p.omega, p.m, p.lambda)
    return (R, Rp, Rpp, error)
end

function _a_value(a_or_params)
    return hasproperty(a_or_params, :a) ? getproperty(a_or_params, :a) : a_or_params
end

function compact_kappa(a_or_params)
    if hasproperty(a_or_params, :kappa)
        return getproperty(a_or_params, :kappa)
    end
    a = _a_value(a_or_params)
    return sqrt((one(a) - a) * (one(a) + a))
end

compact_r_plus(a_or_params) = one(compact_kappa(a_or_params)) + compact_kappa(a_or_params)
compact_r_minus(a_or_params) = one(compact_kappa(a_or_params)) - compact_kappa(a_or_params)

function r_from_x(a_or_params, x)
    rp = compact_r_plus(a_or_params)
    rm = compact_r_minus(a_or_params)
    return (rp - rm * x) / (one(x) - x)
end

function x_from_r(a_or_params, r)
    rp = compact_r_plus(a_or_params)
    rm = compact_r_minus(a_or_params)
    return (r - rp) / (r - rm)
end

function dx_drstar(a_or_params, x)
    kappa = compact_kappa(a_or_params)
    return kappa * (x - one(x))^2 * x /
        (kappa + (x - one(x))^2 + 2 * kappa^2 * x - kappa * x^2)
end

# x = 1 / (1 + exp(-t)), y = 1 - x, log x and log y, evaluated without overflow.
@inline function _logit_parts(t)
    if t >= 0.0
        inverse = exp(-t)
        log_norm = log1p(inverse)
        x = 1.0 / (1.0 + inverse)
        y = inverse / (1.0 + inverse)
        return x, y, -log_norm, -t - log_norm
    end
    forward = exp(t)
    log_norm = log1p(forward)
    x = forward / (1.0 + forward)
    y = 1.0 / (1.0 + forward)
    return x, y, t - log_norm, -log_norm
end

# r*(t) and dr*/dt for x = logistic(t).
@inline function _rstar_parts(kappa, rplus, logkappa2,
        rplus_over_kappa, t)
    x, y, log_x, log_y = _logit_parts(t)
    radial = 2kappa * exp(t)
    value = rplus + radial + logkappa2 +
        rplus_over_kappa * log_x - 2log_y
    slope = radial + rplus_over_kappa * y + 2x
    return value, slope
end

# x from r* through _logit_from_rstar; falls back to r_from_rstar.
function _x_from_rstar(params, target)
    kappa = params.kappa
    a = params.a
    fallback() = x_from_r(params, r_from_rstar(a, target))
    (isfinite(target) && isfinite(kappa) && kappa > 0.0) || return fallback()
    t, converged = _logit_from_rstar(kappa, target)
    x, _, _, _ = _logit_parts(t)
    if !converged || !isfinite(x) || x <= 1.0e-10 || x >= 1.0 - 1.0e-10
        return fallback()
    end
    return x
end

# Invert r*(x) with a safeguarded Newton iteration in t = logit(x), starting from the horizon or far-zone asymptotic
# estimate; returns t and whether it converged. Through _logit_parts, t gives x, 1 - x, their logarithms and
# r - r_+ = 2 kappa e^t with full relative precision at both ends.
@inline function _logit_from_rstar(kappa, target)
    rplus = 1.0 + kappa
    logkappa2 = 2log(kappa)
    rplus_over_kappa = rplus / kappa
    lower = log(nextfloat(0.0))
    upper_x = prevfloat(1.0)
    upper = log(upper_x) - log1p(-upper_x)
    horizon_t = clamp((target - rplus - logkappa2) / rplus_over_kappa, lower, upper)
    far_scale = target - rplus - logkappa2
    infinity_t = clamp(far_scale > 0.0 ? log(far_scale / (2kappa)) : 0.0, lower, upper)
    horizon_value, _ = _rstar_parts(kappa, rplus, logkappa2,
        rplus_over_kappa, horizon_t)
    infinity_value, _ = _rstar_parts(kappa, rplus, logkappa2,
        rplus_over_kappa, infinity_t)
    horizon_error = abs(horizon_value - target)
    infinity_error = abs(infinity_value - target)
    t = horizon_error <= infinity_error ? horizon_t : infinity_t
    converged = false

    for _ in 1:24
        rstar_value, slope = _rstar_parts(kappa, rplus, logkappa2,
            rplus_over_kappa, t)
        value = rstar_value - target
        if abs(value) <= 8eps(max(abs(target), 1.0))
            converged = true
            break
        end
        value < 0.0 ? (lower = t) : (upper = t)
        step = value / slope
        candidate = t - step
        used_newton = isfinite(candidate) && lower < candidate < upper
        if !used_newton
            candidate = 0.5 * (lower + upper)
        end
        candidate == t && break
        t = candidate
        if used_newton && abs(step) <= 4eps(max(abs(t), 1.0))
            converged = true
            break
        end
    end

    return t, converged
end

function _route_boundary_condition(route::Route)
    route.branch == :IN && return IN
    route.branch == :UP && return UP
    throw(ArgumentError("ISEM route branch must be :IN or :UP."))
end

function _route_mode(route::Route)
    mode_type = Mode
    return mode_type(
        route.params.s, route.params.l, route.params.m, route.params.a, route.params.omega, route.params.lambda,
    )
end

function _isem_module()
    return parentmodule(@__MODULE__)
end

function _teukolsky_transformation_module()
    return getfield(_isem_module(), :TeukolskyTransformation)
end

# X and dX/dr* at r; dx/dr* = 2 kappa x / (r^2 + a^2).
function _r_state(route::Route, r)
    x = x_from_r(route.params, r)
    state = route_state(route, x; radius=r)
    return state.X, state.dXdx * (2route.params.kappa*x/(r^2+route.params.a^2))
end

# r* -> (X, dX/dr*, 0.0). Points in the MST region, or beyond the matching point
# when the route has an endpoint basis or plan, are evaluated through r.
gsn_solution_rstar(route::Route) = rs -> _gsn_state_rstar(route, rs)

@inline function _gsn_state_rstar(route::Route, rs)
    x = _x_from_rstar(route.params, rs)
    if _mano_suzuki_takasugi_evaluation_region(route, x) ||
            (x > route.match_x && (route.endpoint_bridge isa ScaledBasis ||
                route.endpoint_plan !== nothing))
        r = r_from_rstar(route.params.a, rs)
        X, dXdrstar = _r_state(route, r)
        return (X, dXdrstar, 0.0)
    end
    state = route_state(route, x)
    dXdrstar = state.dXdx * dx_drstar(route.params, x)
    return (state.X, dXdrstar, 0.0)
end

function gsn_solution_rstar(route::ConjugatedRoute)
    positive_solution = gsn_solution_rstar(route.route)
    return rs -> begin
        X, dXdrstar, residual = positive_solution(rs)
        return (conj(X), conj(dXdrstar), residual)
    end
end

function _unit_gsn_solution(route::Route)
    scale = route.solution_scale
    scaled_solution = gsn_solution_rstar(route)
    scale == 1.0 && return scaled_solution
    return rs -> begin
        X, dXdrstar, residual = scaled_solution(rs)
        return (X / scale, dXdrstar / scale, residual)
    end
end

# The unit-transmission normalisation divides by solution_scale; it is used only
# if the rescaled state stays at least exp(32) below floatmax at sample points.
const _UNIT_STATE_LOG_MARGIN = 32.0

@inline function _unit_point_safe(route::Route, x, limit)
    state = route_state(route, x)
    dXdrstar = state.dXdx * dx_drstar(route.params, x)
    return isfinite(real(state.X)) && isfinite(imag(state.X)) &&
        isfinite(real(dXdrstar)) && isfinite(imag(dXdrstar)) &&
        abs(state.X) <= limit && abs(dXdrstar) <= limit
end

function _unit_basis_safe(route::Route, basis, limit)
    basis === nothing && return true
    local_basis = hasproperty(basis, :basis) ? basis.basis : basis
    for patch in local_basis.patches
        center = patch.center_x
        endpoint = patch.next_x
        _unit_point_safe(route, center, limit) || return false
        _unit_point_safe(route, (center + endpoint) / 2, limit) || return false
        _unit_point_safe(route, endpoint, limit) || return false
    end
    return true
end

# True when dividing by `solution_scale` keeps the state finite, with margin
# _UNIT_STATE_LOG_MARGIN, at x = 1e-5, match_x, 1 - 1e-5 and sample points of
# every bridge patch.
function _unit_state_safe(route::Route)
    scale = route.solution_scale
    scale == 1.0 && return true
    limit = exp(log(floatmax(Float64)) + log(scale) -
        _UNIT_STATE_LOG_MARGIN)
    isfinite(limit) && limit > 0 || return false
    _unit_point_safe(route, 1.0e-5, limit) || return false
    _unit_point_safe(route, route.match_x, limit) || return false
    _unit_point_safe(route, 1.0 - 1.0e-5, limit) || return false
    _unit_basis_safe(route, route.bridge, limit) || return false
    return _unit_basis_safe(route, route.endpoint_bridge, limit)
end

# Certification fields of the GSN metadata (see ISEM.certificate). A strict route's estimated error is its larger
# amplitude error, the incidence error alone when the reflection is taken from the complex path, and NaN when the
# estimate cannot be formed.
function _route_certificate(route::Route)
    route.certified || return (certified=false, estimated_error=route.estimated_error,
        uncertified_stages=route.uncertified_stages)
    error = try
        incidence, reflection = Matching.amplitude_errors(route)
        Matching._complex_path_switch(route) ? incidence : max(incidence, reflection)
    catch err
        absorbable(err) || rethrow()
        NaN
    end
    return (certified=true, estimated_error=Float64(error), uncertified_stages=())
end

# Wrap a route as the package's GSNRadialFunction, with unit GSN transmission when
# representable and the route diagnostics in the metadata.
function gsn_radial_function(route::Route)
    controls = route.controls
    plan = route_plan_of(route)
    threshold_log = any(truncation -> truncation.reason == :threshold_log_companion, route_truncations(route))
    unit_pair = _try_unit_pair(route)
    unit_pair_available = unit_pair !== nothing
    unit_state_available = unit_pair_available && _unit_state_safe(route)
    unit_available = unit_pair_available && unit_state_available
    transmission = unit_available ? 1.0 + 0.0im :
        ComplexF64(route.solution_scale)
    incidence, reflection = unit_available ? unit_pair :
        (route.incidence, route.reflection)
    solution = unit_available ? _unit_gsn_solution(route) :
        gsn_solution_rstar(route)
    metadata = (
        xm = route.match_x,
        tol = controls.tolerance,
        horizon_order = controls.horizon_order,
        ordinary_order = controls.ordinary_order,
        infinity_out_order = route.infinity_out_order,
        infinity_in_order = route.infinity_in_order,
        infinity_endpoint_y = route.infinity_endpoint_y,
        patch_count = plan.ordinary_patches + plan.eikonal_patches +
            plan.endpoint_patches,
        small_frequency_expansion = controls.small_frequency_expansion,
        large_frequency_expansion = controls.large_frequency_expansion,
        frequency_regime = controls.frequency_selection.regime,
        frequency_reason = plan.frequency_reason,
        spin_regime = plan.spin,
        spin_reason = plan.spin_reason,
        endpoint_representation = plan.endpoint,
        horizon_representation = threshold_log ?
            :threshold_log_frobenius : :power_frobenius,
        propagation_representation = plan.propagation,
        matching_representation = plan.matching,
        amplitude_observable = threshold_log ?
            :threshold_log_connection_coefficients : :plane_wave_amplitudes,
        eikonal_candidate = plan.eikonal_candidate,
        ordinary_patches = plan.ordinary_patches,
        eikonal_patches = plan.eikonal_patches,
        endpoint_patches = plan.endpoint_patches,
        horizon_patches = plan.horizon_patches,
        infinity_patches = plan.infinity_patches,
        angular_compression = plan.angular_compression,
        phase_compression = plan.phase_compression,
        solution_scale = route.solution_scale,
        unit_pair_available = unit_pair_available,
        unit_state_available = unit_state_available,
        unit_normalization_available = unit_available,
        endpoint_handoff_x = route.endpoint_plan !== nothing ?
            route.endpoint_plan.seed_x :
            plan.endpoint == :direct_endpoint_basis ? route.match_x : missing,
        endpoint_handoff_scale = route.endpoint_plan !== nothing ?
            route.endpoint_plan.scale :
            plan.endpoint == :direct_endpoint_basis ?
                route.endpoint_bridge.scale : missing,
        normalization = unit_available ?
            :unit_gsn_transmission : :scaled_gsn_transmission,
        _route_certificate(route)...,
    )
    return GSNRadialFunction(
        _route_mode(route),
        _route_boundary_condition(route),
        missing,
        missing,
        rstar_from_r(route.params.a, r_from_x(route.params, route.match_x)),
        controls.horizon_order,
        max(route.infinity_out_order, route.infinity_in_order),
        transmission,
        incidence,
        reflection,
        metadata,
        missing,
        solution,
        UNIT_GSN_TRANS,
        "ISEM",
    )
end

function gsn_radial_function(route::ConjugatedRoute)
    positive = gsn_radial_function(route.route)
    isem = _isem_module()
    conjugate_function = getfield(isem, :_conjugate_gsn_radial_function)
    return conjugate_function(positive, route.params.m, route.params.omega)
end

function _teukolsky_amplitude_factors(route::Route)
    p = route.params
    if route.branch == :IN
        return (
            trans = ConversionFactors.Btrans(p.s, p.m, p.a, p.omega, p.lambda),
            incidence = ConversionFactors.Binc(p.s, p.m, p.a, p.omega, p.lambda),
            reflection = ConversionFactors.Bref(p.s, p.m, p.a, p.omega, p.lambda),
        )
    elseif route.branch == :UP
        return (
            trans = ConversionFactors.Ctrans(p.s, p.m, p.a, p.omega, p.lambda),
            incidence = ConversionFactors.Cinc(p.s, p.m, p.a, p.omega, p.lambda),
            reflection = ConversionFactors.Cref(p.s, p.m, p.a, p.omega, p.lambda),
        )
    end
    throw(ArgumentError("ISEM route branch must be :IN or :UP."))
end

function _normalization_scale_from_factors(route::Route, normalize, factors)
    if normalize == :teukolsky
        return route.solution_scale * factors.trans,
            UNIT_TEUKOLSKY_TRANS
    elseif normalize == :gsn
        return route.solution_scale, UNIT_GSN_TRANS
    end
    throw(ArgumentError("normalize must be :teukolsky or :gsn."))
end

function _normalization_scale(route::Route, normalize)
    return _normalization_scale_from_factors(route, normalize, _teukolsky_amplitude_factors(route))
end

# Teukolsky solution r -> (R, dR/dr, d2R/dr2, error) of a route divided by `scale`
# (default: the transmission in the `normalize` convention, :teukolsky or :gsn).
# `converter` maps (r, X, dX/dx) to (R, dR/dr); real-frequency UP routes use the
# large-r series where it meets the tolerance.
function route_teukolsky_solution(route::Route; normalize=:teukolsky, scale=nothing, converter=nothing)
    p = route.params
    scale_value = scale === nothing ? first(_normalization_scale(route, normalize)) : scale
    pair_converter = converter === nothing ? _teukolsky_converter(p) : converter
    return _teukolsky_solution(route, scale_value, pair_converter)
end

function _teukolsky_solution(route::Route, scale_value, pair_converter)
    p = route.params
    tt = _teukolsky_transformation_module()
    inv_scale = inv(scale_value)
    endpoint = route.branch == :UP && isreal(p.omega) ?
        teukolsky_infinity_series(p,max(route.infinity_out_order,route.controls.ordinary_order),
            max(route.controls.tolerance,_ENDPOINT_SEED_TARGET)) : nothing
    endpoint_scale = ComplexF64(first(_normalization_scale(route,:teukolsky))/scale_value)
    two_wave = _teukolsky_two_wave(route, _teukolsky_amplitude_factors(route))
    pair_value = TeukolskyPairEvaluator(route,pair_converter,inv_scale,endpoint,endpoint_scale,two_wave)
    tuple_value = TeukolskyTupleEvaluator(p, pair_value, getfield(tt, :d2R))
    radial_value = TeukolskyRadialEvaluator(pair_value)
    return TeukolskySolution(tuple_value, pair_value, radial_value)
end

# Converter (r, X, dX/dx) -> (R, R'): the closed form for s = 0; for s = -2, +2 the expanded closed forms beyond
# r - r_+ = 0.1 and the conversion matrix below (the expanded forms cancel near the horizon: R_in of s = -2 off by 1e-1
# at r - r_+ = 1e-4, a = 0.999, where the matrix keeps 1e-5; the matrix loses 2e-8 at r = 1.6e8, omega = 1e-12, where
# the closed form keeps 5e-15); the conversion matrix for the other s.
function _teukolsky_converter(p)
    p.s == 0 && return TeukolskySZeroConverter(p.a)
    abs(p.s) == 2 || return TeukolskyGenericConverter(p)
    closed = p.s == -2 ? TeukolskySMinus2Converter(p.m, p.a, p.omega, p.lambda) :
        TeukolskySPlus2Converter(p.m, p.a, p.omega, p.lambda)
    return TeukolskyZonedConverter(closed, TeukolskyGenericConverter(p), compact_r_plus(p))
end

struct TeukolskyZonedConverter{C,G,T}
    closed::C
    matrix::G
    rplus::T
end

(c::TeukolskyZonedConverter)(r, X, dXdx) = real(r) - c.rplus < 0.1 ? c.matrix(r, X, dXdx) : c.closed(r, X, dXdx)

struct TeukolskyGenericConverter{P}
    params::P
end

function (c::TeukolskyGenericConverter)(r, X, dXdx)
    p = c.params
    vals = Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
        p.s, p.m, p.a, p.omega, p.lambda, r,
    )
    x = x_from_r(p, r)
    dXdrstar = dXdx * dx_drstar(p, x)
    return vals[1, 1] * X + vals[1, 2] * dXdrstar,
        vals[2, 1] * X + vals[2, 2] * dXdrstar
end

struct TeukolskySZeroConverter{T}
    kappa::T
    rminus::T
    a2::T
end

function TeukolskySZeroConverter(a)
    aT = float(a)
    a2 = aT * aT
    kappa = sqrt((one(aT) - aT) * (one(aT) + aT))
    return TeukolskySZeroConverter(kappa, one(kappa) - kappa, a2)
end

function (c::TeukolskySZeroConverter)(r, X, dXdx)
    rho2 = c.a2 + r * r
    invrho = inv(sqrt(rho2))
    offset = r - c.rminus
    R = invrho * X
    Rp = invrho * (
        -(r / rho2) * X + (2 * c.kappa / (offset * offset)) * dXdx)
    return R, Rp
end

struct TeukolskySMinus2Converter{T}
    m::Int
    a::T
    omega::T
    lambda::T
    a2::T
    a3::T
    a4::T
    a5::T
    a6::T
    a7::T
    a8::T
    m2::Int
    lambda2::T
    omega2::T
    omega3::T
    kappa::T
    rminus::T
end

function TeukolskySMinus2Converter(m, a, omega, lambda)
    aT, omegaT, lambdaT = promote(float(a), omega, lambda)
    a2 = aT * aT
    a3 = a2 * aT
    a4 = a2 * a2
    a5 = a4 * aT
    a6 = a3 * a3
    a7 = a6 * aT
    a8 = a4 * a4
    lambda2 = lambdaT * lambdaT
    omega2 = omegaT * omegaT
    omega3 = omega2 * omegaT
    kappa = sqrt(one(aT) - a2)
    rminus = one(kappa) - kappa
    return TeukolskySMinus2Converter{typeof(aT)}(
        Int(m), aT, omegaT, lambdaT, a2, a3, a4, a5, a6, a7, a8, Int(m) * Int(m), lambda2, omega2, omega3, kappa,
        rminus,
    )
end

function (c::TeukolskySMinus2Converter)(r, X, dXdx)
    return _teukolsky_pair_sminus2(
        c.m, c.a, c.omega, c.lambda, r, X, dXdx, c.a2, c.a3, c.a4, c.a5, c.a6, c.a7, c.a8, c.m2, c.lambda2, c.omega2,
        c.omega3, c.kappa, c.rminus,
    )
end

struct TeukolskySPlus2Converter{T}
    m::Int
    a::T
    omega::T
    lambda::T
    a2::T
    a3::T
    a4::T
    a5::T
    a6::T
    a7::T
    a8::T
    m2::Int
    lambda2::T
    omega2::T
    omega3::T
    kappa::T
    rminus::T
end

function TeukolskySPlus2Converter(m, a, omega, lambda)
    aT, omegaT, lambdaT = promote(float(a), omega, lambda)
    a2 = aT * aT
    a3 = a2 * aT
    a4 = a2 * a2
    a5 = a4 * aT
    a6 = a3 * a3
    a7 = a6 * aT
    a8 = a4 * a4
    lambda2 = lambdaT * lambdaT
    omega2 = omegaT * omegaT
    omega3 = omega2 * omegaT
    kappa = sqrt(one(aT) - a2)
    rminus = one(kappa) - kappa
    return TeukolskySPlus2Converter{typeof(aT)}(
        Int(m), aT, omegaT, lambdaT, a2, a3, a4, a5, a6, a7, a8, Int(m) * Int(m), lambda2, omega2, omega3, kappa,
        rminus,
    )
end

function (c::TeukolskySPlus2Converter)(r, X, dXdx)
    return _teukolsky_pair_splus2(
        c.m, c.a, c.omega, c.lambda, r, X, dXdx, c.a2, c.a3, c.a4, c.a5, c.a6, c.a7, c.a8, c.m2, c.lambda2, c.omega2,
        c.omega3, c.kappa, c.rminus,
    )
end

# Explicit s = -2 GSN-to-Teukolsky conversion applied to (X, dX/dx).
function _teukolsky_pair_sminus2(
    m, a, omega, lambda, r, X, dXdx, a2, a3, a4, a5, a6, a7, a8, m2, lambda2, omega2, omega3, kappa, rminus,
)
    delta = a2 + (-2 + r) * r
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r3 * r3
    rshift = r - rminus
    dxdrstar = 2 * kappa * delta / (rshift * rshift * (a2 + r2))
    denom = (-r4) * (2 * lambda + lambda2 - 12im * omega) +
        24 * a3 * m * r * (im - 2 * r * omega) -
        4 * a * m * r2 * (6im + 2im * r * lambda + 3 * r2 * omega) +
        12 * a4 * (-1 - 2im * r * omega + 2 * r2 * omega2) +
        4 * a2 * r * (6 + r * (-3 + 6 * m2 + 6im * omega) +
        2im * r2 * (-3 + lambda) * omega + 3 * r3 * omega2)
    invdenom = inv(denom)
    m11_num = r2 * (
        -4 * a5 * m * r * (-im + r * omega) -
        2 * a * m * r4 * (im + 2 * r2 * omega) -
        2 * a3 * m * r2 * (3im - 2im * r + 4 * r2 * omega) +
        2 * a6 * (-5 - 2im * r * omega + r2 * omega2) +
        a4 * r * (32 + r * (-20 + 2 * m2 - lambda + 6im * omega) -
        10im * r2 * omega + 6 * r3 * omega2) +
        r4 * (-12 + 2 * r * (5 + lambda) -
        r2 * (2 + lambda - 6im * omega) - 2im * r3 * omega +
        2 * r4 * omega2) +
        2 * a2 * r2 * (-12 + r * (19 + lambda) +
        r2 * (-6 + m2 - lambda + 6im * omega) - 4im * r3 * omega +
        3 * r4 * omega2)
    )
    sqrtfac = sqrt((a2 + r2) / delta^2)
    m11 = m11_num * invdenom / (delta^3 * sqrtfac^3)
    m12_dx = (
        2 * r3 * delta * sqrtfac *
        (im * a * m * r + a2 * (-2 - im * r * omega) +
        r * (3 - r - im * r2 * omega))
    ) * invdenom * dxdrstar
    m21_num = 2 * a7 * m * r * (2 + 4im * r * omega - 3 * r2 * omega2) -
        2 * a5 * m * r2 * (4 + r * (-1 + m2 - lambda + 6im * omega) -
        12im * r2 * omega + 9 * r3 * omega2) -
        2 * a3 * m * r4 * (-5 + 2 * lambda + r * (3 + m2 - 2 * lambda +
        16im * omega) - 12im * r2 * omega + 9 * r3 * omega2) -
        2 * a * m * r5 * (6 + r * (-7 + 2 * lambda) +
        r2 * (2 - lambda + 10im * omega) - 4im * r3 * omega +
        3 * r4 * omega2) +
        2 * a8 * (6im - 2 * r * omega - 2im * r2 * omega2 + r3 * omega3) +
        2 * a6 * r * (-36im + r2 * (-12 + 3 * m2 - lambda + 3im * omega) *
        omega - 10im * r3 * omega2 + 4 * r4 * omega3 +
        r * (21im - 2im * m2 + 2im * lambda + 4 * omega)) +
        r5 * (-48im + 12im * r * (6 + lambda) -
        12im * r2 * (3 + lambda - 3im * omega) -
        2 * r4 * (4 + lambda - 9im * omega) * omega -
        8im * r5 * omega2 + 2 * r6 * omega3 +
        r3 * (6im + 3im * lambda + 34 * omega + 4 * lambda * omega)) +
        2 * a2 * r3 * (-48im + 4im * r * (27 + 2 * lambda) +
        im * r2 * (-72 + m2 - 14 * lambda + 30im * omega) +
        r4 * (-16 + 3 * m2 - 3 * lambda + 21im * omega) * omega -
        14im * r5 * omega2 + 4 * r6 * omega3 +
        r3 * (15im + 5im * lambda + 48 * omega + 4 * lambda * omega)) +
        a4 * r2 * (144im + 2im * r * (-90 + 3 * m2 - 8 * lambda) +
        2 * r3 * (-22 + 6 * m2 - 3 * lambda + 15im * omega) * omega -
        36im * r4 * omega2 + 12 * r5 * omega3 +
        r2 * (54im - 4im * m2 + 11im * lambda + 70 * omega + 4 * lambda * omega))
    m21 = im * r * sqrtfac * m21_num * invdenom / ((a2 + r2)^2)
    m22_dx = (
        r2 * sqrtfac *
        (-4 * a3 * m * r * (-im + r * omega) -
        2 * a * m * r2 * (3im - im * r + 2 * r2 * omega) +
        2 * a4 * (-3 - 2im * r * omega + r2 * omega2) +
        a2 * r * (24 + r * (-12 + 2 * m2 - lambda + 6im * omega) -
        12im * r2 * omega + 4 * r3 * omega2) +
        r2 * (-24 + 2 * r * (12 + lambda) -
        r2 * (6 + lambda - 18im * omega) - 8im * r3 * omega +
        2 * r4 * omega2))
    ) * invdenom * dxdrstar
    return m11 * X + m12_dx * dXdx, m21 * X + m22_dx * dXdx
end

# Explicit s = +2 GSN-to-Teukolsky conversion applied to (X, dX/dx).
function _teukolsky_pair_splus2(
    m, a, omega, lambda, r, X, dXdx, a2, a3, a4, a5, a6, a7, a8, m2, lambda2, omega2, omega3, kappa, rminus,
)
    delta = a2 + (-2 + r) * r
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r3 * r3
    sumr2 = a2 + r2
    rshift = r - rminus
    dxdrstar = 2 * kappa * delta / (rshift * rshift * sumr2)
    denom = (-r4) * (24 + 10 * lambda + lambda2 + 12im * omega) -
        24 * a3 * m * r * (im + 2 * r * omega) +
        4 * a * m * r2 * (6im + 2im * r * (4 + lambda) - 3 * r2 * omega) +
        12 * a4 * (-1 + 2im * r * omega + 2 * r2 * omega2) +
        4 * a2 * r * (6 + r * (-3 + 6 * m2 - 6im * omega) -
        2im * r2 * (1 + lambda) * omega + 3 * r3 * omega2)
    common_inv = inv(delta^3 * sumr2 * sqrt(sumr2) * denom)
    m11_num = r2 * delta * (
        -4 * a5 * m * r * (im + r * omega) -
        2 * a3 * m * r2 * (-3im + 2im * r + 4 * r2 * omega) +
        a * m * (2im * r4 - 4 * r6 * omega) +
        2 * a6 * (-5 + 2im * r * omega + r2 * omega2) +
        a4 * r * (32 + r * (-24 + 2 * m2 - lambda - 6im * omega) +
        10im * r2 * omega + 6 * r3 * omega2) +
        r4 * (-12 + 2 * r * (9 + lambda) -
        r2 * (6 + lambda + 6im * omega) + 2im * r3 * omega +
        2 * r4 * omega2) +
        2 * a2 * r2 * (-12 + r * (23 + lambda) +
        r2 * (-10 + m2 - lambda - 6im * omega) + 4im * r3 * omega +
        3 * r4 * omega2)
    )
    m11 = m11_num * common_inv
    m12_dx = (
        2 * r3 * delta * sumr2^2 *
        (-im * a * m * r + a2 * (-2 + im * r * omega) +
        r * (3 - r + im * r2 * omega))
    ) * common_inv * dxdrstar
    m21_num =
        -2 * a7 * m * r * (-2 + 4im * r * omega + 3 * r2 * omega2) -
        2 * a5 * m * r2 * (-4 + r * (3 + m2 - lambda + 2im * omega) +
        4im * r2 * omega + 9 * r3 * omega2) +
        2 * a * m * r5 * (-10 + r * (3 - 2 * lambda) +
        r2 * (2 + lambda + 2im * omega) + 4im * r3 * omega -
        3 * r4 * omega2) -
        2 * a3 * m * r3 *
        (12 + r2 * (3 + m2 - 2 * lambda) + r * (-17 + 2 * lambda) -
        4im * r3 * omega + 9 * r4 * omega2) +
        2 * a8 * (-6im - 2 * r * omega + 2im * r2 * omega2 + r3 * omega3) +
        2 * a6 * r * (16im +
        im * r * (-9 + 2 * m2 - 2 * lambda + 4im * omega) +
        r2 * (-8 + 3 * m2 - lambda + im * omega) * omega +
        6im * r3 * omega2 + 4 * r4 * omega3) +
        r6 * (-4im * lambda - 2 * r3 * (4 + lambda + 5im * omega) * omega +
        2 * r5 * omega3 - 12 * r * (im + omega) +
        r2 * (6im + im * lambda + 18 * omega + 4 * lambda * omega)) +
        a4 * r2 * (-16im + 2im * r * (10 + m2 + 6 * lambda -
        12im * omega) + 2 * r3 * (-14 + 6 * m2 - 3 * lambda -
        3im * omega) * omega + 12im * r4 * omega2 + 12 * r5 * omega3 +
        r2 * (-2im - 4im * m2 - 7im * lambda + 22 * omega +
        4 * lambda * omega)) +
        2 * a2 * r4 * (-4im * lambda +
        im * r * (-4 + 3 * m2 + 6 * lambda + 6im * omega) +
        3 * r3 * (-4 + m2 - lambda - 3im * omega) * omega +
        2im * r4 * omega2 + 4 * r5 * omega3 +
        r2 * (5im - 4im * m2 - im * lambda + 24 * omega +
        4 * lambda * omega))
    m21 = -im * r * m21_num * common_inv
    m22_dx = (
        r2 * sumr2^2 *
        (-4 * a3 * m * r * (im + r * omega) -
        2 * a * m * r2 * (im - 3im * r + 2 * r2 * omega) +
        2 * a4 * (-3 + 2im * r * omega + r2 * omega2) +
        r3 * (2 * lambda - r * (2 + lambda + 10im * omega) + 2 * r3 * omega2) +
        a2 * r * (8 + 2 * m2 * r - r * lambda + 2im * r * omega +
        4im * r2 * omega + 4 * r3 * omega2))
    ) * common_inv * dxdrstar
    return m11 * X + m12_dx * dXdx, m21 * X + m22_dx * dXdx
end

# Wrap a route as the package's TeukolskyRadialFunction, normalised to unit
# Teukolsky (:teukolsky) or unit GSN (:gsn) transmission.
function teukolsky_radial_function(route::Route; normalize=:teukolsky)
    factors = _teukolsky_amplitude_factors(route)
    scale, convention = _normalization_scale_from_factors(route, normalize, factors)
    solution = route_teukolsky_solution(route; normalize=normalize, scale=scale)
    return TeukolskyRadialFunction(
        _route_mode(route), _route_boundary_condition(route), route.solution_scale * factors.trans / scale,
        route.incidence * factors.incidence / scale, route.reflection * factors.reflection / scale, missing,
        gsn_radial_function(route), solution, convention,
    )
end

function teukolsky_radial_function(route::ConjugatedRoute; normalize=:teukolsky)
    positive = teukolsky_radial_function(route.route; normalize=normalize)
    isem = _isem_module()
    conjugate_function =
        getfield(isem, :_conjugate_teukolsky_radial_function)
    return conjugate_function(positive, route.params.m, route.params.omega)
end

y_branch_supported(route::Route) =
    (route.params.s == -2 && route.branch == :IN) || (route.params.s == 2 && route.branch == :UP)
y_branch_supported(route::ConjugatedRoute) =
    y_branch_supported(route.route)

function _check_y_branch(route::Route)
    y_branch_supported(route) && return nothing
    error("ISEM Y transformation supports only s = -2 with IN and s = +2 with UP.")
end

function _route_y_amplitudes(route::Route)
    p = route.params
    if p.s == -2 && route.branch == :IN
        return (
            trans = route.solution_scale *
                ConversionFactors.Btrans(p.s, p.m, p.a, p.omega, p.lambda),
            incidence = ConversionFactors.Binc(p.s, p.m, p.a, p.omega, p.lambda) * route.incidence,
            reflection = ConversionFactors.Bref(p.s, p.m, p.a, p.omega, p.lambda) * route.reflection,
        )
    elseif p.s == 2 && route.branch == :UP
        return (
            trans = route.solution_scale *
                ConversionFactors.Ctrans(p.s, p.m, p.a, p.omega, p.lambda),
            incidence = ConversionFactors.Cinc(p.s, p.m, p.a, p.omega, p.lambda) * route.incidence,
            reflection = ConversionFactors.Cref(p.s, p.m, p.a, p.omega, p.lambda) * route.reflection,
        )
    end
    error("unsupported ISEM Y branch.")
end

# Y-function solution built from the GSN solution; supported for s = -2 IN and s = +2 UP.
function route_y_solution(route::Route; amplitudes=nothing)
    _check_y_branch(route)
    return first(_route_y_solution_checked(route, amplitudes === nothing ? _route_y_amplitudes(route) : amplitudes))
end

# r of the matching point of a real-frequency route (the converted Y uses horizon_in below it).
function _route_match_r(route::Route)
    p = route.params
    (p.omega isa Real && p.a isa Float64) || return nothing
    kappa = sqrt(1 - p.a^2)
    x = route.match_x
    return (1 + kappa - (1 - kappa) * x) / (1 - x)
end

# The Y solution of a route and its near-horizon self-check mismatch: an RstarYSolution for s = -2 IN at real omega
# in binary64, the Y converted from X otherwise (s = +2 UP; no self-check).
function _route_y_solution_checked(route::Route, amps)
    p = route.params
    iszero(p.omega) && error("ISEM Y transformation requires nonzero omega.")
    if p.s == -2 && route.branch === :IN && p.a isa Float64 && p.omega isa Float64
        y = RstarYSolution(route, amps)
        return y, y.mismatch
    end
    tt = _teukolsky_transformation_module()
    ysoln = getfield(tt, :GSN_to_Y_solution_from_matrix)(
        r -> rstar_from_r(p.a, r),
        r -> Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
            p.s, p.m, p.a, p.omega, p.lambda, r),
        gsn_solution_rstar(route),
        p.s, p.m, p.a, p.omega, p.lambda, amps.trans)
    return ysoln, 0.0
end

# Y of s = -2 IN near the horizon from the Teukolsky horizon series,
# Y = trans e^{-i phi} R_in / r^2 with d phi/dr = K / Delta, R_in the unit horizon
# series and `trans` the Teukolsky transmission amplitude of the solution, for
# r - r_+ < kappa (half the series' radius 2 kappa). The conversion Y = C11 X +
# C12 dX/dr* cancels there: to O(Delta^2) at the horizon (3e-7 at r - r_+ = 1e-3
# for (a, omega) = (0.8, 0.9) with X good to 1e-13), and by 1e4 to 1e6 for
# l = 3, omega = 2.3 - 3 at a >= 0.8, where Y is small next to C11 X and C12 X'
# (4e-7 at r_+ + 0.05 for (0.9, 2.35)).
# The series itself cancels for a large horizon exponent u = r_+ k / kappa: at
# r_+ + kappa its sum |terms| / |sum| was 1e17 for (a, m, omega) = (0.99, -2, 14)
# (u = 120; Y off by 8e-3 at r_+ + 0.75 kappa) and 1e12 for omega = 8. Where
# eps times that cancellation exceeds NEAR_HORIZON_SERIES_CANCELLATION_TOL the
# converted Y is used instead (1e-9 to 2e-8 there against the 256-bit reference).
#
# Self-check: the converted Y must agree with the series at r_+ + kappa / 2 and
# r_+ + kappa, at those of the two points where the series is used; returns the
# wrapped solution and the mismatch. A mismatch means either a wrong
# near-horizon solution or a conversion that loses that much just beyond
# r_+ + kappa (see y_radial_function).
# near_horizon_y wraps a Y given only as a function of r (the Riccati and linear routes); the Y of an ISEM route has
# the same series in its RstarYSolution.
const NEAR_HORIZON_CHECK_TOL = 1e-7
const NEAR_HORIZON_SERIES_CANCELLATION_TOL = 1e-8
const SERIES_TAIL_TOL = 1e-14

function near_horizon_y(ysoln, s, m, a, omega, lambda, trans)
    (s == -2 && omega isa Real && !iszero(omega) && trans isa Number && isfinite(trans) &&
        !iszero(trans)) || return ysoln, 0.0
    kappa = sqrt(1 - Float64(a)^2)
    kappa > 1e-3 || return ysoln, 0.0
    rplus = 1 + kappa
    rminus = 1 - kappa
    series_y(r) = _series_y(s, m, a, omega, lambda, trans, rplus, rminus, r)
    r_max = rplus + kappa
    mismatch = 0.0
    for f in (0.5, 1.0)
        S, _, tail, usable = series_y(rplus + f * kappa)
        tail <= SERIES_TAIL_TOL || return ysoln, 0.0
        usable || continue
        mismatch = max(mismatch, abs(ysoln(rplus + f * kappa)[1] / S - 1))
    end
    return (r -> begin
        rplus < r < r_max || return ysoln(r)
        Y, dY, tail, usable = series_y(r)
        (tail <= SERIES_TAIL_TOL && usable) || return ysoln(r)
        # (Y, Y', X, error estimate): no error estimate here, as for the converted Y
        return (Y, dY, ysoln(r)[3], 0.0)
    end), mismatch
end

# The horizon-series Y at r: (Y, dY/dr, truncation, whether the cancellation passes, R_in, dR_in/dr, cancellation),
# Y = trans e^{-i phi} R_in / r^2.
@inline function _series_y(s, m, a, omega, lambda, trans, rplus, rminus, r)
    Rin, dRin, _, _, tail, cancellation = teukolsky_horizon_pair(s, m, a, omega, lambda, r)
    phase = omega * rstar_from_r(a, r) -
        a * m / (rplus - rminus) * log((r - rplus) / (r - rminus))
    delta = r^2 - 2r + a^2
    K = (r^2 + a^2) * omega - a * m
    e = trans * exp(-im * phase)
    Y = e * Rin / r^2
    dY = e * (dRin / r^2 - 2Rin / r^3 - im * K / delta * Rin / r^2)
    return Y, dY, tail, eps(Float64) * cancellation <= NEAR_HORIZON_SERIES_CANCELLATION_TOL, Rin, dRin, cancellation
end

_warn_near_horizon_mismatch(mismatch, s, m, a, omega) =
    mismatch <= NEAR_HORIZON_CHECK_TOL || @warn "Y_radial: near-horizon self-check: the Teukolsky " *
        "series or Taylor chain and the Y converted from X differ by $(mismatch); " *
        "Y may be inaccurate." s m a omega maxlog = 20

# Teukolsky Taylor chain near the horizon (G-005).
# What: Y of s = -2 IN at real omega on (r_start, r_check) comes from Taylor series of the Teukolsky equation itself
#   (NearHorizonTeukolsky), started from the horizon series at r_start = r_+ + f kappa; below r_start the series, from
#   r_check on the converted Y. r_check lies a little beyond the matching point of the route (x = match_x + 0.05), where
#   the converted Y takes the infinity bases.
# Why: between the series and the matching point the converted Y = C11 X + C12 X' takes the X of the horizon_in basis,
#   and the conversion cancels by 1e4 - 4e4: (s, l, m, a, omega) = (-2, 3, 2, 0.9, 2.37) was off by 8e-7 at r = 2 - 3.4
#   with X off by 4e-10, (-2, 2, 1, 0.9, 1.2595) by 1e-7, (-2, 3, 3, 0.99, 1.6987) by 3e-8 (H-008: 3e-14 - 1e-12 with
#   the chain). f is the largest of NEAR_HORIZON_TEUKOLSKY_START where the series truncates (SERIES_TAIL_TOL) and cancels
#   by at most NEAR_HORIZON_TEUKOLSKY_START_CANCELLATION / eps, so that the start value is good to that level and a
#   change of f between neighbouring omega leaves only a small step: with the gate at 1e-12 the chains from f = 1/2 and
#   1/4 differed by 4e-11 at the switch of (2, -2, 0.99, 4.65); the gate is 1e-13.
# Self-check: the chain against the converted Y at r_check, where the two hand over; the difference is the `mismatch`
#   of the solution, warned about above NEAR_HORIZON_CHECK_TOL; the Y is returned either way.
# When there is no chain: only for structural reasons: no start point passes the series test, the matching point lies
#   inside it, or the chain cannot keep its step-control invariant. Then Y is the series below r_+ + kappa and the
#   converted Y above, with the self-check of near_horizon_y.
const NEAR_HORIZON_TEUKOLSKY_START = (0.5, 0.25, 0.125, 0.0625, 0.03125)
const NEAR_HORIZON_TEUKOLSKY_START_CANCELLATION = 1e-13

# Two Teukolsky series at infinity: Y of s = -2 IN at real omega > 0 is e^{-i psi} (B_inc R_in + B_ref R_out) / r^2
# wherever their truncation stays below the conversion's own error, eps (2 omega r)^4 (at least
# TWO_WAVE_SERIES_TOL); the converted Y = C11 X + C12 X' elsewhere. The conversion
# amplifies the rounding of (X, X') by about 16 omega^4 r^4 (Y at r = 160: 2e-6 at (a, l, omega) = (0.9, 2, 3), 7e-6
# at omega = 6), the series not at all (2e-13, 3e-13), given an accurate B_ref (Matching._complex_path_reflection).
# Coefficients are built per Y function and read only, so evaluation is thread safe and allocates nothing.
const TWO_WAVE_SERIES_TOL = 1e-14
const TWO_WAVE_SERIES_ORDER = 400

_two_wave_series_ok(err, omega, r) = err <= max(TWO_WAVE_SERIES_TOL, eps(Float64) * (2 * omega * r)^4)

# Radii scanned once per Y function for the first one where the series qualify (their truncation falls with r).
const TWO_WAVE_RADII = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0,
    250.0, 400.0, 1000.0)

# The two infinity series of s = -2 and the first of TWO_WAVE_RADII beyond r_+ from which they qualify (Inf: none).
function _two_wave_series(m, a, omega, lambda, rplus)
    cin = _infinity_in_coefficients(-2, m, a, omega, lambda, TWO_WAVE_SERIES_ORDER)
    cout = _infinity_out_coefficients(-2, m, a, omega, lambda, TWO_WAVE_SERIES_ORDER)
    i = findfirst(r -> r > rplus &&
        _two_wave_series_ok(_infinity_solutions(-2, a, omega, ComplexF64(r), cin, cout)[5], omega, r), TWO_WAVE_RADII)
    return cin, cout, i === nothing ? Inf : TWO_WAVE_RADII[i]
end

# Y of s = -2 IN at real omega in binary64 from an ISEM route: one read-only struct with the pieces described above,
# from the horizon out the horizon series, the Teukolsky Taylor chain, the converted Y and the two infinity series.
# y(r) is the Y_solution contract (Y, dY/dr, X, 0); y_at_rstar(y, rstar) evaluates the same pieces at the tortoise
# coordinate, with z = r - r_+, x and log x taken from t = logit(x) of r* (_logit_from_rstar) at their full relative
# precision: near the horizon Y is then good to eps times its condition number |r* dlog Y/dr*| ~ 2 |r*| |kappa / r_+
# - i k| (Y ~ Delta^2 e^{-2 i k r*}), where y(r_from_rstar(r*)) carries the rounding of r, eps r_+ / z (1e-8 to 5e-4
# at r* = -60, and r rounds to r_+ below r* ~ -120 at a = 0.9). For y_at_rstar the coefficients of the R_in horizon
# series are summed once, up to z_series: with a chain its start, where y(r) hands over too; without it the largest
# 2^-j kappa (j = 0..12) where the series passes the per-point test of near_horizon_y, which smaller z then pass too.
# That precision is the one of the series zone; where there is no series (kappa <= 1e-3) y_at_rstar converts X as y(r)
# does. Between r_check and r_two both evaluations convert X, which passes the rounding of its inputs on enlarged (to
# 1e-9 at (l, m, a, omega) = (2, 1, 0.9, 1.2595)); they agree there to that level only.
# The fields are the route and plain data, not closures over them: a type that nests a closure per piece makes every
# dynamically dispatched call of the solution 3 - 4 us slower (it is the type parameter of the Y of Y_radial).
struct RstarYSolution{R,C} <: Function
    a::Float64
    m::Int
    omega::Float64
    lambda::Float64
    rplus::Float64                  # 1 +- sqrt(1 - a^2), as y(r) takes them
    rminus::Float64
    kappa::Float64                  # of the route parameters, sqrt((1 - a)(1 + a)): y_at_rstar
    trans::ComplexF64
    route::R
    conversion::C                   # (r, r*) -> (C11, C12, C21, C22)
    near_until::Float64             # y(r): series or chain for r_+ < r < near_until (r_check, r_+ + kappa, or r_+: neither)
    r_start::Float64                # start of the chain: the series for r <= r_start
    chain::Union{Nothing,TeukolskyTaylorChain}
    r_check::Float64                # end of the chain (0 without it)
    mismatch::Float64               # near-horizon self-check
    series_rho::ComplexF64          # y_at_rstar: R = trans series_norm z^series_rho sum_n series[n + 1] z^n, z < z_series
    series_norm::ComplexF64
    series::Vector{ComplexF64}
    z_series::Float64
    incidence::ComplexF64           # Y = e^{-i psi} (incidence R_in + reflection R_out) / r^2 for r >= r_two
    reflection::ComplexF64
    cin::Vector{ComplexF64}
    cout::Vector{ComplexF64}
    r_two::Float64                  # Inf where the two series qualify nowhere
end

function RstarYSolution(route::Route, amps)
    p = route.params
    a, m, omega, lambda = p.a, p.m, p.omega, p.lambda
    kappa = sqrt(1 - a^2)
    rplus = 1 + kappa
    rminus = 1 - kappa
    trans = ComplexF64(amps.trans)
    conversion = _GSN_to_Y_coeffs_minus2_in(m, a, omega, lambda)
    series(r) = _series_y(-2, m, a, omega, lambda, trans, rplus, rminus, r)
    converted(r) = first(_converted_y_at_r(route, conversion, a, r))
    near_until, r_start, r_check, chain, mismatch, z_series = rplus, rplus, 0.0, nothing, 0.0, 0.0
    if isfinite(trans) && !iszero(trans) && kappa > 1e-3
        r_stop = _route_match_r(route)
        for f in NEAR_HORIZON_TEUKOLSKY_START
            _, _, tail, _, Rin, dRin, cancellation = series(rplus + f * kappa)
            (tail <= SERIES_TAIL_TOL && eps(Float64) * cancellation <= NEAR_HORIZON_TEUKOLSKY_START_CANCELLATION) || continue
            r_stop > rplus + f * kappa || break
            x_check = min((r_stop - rplus) / (r_stop - rminus) + 0.05, 0.95)
            check = (rplus - rminus * x_check) / (1 - x_check)
            chain = teukolsky_taylor_chain(-2, m, a, omega, lambda, rplus + f * kappa, Rin, dRin, check)
            chain === nothing || ((near_until, r_start, r_check) = (check, rplus + f * kappa, check))
            break
        end
        if chain !== nothing
            # The chain serves up to r_check: at the matching point itself the converted Y still takes horizon_in.
            mismatch = abs(_chain_y(chain, m, a, omega, trans, rplus, rminus, r_check)[1] / converted(r_check) - 1)
            z_series = r_start - rplus
        else
            near_until = rplus + kappa
            for f in (0.5, 1.0)
                S, _, tail, usable = series(rplus + f * kappa)
                if !(tail <= SERIES_TAIL_TOL)
                    near_until, mismatch = rplus, 0.0
                    break
                end
                usable || continue
                mismatch = max(mismatch, abs(converted(rplus + f * kappa) / S - 1))
                z_series = f * kappa
            end
            for j in 2:12       # the series cancels at r_+ + kappa / 2 already: the largest smaller 2^-j kappa
                z_series == 0 || break
                _, _, tail, usable = series(rplus + kappa / 2^j)
                tail <= SERIES_TAIL_TOL && usable && (z_series = kappa / 2^j)
            end
        end
    end
    cin, cout, r_two = omega > 0 && a >= 0 ? _two_wave_series(m, a, omega, lambda, rplus) :
        (ComplexF64[], ComplexF64[], Inf)
    # Below the end of the chain the chain is the more accurate Y (the two series carry the error of the matched B_ref:
    # 9e-7 at r = 2 - 3.4 for (l, m, a, omega) = (3, 2, 0.9, 2.37)).
    r_two = max(r_two, r_check)
    h = _horizon_frobenius(-2, m, a, omega, lambda, p.kappa)
    coefficients = ComplexF64[1.0]
    if z_series > 0
        partial = 1.0 + 0.0im
        zn = 1.0
        for n in 1:400
            push!(coefficients, _frobenius_coefficient(h, coefficients, h.rho_in, n))
            zn *= z_series
            term = coefficients[end] * zn
            partial += term
            n >= 4 && abs(term) <= eps(Float64) * abs(partial) && break
        end
    end
    return RstarYSolution(a, m, omega, lambda, rplus, rminus, p.kappa, trans, route, conversion, near_until, r_start,
        chain, r_check, mismatch, ComplexF64(h.rho_in), ComplexF64(h.norm_in), coefficients, z_series,
        ComplexF64(amps.incidence), ComplexF64(amps.reflection), cin, cout, r_two)
end

# Y, dY/dr of the Teukolsky Taylor chain at r.
@inline function _chain_y(chain, m, a, omega, trans, rplus, rminus, r)
    R, dR = teukolsky_taylor_state(chain, r)
    phase = omega * rstar_from_r(a, r) - a * m / (rplus - rminus) * log((r - rplus) / (r - rminus))
    delta = r^2 - 2r + a^2
    K = (r^2 + a^2) * omega - a * m
    e = trans * exp(-im * phase)
    return e * R / r^2, e * (dR / r^2 - 2R / r^3 - im * K / delta * R / r^2)
end

# Y converted from X at r, (C11 X + C12 X', C21 X + C22 X', X, 0), as TeukolskyTransformation evaluates it.
function _converted_y_at_r(route::Route, conversion, a, r)
    rs = rstar_from_r(a, r)
    Xvals = _gsn_state_rstar(route, rs)
    X = Xvals[1]
    Xp = Xvals[2]
    C11, C12, C21, C22 = conversion(r, rs)
    Y = C11 * X + C12 * Xp
    return (Y, C21 * X + C22 * Xp, X, zero(abs(Y)))
end

function (y::RstarYSolution)(r)
    if isfinite(y.r_two) && r isa Real && r >= y.r_two
        Rin, dRin, Rout, dRout, err = _infinity_solutions(-2, y.a, y.omega, ComplexF64(r), y.cin, y.cout)
        if _two_wave_series_ok(err, y.omega, r)
            R = y.incidence * Rin + y.reflection * Rout
            dR = y.incidence * dRin + y.reflection * dRout
            psi = y.omega * rstar_from_r(y.a, r) - y.a * y.m / (y.rplus - y.rminus) * log((r - y.rplus) / (r - y.rminus))
            delta = r^2 - 2r + y.a^2
            K = (r^2 + y.a^2) * y.omega - y.a * y.m
            e = cis(-psi) / r^2
            X = _gsn_state_rstar(y.route, rstar_from_r(y.a, r))[1]
            return (e * R, e * (dR - (im * K / delta + 2 / r) * R), X, zero(Float64))
        end
    end
    y.rplus < r < y.near_until || return _converted_y_at_r(y.route, y.conversion, y.a, r)
    if y.chain === nothing || r <= y.r_start
        Y, dY, tail, usable = _series_y(-2, y.m, y.a, y.omega, y.lambda, y.trans, y.rplus, y.rminus, r)
        (tail <= SERIES_TAIL_TOL && usable) || return _converted_y_at_r(y.route, y.conversion, y.a, r)
    else
        Y, dY = _chain_y(y.chain, y.m, y.a, y.omega, y.trans, y.rplus, y.rminus, r)
    end
    # (Y, Y', X, error estimate): no error estimate here, as for the converted Y
    return (Y, dY, _gsn_state_rstar(y.route, rstar_from_r(y.a, r))[1], 0.0)
end

# X and dX/dr* of the route at x, r, evaluated as gsn_solution_rstar does at the same x (the converted Y passes the
# rounding of X' on enlarged, so y(r) and y_at_rstar are kept alike): through r where the route needs the radius (MST
# region, endpoint basis or plan beyond the matching point), through x elsewhere. The route's results are not always
# inferred, hence the assertions.
function _route_x_state(route::Route, x, r)
    p = route.params
    if _mano_suzuki_takasugi_evaluation_region(route, x) ||
            (x > route.match_x && (route.endpoint_bridge isa ScaledBasis || route.endpoint_plan !== nothing))
        state = route_state(route, x; radius=r)
        return ComplexF64(state.X)::ComplexF64, ComplexF64(state.dXdx)::ComplexF64 * (2 * p.kappa * x / (r^2 + p.a^2))
    end
    state = route_state(route, x)
    return ComplexF64(state.X)::ComplexF64, ComplexF64(state.dXdx)::ComplexF64 * dx_drstar(p, x)
end

"""
    y_at_rstar(y::RstarYSolution, rstar)

(Y, dY/dr, X) at the tortoise coordinate `rstar`; in the horizon series zone good to eps times the condition number
of Y (see RstarYSolution). A point where the inversion of r*(x) does not converge (r - r_+ below the binary64 range)
is evaluated by y(r_from_rstar(a, rstar)).
"""
function y_at_rstar(y::RstarYSolution, rstar::Real)
    rs = Float64(rstar)
    kappa = y.kappa
    rplus = 1 + kappa
    t, converged = isfinite(rs) ? _logit_from_rstar(kappa, rs) : (NaN, false)
    if !converged
        Y, dY, X = y(r_from_rstar(y.a, rs))
        return (ComplexF64(Y)::ComplexF64, ComplexF64(dY)::ComplexF64, ComplexF64(X)::ComplexF64)
    end
    # One more Newton step: _logit_from_rstar stops at a residual of 8 ulps of r*, which Y carries times |dlog Y/dr*|.
    value, slope = _rstar_parts(kappa, rplus, 2log(kappa), rplus / kappa, t)
    step = (value - rs) / slope
    isfinite(step) && (t -= step)
    x, _, log_x, log_1mx = _logit_parts(t)
    z = 2 * kappa * exp(t)
    r = rplus + z
    X, dX = _route_x_state(y.route, x, r)
    K = (r^2 + y.a^2) * y.omega - y.a * y.m                                    # d psi/dr = K / Delta, Delta = z (z + 2 kappa)
    if z < y.z_series
        # Horizon series: R = trans norm z^rho S(z), rho = 2 - i u, so Y = trans norm (z / r)^2 e^{i theta} S with
        # theta = -u log z - psi, psi = omega r* - a m / (2 kappa) log x. With r* = (r_+ / kappa) t + F and
        # log x = t + log(1 - x), theta = -2 u t - u log(2 kappa) - omega F + a m / (2 kappa) log(1 - x): the parts of
        # omega r* and u log z that grow like r* cancel analytically, so theta carries the error of t times 2u and not
        # eps (omega + |m| Omega_H) |r*|. What remains at high m is the rounding of k = omega - m Omega_H itself.
        c = y.series
        S = c[end]
        dS = 0.0im
        @inbounds for n in length(c)-1:-1:1
            dS = dS * z + S
            S = S * z + c[n]
        end
        rho = y.series_rho
        u = -imag(rho)
        F = rplus + z + 2 * log(kappa) + (rplus / kappa - 2) * log_1mx
        theta = -2u * t - u * log(2 * kappa) - y.omega * F + y.a * y.m / (2 * kappa) * log_1mx
        Y1 = y.trans * y.series_norm * cis(theta) * z / r^2       # Y / (z S); no 1 / z below, so Y, dY -> 0 with z
        return (Y1 * z * S, Y1 * ((rho - im * K / (z + 2 * kappa) - 2 * z / r) * S + z * dS), X)
    end
    e = cis(-(y.omega * rs - y.a * y.m / (2 * kappa) * log_x)) / r^2       # e^{-i psi} / r^2
    dpsi = K / (z * (z + 2 * kappa))
    if r < y.r_check
        R, dR = teukolsky_taylor_state(y.chain, r)
        return (e * y.trans * R, e * y.trans * (dR - (im * dpsi + 2 / r) * R), X)
    elseif isfinite(y.r_two) && r >= y.r_two
        Rin, dRin, Rout, dRout, err = _infinity_solutions(-2, y.a, y.omega, ComplexF64(r), y.cin, y.cout)
        if _two_wave_series_ok(err, y.omega, r)
            R = y.incidence * Rin + y.reflection * Rout
            return (e * R, e * (y.incidence * dRin + y.reflection * dRout - (im * dpsi + 2 / r) * R), X)
        end
    end
    C11, C12, C21, C22 = y.conversion(r, rs)
    return (C11 * X + C12 * dX, C21 * X + C22 * dX, X)
end

# Wrap a route as a YRadialFunction (s = -2 IN or s = +2 UP), with the GSN
# amplitudes converted by the Teukolsky factors; a near-horizon mismatch above
# NEAR_HORIZON_CHECK_TOL warns.
function y_radial_function(route::Route)
    _check_y_branch(route)
    amps = _route_y_amplitudes(route)
    ysoln, mismatch = _route_y_solution_checked(route, amps)
    p = route.params
    _warn_near_horizon_mismatch(mismatch, p.s, p.m, p.a, p.omega)
    return _y_radial_function(route, amps, ysoln)
end

function _y_radial_function(route::Route, amps, ysoln)
    isem = _isem_module()
    isdefined(isem, :YRadialFunction) || error("ISEM.YRadialFunction is not loaded.")
    y_type = getfield(isem, :YRadialFunction)
    xsoln = r -> gsn_solution_rstar(route)(rstar_from_r(route.params.a, r))[1]
    yscalar = r -> ysoln(r)[1]
    return y_type(
        _route_mode(route), _route_boundary_condition(route), amps.trans, amps.incidence, amps.reflection, missing,
        missing, xsoln, yscalar, ysoln, UNIT_TEUKOLSKY_TRANS,
    )
end

function y_radial_function(route::ConjugatedRoute)
    positive = y_radial_function(route.route)
    isem = _isem_module()
    conjugate_function = getfield(isem, :_conjugate_y_radial_function)
    return conjugate_function(positive, route.params.m, route.params.omega)
end

end
