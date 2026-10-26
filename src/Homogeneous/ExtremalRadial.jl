module ExtremalRadial

using OrdinaryDiffEq
using StaticArrays
using TaylorSeries

using ..Kerr
using ..Coordinates
using ..InitialConditions
using ..Potentials
using ..Solutions
using ..Transformation

export is_exact_extremal_spin, extremal_horizon_detuning
export extremal_teukolsky_horizon_coefficients
export extremal_teukolsky_horizon_state
export extremal_gsn_horizon_state, extremal_horizon_normalization
export extremal_horizon_closed_normalization, extremal_gsn_horizon_coefficients
export extremal_horizon_start
export solve_extremal_gsn, extremal_complex_path_amplitudes

const _NORMALIZATION_SERIES_ORDER = 48
const _SERIES_ORDER = 28

"""
    is_exact_extremal_spin(a)

True when `a` is exactly +1 or -1.
"""
is_exact_extremal_spin(a) = isreal(a) && isone(abs(real(a)))

function _checked_extremal_sign(a)
    is_exact_extremal_spin(a) || throw(ArgumentError(
        "the exact-extremal radial basis requires a = +1 or a = -1"))
    return sign(real(a))
end

"""
    extremal_horizon_detuning(a, m, omega)

Horizon detuning `2omega - m sign(a)`, i.e. `2(omega - m Omega_H)` at exact extremality.
"""
extremal_horizon_detuning(a, m::Integer, omega) =
    2omega - m * _checked_extremal_sign(a)

# Rejects the synchronous frequency omega = m Omega_H, a branch point of the extremal problem.
function _require_nonsynchronous(a, m, omega)
    detuning = extremal_horizon_detuning(a, m, omega)
    iszero(detuning) && throw(DomainError(omega,
        "omega = m*Omega_H is a synchronous extremal branch point; " *
        "IN/OUT plane-wave amplitudes and a simple-pole QNM residue are not defined there"))
    return detuning
end

# (alpha, beta) of the horizon behavior exp(alpha/h) h^beta, h = r - 1, for each branch.
function _branch_parameters(s::Int, m::Int, a, omega, branch::Symbol)
    detuning = _require_nonsynchronous(a, m, omega)
    if branch === :ingoing
        return im * detuning, -2s - 2im * omega
    elseif branch === :outgoing
        return -im * detuning, 2im * omega
    end
    throw(ArgumentError("extremal horizon branch must be :ingoing or :outgoing"))
end

"""
    extremal_teukolsky_horizon_coefficients(s, m, a, omega, lambda, branch; order)

Exact-extremal Teukolsky horizon expansion `R = exp(alpha/h) h^beta sum_n c_n h^n`, `h = r - 1`, for
`branch = :ingoing` or `:outgoing` (`c_0 = 1`). Returns `(alpha, beta, coefficients)`.
"""
function extremal_teukolsky_horizon_coefficients(
        s::Int, m::Int, a, omega, lambda, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER)
    order >= 0 || throw(ArgumentError("series order must be nonnegative"))
    alpha, beta = _branch_parameters(s, m, a, omega, branch)
    detuning = extremal_horizon_detuning(a, m, omega)
    coefficient_type = promote_type(
        typeof(complex(omega)), typeof(complex(lambda)), typeof(complex(a)))
    coefficients = zeros(coefficient_type, order + 1)
    coefficients[1] = one(coefficient_type)
    C0 = beta * (beta + 2s + 1) + 4omega^2 +
        2detuning * omega - lambda
    C1 = 4omega^2 + 2im * s * omega
    C2 = omega^2
    if order > 0
        for n in 0:(order - 1)
            Dn = n * (n - 1) + (2beta + 2s + 2) * n + C0
            previous1 = n >= 1 ? coefficients[n] : zero(coefficient_type)
            previous2 = n >= 2 ? coefficients[n - 1] : zero(coefficient_type)
            denominator = 2alpha * (n + 1)
            iszero(denominator) && throw(DomainError(denominator,
                "the nonsynchronous extremal recurrence encountered a zero denominator"))
            coefficients[n + 2] =
                (Dn * coefficients[n + 1] + C1 * previous1 +
                 C2 * previous2) / denominator
        end
    end
    return (alpha=alpha, beta=beta, coefficients=coefficients)
end

# Value and h derivative of sum coefficients[n] h^(n-1).
function _poly_pair(coefficients, h)
    value = coefficients[end]
    derivative = zero(eltype(coefficients))
    for n in (length(coefficients) - 1):-1:1
        derivative = derivative * h + value
        value = value * h + coefficients[n]
    end
    return value, derivative
end

"""
    extremal_teukolsky_horizon_state(s, m, a, omega, lambda, r, branch; order)

`(R, dR/dr)` of the exact-extremal horizon branch at radius `r`.
"""
function extremal_teukolsky_horizon_state(
        s::Int, m::Int, a, omega, lambda, r, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER)
    _checked_extremal_sign(a)
    h = r - one(r)
    iszero(h) && throw(DomainError(r, "the radial state is singular at r = r_+"))
    series = extremal_teukolsky_horizon_coefficients(
        s, m, a, omega, lambda, branch; order)
    value, derivative = _poly_pair(series.coefficients, h)
    prefactor = exp(series.alpha / h) * h^series.beta
    R = prefactor * value
    dRdr = prefactor *
        (derivative + (series.beta / h - series.alpha / h^2) * value)
    return SVector(R, dRdr)
end

_bigreal(x::Real) = BigFloat(x)
_bigcomplex(z) = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))

# The horizon branch converted to the GSN pair (X, dX/dr*), before normalization.
function _raw_extremal_gsn_horizon_state(
        s::Int, m::Int, a, omega, lambda, r, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER)
    radial = extremal_teukolsky_horizon_state(
        s, m, a, omega, lambda, r, branch; order)
    matrix = Solutions.
        Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
            s, m, a, omega, lambda, r)
    return matrix \ radial
end

# Factor that removes the leading horizon behavior of the raw GSN branch, leaving a constant as h -> 0.
function _leading_factor(a, m, omega, h, branch::Symbol)
    detuning = extremal_horizon_detuning(a, m, omega)
    if branch === :ingoing
        return exp(-im * detuning / h) * h^(2im * omega)
    end
    return exp(im * detuning / h) * h^(-2im * omega)
end

function _normalization_sample(s, m, a, omega, lambda, h, branch, order)
    state = _raw_extremal_gsn_horizon_state(
        s, m, a, omega, lambda, one(h) + h, branch; order)
    return state[1] * _leading_factor(a, m, omega, h, branch)
end

"""
    extremal_horizon_closed_normalization(s, m, a, omega, lambda, branch)

Leading amplitude of the raw GSN horizon branch at exact extremality in closed form, with
`delta = 2omega - sign(a) m` formed once: `N_in = sqrt(2)` (s = 0), `-4 sqrt(2) i delta` (s = -1),
`(2 sign(a) m + 2i(2 + lambda)) / (sqrt(2) delta)` (s = +1), `-4 sqrt(2) delta^2` (s = -2),
`-eta(1) / (2 sqrt(2) delta^2)` (s = +2), and `N_out = 2 eta(1) / N_in` (from the Wronskian of the two
branches), `eta` the GSN transformation function at r = 1. The general conversion factors cancel as
eps (m^2 + 4|omega|^2) / |delta|^2 there; these agree with the 384-bit extrapolation to 4e-14 at
|delta| = 0.3. Generic in the number type.
"""
function extremal_horizon_closed_normalization(s::Int, m::Int, a, omega, lambda, branch::Symbol)
    sigma = _checked_extremal_sign(a)
    detuning = _require_nonsynchronous(a, m, omega)
    eta1 = Transformation.eta(s, m, a, omega, lambda, one(real(float(detuning))))
    iszero(eta1) && throw(DomainError(eta1, "eta(r = 1) = 0: the GSN transformation is singular"))
    root2 = sqrt(2 * one(real(float(detuning))))
    incoming = s == 0 ? complex(root2) :
        s == -1 ? -4root2 * im * detuning :
        s == 1 ? (2sigma * m + 2im * (2 + lambda)) / (root2 * detuning) :
        s == -2 ? -4root2 * detuning^2 :
        s == 2 ? -eta1 / (2root2 * detuning^2) :
        throw(DomainError(s, "Currently only spin weight s of 0, +/-1, +/-2 are supported"))
    branch === :ingoing && return incoming
    branch === :outgoing && return 2eta1 / incoming
    throw(ArgumentError("extremal horizon branch must be :ingoing or :outgoing"))
end

"""
    extremal_horizon_normalization(s, m, a, omega, lambda, branch; precision_bits=nothing, order)

Leading amplitude of the raw GSN horizon branch. By default the closed form
(`extremal_horizon_closed_normalization`, in the working precision); with an explicit `precision_bits`,
the Richardson extrapolation to `h = r - 1 -> 0` of samples at `h`, `h/2`, `h/4` in BigFloat arithmetic.
Returns `(value, drift, base_h, precision_bits, order)`, with `drift` an estimate of the extrapolation error
(zero for the closed form, whose `base_h` is `nothing`).
"""
function extremal_horizon_normalization(
        s::Int, m::Int, a, omega, lambda, branch::Symbol;
        precision_bits::Union{Int,Nothing}=nothing,
        order::Int=_NORMALIZATION_SERIES_ORDER)
    _require_nonsynchronous(a, m, omega)
    if precision_bits === nothing
        value = extremal_horizon_closed_normalization(s, m, a, omega, lambda, branch)
        R = real(typeof(value))
        return (value=value, drift=zero(R), base_h=nothing,
            precision_bits=R <: BigFloat ? precision(R) : 53, order=order)
    end
    return setprecision(precision_bits) do
        abig = _bigreal(real(a))
        omegabig = _bigcomplex(omega)
        lambdabig = _bigcomplex(lambda)
        detuning = extremal_horizon_detuning(abig, m, omegabig)
        frequency_scale = max(one(BigFloat), abs(omegabig))
        base_h = min(BigFloat("1e-5"),
            BigFloat("1e-3") * abs(detuning) / frequency_scale)
        base_h = max(base_h, BigFloat("1e-10"))
        q1 = _normalization_sample(
            s, m, abig, omegabig, lambdabig, base_h, branch, order)
        q2 = _normalization_sample(
            s, m, abig, omegabig, lambdabig, base_h / 2, branch, order)
        q4 = _normalization_sample(
            s, m, abig, omegabig, lambdabig, base_h / 4, branch, order)
        richardson12 = 2q2 - q1
        richardson24 = 2q4 - q2
        leading = (4richardson24 - richardson12) / 3
        drift = abs(leading - richardson24) /
            max(one(BigFloat), abs(leading), abs(richardson24))
        return (value=leading, drift=drift, base_h=base_h,
            precision_bits=precision_bits, order=order)
    end
end

"""
    extremal_gsn_horizon_coefficients(s, m, a, omega, lambda, branch; order)

GSN horizon expansion at exact extremality, `X = exp(alpha/h) h^gamma S(h)`, `h = r - 1`,
`S = sum_n x_n h^n` with `x_0 = 1` (the normalized branch), from the GSN equation itself: with
`rho = r^2 + 1`, `K = rho omega - sign(a) m = delta + 2 omega h + omega h^2` (exact), `U = -(K/rho)^2 + h^2 W`
(`W = sW / rho^2`, Potentials) and `F = h^2 V` (`V = sF_delta` with Delta = 1), the equation
`X'' + (2/h - G) X' - (rho^2 U / h^4) X = 0` (`G = rho'/rho + rho V`) gives
`2 alpha (n+1) x_{n+1} = [n(n-1) + (2 gamma + 2) n + P_0] x_n + sum_{k>=1} P_k x_{n-k} - sum_{k>=0} G_k (n-1-k) x_{n-1-k}`
with `P = [alpha^2 - 2 alpha gamma h + (gamma^2 + gamma) h^2 + G (alpha h^2 - gamma h^3) + K^2 - rho^2 W h^2] / h^2`
(its h^0 and h^1 numerator terms vanish identically). Ingoing: `alpha = i delta`, `gamma = -2i omega`;
outgoing: `alpha = -i delta`, `gamma = 2i omega`. Returns `(alpha, gamma, coefficients)`.
"""
function extremal_gsn_horizon_coefficients(
        s::Int, m::Int, a, omega, lambda, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER)
    order >= 0 || throw(ArgumentError("series order must be nonnegative"))
    _checked_extremal_sign(a)
    detuning = _require_nonsynchronous(a, m, omega)
    alpha, gamma = branch === :ingoing ? (im * detuning, -2im * omega) :
        branch === :outgoing ? (-im * detuning, 2im * omega) :
        throw(ArgumentError("extremal horizon branch must be :ingoing or :outgoing"))
    C = promote_type(typeof(alpha), typeof(complex(lambda)))
    h = Taylor1(C, order + 2)
    r = 1 + h
    rho = r^2 + a * a
    K = detuning + h * (2omega + omega * h)
    W = Potentials.sW(s, m, a, omega, lambda, r) / rho^2
    V = Potentials.sF_delta(s, m, a, omega, lambda, r, one(C))
    G = 2r / rho + rho * V
    numerator = alpha^2 - 2alpha * gamma * h + (gamma^2 + gamma) * h^2 +
        G * (alpha * h^2 - gamma * h^3) + K^2 - rho^2 * W * h^2
    P(k) = getcoeff(numerator, k + 2)
    Gk(k) = getcoeff(G, k)
    coefficients = zeros(C, order + 1)
    coefficients[1] = one(C)
    for n in 0:(order - 1)
        acc = (n * (n - 1) + (2gamma + 2) * n + P(0)) * coefficients[n + 1]
        for k in 1:n
            acc += P(k) * coefficients[n - k + 1]
        end
        for k in 0:(n - 1)
            acc -= Gk(k) * (n - 1 - k) * coefficients[n - k]
        end
        coefficients[n + 2] = acc / (2alpha * (n + 1))
    end
    return (alpha=alpha, gamma=gamma, coefficients=coefficients, p0=P(0))
end

"""
    extremal_horizon_start(s, m, a, omega, lambda; cap=0.15)

Start `h = r - 1` of the asymptotic horizon expansion: `min(cap, |alpha| / max(32, |P_0|))`, `alpha = i delta` and
`P_0` the constant term of the recurrence (`extremal_gsn_horizon_coefficients`), where its terms
`x_n h^n ~ n! (h / 2|alpha|)^n` stay far below eps before they grow.
"""
function extremal_horizon_start(s::Int, m::Int, a, omega, lambda; cap=0.15)
    series = extremal_gsn_horizon_coefficients(s, m, a, omega, lambda, :ingoing; order=0)
    return min(cap, abs(series.alpha) / max(32, abs(series.p0)))
end

"""
    extremal_gsn_horizon_state(s, m, a, omega, lambda, r, branch; order, normalization=nothing)

Normalized GSN horizon state `(X, dX/dr*)` of `branch` at radius `r` (leading amplitude 1), from
`extremal_gsn_horizon_coefficients`; a `normalization` other than the closed form (an explicit-precision
result of `extremal_horizon_normalization`) rescales it by `N_closed / N`. Off the real axis the state is the
analytic continuation from real `h = r - 1 > 0` (principal `h^gamma`).
"""
function extremal_gsn_horizon_state(
        s::Int, m::Int, a, omega, lambda, r, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER, normalization=nothing)
    h = r - one(r)
    iszero(h) && throw(DomainError(r, "the radial state is singular at r = r_+"))
    series = extremal_gsn_horizon_coefficients(s, m, a, omega, lambda, branch; order)
    # The expansion is asymptotic (x_n ~ n!/(2 alpha)^n): summed up to its smallest term, whose size is the
    # truncation error.
    # The coefficients overflow at high order for small detuning (n!/(2 alpha)^n); only finite ones are summed.
    finite_terms = something(findfirst(!isfinite, series.coefficients), order + 2) - 1
    terms = abs.(view(series.coefficients, 1:finite_terms)) .* abs(h) .^ (0:(finite_terms - 1))
    last_term = argmin(terms)
    value, derivative = _poly_pair(view(series.coefficients, 1:last_term), h)
    prefactor = exp(series.alpha / h) * h^series.gamma
    rho = r^2 + a * a
    X = prefactor * value
    dX = (h^2 / rho) * prefactor * (derivative + (series.gamma / h - series.alpha / h^2) * value)
    normalization === nothing && return SVector(X, dX)
    # The branch with leading amplitude 1; a normalization other than the closed form (an explicit
    # Richardson value) rescales it by N_closed / N.
    closed = extremal_horizon_closed_normalization(s, m, a, omega, lambda, branch)
    closed == normalization.value && return SVector(X, dX)
    factor = closed / convert(typeof(closed), normalization.value)
    return SVector(X * factor, dX * factor)
end

# Decomposes a GSN state at r into the normalized ingoing and outgoing horizon branches.
function _horizon_decomposition(
        s, m, a, omega, lambda, state, r;
        order, ingoing_normalization, outgoing_normalization)
    state = SVector(state[1], state[2])
    ingoing = extremal_gsn_horizon_state(
        s, m, a, omega, lambda, r, :ingoing;
        order, normalization=ingoing_normalization)
    outgoing = extremal_gsn_horizon_state(
        s, m, a, omega, lambda, r, :outgoing;
        order, normalization=outgoing_normalization)
    basis = SMatrix{2,2}(ingoing[1], ingoing[2], outgoing[1], outgoing[2])
    coefficients = basis \ state
    return (ingoing=coefficients[1], outgoing=coefficients[2])
end

# Asymptotic GSN solution f(r) exp(+-i omega r*) at infinity and its r* derivative, for each branch.
function _infinity_pair(s, m, a, omega, lambda, r, rs, order, branch)
    coefficient = branch === :outgoing ?
        Solutions.coefficient_sequence(Solutions.outgoing_coefficient_at_inf,
            s, m, a, omega, lambda) :
        Solutions.coefficient_sequence(Solutions.ingoing_coefficient_at_inf,
            s, m, a, omega, lambda)
    value = InitialConditions.fansatz(coefficient, omega, r; order)
    derivative = InitialConditions.dfansatz_dr(coefficient, omega, r; order)
    sign = branch === :outgoing ? one(omega) : -one(omega)
    phase = exp(im * sign * omega * rs)
    q = _extremal_q(r)
    return SVector(
        phase * value,
        phase * (q * derivative + im * sign * omega * value),
    )
end

# Decomposes a GSN state at large r into the ingoing and outgoing infinity solutions.
function _infinity_decomposition(s, m, a, omega, lambda, state, r, rs, order)
    state = SVector(state[1], state[2])
    ingoing = _infinity_pair(
        s, m, a, omega, lambda, r, rs, order, :ingoing)
    outgoing = _infinity_pair(
        s, m, a, omega, lambda, r, rs, order, :outgoing)
    basis = SMatrix{2,2}(ingoing[1], ingoing[2], outgoing[1], outgoing[2])
    coefficients = basis \ state
    return (ingoing=coefficients[1], outgoing=coefficients[2])
end

# Piecewise Taylor solution in r: each patch holds a series about `center` valid on [left, right].
# The solution returns (X, dX/dr*) for :gsn, or converts the Teukolsky pair with `transform`.
struct ExtremalSeriesPatch{T,C}
    left::T
    right::T
    center::T
    coefficients::C
end

struct ExtremalSeriesSolution{T,P,F}
    a::T
    patches::P
    representation::Symbol
    transform::F
end

function _evaluate_polynomial(coefficients, z)
    value = coefficients[end]
    derivative = zero(eltype(coefficients))
    for index in (length(coefficients) - 1):-1:1
        derivative = derivative * z + value
        value = value * z + coefficients[index]
    end
    return value, derivative
end

function (solution::ExtremalSeriesSolution)(rs)
    r = Coordinates.r_from_rstar(solution.a, rs)
    for patch in solution.patches
        if patch.left <= r <= patch.right
            value, derivative = _evaluate_polynomial(
                patch.coefficients, r - patch.center)
            if solution.representation === :gsn
                return SVector(value, _extremal_q(r) * derivative)
            end
            return solution.transform(r, SVector(value, derivative))
        end
    end
    throw(DomainError(rs, "requested point lies outside the extremal series solution"))
end

# Taylor coefficients about r0 of y'' = A(r) y' + B(r) y with (y, y') = state at r0.
function _ode_series_coefficients(A, B, r0, state, order)
    Aseries = taylor_expand(A, r0; order=order)
    Bseries = taylor_expand(B, r0; order=order)
    coefficient_type = promote_type(typeof(state[1]), typeof(state[2]))
    coefficients = zeros(coefficient_type, order + 1)
    coefficients[1] = state[1]
    coefficients[2] = state[2]
    for n in 0:(order - 2)
        total = zero(coefficient_type)
        for k in 0:n
            total += getcoeff(Aseries, k) * (n - k + 1) *
                coefficients[n - k + 2]
            total += getcoeff(Bseries, k) * coefficients[n - k + 1]
        end
        coefficients[n + 3] = total / ((n + 2) * (n + 1))
    end
    return coefficients
end

# Relative size of the last three terms at the step, used as the truncation-error estimate.
function _patch_tail(coefficients, step)
    order = length(coefficients) - 1
    tail = sum(abs(coefficients[index + 1] * step^index)
        for index in max(0, order - 2):order)
    value, derivative = _evaluate_polynomial(coefficients, step)
    scale = max(one(tail), abs(value), abs(step * derivative))
    return tail / scale
end

# Propagates from rstart to rend with Taylor patches; each step is at most 0.22 (r - 1) and is halved
# until the tail estimate is below tolerance.
function _series_propagate(A, B, a, rstart, rend, state;
        representation::Symbol, transform=identity, order::Int=_SERIES_ORDER,
        tolerance=1e-12)
    direction = sign(rend - rstart)
    direction == 0 && throw(ArgumentError("series interval must be nonzero"))
    patches = ExtremalSeriesPatch[]
    r = rstart
    current = SVector(state[1], state[2])
    attempts = 0
    while direction * (rend - r) > 16eps(float(max(abs(r), abs(rend), 1)))
        attempts += 1
        attempts <= 2000 || error("extremal series propagation exceeded 2000 patches")
        maximum_step = min(abs(rend - r), 0.22 * abs(r - one(r)))
        step = direction * maximum_step
        accepted = false
        coefficients = nothing
        for _ in 1:16
            coefficients = _ode_series_coefficients(A, B, r, current, order)
            if _patch_tail(coefficients, step) <= max(tolerance, eps(Float64))
                accepted = true
                break
            end
            step /= 2
        end
        accepted || error("extremal series patch failed its truncation gate at r = $r (h = $(r - one(r))), last step $step")
        rnext = r + step
        push!(patches, ExtremalSeriesPatch(
            min(r, rnext), max(r, rnext), r, coefficients))
        current = SVector(_evaluate_polynomial(coefficients, step)...)
        r = rnext
    end
    return ExtremalSeriesSolution(a, patches, representation, transform)
end

# Coefficients of the GSN equation in r at |a| = 1, X'' + A X' + B X = 0 with A = F / q - q' / q, B = U / q^2,
# q = Delta / rho, in the factored form of the horizon expansion: h = r - 1 (exact near the horizon), Delta = h^2,
# F = h^2 V (V = sF_delta with Delta = 1), U = -(K / rho)^2 + h^2 W (W = sW / rho^2), K = delta + 2 omega h + omega h^2.
# Delta = r^2 - 2r + 1 cancels as h -> 0 and put a relative error of about eps / h^2 into A and B.
# q = Delta / rho at |a| = 1 with Delta = (r - 1)^2.
_extremal_q(r) = (r - 1)^2 / (r^2 + 1)

function _extremal_gsn_a(s, m, a, omega, lambda, r)
    h = r - 1
    rho = r^2 + 1
    V = Potentials.sF_delta(s, m, a, omega, lambda, r, one(h))
    return rho * V - 2 * (rho - r * h) / (rho * h)
end

function _extremal_gsn_b(s, m, a, omega, lambda, r)
    h = r - 1
    K = (2omega - a * m) + h * (2omega + omega * h)
    return (h^2 * Potentials.sW(s, m, a, omega, lambda, r) - K^2) / h^4
end

_extremal_gsn_ab(s, m, a, omega, lambda) = (r -> _extremal_gsn_a(s, m, a, omega, lambda, r),
    r -> _extremal_gsn_b(s, m, a, omega, lambda, r))

# Series solution of the GSN equation in r (:gsn) or of the Teukolsky equation (:teukolsky, converted
# back to GSN), started from the horizon state (IN) or the infinity state (UP).
function _series_solution(s, m, a, omega, lambda, rsin, rsout,
        boundary::Symbol, representation::Symbol, horizon_state, infinity_state;
        order, tolerance)
    rin = Coordinates.r_from_rstar(a, rsin)
    rout = Coordinates.r_from_rstar(a, rsout)
    q(r) = _extremal_q(r)
    if representation === :gsn
        A, B = _extremal_gsn_ab(s, m, a, omega, lambda)
        initial = boundary === :IN ?
            SVector(horizon_state[1], horizon_state[2] / q(rin)) :
            SVector(infinity_state[1], infinity_state[2] / q(rout))
        return _series_propagate(A, B, a,
            boundary === :IN ? rin : rout,
            boundary === :IN ? rout : rin,
            initial; representation=:gsn, order, tolerance)
    end

    A = r -> -2 * (s + 1) * (r - 1) / Kerr.Delta(a, r)
    B = r -> Potentials.VT(s, m, a, omega, lambda, r) / Kerr.Delta(a, r)
    matrix(r) = Solutions.
        Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
            s, m, a, omega, lambda, r)
    initial_gsn = boundary === :IN ? horizon_state : infinity_state
    initial_r = boundary === :IN ? rin : rout
    initial = matrix(initial_r) * initial_gsn
    transform = (r, radial) -> matrix(r) \ radial
    return _series_propagate(A, B, a,
        boundary === :IN ? rin : rout,
        boundary === :IN ? rout : rin,
        initial; representation=:teukolsky, transform, order, tolerance)
end

# Numerical integration in r* of the linear GSN equation or of its Riccati form.
function _differential_solution(s, m, a, omega, lambda, rsin, rsout,
        boundary::Symbol, method::Symbol, horizon_state, infinity_state;
        data_type, ode_algorithm, tolerance)
    initial = boundary === :IN ? horizon_state : infinity_state
    span = boundary === :IN ? (rsin, rsout) : (rsout, rsin)
    parameters = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    if method === :linear
        problem = ODEProblem(Solutions.GSN_linear_eqn,
            SVector(data_type(initial[1]), data_type(initial[2])), span, parameters)
        return solve(problem, ode_algorithm;
            reltol=tolerance, abstol=tolerance, maxiters=1_000_000)
    end
    phase, derivative = Solutions.PhiPhiprime_from_XXprime(
        initial[1], initial[2])
    problem = ODEProblem(Solutions.GSN_Riccati_eqn,
        SVector(data_type(phase), data_type(derivative)), span, parameters)
    phi_solution = solve(problem, ode_algorithm;
        reltol=tolerance, abstol=tolerance, maxiters=1_000_000)
    return Solutions.Xsoln_from_Phisoln(phi_solution), phi_solution
end

"""
    solve_extremal_gsn(s, m, a, omega, lambda, boundary, rsin, rsout; method=:gsn_isem, kwargs...)

Exact-extremal (`|a| = 1`) GSN solution with `boundary = :IN` or `:UP`, computed on `[rsin, rsout]` by
`method` (`:gsn_isem` Taylor patches on the GSN equation, `:isem` patches on the Teukolsky equation,
`:linear` or `:riccati` ODE integration) and extended outside by the horizon and infinity
expansions. Returns a NamedTuple with the solution, unit transmission amplitude, the incidence and
reflection amplitudes and the horizon normalizations.
"""
function solve_extremal_gsn(
        s::Int, m::Int, a, omega, lambda, boundary::Symbol,
        rsin, rsout;
        method::Symbol=:gsn_isem,
        horizon_order::Int=_NORMALIZATION_SERIES_ORDER,
        infinity_order::Int=12,
        series_order::Int=_SERIES_ORDER,
        data_type=Solutions._DEFAULTDATATYPE,
        ode_algorithm=Solutions._DEFAULTSOLVER,
        tolerance=Solutions._DEFAULTTOLERANCE)
    _checked_extremal_sign(a)
    _require_nonsynchronous(a, m, omega)
    boundary in (:IN, :UP) || throw(ArgumentError(
        "the extremal fundamental solver supports only :IN and :UP"))
    rin = Coordinates.r_from_rstar(a, rsin)
    rout = Coordinates.r_from_rstar(a, rsout)
    ingoing_normalization = extremal_horizon_normalization(
        s, m, a, omega, lambda, :ingoing; order=horizon_order)
    outgoing_normalization = extremal_horizon_normalization(
        s, m, a, omega, lambda, :outgoing; order=horizon_order)
    horizon_state = extremal_gsn_horizon_state(
        s, m, a, omega, lambda, rin, :ingoing;
        order=horizon_order, normalization=ingoing_normalization)
    infinity_state = _infinity_pair(
        s, m, a, omega, lambda, rout, rsout,
        infinity_order, :outgoing)

    phi_solution = nothing
    numerical = if method === :linear
        _differential_solution(s, m, a, omega, lambda, rsin, rsout,
            boundary, :linear, horizon_state, infinity_state;
            data_type, ode_algorithm, tolerance)
    elseif method === :riccati
        value, phi_solution = _differential_solution(
            s, m, a, omega, lambda, rsin, rsout,
            boundary, :riccati, horizon_state, infinity_state;
            data_type, ode_algorithm, tolerance)
        value
    elseif method === :isem
        _series_solution(s, m, a, omega, lambda, rsin, rsout,
            boundary, :teukolsky, horizon_state, infinity_state;
            order=series_order, tolerance)
    elseif method === :gsn_isem
        _series_solution(s, m, a, omega, lambda, rsin, rsout,
            boundary, :gsn, horizon_state, infinity_state;
            order=series_order, tolerance)
    else
        throw(ArgumentError("unsupported exact-extremal method $method"))
    end

    endpoint_state = boundary === :IN ? numerical(rsout) : numerical(rsin)
    amplitudes = if boundary === :IN
        _infinity_decomposition(
            s, m, a, omega, lambda, endpoint_state, rout, rsout,
            infinity_order)
    else
        _horizon_decomposition(
            s, m, a, omega, lambda, endpoint_state, rin;
            order=horizon_order, ingoing_normalization,
            outgoing_normalization)
    end

    function full_solution(rs)
        r = Coordinates.r_from_rstar(a, rs)
        if rs < rsin
            if boundary === :IN
                return extremal_gsn_horizon_state(
                    s, m, a, omega, lambda, r, :ingoing;
                    order=horizon_order, normalization=ingoing_normalization)
            end
            incoming = extremal_gsn_horizon_state(
                s, m, a, omega, lambda, r, :ingoing;
                order=horizon_order, normalization=ingoing_normalization)
            outgoing = extremal_gsn_horizon_state(
                s, m, a, omega, lambda, r, :outgoing;
                order=horizon_order, normalization=outgoing_normalization)
            return amplitudes.ingoing * incoming + amplitudes.outgoing * outgoing
        elseif rs > rsout
            if boundary === :UP
                return _infinity_pair(
                    s, m, a, omega, lambda, r, rs,
                    infinity_order, :outgoing)
            end
            incoming = _infinity_pair(
                s, m, a, omega, lambda, r, rs,
                infinity_order, :ingoing)
            outgoing = _infinity_pair(
                s, m, a, omega, lambda, r, rs,
                infinity_order, :outgoing)
            return amplitudes.ingoing * incoming + amplitudes.outgoing * outgoing
        end
        return numerical(rs)
    end

    return (
        numerical=numerical,
        riccati=phi_solution,
        solution=full_solution,
        transmission=data_type(1),
        incidence=boundary === :IN ? amplitudes.ingoing : amplitudes.outgoing,
        reflection=boundary === :IN ? amplitudes.outgoing : amplitudes.ingoing,
        amplitudes=amplitudes,
        ingoing_normalization=ingoing_normalization,
        outgoing_normalization=outgoing_normalization,
        horizon_state=horizon_state,
        infinity_state=infinity_state,
        method=method,
    )
end

# Tortoise coordinate at a = +1 continued along a path off the real axis (the branch of log(r - 1)
# that is continuous from r > 1); it reduces to Coordinates.rstar_from_r on the real axis.
_complex_extremal_rstar(r) = r - log(oftype(real(r), 4)) + 2 * log(r - 1) - 2 / (r - 1)

# Taylor patches of y'' = A(r) y' + B(r) y along the straight segments through `nodes`. The step is
# `frac` times the distance to the nearest singular point (r = 1, r = +-i), capped at `zmax` over the
# local wavenumber `wavenumber(r)`, so it depends only on the path and nearby frequencies share one
# patch grid. The last step of each segment ends exactly on the next node.
function _complex_path_propagate(A, B, nodes, state; order, frac, zmax, wavenumber)
    y = SVector(state[1], state[2])
    patches = 0
    tail = 0.0
    for j in 1:(length(nodes) - 1)
        r = nodes[j]
        rend = nodes[j + 1]
        while true
            distance = min(abs(r - 1), abs(r - im), abs(r + im))
            reach = min(frac * distance, zmax / wavenumber(r))
            dr = rend - r
            final = abs(dr) <= reach
            step = final ? dr : dr / abs(dr) * reach
            coefficients = _ode_series_coefficients(A, B, r, y, order)
            tail = max(tail, Float64(_patch_tail(coefficients, step)))
            y = SVector(_evaluate_polynomial(coefficients, step)...)
            patches += 1
            patches <= 50000 ||
                error("extremal complex-path propagation exceeded 50000 patches")
            final && break
            r += step
        end
    end
    return y, patches, tail
end

# Taylor coefficients about r0 of y'' = A y' + B y together with its frequency derivative
# z'' = A z' + B z + dA y' + dB y, from (y, y', z, z') at r0.
function _variational_series_coefficients(A, B, dA, dB, r0, state, order)
    As = taylor_expand(A, r0; order=order)
    Bs = taylor_expand(B, r0; order=order)
    dAs = taylor_expand(dA, r0; order=order)
    dBs = taylor_expand(dB, r0; order=order)
    T = promote_type(typeof(state[1]), typeof(state[3]))
    y = zeros(T, order + 1)
    z = zeros(T, order + 1)
    y[1], y[2], z[1], z[2] = state[1], state[2], state[3], state[4]
    for n in 0:(order - 2)
        ty = zero(T)
        tz = zero(T)
        for k in 0:n
            a = getcoeff(As, k)
            b = getcoeff(Bs, k)
            yp = (n - k + 1) * y[n - k + 2]
            ty += a * yp + b * y[n - k + 1]
            tz += a * (n - k + 1) * z[n - k + 2] + b * z[n - k + 1] +
                getcoeff(dAs, k) * yp + getcoeff(dBs, k) * y[n - k + 1]
        end
        y[n + 3] = ty / ((n + 2) * (n + 1))
        z[n + 3] = tz / ((n + 2) * (n + 1))
    end
    return y, z
end

# Five-point central difference of an analytic function of omega.
_omega_derivative(f, omega, h) =
    (8 * (f(omega + h) - f(omega - h)) - (f(omega + 2h) - f(omega - 2h))) / (12h)

function _complex_path_propagate_variational(A, B, dA, dB, nodes, state;
        order, frac, zmax, wavenumber)
    y = SVector(state[1], state[2], state[3], state[4])
    patches = 0
    tail = 0.0
    for j in 1:(length(nodes) - 1)
        r = nodes[j]
        rend = nodes[j + 1]
        while true
            distance = min(abs(r - 1), abs(r - im), abs(r + im))
            reach = min(frac * distance, zmax / wavenumber(r))
            dr = rend - r
            final = abs(dr) <= reach
            step = final ? dr : dr / abs(dr) * reach
            cy, cz = _variational_series_coefficients(A, B, dA, dB, r, y, order)
            tail = max(tail, Float64(_patch_tail(cy, step)))
            vy, dy = _evaluate_polynomial(cy, step)
            vz, dz = _evaluate_polynomial(cz, step)
            y = SVector(vy, dy, vz, dz)
            patches += 1
            patches <= 50000 ||
                error("extremal complex-path propagation exceeded 50000 patches")
            final && break
            r += step
        end
    end
    return y, patches, tail
end

# End point rmid + L exp(i phi) of the outer leg with Im(omega r*) = 0, where the two infinity
# branches exp(+-i omega r*) have equal size and the amplitude decomposition is well conditioned;
# phi starts at -arg(omega) and is corrected by secant steps for the log(r - 1) part of r*.
function _balanced_infinity_endpoint(omega, rmid, L)
    g(phi) = imag(omega * _complex_extremal_rstar(rmid + L * cis(phi)))
    phi0 = -angle(omega)
    phi1 = phi0 + oftype(phi0, 0.01)
    g0, g1 = g(phi0), g(phi1)
    for _ in 1:30
        g1 == g0 && break
        phi2 = phi1 - g1 * (phi1 - phi0) / (g1 - g0)
        phi0, g0 = phi1, g1
        phi1 = phi2
        g1 = g(phi1)
        abs(g1) <= 1e-10 * max(one(g1), abs(omega) * L) && break
    end
    return rmid + L * cis(phi1)
end

"""
    extremal_complex_path_amplitudes(s, m, a, omega, lambda; kwargs...)

Incidence and reflection amplitudes of the exact-extremal (`a = +1`) GSN IN solution (unit
transmission) for complex `omega`, from a path in the complex `r` plane. On the real axis the
horizon IN branch `exp(alpha/h) h^beta`, `alpha = i(2 omega - m)`, is the dominant branch for
`Re(alpha) > 0` and fixes no admixture of the outgoing branch, while at large real `r` the
outgoing infinity branch outgrows the ingoing one by `exp(2 |Im omega| r*)`. The path therefore
starts at `h = h0 exp(i(arg alpha - pi))`, where the IN branch is recessive and fixed by its
asymptotic series; passes `1 + 0.3 exp(i theta/2)` (theta the starting argument), the side reached
by continuity from real frequencies; reaches `r = rmid`; and leaves along `arg(r - rmid) = -arg omega`,
where the two infinity branches have equal size. Returns `(incidence, reflection, patches, tail,
endpoint)`, with `tail` the largest relative truncation estimate of the Taylor patches.
"""
function extremal_complex_path_amplitudes(s::Int, m::Int, a, omega, lambda;
        h0=nothing, rmid=2.0, rout=60.0, frac=0.1, zmax=1.5,
        horizon_order::Int=_NORMALIZATION_SERIES_ORDER, infinity_order::Int=12,
        series_order::Int=_SERIES_ORDER, dlambda=nothing, path_omega=omega)
    _checked_extremal_sign(a) > 0 || throw(ArgumentError(
        "the complex-path amplitudes are implemented for a = +1"))
    _require_nonsynchronous(a, m, omega)
    a = one(real(omega))
    alpha = im * (2omega - m)
    path_alpha = im * (2path_omega - m)
    theta = angle(path_alpha) - oftype(real(alpha), pi)
    theta <= -oftype(theta, pi) && (theta += 2 * oftype(theta, pi))
    # Start of the asymptotic horizon state: |alpha| / max(32, |P_0|) (extremal_horizon_start), at most 0.02.
    h0 === nothing && (h0 = extremal_horizon_start(s, m, a, omega, lambda; cap=oftype(real(alpha), 0.02)))
    rin = 1 + h0 * cis(theta)
    rturn = 1 + oftype(real(alpha), 0.3) * cis(theta / 2)
    rmidc = complex(oftype(real(alpha), rmid))
    rend = _balanced_infinity_endpoint(path_omega, rmidc, rout - rmid)
    q(r) = _extremal_q(r)
    # everything below as a function of omega on the fixed path, with lambda carried to
    # first order along omega
    lam(w) = dlambda === nothing ? lambda : lambda + (w - omega) * dlambda
    function start(w)
        normalization = extremal_horizon_normalization(
            s, m, a, w, lam(w), :ingoing; order=horizon_order)
        horizon = extremal_gsn_horizon_state(
            s, m, a, w, lam(w), rin, :ingoing; order=horizon_order, normalization)
        return SVector(horizon[1], horizon[2] / q(rin))
    end
    Aw(w, r) = _extremal_gsn_a(s, m, a, w, lam(w), r)
    Bw(w, r) = _extremal_gsn_b(s, m, a, w, lam(w), r)
    A = r -> Aw(omega, r)
    B = r -> Bw(omega, r)
    nodes = [rin, rturn, rmidc, rend]
    wavenumber = r -> abs(omega) + abs(alpha) / abs(r - 1)^2
    rs = _complex_extremal_rstar(rend)
    basis(w) = (inn = _infinity_pair(s, m, a, w, lam(w), rend, rs, infinity_order, :ingoing);
        out = _infinity_pair(s, m, a, w, lam(w), rend, rs, infinity_order, :outgoing);
        SMatrix{2,2}(inn[1], inn[2], out[1], out[2]))
    initial = start(omega)
    if dlambda === nothing
        y, patches, tail = _complex_path_propagate(A, B, nodes, initial;
            order=series_order, frac, zmax, wavenumber)
        state = SVector(y[1], q(rend) * y[2])
        coefficients = basis(omega) \ state
        return (incidence=coefficients[1], reflection=coefficients[2],
            patches=patches, tail=tail, endpoint=rend, state=state, initial=initial)
    end
    # frequency derivative of the IN solution on the same path (variational equation)
    hw = oftype(real(omega), 1.0e-3) * max(one(real(omega)), abs(omega))
    dinitial = _omega_derivative(start, omega, hw)
    dA = r -> _omega_derivative(w -> Aw(w, r), omega, hw)
    dB = r -> _omega_derivative(w -> Bw(w, r), omega, hw)
    y, patches, tail = _complex_path_propagate_variational(A, B, dA, dB, nodes,
        SVector(initial[1], initial[2], dinitial[1], dinitial[2]);
        order=series_order, frac, zmax, wavenumber)
    qend = q(rend)
    state = SVector(y[1], qend * y[2])
    dstate = SVector(y[3], qend * y[4])
    M = basis(omega)
    coefficients = M \ state
    dM = _omega_derivative(basis, omega, hw)
    dcoefficients = M \ (dstate - dM * coefficients)
    return (incidence=coefficients[1], reflection=coefficients[2],
        dincidence=dcoefficients[1], dreflection=dcoefficients[2],
        patches=patches, tail=tail, endpoint=rend, state=state, dstate=dstate,
        initial=initial, dinitial=dinitial)
end

end
