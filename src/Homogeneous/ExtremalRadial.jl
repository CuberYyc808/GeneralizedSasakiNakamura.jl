module ExtremalRadial

using DifferentialEquations
using StaticArrays
using TaylorSeries

using ..Kerr
using ..Coordinates
using ..InitialConditions
using ..Potentials
using ..Solutions

export is_exact_extremal_spin, extremal_horizon_detuning
export extremal_teukolsky_horizon_coefficients
export extremal_teukolsky_horizon_state, extremal_synchronous_exponents
export extremal_gsn_horizon_state, extremal_horizon_normalization
export solve_extremal_gsn

const _EXACT_SPIN_ATOL = 16eps(Float64)
const _SYNCHRONOUS_RTOL = 128eps(Float64)
const _NORMALIZATION_PRECISION_BITS = 384
const _NORMALIZATION_SERIES_ORDER = 48
const _SERIES_ORDER = 28
const _NORMALIZATION_CACHE = Dict{Any,Any}()
const _NORMALIZATION_CACHE_LOCK = ReentrantLock()

is_exact_extremal_spin(a) = isreal(a) &&
    abs(abs(float(real(a))) - 1.0) <= _EXACT_SPIN_ATOL

function _checked_extremal_sign(a)
    is_exact_extremal_spin(a) || throw(ArgumentError(
        "the exact-extremal radial basis requires a = +1 or a = -1"))
    return sign(real(a))
end

extremal_horizon_detuning(a, m::Integer, omega) =
    2omega - m * _checked_extremal_sign(a)

function _synchronous_scale(a, m, omega)
    return max(one(abs(omega)), abs(2omega), abs(m * _checked_extremal_sign(a)))
end

function _require_nonsynchronous(a, m, omega)
    detuning = extremal_horizon_detuning(a, m, omega)
    threshold = _SYNCHRONOUS_RTOL * _synchronous_scale(a, m, omega)
    abs(detuning) > threshold || throw(DomainError(omega,
        "omega = m*Omega_H is a synchronous extremal branch point; " *
        "IN/OUT plane-wave amplitudes and a simple-pole QNM residue are not defined there"))
    return detuning
end

function _branch_parameters(s::Int, m::Int, a, omega, branch::Symbol)
    detuning = _require_nonsynchronous(a, m, omega)
    if branch === :ingoing
        return im * detuning, -2s - 2im * omega
    elseif branch === :outgoing
        return -im * detuning, 2im * omega
    end
    throw(ArgumentError("extremal horizon branch must be :ingoing or :outgoing"))
end

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

function extremal_synchronous_exponents(s::Int, lambda, omega)
    discriminant = (2s + 1)^2 - 4 * (4omega^2 - lambda)
    root = sqrt(complex(discriminant))
    return ((-(2s + 1) + root) / 2, (-(2s + 1) - root) / 2)
end

function _poly_pair(coefficients, h)
    value = coefficients[end]
    derivative = zero(eltype(coefficients))
    for n in (length(coefficients) - 1):-1:1
        derivative = derivative * h + value
        value = value * h + coefficients[n]
    end
    return value, derivative
end

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

function extremal_horizon_normalization(
        s::Int, m::Int, a, omega, lambda, branch::Symbol;
        precision_bits::Int=_NORMALIZATION_PRECISION_BITS,
        order::Int=_NORMALIZATION_SERIES_ORDER)
    _require_nonsynchronous(a, m, omega)
    cache_key = (s, m, a, omega, lambda, branch, precision_bits, order)
    cached = lock(_NORMALIZATION_CACHE_LOCK) do
        get(_NORMALIZATION_CACHE, cache_key, nothing)
    end
    cached === nothing || return cached
    result = setprecision(precision_bits) do
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
    lock(_NORMALIZATION_CACHE_LOCK) do
        _NORMALIZATION_CACHE[cache_key] = result
    end
    return result
end

function extremal_gsn_horizon_state(
        s::Int, m::Int, a, omega, lambda, r, branch::Symbol;
        order::Int=_NORMALIZATION_SERIES_ORDER, normalization=nothing)
    normalization === nothing &&
        (normalization = extremal_horizon_normalization(
            s, m, a, omega, lambda, branch; order))
    raw = _raw_extremal_gsn_horizon_state(
        s, m, a, omega, lambda, r, branch; order)
    scale = convert(eltype(raw), normalization.value)
    return raw / scale
end

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

function _infinity_pair(s, m, a, omega, lambda, r, rs, order, branch)
    coefficient = branch === :outgoing ?
        n -> Solutions.outgoing_coefficient_at_inf(
            s, m, a, omega, lambda, n) :
        n -> Solutions.ingoing_coefficient_at_inf(
            s, m, a, omega, lambda, n)
    value = InitialConditions.fansatz(coefficient, omega, r; order)
    derivative = InitialConditions.dfansatz_dr(coefficient, omega, r; order)
    sign = branch === :outgoing ? one(omega) : -one(omega)
    phase = exp(im * sign * omega * rs)
    q = Kerr.Delta(a, r) / (r^2 + a^2)
    return SVector(
        phase * value,
        phase * (q * derivative + im * sign * omega * value),
    )
end

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
                q = Kerr.Delta(solution.a, r) / (r^2 + solution.a^2)
                return SVector(value, q * derivative)
            end
            return solution.transform(r, SVector(value, derivative))
        end
    end
    throw(DomainError(rs, "requested point lies outside the extremal series solution"))
end

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

function _patch_tail(coefficients, step)
    order = length(coefficients) - 1
    tail = sum(abs(coefficients[index + 1] * step^index)
        for index in max(0, order - 2):order)
    value, derivative = _evaluate_polynomial(coefficients, step)
    scale = max(one(tail), abs(value), abs(step * derivative))
    return tail / scale
end

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
        accepted || error("extremal series patch failed its truncation gate")
        rnext = r + step
        push!(patches, ExtremalSeriesPatch(
            min(r, rnext), max(r, rnext), r, coefficients))
        current = SVector(_evaluate_polynomial(coefficients, step)...)
        r = rnext
    end
    return ExtremalSeriesSolution(a, patches, representation, transform)
end

function _series_solution(s, m, a, omega, lambda, rsin, rsout,
        boundary::Symbol, representation::Symbol, horizon_state, infinity_state;
        order, tolerance)
    rin = Coordinates.r_from_rstar(a, rsin)
    rout = Coordinates.r_from_rstar(a, rsout)
    q(r) = Kerr.Delta(a, r) / (r^2 + a^2)
    dq(r) = ((2r - 2) * (r^2 + a^2) - 2r * Kerr.Delta(a, r)) /
        (r^2 + a^2)^2
    if representation === :gsn
        A = r -> Potentials.sF(s, m, a, omega, lambda, r) / q(r) - dq(r) / q(r)
        B = r -> Potentials.sU(s, m, a, omega, lambda, r) / q(r)^2
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

end
