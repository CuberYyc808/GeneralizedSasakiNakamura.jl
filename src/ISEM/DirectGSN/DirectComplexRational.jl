module DirectComplexRational

using ....Coordinates: r_from_rstar, rstar_from_r
using ..DirectParameters:
    direct_gsn_controls,
    direct_gsn_parameters
using ..DirectCoefficientTables:
    DirectEndpointCoefficientSet,
    DirectRationalCoefficients,
    direct_gsn_coefficients
using ..DirectOrdinaryPointExpansion:
    direct_denom_ok,
    direct_endpoint_ab_series,
    direct_endpoint_power_lookup,
    direct_ordinary_ab_series!,
    direct_ordinary_point_coeffs!,
    direct_poly_pair,
    direct_poly_value,
    direct_poly_triple,
    direct_series_issue
using ..DirectOrdinaryPointExpansion:
    direct_shifted_ab!,
    direct_shifted_infinity_ab!
using ..DirectLocalSolutionAtZero:
    direct_horizon_exponent,
    direct_zero_local_solution
using ..DirectLFE:
    DDComplex,
    dc_add,
    dc_div,
    dc_imul,
    dc_mul,
    dc_neg,
    dc_scale,
    dc_sub,
    dc_value
using ..DirectNearExtreme: factor_values_complex_dd
using ..DirectLocalSolutionAtInfinity: direct_infinity_local_solution
using ..DirectIteration:
    _best_infinity_order_metrics,
    _horizon_eval_condition,
    _horizon_tail_metric,
    _infinity_adjacent_metric,
    _infinity_residual_metric,
    direct_endpoint_scale
using ..DirectMSTInfinity:
    direct_abel_denominator,
    direct_mst_principal_plan,
    direct_mst_principal_state

export DirectComplexRationalEvaluator
export direct_complex_rational_build, direct_complex_rational_state_r
export direct_complex_up_horizon_in_candidate
export direct_complex_up_initial_match_candidates
export direct_complex_offpole_match_plateau_candidate

const _DEFAULT_ORDER = 40
const _NEAR_REAL_MATCH_X = 0.7
const _NEAR_REAL_MAX_ABS_RE = 1.0
const _NEAR_REAL_MAX_RATIO = 1.0e-3
const _COORDINATE_TOLERANCE = 2.0e-12
const _COORDINATE_MAX_STEPS = 20_000
const _HORIZON_COORDINATE_STEP_BUDGET = 8192
const _MATCH_CONDITION_RETRY = 1.0e4
const _MATCH_CONSENSUS_KAPPA_MAX = 0.45
const _MATCH_CONSENSUS_IN_CANCELLATION_MIN = 200.0
const _MATCH_CONSENSUS_IN_CONDITION_MIN = 1.0e4
const _MATCH_CONSENSUS_IN_POLE_KAPPA_MAX = 0.05
const _MATCH_CONSENSUS_IN_POLE_IN_CANCELLATION_MIN = 1.0e8
const _MATCH_CONSENSUS_IN_POLE_REF_CANCELLATION_MIN = 10.0
const _MATCH_CONSENSUS_IN_POLE_CONDITION_MIN = 100.0
const _MATCH_CONSENSUS_IN_POLE_RESIDUAL_MAX = 1.0e-8
const _MATCH_CONSENSUS_IN_POLE_AGREEMENT_MAX = 1.0e-11
const _MATCH_CONSENSUS_IN_POLE_CONDITION_RATIO_MAX = 0.02
const _MATCH_CONSENSUS_UP_CANCELLATION_MIN = 250.0
const _MATCH_CONSENSUS_UP_CONDITION_MIN = 700.0
const _MATCH_CONSENSUS_AGREEMENT_MAX = 1.0e-9
const _MATCH_CONSENSUS_SEPARATION_MIN = 1.0e-9
const _MATCH_CONSENSUS_SEPARATION_RATIO_MIN = 10.0
const _OFFPOLE_PLATEAU_EARLY_STABILITY_MAX = 1.0e-13
const _HORIZON_STOKES_KAPPA_MAX = 0.05
const _HORIZON_STOKES_INITIAL_ANGLE = pi / 36
const _HORIZON_STOKES_ANGLE_STEP = pi / 144
const _HORIZON_STOKES_MAX_ANGLE_STEPS = 4
const _HORIZON_STOKES_AGREEMENT_MAX = 2.0e-11
const _HORIZON_STOKES_SEPARATION_MIN = 1.0e-10
const _HORIZON_STOKES_SEPARATION_RATIO_MIN = 8.0
const _PATH_DISK_SAFETY = 0.1
const _STEP_SAFETY = 0.4
const _MAX_PATCHES = 2048
const _EVALUATION_CHAIN_MAX = 256
const _PURE_IMAG_LATERAL_ANGLE = 3pi / 8
const _PURE_IMAG_LATERAL_RATIO = 1.0e-3
const _PURE_IMAG_LATERAL_ABS_MAX = 0.1
const _SCALED_Y_CHART_SOURCE = 0.125
const _SCALED_Y_CHART_GROWTH = 4.0
const _SCALED_Y_PHYSICAL_HANDOFF = 1.0e-2
const _SCALED_X_PHYSICAL_HANDOFF = 1.0e-2
const _HORIZON_RESONANCE_TOL = 2048eps(Float64)
const _PRINCIPAL_MST_Y_SCALES =
    (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24)
const _RHO_MAGNITUDES = ntuple(index -> Float64(2.0^(index - 1)), 17)
const _INFINITY_RHO_MAGNITUDES =
    ntuple(index -> Float64(2.0^(index - 1)), 33)

struct DirectComplexRationalSettings
    order::Int
    reduced_order::Int
    horizon_target::Float64
    real_horizon_target::Float64
    infinity_target::Float64
    step_target::Float64
end

@inline function _use_near_real_match(omega)
    abs_re = abs(real(omega))
    return 0.0 < abs_re <= _NEAR_REAL_MAX_ABS_RE &&
        abs(imag(omega)) <= _NEAR_REAL_MAX_RATIO * abs_re
end

function DirectComplexRationalSettings(order::Integer, tolerance::Real)
    resolved_order = Int(order)
    resolved_order >= 10 ||
        throw(ArgumentError("complex rational order must be at least 10."))
    resolved_tolerance = Float64(tolerance)
    isfinite(resolved_tolerance) && resolved_tolerance > 0 ||
        throw(ArgumentError("complex rational tolerance must be finite and positive."))
    return DirectComplexRationalSettings(
        resolved_order,
        max(2, resolved_order - 8),
        max(1.0e-12, 100 * resolved_tolerance),
        max(1.0e-11, 1000 * resolved_tolerance),
        max(1.0e-10, 10_000 * resolved_tolerance),
        max(2.0e-13, 20 * resolved_tolerance),
    )
end

struct DirectComplexRationalState
    X::ComplexF64
    dXdx::ComplexF64
end

struct DirectComplexRationalScratch
    pq::Vector{ComplexF64}
    avec::Vector{ComplexF64}
    bvec::Vector{ComplexF64}
    coeffs1::Vector{ComplexF64}
    coeffs2::Vector{ComplexF64}
end

function DirectComplexRationalScratch(coefficients, order::Integer)
    size = Int(order) + 1
    return DirectComplexRationalScratch(
        Vector{ComplexF64}(undef, coefficients.ordinary.value_count),
        Vector{ComplexF64}(undef, size),
        Vector{ComplexF64}(undef, size),
        Vector{ComplexF64}(undef, size),
        Vector{ComplexF64}(undef, size),
    )
end

struct DirectComplexRationalEvaluator{P,C,S}
    params::P
    coefficients::C
    settings::S
    match_x::Float64
    match_state::DirectComplexRationalState
end

@inline _finite_complex(value) = isfinite(real(value)) && isfinite(imag(value))
@inline _rplus(params) = 1.0 + params.kappa
@inline _rminus(params) = 1.0 - params.kappa

@inline function _compact_r(params, x)
    return (_rplus(params) - x * _rminus(params)) / (1 - x)
end

@inline function _compact_x(params, r)
    return (r - _rplus(params)) / (r - _rminus(params))
end

@inline _ray_sign(q) = real(q) >= 0 ? 1.0 : -1.0

function _two_ray_direction(q)
    iszero(q) && throw(DomainError(q, "two-ray phase frequency cannot be zero."))
    if imag(q) < 0 &&
            abs(q) <= _PURE_IMAG_LATERAL_ABS_MAX &&
            abs(real(q)) <= _PURE_IMAG_LATERAL_RATIO * abs(imag(q))
        stokes_direction = cis(-angle(q))
        side = real(q) < 0 ? -1.0 : 1.0
        direction =
            cos(_PURE_IMAG_LATERAL_ANGLE) +
            side * sin(_PURE_IMAG_LATERAL_ANGLE) * stokes_direction
        return ComplexF64(direction / abs(direction))
    end
    return ComplexF64(_ray_sign(q) * cis(-angle(q)))
end

@inline function _coordinate_rhs(params, direction, x)
    r = _compact_r(params, x)
    return direction * (2 * params.kappa * x) /
        (r * r + params.a * params.a)
end

@inline function _coordinate_rk4_step(params, direction, x, h)
    k1 = _coordinate_rhs(params, direction, x)
    k2 = _coordinate_rhs(params, direction, x + 0.5h * k1)
    k3 = _coordinate_rhs(params, direction, x + 0.5h * k2)
    k4 = _coordinate_rhs(params, direction, x + h * k3)
    return ComplexF64(x + (h / 6) * (k1 + 2k2 + 2k3 + k4))
end

@inline function _coordinate_step_error(full, half)
    delta = abs(half - full) / 15
    xscale = max(abs(half), floatmin(Float64))
    yscale = max(abs(1 - half), floatmin(Float64))
    return max(delta / xscale, delta / yscale)
end

@inline function _coordinate_logu_rhs(params, direction, logu)
    u = exp(logu)
    r = 2 * params.kappa * u + _rminus(params)
    return direction * (2 * params.kappa * (u - 1)) /
        (r * r + params.a * params.a)
end

@inline function _coordinate_logu_rk4_step(params, direction, logu, h)
    k1 = _coordinate_logu_rhs(params, direction, logu)
    k2 = _coordinate_logu_rhs(params, direction, logu + 0.5h * k1)
    k3 = _coordinate_logu_rhs(params, direction, logu + 0.5h * k2)
    k4 = _coordinate_logu_rhs(params, direction, logu + h * k3)
    return ComplexF64(logu + (h / 6) * (k1 + 2k2 + 2k3 + k4))
end

@inline function _coordinate_logu_step_error(full, half)
    return abs(half - full) / 15
end

function _integrate_infinity_path(
    params,
    q,
    rho_target;
    match_x,
    initial_path=nothing,
    angle_offset=0.0,
)
    target = Float64(rho_target)
    isfinite(target) && target > 0 ||
        throw(ArgumentError("infinity rho target must be finite and positive."))
    resolved_angle_offset = Float64(angle_offset)
    isfinite(resolved_angle_offset) ||
        throw(ArgumentError("two-ray angle offset must be finite."))
    direction = _two_ray_direction(q) * cis(resolved_angle_offset)
    rho, logu, u, x, nodes, logx, logy, accepted_steps, rejected_steps =
        if initial_path === nothing
            initial_x = ComplexF64(match_x)
            iszero(imag(initial_x)) && 0 < real(initial_x) < 1 ||
                throw(ArgumentError(
                    "two-ray match_x must be real and lie in (0,1)."))
            initial_u = inv(1 - initial_x)
            (
                0.0,
                log(initial_u),
                initial_u,
                initial_x,
                ComplexF64[initial_x],
                ComplexF64(log(real(initial_x))),
                ComplexF64(log(1 - real(initial_x))),
                0,
                0,
            )
        else
            initial_path.kind == :rotated ||
                throw(ArgumentError("only rotated paths can be extended."))
            target > initial_path.rho ||
                throw(ArgumentError(
                    "extended infinity path must increase rho."))
            abs(initial_path.direction - direction) <= 16eps(Float64) ||
                throw(ArgumentError("coordinate path direction mismatch."))
            initial_u = hasproperty(initial_path, :u) ?
                initial_path.u : inv(1 - initial_path.x)
            initial_logu = hasproperty(initial_path, :logu) ?
                initial_path.logu : log(initial_u)
            (
                initial_path.rho,
                initial_logu,
                initial_u,
                initial_path.x,
                copy(initial_path.nodes),
                initial_path.logx,
                initial_path.logy,
                initial_path.accepted_steps,
                initial_path.rejected_steps,
            )
        end
    h = min(0.25, (target - rho) / 8)

    while target - rho > 16eps(Float64) * max(1.0, target)
        accepted_steps < _COORDINATE_MAX_STEPS ||
            error("two-ray coordinate path exceeded step limit.")
        h = min(abs(h), target - rho)
        full = _coordinate_logu_rk4_step(
            params, direction, logu, h)
        half1 = _coordinate_logu_rk4_step(
            params, direction, logu, 0.5h)
        candidate_logu = _coordinate_logu_rk4_step(
            params, direction, half1, 0.5h)
        candidate_u = exp(candidate_logu)
        candidate_x = 1 - inv(candidate_u)
        finite = _finite_complex(candidate_logu) &&
            _finite_complex(candidate_u) &&
            _finite_complex(candidate_x) &&
            !iszero(candidate_u) && !iszero(candidate_u - 1) &&
            !iszero(candidate_x) && !iszero(1 - candidate_x)
        error_estimate = finite ?
            _coordinate_logu_step_error(full, candidate_logu) : Inf
        if finite && error_estimate <= _COORDINATE_TOLERANCE
            old_logu = logu
            old_u = u
            logu = candidate_logu
            u = candidate_u
            x = candidate_x
            rho += h
            logy += old_logu - logu
            logx += log((u - 1) / (old_u - 1)) + log(old_u / u)
            push!(nodes, x)
            accepted_steps += 1
            factor = iszero(error_estimate) ? 2.0 :
                clamp(0.9 * (_COORDINATE_TOLERANCE / error_estimate)^0.2,
                    0.5, 2.0)
            h *= factor
        else
            h *= 0.5
            rejected_steps += 1
            abs(h) > eps(Float64) * max(1.0, abs(rho)) ||
                error("two-ray coordinate path reached minimum step.")
        end
    end

    return (;
        rho=target,
        x,
        u,
        logu,
        nodes,
        logx,
        logy,
        direction,
        angle_offset=resolved_angle_offset,
        kind=:rotated,
        accepted_steps,
        rejected_steps,
    )
end

function _integrate_coordinate_path(
    params,
    q,
    rho_target;
    match_x,
    initial_path=nothing,
    angle_offset=0.0,
)
    target = Float64(rho_target)
    isfinite(target) && !iszero(target) ||
        throw(ArgumentError("two-ray rho target must be finite and nonzero."))
    target > 0 && return _integrate_infinity_path(
        params,
        q,
        target;
        match_x,
        initial_path,
        angle_offset,
    )
    resolved_angle_offset = Float64(angle_offset)
    isfinite(resolved_angle_offset) ||
        throw(ArgumentError("two-ray angle offset must be finite."))
    direction = _two_ray_direction(q) * cis(resolved_angle_offset)
    rho, x, nodes, logx, logy, accepted_steps, rejected_steps =
        if initial_path === nothing
            initial_x = ComplexF64(match_x)
            iszero(imag(initial_x)) && 0 < real(initial_x) < 1 ||
                throw(ArgumentError("two-ray match_x must be real and lie in (0,1)."))
            (
                0.0,
                initial_x,
                ComplexF64[initial_x],
                ComplexF64(log(real(initial_x))),
                ComplexF64(log(1 - real(initial_x))),
                0,
                0,
            )
        else
            initial_path.kind == :rotated ||
                throw(ArgumentError("only rotated paths can be extended."))
            sign(target) == sign(initial_path.rho) &&
                    abs(target) > abs(initial_path.rho) ||
                throw(ArgumentError("extended rho target must continue the existing ray."))
            abs(initial_path.direction - direction) <= 16eps(Float64) ||
                throw(ArgumentError("coordinate path direction mismatch."))
            (
                initial_path.rho,
                initial_path.x,
                copy(initial_path.nodes),
                initial_path.logx,
                initial_path.logy,
                initial_path.accepted_steps,
                initial_path.rejected_steps,
            )
        end
    step_sign = sign(target - rho)
    h = step_sign * min(0.25, abs(target - rho) / 8)

    while abs(target - rho) > 16eps(Float64) * max(1.0, abs(target))
        accepted_steps < _COORDINATE_MAX_STEPS ||
            error("two-ray coordinate path exceeded step limit.")
        h = step_sign * min(abs(h), abs(target - rho))
        full = _coordinate_rk4_step(params, direction, x, h)
        half1 = _coordinate_rk4_step(params, direction, x, 0.5h)
        candidate = _coordinate_rk4_step(params, direction, half1, 0.5h)
        finite = _finite_complex(candidate) && !iszero(candidate) &&
            !iszero(1 - candidate)
        error_estimate = finite ? _coordinate_step_error(full, candidate) : Inf
        if finite && error_estimate <= _COORDINATE_TOLERANCE
            oldx = x
            x = candidate
            rho += h
            logx += log(x / oldx)
            logy += log((1 - x) / (1 - oldx))
            push!(nodes, x)
            accepted_steps += 1
            factor = iszero(error_estimate) ? 2.0 :
                clamp(0.9 * (_COORDINATE_TOLERANCE / error_estimate)^0.2,
                    0.5, 2.0)
            h *= factor
        else
            h *= 0.5
            rejected_steps += 1
            abs(h) > eps(Float64) * max(1.0, abs(rho)) ||
                error("two-ray coordinate path reached minimum step.")
        end
    end

    return (;
        rho=target,
        x,
        nodes,
        logx,
        logy,
        direction,
        angle_offset=resolved_angle_offset,
        kind=:rotated,
        accepted_steps,
        rejected_steps,
    )
end

@inline function _ode_ab!(scratch, coefficients, x)
    direct_ordinary_ab_series!(
        scratch.avec, scratch.bvec, scratch.pq, coefficients, x, 0)
    return scratch.avec[1], scratch.bvec[1]
end

@inline function _state_score_from_ab(A, B, full, reduced, second)
    scale = max(
        abs(full.X), abs(full.dXdx), abs(reduced.X), abs(reduced.dXdx),
        floatmin(Float64))
    adjacent = max(
        abs(full.X - reduced.X),
        abs(full.dXdx - reduced.dXdx)) / scale
    residual_scale = abs(second) + abs(A * full.dXdx) + abs(B * full.X)
    residual = abs(second + A * full.dXdx + B * full.X) /
        max(residual_scale, floatmin(Float64))
    return max(adjacent, residual), adjacent, residual
end

function _horizon_state(
    coefficients,
    solution,
    kind,
    x,
    logx,
    order,
)
    coeffs = @view solution.coefficients[1:(order + 1)]
    series, dseries, ddseries = direct_poly_triple(coeffs, x)
    mu = solution.exponent
    prefactor = exp(mu * logx)
    log_derivative = mu / x
    log_second = -mu / x^2
    scale = direct_endpoint_scale(kind, coefficients.params)
    value = scale * prefactor * series
    derivative = scale * prefactor * (dseries + log_derivative * series)
    second = scale * prefactor * (
        ddseries + 2 * log_derivative * dseries +
        (log_derivative^2 + log_second) * series)
    if hasproperty(solution, :log_coefficient) &&
            !iszero(solution.log_coefficient)
        log_order = min(order, length(solution.log_coefficients) - 1)
        log_coeffs = @view solution.log_coefficients[1:(log_order + 1)]
        log_series, log_dseries, log_ddseries =
            direct_poly_triple(log_coeffs, x)
        log_mu = solution.log_exponent
        log_prefactor = exp(log_mu * logx)
        log_first = log_mu / x
        log_second_derivative = -log_mu / x^2
        logarithm = logx
        coefficient = solution.log_coefficient
        log_value = coefficient * log_prefactor * logarithm * log_series
        log_derivative_value = coefficient * log_prefactor * (
            logarithm * (log_dseries + log_first * log_series) +
            log_series / x)
        log_second_value = coefficient * log_prefactor * (
            logarithm * (
                log_ddseries +
                2 * log_first * log_dseries +
                (log_first^2 + log_second_derivative) * log_series) +
            2 * (log_dseries + log_first * log_series) / x -
            log_series / x^2)
        value += scale * log_value
        derivative += scale * log_derivative_value
        second += scale * log_second_value
    end
    return (X=ComplexF64(value), dXdx=ComplexF64(derivative),
        second=ComplexF64(second))
end

function _horizon_numerator(coeffs, a_power, b_power, mu, n::Int)
    numerator = zero(ComplexF64)
    @inbounds for j in 0:(n - 1)
        numerator +=
            a_power(j) * (mu + n - j - 1) * coeffs[n - j]
    end
    @inbounds for j in -1:(n - 2)
        numerator += b_power(j) * coeffs[n - j - 1]
    end
    return ComplexF64(numerator)
end

@inline _dc_zero(value::DDComplex) =
    iszero(value.high) && iszero(value.low)

function _dc_sqrt_real(value::DDComplex, initial::Float64)
    root = DDComplex(initial)
    @inbounds for _ in 1:3
        root = dc_scale(dc_add(root, dc_div(value, root)), 0.5)
    end
    return root
end

function _horizon_exponent_dd(params, branch::Symbol)
    one = DDComplex(1)
    a = DDComplex(params.a)
    kappa_squared = dc_mul(dc_sub(one, a), dc_add(one, a))
    kappa = _dc_sqrt_real(kappa_squared, params.kappa)
    r_plus = dc_add(one, kappa)
    numerator = dc_sub(
        dc_scale(dc_mul(r_plus, DDComplex(params.omega)), 2.0),
        dc_scale(a, Float64(params.m)),
    )
    nu_h = dc_div(numerator, dc_scale(kappa, 2.0))
    mu = dc_imul(nu_h)
    branch in (:in, :ingoing, :IN) && return dc_neg(mu)
    branch in (:out, :outgoing, :OUT) && return mu
    throw(ArgumentError("horizon branch must be :in or :out."))
end

function _horizon_numerator_dd(
    coeffs::Vector{DDComplex},
    a_power,
    b_power,
    mu::DDComplex,
    n::Int,
)
    numerator = DDComplex(0)
    @inbounds for j in 0:(n - 1)
        factor = dc_add(mu, DDComplex(Float64(n - j - 1)))
        term = dc_mul(_as_dd(a_power(j)), factor)
        numerator = dc_add(
            numerator, dc_mul(term, coeffs[n - j]))
    end
    @inbounds for j in -1:(n - 2)
        numerator = dc_add(
            numerator,
            dc_mul(_as_dd(b_power(j)), coeffs[n - j - 1]),
        )
    end
    return numerator
end

@inline _as_dd(value::DDComplex) = value
@inline _as_dd(value) = DDComplex(value)

function _rational_series_dd(numerator, denominator, count::Int)
    first = findfirst(value -> !_dc_zero(value), denominator)
    first === nothing &&
        error("noninvertible q0 in compensated horizon P/Q recurrence")
    shift = first - 1
    @inbounds for index in 1:min(shift, length(numerator))
        _dc_zero(numerator[index]) ||
            error("compensated horizon P/Q has an unmatched pole")
    end
    out = Vector{DDComplex}(undef, count)
    qinv = dc_div(DDComplex(1), denominator[first])
    @inbounds for n in 0:(count - 1)
        pindex = n + shift
        value = pindex < length(numerator) ?
            numerator[pindex + 1] : DDComplex(0)
        for k in 1:min(n, length(denominator) - shift - 1)
            value = dc_sub(value, dc_mul(
                denominator[shift + k + 1], out[n - k + 1]))
        end
        out[n + 1] = dc_mul(value, qinv)
    end
    return out
end

function _horizon_ab_dd(coefficients, order::Int)
    values = factor_values_complex_dd(coefficients.params)
    spin = coefficients.params.s
    b_length = spin == 0 ? 11 : 17
    q_start = 23 + b_length
    expected = spin == 0 ? 46 : 58
    length(values) == expected ||
        error("invalid compensated horizon factor layout")
    a_count = order + 2
    b_count = order + 3
    avec = _rational_series_dd(
        @view(values[1:11]), @view(values[12:22]), a_count)
    bvec = _rational_series_dd(
        @view(values[23:(q_start - 1)]),
        @view(values[q_start:expected]),
        b_count,
    )
    return avec, bvec
end

@inline function _dd_power_lookup(values, first_power::Int, power::Int)
    index = power - first_power + 1
    return 1 <= index <= length(values) ?
        values[index] : DDComplex(0)
end

function _horizon_log_forcing(coeffs, a_power, mu, q::Int)
    forcing = (2 * (mu + q) - 1 + a_power(-1)) * coeffs[q + 1]
    @inbounds for j in 0:(q - 1)
        forcing += a_power(j) * coeffs[q - j]
    end
    return ComplexF64(forcing)
end

function _horizon_plain_coefficients(
    a_power,
    b_power,
    mu,
    order::Int,
)
    coeffs = Vector{ComplexF64}(undef, order + 1)
    coeffs[1] = 1.0 + 0.0im
    a_minus_one = a_power(-1)
    b_minus_two = b_power(-2)
    @inbounds for n in 1:order
        numerator = _horizon_numerator(
            coeffs, a_power, b_power, mu, n)
        denominator = (mu + n) * (mu + n - 1) +
            a_minus_one * (mu + n) + b_minus_two
        scale = abs((mu + n) * (mu + n - 1)) +
            abs(a_minus_one * (mu + n)) + abs(b_minus_two)
        direct_denom_ok(denominator, scale) ||
            error("resonant companion horizon recurrence is ill-conditioned at n=$n")
        coeffs[n + 1] = -numerator / denominator
        direct_series_issue(coeffs, n + 1) == :none ||
            error("resonant companion horizon recurrence lost Float64 credibility at n=$n")
    end
    return coeffs
end

function _resonant_horizon_solution(
    coefficients,
    branch::Symbol,
    order::Int,
)
    other_branch = branch in (:in, :ingoing, :IN) ? :out : :in
    mu = ComplexF64(direct_horizon_exponent(
        coefficients.params, branch))
    log_mu = ComplexF64(direct_horizon_exponent(
        coefficients.params, other_branch))
    difference = log_mu - mu
    resonance = round(Int, real(difference))
    resonance >= 1 &&
        abs(difference - resonance) <=
            _HORIZON_RESONANCE_TOL * max(1.0, abs(difference)) ||
        error("horizon recurrence failure is not a positive-integer Frobenius resonance")
    resonance <= order ||
        error("horizon Frobenius resonance lies beyond the requested order")

    a_coeffs, b_coeffs =
        direct_endpoint_ab_series(coefficients, :H, order)
    a_dd, b_dd = _horizon_ab_dd(coefficients, order)
    a_power = power -> ComplexF64(
        dc_value(_dd_power_lookup(a_dd, -1, power)))
    b_power = power -> ComplexF64(
        dc_value(_dd_power_lookup(b_dd, -2, power)))
    companion = _horizon_plain_coefficients(
        a_power, b_power, log_mu, order)
    solution = Vector{ComplexF64}(undef, order + 1)
    solution[1] = 1.0 + 0.0im
    a_minus_one = a_power(-1)
    b_minus_two = b_power(-2)
    log_coefficient = zero(ComplexF64)

    @inbounds for n in 1:order
        numerator = _horizon_numerator(
            solution, a_power, b_power, mu, n)
        denominator = (mu + n) * (mu + n - 1) +
            a_minus_one * (mu + n) + b_minus_two
        if n == resonance
            forcing = _horizon_log_forcing(
                companion, a_power, log_mu, 0)
            !iszero(forcing) && isfinite(real(forcing)) &&
                isfinite(imag(forcing)) ||
                error("singular logarithmic Frobenius forcing at the horizon")
            log_coefficient = ComplexF64(-numerator / forcing)
            solution[n + 1] = zero(ComplexF64)
        else
            if n > resonance
                numerator += log_coefficient * _horizon_log_forcing(
                    companion, a_power, log_mu, n - resonance)
            end
            scale = abs((mu + n) * (mu + n - 1)) +
                abs(a_minus_one * (mu + n)) + abs(b_minus_two)
            direct_denom_ok(denominator, scale) ||
                error("logarithmic horizon recurrence is ill-conditioned at n=$n")
            solution[n + 1] = -numerator / denominator
        end
        direct_series_issue(solution, n + 1) == :none ||
            error("logarithmic horizon recurrence lost Float64 credibility at n=$n")
    end

    return (
        branch=branch,
        exponent=mu,
        coefficients=solution,
        A=ComplexF64.(dc_value.(a_dd)),
        B=ComplexF64.(dc_value.(b_dd)),
        variable=:x,
        requested_order=order,
        effective_order=order,
        truncation_reason=:none,
        representation=:log_frobenius,
        resonance_order=resonance,
        log_exponent=log_mu,
        log_coefficient,
        log_coefficients=companion,
    )
end

function _factorized_horizon_solution(
    coefficients,
    branch::Symbol,
    order::Int,
)
    other_branch = branch in (:in, :ingoing, :IN) ? :out : :in
    mu_dd = _horizon_exponent_dd(coefficients.params, branch)
    other_mu_dd = _horizon_exponent_dd(
        coefficients.params, other_branch)
    mu = ComplexF64(dc_value(mu_dd))
    other_mu = ComplexF64(dc_value(other_mu_dd))
    a_coeffs, b_coeffs =
        direct_endpoint_ab_series(coefficients, :H, order)
    a_power = direct_endpoint_power_lookup(a_coeffs, -1)
    b_power = direct_endpoint_power_lookup(b_coeffs, -2)
    solution = Vector{ComplexF64}(undef, order + 1)
    solution_dd = Vector{DDComplex}(undef, order + 1)
    solution[1] = 1.0 + 0.0im
    solution_dd[1] = DDComplex(1)

    @inbounds for n in 1:order
        numerator = _horizon_numerator_dd(
            solution_dd, a_power, b_power, mu_dd, n)
        detuning = dc_sub(
            dc_add(mu_dd, DDComplex(Float64(n))), other_mu_dd)
        detuning_value = ComplexF64(dc_value(detuning))
        resonance_scale = max(
            1.0, abs(mu), abs(other_mu), Float64(n))
        (_dc_zero(detuning) ||
            abs(detuning_value) <=
                _HORIZON_RESONANCE_TOL * resonance_scale) &&
            return _resonant_horizon_solution(
                coefficients, branch, order)
        denominator = dc_scale(detuning, Float64(n))
        value_dd = dc_div(dc_neg(numerator), denominator)
        value = ComplexF64(dc_value(value_dd))
        isfinite(real(value)) && isfinite(imag(value)) ||
            error("factorized horizon recurrence produced a nonfinite coefficient at n=$n")
        solution[n + 1] = value
        solution_dd[n + 1] = value_dd
        issue = direct_series_issue(solution, n + 1)
        issue in (:none, :consecutive_ratio_growth) ||
            error("factorized horizon recurrence lost Float64 credibility at n=$n: $issue")
    end

    return (
        branch=branch,
        exponent=mu,
        coefficients=solution,
        A=a_coeffs,
        B=b_coeffs,
        variable=:x,
        requested_order=order,
        effective_order=order,
        truncation_reason=:none,
        representation=:factorized_frobenius,
        resonance_order=0,
        log_exponent=other_mu,
        log_coefficient=zero(ComplexF64),
        log_coefficients=ComplexF64[],
    )
end

function _complex_horizon_solution(
    coefficients,
    branch::Symbol,
    order::Int,
)
    try
        solution = direct_zero_local_solution(
            coefficients, branch, order)
        if solution.effective_order < order &&
                solution.truncation_reason == :small_recurrence_denominator
            return _factorized_horizon_solution(
                coefficients, branch, order)
        end
        return solution
    catch error
        occursin(
            "direct GSN horizon recurrence lost Float64 credibility",
            sprint(showerror, error),
        ) || rethrow()
        return _factorized_horizon_solution(
            coefficients, branch, order)
    end
end

function _infinity_state(
    coefficients,
    solution,
    kind,
    x,
    logy,
    order,
)
    y = 1 - x
    coeffs = @view solution.coefficients[1:(order + 1)]
    variable_scale = hasproperty(solution, :variable_scale) ?
        solution.variable_scale : 1.0 + 0.0im
    variable = y / variable_scale
    series, dseries, ddseries = direct_poly_triple(coeffs, variable)
    rho = solution.rho
    sigma = solution.sigma
    prefactor = exp(sigma / y + rho * logy)
    log_derivative = -sigma / y^2 + rho / y
    log_second = 2 * sigma / y^3 - rho / y^2
    inv_variable_scale = inv(variable_scale)
    scale = direct_endpoint_scale(kind, coefficients.params)
    value = scale * prefactor * series
    derivative_y = scale * prefactor * (
        dseries * inv_variable_scale + log_derivative * series)
    second_y = scale * prefactor * (
        ddseries * inv_variable_scale^2 +
        2 * log_derivative * dseries * inv_variable_scale +
        (log_derivative^2 + log_second) * series)
    return (X=ComplexF64(value), dXdx=ComplexF64(-derivative_y),
        second=ComplexF64(second_y))
end

function _scaled_infinity_order_metrics(
    A,
    B,
    coefficients,
    solution,
    kind,
    path,
    order::Int,
)
    full = _infinity_state(
        coefficients, solution, kind, path.x, path.logy, order)
    previous = _infinity_state(
        coefficients, solution, kind, path.x, path.logy, max(1, order - 1))
    score, adjacent, residual = _state_score_from_ab(
        A, B, full, previous, full.second)
    variable = (1 - path.x) / solution.variable_scale
    polynomial = direct_poly_value(
        @view(solution.coefficients[1:(order + 1)]), variable)
    tail = abs(solution.coefficients[order + 1] * variable^order) /
        max(abs(polynomial), floatmin(Float64))
    return (; order, score=max(score, tail), adjacent, residual, tail)
end

function _best_scaled_infinity_metrics(
    A,
    B,
    coefficients,
    solution,
    kind,
    path,
    settings,
)
    max_order = min(settings.order, solution.effective_order)
    min_order = min(12, max_order)
    best = nothing
    for order in min_order:max_order
        candidate = _scaled_infinity_order_metrics(
            A, B, coefficients, solution, kind, path, order)
        if best === nothing || candidate.score < best.score
            best = candidate
        end
    end
    return best
end

function _horizon_check_from_ab(
    A,
    B,
    coefficients,
    solution,
    kind,
    path,
    settings,
)
    full_order = min(settings.order, solution.effective_order)
    reduced_order = max(2, full_order - 8)
    full = _horizon_state(
        coefficients, solution, kind, path.x, path.logx, full_order)
    reduced = _horizon_state(
        coefficients, solution, kind, path.x, path.logx, reduced_order)
    score, adjacent, residual = _state_score_from_ab(
        A, B, full, reduced, full.second)
    return (; score, adjacent, residual, order=full_order,
        roundoff=0.0, tail=0.0,
        effective_order=solution.effective_order,
        truncation_reason=solution.truncation_reason,
        state=DirectComplexRationalState(full.X, full.dXdx))
end

function _horizon_check_from_endpoint(
    coefficients,
    solution,
    kind,
    path,
    settings,
)
    full_order = min(settings.order, solution.effective_order)
    reduced_order = max(2, full_order - 8)
    full = _horizon_state(
        coefficients, solution, kind, path.x, path.logx, full_order)
    reduced = _horizon_state(
        coefficients, solution, kind, path.x, path.logx, reduced_order)
    A = direct_poly_value(solution.A, path.x) / path.x
    B = direct_poly_value(solution.B, path.x) / path.x^2
    score, adjacent, residual = _state_score_from_ab(
        A, B, full, reduced, full.second)
    coeffs = @view solution.coefficients[1:(full_order + 1)]
    mu = solution.exponent
    roundoff = 8eps(Float64) * _horizon_eval_condition(
        coeffs, mu, path.x)
    tail = _horizon_tail_metric(coeffs, path.x)
    return (; score=max(score, tail), adjacent, residual, order=full_order,
        roundoff, tail,
        effective_order=solution.effective_order,
        truncation_reason=solution.truncation_reason,
        state=DirectComplexRationalState(full.X, full.dXdx))
end

function _infinity_check_from_ab(
    A,
    B,
    coefficients,
    solution,
    kind,
    path,
    settings,
)
    y = 1 - path.x
    variable_scale = hasproperty(solution, :variable_scale) ?
        solution.variable_scale : 1.0 + 0.0im
    best = if isone(variable_scale)
        _best_infinity_order_metrics(
            solution.A, solution.B, solution.coefficients,
            solution.rho, solution.sigma, y)
    else
        _best_scaled_infinity_metrics(
            A, B, coefficients, solution, kind, path, settings)
    end
    order = min(best.order, solution.effective_order)
    reduced_order = max(2, order - 8)
    full = _infinity_state(
        coefficients, solution, kind, path.x, path.logy, order)
    reduced = _infinity_state(
        coefficients, solution, kind, path.x, path.logy, reduced_order)
    score, adjacent, residual = _state_score_from_ab(
        A, B, full, reduced, full.second)
    return (;
        score=max(best.score, score),
        adjacent,
        residual,
        roundoff=0.0,
        order,
        state=DirectComplexRationalState(full.X, full.dXdx),
    )
end

function _complex_infinity_solution(coefficients, branch::Symbol, order::Int)
    omega = coefficients.params.omega
    if !iszero(omega) && abs(omega) <= _SCALED_Y_PHYSICAL_HANDOFF
        return direct_infinity_local_solution(
            coefficients,
            branch,
            order;
            variable_scale=omega,
        )
    end
    try
        return direct_infinity_local_solution(coefficients, branch, order)
    catch exception
        message = sprint(showerror, exception)
        occursin(
            "direct GSN infinity recurrence lost Float64 credibility",
            message,
        ) || rethrow()
        return direct_infinity_local_solution(
            coefficients,
            branch,
            order;
            variable_scale=omega,
        )
    end
end

function _horizon_candidate!(
    scratch,
    coefficients,
    solutions,
    kinds,
    path,
    settings,
    endpoint_rational=false,
)
    checks = if endpoint_rational
        map((solution, kind) -> _horizon_check_from_endpoint(
            coefficients, solution, kind, path, settings),
            solutions, kinds)
    else
        A, B = _ode_ab!(scratch, coefficients, path.x)
        map((solution, kind) -> _horizon_check_from_ab(
            A, B, coefficients, solution, kind, path, settings),
            solutions, kinds)
    end
    pair_abel_error = if length(checks) == 2
        try
            denominator = direct_abel_denominator(
                coefficients, :horizon, path.x)
            denominator.status == "OK" || error(denominator.status)
            _abel_normalize_second(
                checks[1].state,
                checks[2].state,
                denominator.denominator,
            ).corrected_error
        catch
            Inf
        end
    else
        0.0
    end
    return (;
        path,
        checks,
        score=maximum(check.score for check in checks),
        pair_abel_error,
    )
end

@inline _pair_abel_ok(candidate, target) =
    target === nothing || candidate.pair_abel_error <= target

function _infinity_candidate!(
    scratch,
    coefficients,
    solutions,
    kinds,
    path,
    settings,
)
    A, B = _ode_ab!(scratch, coefficients, path.x)
    checks = map((solution, kind) -> _infinity_check_from_ab(
        A, B, coefficients, solution, kind, path, settings),
        solutions, kinds)
    return (; path, checks, score=maximum(check.score for check in checks))
end

function _select_real_horizon_path(
    coefficients,
    solutions,
    kinds,
    match_x,
    settings,
    ;
    pair_abel_target=nothing,
)
    best = nothing
    structural = nothing
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    for exponent in 1:40
        seed_x = match_x * exp2(-exponent)
        path = (;
            rho=NaN,
            x=ComplexF64(seed_x),
            nodes=ComplexF64[match_x, seed_x],
            logx=ComplexF64(log(seed_x)),
            logy=ComplexF64(log1p(-seed_x)),
            direction=ComplexF64(1),
            angle_offset=0.0,
            kind=:real,
            accepted_steps=0,
            rejected_steps=0,
        )
        candidate = _horizon_candidate!(
            scratch, coefficients, solutions, kinds, path, settings, true)
        if isfinite(candidate.score) &&
                (best === nothing || candidate.score < best.score)
            best = candidate
        end
        structural_ok = all(check ->
            check.effective_order == settings.order &&
            check.truncation_reason == :none &&
            check.adjacent <= settings.horizon_target &&
            check.tail <= settings.horizon_target &&
            check.residual <= settings.infinity_target,
            candidate.checks)
        structural_ok &= _pair_abel_ok(candidate, pair_abel_target)
        structural === nothing && structural_ok &&
            (structural = candidate)
        candidate.score <= settings.real_horizon_target &&
            _pair_abel_ok(candidate, pair_abel_target) && return candidate
    end
    best_check = best === nothing ? nothing : reduce(
        (left, right) -> left.score >= right.score ? left : right,
        best.checks)
    if _principal_sfe_evaluator_axis(coefficients.params.omega) &&
            best !== nothing && best.score <= 1.0e-8
        return best
    end
    structural !== nothing && return structural
    error("no certified real-axis horizon path; best_score=" *
        "$(best === nothing ? Inf : best.score), best_x=" *
        "$(best === nothing ? NaN : best.path.x), adjacent=" *
        "$(best_check === nothing ? Inf : best_check.adjacent), residual=" *
        "$(best_check === nothing ? Inf : best_check.residual), roundoff=" *
        "$(best_check === nothing ? Inf : best_check.roundoff), order=" *
        "$(best_check === nothing ? 0 : best_check.effective_order), reason=" *
        "$(best_check === nothing ? :none : best_check.truncation_reason), tail=" *
        "$(best_check === nothing ? Inf : best_check.tail), pair_abel_error=" *
        "$(best === nothing ? Inf : best.pair_abel_error).")
end

function _select_rotated_horizon_path(
    coefficients,
    solutions,
    kinds,
    p,
    match_x,
    settings,
    angle_offset=0.0,
    ;
    max_best_score=nothing,
    pair_abel_target=nothing,
)
    best = nothing
    path = nothing
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    for magnitude in _RHO_MAGNITUDES
        path = _integrate_coordinate_path(
            coefficients.params, p, -magnitude;
            match_x, initial_path=path, angle_offset)
        candidate = _horizon_candidate!(
            scratch, coefficients, solutions, kinds, path, settings)
        if isfinite(candidate.score) &&
                (best === nothing || candidate.score < best.score)
            best = candidate
        end
        candidate.score <= settings.horizon_target &&
            _pair_abel_ok(candidate, pair_abel_target) && return candidate
        if path.accepted_steps > _HORIZON_COORDINATE_STEP_BUDGET
            max_best_score === nothing &&
                error("two-ray coordinate path exceeded horizon search budget; " *
                    "accepted_steps=$(path.accepted_steps), best_score=" *
                    "$(best === nothing ? Inf : best.score).")
            break
        end
    end
    if max_best_score !== nothing && best !== nothing &&
            best.score <= max_best_score &&
            _pair_abel_ok(best, pair_abel_target)
        return best
    end
    error("no certified two-ray horizon path; best_score=" *
        "$(best === nothing ? Inf : best.score), pair_abel_error=" *
        "$(best === nothing ? Inf : best.pair_abel_error).")
end

function _select_horizon_path(
    coefficients,
    solutions,
    kinds,
    p,
    match_x,
    settings,
    ;
    pair_abel_target=nothing,
    fallback_pair_abel_target=pair_abel_target,
)
    try
        return _select_rotated_horizon_path(
            coefficients, solutions, kinds, p, match_x, settings;
            pair_abel_target)
    catch exception
        message = sprint(showerror, exception)
        expected = occursin("two-ray coordinate path", message) ||
            occursin("no certified two-ray horizon path", message)
        expected || rethrow()
        try
            return _select_real_horizon_path(
                coefficients, solutions, kinds, match_x, settings;
                pair_abel_target=fallback_pair_abel_target)
        catch fallback_exception
            error(sprint(showerror, fallback_exception) *
                " Rotated-path failure: " * message)
        end
    end
end

function _select_infinity_path(
    coefficients,
    solutions,
    kinds,
    omega,
    match_x,
    settings,
)
    best = nothing
    path = nothing
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    for magnitude in _INFINITY_RHO_MAGNITUDES
        path = _integrate_coordinate_path(
            coefficients.params, omega, magnitude; match_x,
            initial_path=path)
        candidate = _infinity_candidate!(
            scratch, coefficients, solutions, kinds, path, settings)
        if isfinite(candidate.score) &&
                (best === nothing || candidate.score < best.score)
            best = candidate
        end
        candidate.score <= settings.infinity_target && return candidate
    end
    best_check = best === nothing ? nothing : reduce(
        (left, right) -> left.score >= right.score ? left : right,
        best.checks)
    error("no certified two-ray infinity path; best_score=" *
        "$(best === nothing ? Inf : best.score), rho=" *
        "$(best === nothing ? NaN : best.path.rho), y=" *
        "$(best === nothing ? NaN : 1 - best.path.x), adjacent=" *
        "$(best_check === nothing ? Inf : best_check.adjacent), residual=" *
        "$(best_check === nothing ? Inf : best_check.residual), roundoff=" *
        "$(best_check === nothing ? Inf : best_check.roundoff), order=" *
        "$(best_check === nothing ? 0 : best_check.order).")
end

@inline _principal_mst_axis(omega) =
    iszero(real(omega)) && imag(omega) < 0 &&
    abs(omega) <= _SCALED_Y_PHYSICAL_HANDOFF

@inline _principal_sfe_evaluator_axis(omega) =
    iszero(real(omega)) && imag(omega) < 0 && abs(omega) <= 0.1

@inline _mst_branch(kind::Symbol) =
    kind == :infinity_in ? :in :
    kind == :infinity_out ? :out :
    throw(ArgumentError("principal MST infinity kind must be in or out."))

function _principal_mst_candidate!(
    coefficients,
    plan,
    solutions,
    kinds,
    path,
    settings,
)
    length(kinds) == 2 &&
        kinds[1] == :infinity_in &&
        kinds[2] == :infinity_out ||
        throw(ArgumentError(
            "principal MST projective correction requires in/out infinity pair."))
    A, B = _ode_ab!(
        DirectComplexRationalScratch(coefficients, settings.order),
        coefficients,
        path.x,
    )
    local_checks = map(solutions, kinds) do solution, kind
        _infinity_check_from_ab(
            A, B, coefficients, solution, kind, path, settings)
    end
    mst = direct_mst_principal_state(
        coefficients, plan, :out, path.x)
    mst_state = DirectComplexRationalState(mst.X, mst.dXdx)
    normalize(state) = begin
        scale = max(abs(state.X), abs(state.dXdx))
        isfinite(scale) && scale > 0 ||
            error("principal MST projective state is zero or nonfinite.")
        DirectComplexRationalState(
            ComplexF64(state.X / scale),
            ComplexF64(state.dXdx / scale),
        )
    end
    incoming = normalize(local_checks[1].state)
    outgoing = normalize(local_checks[2].state)
    principal = normalize(mst_state)
    numerator = principal.X * outgoing.dXdx -
        outgoing.X * principal.dXdx
    denominator = incoming.X * principal.dXdx -
        principal.X * incoming.dXdx
    _finite_complex(denominator) && !iszero(denominator) ||
        error("principal MST projective correction is singular.")
    mixing = ComplexF64(numerator / denominator)
    corrected = normalize(DirectComplexRationalState(
        outgoing.X + mixing * incoming.X,
        outgoing.dXdx + mixing * incoming.dXdx,
    ))
    checks = (
        local_checks[1],
        (;
            score=mst.estimated_relerr,
            adjacent=mst.estimated_relerr,
            residual=mst.estimated_relerr,
            roundoff=0.0,
            order=settings.order,
            state=corrected,
        ),
    )
    pair_abel_error = if length(checks) == 2
        denominator = direct_abel_denominator(
            coefficients, :infinity, path.x)
        denominator.status == "OK" || error(denominator.status)
        _abel_normalize_second(
            checks[1].state,
            checks[2].state,
            denominator.denominator,
        ).corrected_error
    else
        0.0
    end
    return (;
        path,
        checks,
        score=max(
            mst.estimated_relerr,
            pair_abel_error,
        ),
        pair_abel_error,
        projective_mixing=mixing,
    )
end

function _select_principal_mst_infinity_path(
    coefficients,
    solutions,
    kinds,
    omega,
    match_x,
    settings,
)
    branches = (:out,)
    plan = direct_mst_principal_plan(coefficients; branches)
    best = nothing
    omega_scale = abs(omega)
    for scaled_y in _PRINCIPAL_MST_Y_SCALES
        y = omega_scale * scaled_y
        16eps(Float64) < y < 1 - match_x || continue
        seed_x = 1 - y
        path_y = 1 - match_x
        nodes = ComplexF64[match_x]
        while path_y > y * (1 + 16eps(Float64))
            next_y = max(y, path_y / _SCALED_Y_CHART_GROWTH)
            next_y < path_y || break
            push!(nodes, ComplexF64(1 - next_y))
            path_y = next_y
        end
        last(nodes) == ComplexF64(seed_x) ||
            push!(nodes, ComplexF64(seed_x))
        path = (;
            rho=inv(scaled_y),
            x=ComplexF64(seed_x),
            nodes,
            logx=ComplexF64(log(seed_x)),
            logy=ComplexF64(log(y)),
            direction=ComplexF64(1),
            angle_offset=0.0,
            kind=:principal_mst,
            accepted_steps=length(nodes) - 1,
            rejected_steps=0,
        )
        candidate = try
            _principal_mst_candidate!(
                coefficients, plan, solutions, kinds, path, settings)
        catch
            nothing
        end
        candidate === nothing && continue
        if best === nothing || candidate.score < best.score
            best = candidate
        end
        candidate.score <= settings.infinity_target && return candidate
    end
    best !== nothing && best.score <= 32settings.infinity_target && return best
    error("no certified principal MST infinity seed; best_score=" *
        "$(best === nothing ? Inf : best.score).")
end

function _series_state_condition(coeffs, h, value, derivative, order)
    absolute_h = abs(h)
    value_scale = 0.0
    derivative_scale = 0.0
    value_power = 1.0
    derivative_power = 1.0
    @inbounds for degree in 0:order
        coefficient = abs(coeffs[degree + 1])
        value_scale += coefficient * value_power
        if degree > 0
            derivative_scale += degree * coefficient * derivative_power
            derivative_power *= absolute_h
        end
        value_power *= absolute_h
    end
    return max(
        value_scale / max(abs(value), floatmin(Float64)),
        derivative_scale / max(abs(derivative), floatmin(Float64)),
    )
end

function _local_ab_series!(
    scratch,
    coefficients,
    x,
    order,
    eps_limit,
    shifted,
)
    if !shifted
        a_terms, b_terms = direct_ordinary_ab_series!(
            scratch.avec, scratch.bvec, scratch.pq, coefficients, x, order)
        return a_terms, b_terms, :ordinary, 0.0
    end

    shift_condition = Inf
    if abs(x) <= abs(1 - x)
        center = iszero(imag(x)) ? real(x) : x
        accepted, a_terms, b_terms, condition = direct_shifted_ab!(
            scratch.avec, scratch.bvec, scratch.pq,
            coefficients.horizon, center, order;
            eps_limit=10eps_limit)
        shift_condition = condition
        accepted && return a_terms, b_terms, :horizon_shift, condition
    else
        y = 1 - x
        center = iszero(imag(y)) ? real(y) : y
        accepted, a_terms, b_terms, condition = direct_shifted_infinity_ab!(
            scratch.avec, scratch.bvec, scratch.pq,
            coefficients.infinity, center, order;
            eps_limit=10eps_limit)
        shift_condition = condition
        if accepted
            @inbounds for degree in 0:(a_terms - 1)
                scratch.avec[degree + 1] *= iseven(degree) ? -1.0 : 1.0
            end
            @inbounds for degree in 0:(b_terms - 1)
                scratch.bvec[degree + 1] *= iseven(degree) ? 1.0 : -1.0
            end
            return a_terms, b_terms, :infinity_shift, condition
        end
    end
    a_terms, b_terms = direct_ordinary_ab_series!(
        scratch.avec, scratch.bvec, scratch.pq, coefficients, x, order)
    return a_terms, b_terms, :ordinary_shift_fallback, shift_condition
end

function _local_step_from_ab!(
    scratch,
    state1,
    state2,
    h,
    settings,
    a_terms,
    b_terms,
    representation,
    coefficient_condition,
)
    order = settings.order
    direct_ordinary_point_coeffs!(
        scratch.coeffs1, scratch.avec, scratch.bvec,
        state1.X, state1.dXdx, order)
    direct_ordinary_point_coeffs!(
        scratch.coeffs2, scratch.avec, scratch.bvec,
        state2.X, state2.dXdx, order)

    X1, dX1, ddX1 = direct_poly_triple(scratch.coeffs1, h)
    X2, dX2, ddX2 = direct_poly_triple(scratch.coeffs2, h)
    low1 = direct_poly_pair(
        @view(scratch.coeffs1[1:(settings.reduced_order + 1)]), h)
    low2 = direct_poly_pair(
        @view(scratch.coeffs2[1:(settings.reduced_order + 1)]), h)
    full1 = DirectComplexRationalState(ComplexF64(X1), ComplexF64(dX1))
    full2 = DirectComplexRationalState(ComplexF64(X2), ComplexF64(dX2))
    A = direct_poly_value(@view(scratch.avec[1:a_terms]), h)
    B = direct_poly_value(@view(scratch.bvec[1:b_terms]), h)
    score1, adjacent1, residual1 = _state_score_from_ab(
        A, B, full1,
        DirectComplexRationalState(ComplexF64(low1[1]), ComplexF64(low1[2])),
        ddX1)
    score2, adjacent2, residual2 = _state_score_from_ab(
        A, B, full2,
        DirectComplexRationalState(ComplexF64(low2[1]), ComplexF64(low2[2])),
        ddX2)
    tail_range = (order - 3):(order + 1)
    tail1 = maximum(abs(scratch.coeffs1[index] * h^(index - 1))
        for index in tail_range) / max(abs(X1), floatmin(Float64))
    tail2 = maximum(abs(scratch.coeffs2[index] * h^(index - 1))
        for index in tail_range) / max(abs(X2), floatmin(Float64))
    condition1 = _series_state_condition(
        scratch.coeffs1, h, X1, dX1, order)
    condition2 = _series_state_condition(
        scratch.coeffs2, h, X2, dX2, order)
    return (;
        state1=full1,
        state2=full2,
        score=max(score1, score2, tail1, tail2),
        adjacent=max(adjacent1, adjacent2),
        residual=max(residual1, residual2),
        tail=max(tail1, tail2),
        roundoff=8eps(Float64) * max(condition1, condition2),
        representation,
        target_representation=representation,
        coefficient_condition,
    )
end

function _local_step!(
    scratch,
    coefficients,
    x,
    state1,
    state2,
    h,
    settings,
    shifted,
)
    order = settings.order
    a_terms, b_terms, representation, coefficient_condition =
        _local_ab_series!(
            scratch, coefficients, x, order, settings.step_target, shifted)
    candidate = _local_step_from_ab!(
        scratch,
        state1,
        state2,
        h,
        settings,
        a_terms,
        b_terms,
        representation,
        coefficient_condition,
    )
    if representation == :ordinary ||
            representation == :ordinary_shift_fallback
        A, B = _ode_ab!(scratch, coefficients, x + h)
        score1, adjacent1, residual1 = _state_score_from_ab(
            A,
            B,
            candidate.state1,
            DirectComplexRationalState(
                direct_poly_pair(
                    @view(scratch.coeffs1[1:(settings.reduced_order + 1)]),
                    h,
                )...,
            ),
            direct_poly_triple(scratch.coeffs1, h)[3],
        )
        score2, adjacent2, residual2 = _state_score_from_ab(
            A,
            B,
            candidate.state2,
            DirectComplexRationalState(
                direct_poly_pair(
                    @view(scratch.coeffs2[1:(settings.reduced_order + 1)]),
                    h,
                )...,
            ),
            direct_poly_triple(scratch.coeffs2, h)[3],
        )
        return merge(candidate, (;
            score=max(candidate.tail, score1, score2),
            adjacent=max(adjacent1, adjacent2),
            residual=max(residual1, residual2),
        ))
    end
    return candidate
end

function _scaled_rational(
    rational::DirectRationalCoefficients,
    variable_scale,
    numerator_factor,
)
    scale = ComplexF64(variable_scale)
    numerator = Vector{ComplexF64}(undef, length(rational.numerator))
    denominator = Vector{ComplexF64}(undef, length(rational.denominator))
    power = 1.0 + 0.0im
    @inbounds for index in eachindex(numerator)
        numerator[index] =
            ComplexF64(numerator_factor * rational.numerator[index] * power)
        power *= scale
    end
    power = 1.0 + 0.0im
    @inbounds for index in eachindex(denominator)
        denominator[index] = ComplexF64(rational.denominator[index] * power)
        power *= scale
    end
    denominator_norm = maximum(abs, denominator)
    isfinite(denominator_norm) && denominator_norm > 0 ||
        error("scaled infinity rational denominator is nonfinite.")
    numerator ./= denominator_norm
    denominator ./= denominator_norm
    return DirectRationalCoefficients(numerator, denominator)
end

function _scaled_infinity_endpoint(endpoint, variable_scale)
    scale = ComplexF64(variable_scale)
    iszero(scale) &&
        throw(DomainError(scale, "scaled infinity propagation requires nonzero omega."))
    A = _scaled_rational(endpoint.A, scale, 1.0 + 0.0im)
    B = _scaled_rational(endpoint.B, scale, inv(scale * scale))
    return DirectEndpointCoefficientSet(A, B, endpoint.basis)
end

function _scaled_horizon_endpoint(endpoint, variable_scale)
    scale = ComplexF64(variable_scale)
    iszero(scale) &&
        throw(DomainError(scale, "scaled horizon propagation requires nonzero scale."))
    A = _scaled_rational(endpoint.A, scale, 1.0 + 0.0im)
    B = _scaled_rational(endpoint.B, scale, 1.0 + 0.0im)
    return DirectEndpointCoefficientSet(A, B, endpoint.basis)
end

function _scaled_y_step!(
    scratch,
    endpoint,
    variable,
    state1,
    state2,
    h,
    settings,
)
    accepted, a_terms, b_terms, coefficient_condition =
        direct_shifted_infinity_ab!(
            scratch.avec,
            scratch.bvec,
            scratch.pq,
            endpoint,
            variable,
            settings.order;
            eps_limit=10settings.step_target,
        )
    if !accepted && !isfinite(coefficient_condition)
        return (;
            state1,
            state2,
            score=Inf,
            adjacent=Inf,
            residual=Inf,
            tail=Inf,
            roundoff=Inf,
            representation=:scaled_y_shift,
            target_representation=:scaled_y_shift,
            coefficient_condition,
        )
    end
    return _local_step_from_ab!(
        scratch,
        state1,
        state2,
        h,
        settings,
        a_terms,
        b_terms,
        :scaled_y_shift,
        coefficient_condition,
    )
end

function _scaled_x_step!(
    scratch,
    endpoint,
    variable,
    state1,
    state2,
    h,
    settings,
)
    accepted, a_terms, b_terms, coefficient_condition = direct_shifted_ab!(
        scratch.avec,
        scratch.bvec,
        scratch.pq,
        endpoint,
        variable,
        settings.order;
        eps_limit=10settings.step_target,
    )
    if !accepted && !isfinite(coefficient_condition)
        return (;
            state1,
            state2,
            score=Inf,
            adjacent=Inf,
            residual=Inf,
            tail=Inf,
            roundoff=Inf,
            representation=:scaled_x_shift,
            target_representation=:scaled_x_shift,
            coefficient_condition,
        )
    end
    return _local_step_from_ab!(
        scratch,
        state1,
        state2,
        h,
        settings,
        a_terms,
        b_terms,
        :scaled_x_shift,
        coefficient_condition,
    )
end

function _propagate_scaled_y_pair!(
    scratch,
    endpoint,
    variable_scale,
    variable0,
    state1,
    state2,
    target,
    settings,
)
    variable = ComplexF64(variable0)
    target_variable = ComplexF64(target)
    endpoint_zero = inv(ComplexF64(variable_scale))
    patches = 0
    rejected = 0
    max_score = 0.0
    while abs(target_variable - variable) >
            100eps(Float64) * max(1.0, abs(variable))
        patches < _MAX_PATCHES ||
            error("complex rational scaled-y patch limit reached.")
        remaining = target_variable - variable
        radius = min(abs(variable), abs(endpoint_zero - variable))
        h = abs(remaining) <= _STEP_SAFETY * radius ? remaining :
            _STEP_SAFETY * radius * remaining / abs(remaining)
        accepted = nothing
        best_score = Inf
        best_candidate = nothing
        best_h = h
        for _ in 1:32
            candidate = _scaled_y_step!(
                scratch,
                endpoint,
                variable,
                state1,
                state2,
                h,
                settings,
            )
            if isfinite(candidate.score) && candidate.score < best_score
                best_score = candidate.score
                best_candidate = candidate
                best_h = h
            end
            if isfinite(candidate.score) &&
                    candidate.score <= settings.step_target
                accepted = candidate
                break
            end
            h *= 0.5
            rejected += 1
        end
        accepted === nothing && error(
            "complex rational scaled-y step did not certify; Y=$variable, " *
            "target=$target_variable, best_h=$best_h, best_score=$best_score, " *
            "adjacent=$(best_candidate === nothing ? Inf : best_candidate.adjacent), " *
            "residual=$(best_candidate === nothing ? Inf : best_candidate.residual), " *
            "tail=$(best_candidate === nothing ? Inf : best_candidate.tail), " *
            "coefficient_condition=$(best_candidate === nothing ? Inf : best_candidate.coefficient_condition).")
        variable += h
        state1 = accepted.state1
        state2 = accepted.state2
        max_score = max(max_score, accepted.score)
        patches += 1
    end
    return (; variable, state1, state2, patches, rejected, max_score)
end

function _propagate_scaled_x_pair!(
    scratch,
    endpoint,
    variable_scale,
    variable0,
    state1,
    state2,
    target,
    settings,
)
    variable = ComplexF64(variable0)
    target_variable = ComplexF64(target)
    infinity = inv(ComplexF64(variable_scale))
    patches = 0
    rejected = 0
    max_score = 0.0
    while abs(target_variable - variable) >
            100eps(Float64) * max(1.0, abs(variable))
        patches < _MAX_PATCHES ||
            error("complex rational scaled-x patch limit reached.")
        remaining = target_variable - variable
        radius = min(abs(variable), abs(infinity - variable))
        h = abs(remaining) <= _STEP_SAFETY * radius ? remaining :
            _STEP_SAFETY * radius * remaining / abs(remaining)
        accepted = nothing
        best_score = Inf
        best_candidate = nothing
        best_h = h
        for _ in 1:32
            candidate = _scaled_x_step!(
                scratch,
                endpoint,
                variable,
                state1,
                state2,
                h,
                settings,
            )
            if isfinite(candidate.score) && candidate.score < best_score
                best_score = candidate.score
                best_candidate = candidate
                best_h = h
            end
            if isfinite(candidate.score) &&
                    candidate.score <= settings.step_target
                accepted = candidate
                break
            end
            h *= 0.5
            rejected += 1
        end
        accepted === nothing && error(
            "complex rational scaled-x step did not certify; Z=$variable, " *
            "target=$target_variable, best_h=$best_h, best_score=$best_score, " *
            "adjacent=$(best_candidate === nothing ? Inf : best_candidate.adjacent), " *
            "residual=$(best_candidate === nothing ? Inf : best_candidate.residual), " *
            "tail=$(best_candidate === nothing ? Inf : best_candidate.tail), " *
            "coefficient_condition=$(best_candidate === nothing ? Inf : best_candidate.coefficient_condition).")
        variable += h
        state1 = accepted.state1
        state2 = accepted.state2
        max_score = max(max_score, accepted.score)
        patches += 1
    end
    return (; variable, state1, state2, patches, rejected, max_score)
end

@inline function _residual_plateau_step(candidate, target)
    return isfinite(candidate.score) &&
        candidate.adjacent <= target &&
        candidate.tail <= target &&
        candidate.coefficient_condition <= target &&
        candidate.residual <= 10target
end

function _propagate_pair!(
    scratch,
    coefficients,
    x0,
    state1,
    state2,
    target,
    settings,
)
    x = ComplexF64(x0)
    target_x = ComplexF64(target)
    patches = 0
    rejected = 0
    max_score = 0.0
    while abs(target_x - x) > 100eps(Float64) * max(1.0, abs(x))
        patches < _MAX_PATCHES || error("complex rational patch limit reached.")
        remaining = target_x - x
        radius = min(abs(x), abs(1 - x))
        h = abs(remaining) <= _STEP_SAFETY * radius ? remaining :
            _STEP_SAFETY * radius * remaining / abs(remaining)
        initial_h = h
        accepted = nothing
        best_score = Inf
        best_candidate = nothing
        best_h = h
        for shifted in (false, true)
            h = initial_h
            plateau_candidate = nothing
            plateau_h = h
            for _ in 1:32
                candidate = _local_step!(
                    scratch, coefficients, x, state1, state2, h, settings,
                    shifted)
                if isfinite(candidate.score) && candidate.score < best_score
                    best_score = candidate.score
                    best_candidate = candidate
                    best_h = h
                end
                if plateau_candidate === nothing &&
                        _residual_plateau_step(candidate, settings.step_target)
                    plateau_candidate = candidate
                    plateau_h = h
                end
                if isfinite(candidate.score) &&
                        candidate.score <= settings.step_target
                    accepted = candidate
                    break
                end
                h *= 0.5
                rejected += 1
            end
            if accepted === nothing && plateau_candidate !== nothing
                accepted = plateau_candidate
                h = plateau_h
            end
            accepted === nothing || break
        end
        accepted === nothing && error(
            "complex rational local step did not certify; x=$x, " *
            "target=$target_x, best_h=$best_h, best_score=$best_score, " *
            "adjacent=$(best_candidate === nothing ? Inf : best_candidate.adjacent), " *
            "residual=$(best_candidate === nothing ? Inf : best_candidate.residual), " *
            "tail=$(best_candidate === nothing ? Inf : best_candidate.tail), " *
            "roundoff=$(best_candidate === nothing ? Inf : best_candidate.roundoff), " *
            "representation=$(best_candidate === nothing ? :none : best_candidate.representation), " *
            "target_representation=$(best_candidate === nothing ? :none : best_candidate.target_representation), " *
            "coefficient_condition=$(best_candidate === nothing ? Inf : best_candidate.coefficient_condition).")
        x += h
        state1 = accepted.state1
        state2 = accepted.state2
        max_score = max(max_score, accepted.score)
        patches += 1
    end
    return (; x, state1, state2, patches, rejected, max_score)
end

function _propagate_pair(coefficients, x0, state1, state2, target, settings)
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    return _propagate_pair!(
        scratch, coefficients, x0, state1, state2, target, settings)
end

@inline function _seed_bundle(seed)
    return (;
        x=seed.path.x,
        state1=seed.checks[1].state,
        state2=seed.checks[end].state,
        patches=0,
        rejected=0,
        max_score=seed.score,
        scaled_y_patches=0,
    )
end

function _propagate_on_path(
    coefficients,
    path,
    state1,
    state2,
    settings,
    ;
    variable_scale=1.0 + 0.0im,
)
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    patches = 0
    rejected = 0
    max_score = 0.0
    x = path.x
    source_index = length(path.nodes)
    scaled_y_patches = 0
    solution_scale = ComplexF64(variable_scale)
    if !isone(solution_scale)
        while source_index > 1 &&
                abs(1 - x) < _SCALED_Y_PHYSICAL_HANDOFF
            source_y = 1 - x
            propagation_scale = source_y / _SCALED_Y_CHART_SOURCE
            target_magnitude = min(
                _SCALED_Y_PHYSICAL_HANDOFF,
                _SCALED_Y_CHART_GROWTH * abs(source_y),
            )
            target_index = source_index - 1
            for candidate_index in (source_index - 1):-1:1
                candidate_y = 1 - path.nodes[candidate_index]
                abs(candidate_y) <= target_magnitude || break
                target_index = candidate_index
            end
            target_y = 1 - path.nodes[target_index]
            source_variable = source_y / propagation_scale
            target_variable = target_y / propagation_scale
            endpoint = _scaled_infinity_endpoint(
                coefficients.infinity, propagation_scale)
            state1_y = DirectComplexRationalState(
                state1.X, ComplexF64(-propagation_scale * state1.dXdx))
            state2_y = DirectComplexRationalState(
                state2.X, ComplexF64(-propagation_scale * state2.dXdx))
            segment = _propagate_scaled_y_pair!(
                scratch,
                endpoint,
                propagation_scale,
                source_variable,
                state1_y,
                state2_y,
                target_variable,
                settings,
            )
            x = path.nodes[target_index]
            state1 = DirectComplexRationalState(
                segment.state1.X,
                ComplexF64(-segment.state1.dXdx / propagation_scale),
            )
            state2 = DirectComplexRationalState(
                segment.state2.X,
                ComplexF64(-segment.state2.dXdx / propagation_scale),
            )
            source_index = target_index
            patches += segment.patches
            scaled_y_patches += segment.patches
            rejected += segment.rejected
            max_score = max(max_score, segment.max_score)
        end
    end
    while source_index > 1
        target_index = source_index - 1
        radius = _PATH_DISK_SAFETY * min(abs(x), abs(1 - x))
        for candidate_index in (source_index - 2):-1:1
            abs(path.nodes[candidate_index] - x) <= radius || break
            target_index = candidate_index
        end
        target = path.nodes[target_index]
        source_index = target_index
        abs(target - x) <= 100eps(Float64) * max(1.0, abs(x)) && continue
        segment = _propagate_pair!(
            scratch, coefficients, x, state1, state2, target, settings)
        x = segment.x
        state1 = segment.state1
        state2 = segment.state2
        patches += segment.patches
        rejected += segment.rejected
        max_score = max(max_score, segment.max_score)
    end
    return (;
        x,
        state1,
        state2,
        patches,
        rejected,
        max_score,
        scaled_y_patches,
    )
end

function _propagate_forward_on_path(
    coefficients,
    path,
    state1,
    state2,
    settings,
    ;
    variable_scale=1.0 + 0.0im,
)
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    patches = 0
    rejected = 0
    max_score = 0.0
    x = first(path.nodes)
    source_index = 1
    last_index = length(path.nodes)
    scaled_y_patches = 0
    solution_scale = ComplexF64(variable_scale)

    while source_index < last_index &&
            (isone(solution_scale) ||
             abs(1 - path.nodes[source_index + 1]) >=
                _SCALED_Y_PHYSICAL_HANDOFF)
        target_index = source_index + 1
        radius = _PATH_DISK_SAFETY * min(abs(x), abs(1 - x))
        for candidate_index in (source_index + 2):last_index
            candidate = path.nodes[candidate_index]
            if !isone(solution_scale) &&
                    abs(1 - candidate) < _SCALED_Y_PHYSICAL_HANDOFF
                break
            end
            abs(candidate - x) <= radius || break
            target_index = candidate_index
        end
        target = path.nodes[target_index]
        source_index = target_index
        abs(target - x) <= 100eps(Float64) * max(1.0, abs(x)) && continue
        segment = _propagate_pair!(
            scratch, coefficients, x, state1, state2, target, settings)
        x = segment.x
        state1 = segment.state1
        state2 = segment.state2
        patches += segment.patches
        rejected += segment.rejected
        max_score = max(max_score, segment.max_score)
    end

    if !isone(solution_scale)
        while source_index < last_index
            source_y = 1 - x
            propagation_scale = source_y / 0.5
            target_magnitude = max(
                abs(1 - path.nodes[last_index]),
                abs(source_y) / _SCALED_Y_CHART_GROWTH,
            )
            target_index = source_index + 1
            for candidate_index in (source_index + 1):last_index
                candidate_y = 1 - path.nodes[candidate_index]
                abs(candidate_y) >= target_magnitude || break
                target_index = candidate_index
            end
            target_y = 1 - path.nodes[target_index]
            endpoint = _scaled_infinity_endpoint(
                coefficients.infinity, propagation_scale)
            state1_y = DirectComplexRationalState(
                state1.X, ComplexF64(-propagation_scale * state1.dXdx))
            state2_y = DirectComplexRationalState(
                state2.X, ComplexF64(-propagation_scale * state2.dXdx))
            segment = _propagate_scaled_y_pair!(
                scratch,
                endpoint,
                propagation_scale,
                source_y / propagation_scale,
                state1_y,
                state2_y,
                target_y / propagation_scale,
                settings,
            )
            x = path.nodes[target_index]
            state1 = DirectComplexRationalState(
                segment.state1.X,
                ComplexF64(-segment.state1.dXdx / propagation_scale),
            )
            state2 = DirectComplexRationalState(
                segment.state2.X,
                ComplexF64(-segment.state2.dXdx / propagation_scale),
            )
            source_index = target_index
            patches += segment.patches
            scaled_y_patches += segment.patches
            rejected += segment.rejected
            max_score = max(max_score, segment.max_score)
        end
    end
    return (;
        x,
        state1,
        state2,
        patches,
        rejected,
        max_score,
        scaled_y_patches,
    )
end

function _propagate_forward_horizon_path(
    coefficients,
    path,
    state1,
    state2,
    settings,
)
    scratch = DirectComplexRationalScratch(coefficients, settings.order)
    patches = 0
    rejected = 0
    max_score = 0.0
    x = first(path.nodes)
    source_index = 1
    last_index = length(path.nodes)
    scaled_x_patches = 0

    while source_index < last_index &&
            abs(path.nodes[source_index + 1]) >=
                _SCALED_X_PHYSICAL_HANDOFF
        target_index = source_index + 1
        radius = _PATH_DISK_SAFETY * min(abs(x), abs(1 - x))
        for candidate_index in (source_index + 2):last_index
            candidate = path.nodes[candidate_index]
            abs(candidate) >= _SCALED_X_PHYSICAL_HANDOFF || break
            abs(candidate - x) <= radius || break
            target_index = candidate_index
        end
        target = path.nodes[target_index]
        source_index = target_index
        abs(target - x) <= 100eps(Float64) * max(1.0, abs(x)) && continue
        segment = _propagate_pair!(
            scratch, coefficients, x, state1, state2, target, settings)
        x = segment.x
        state1 = segment.state1
        state2 = segment.state2
        patches += segment.patches
        rejected += segment.rejected
        max_score = max(max_score, segment.max_score)
    end

    while source_index < last_index
        source_x = x
        propagation_scale = source_x / 0.5
        target_magnitude = max(
            abs(path.nodes[last_index]),
            abs(source_x) / _SCALED_Y_CHART_GROWTH,
        )
        target_index = source_index + 1
        for candidate_index in (source_index + 1):last_index
            candidate_x = path.nodes[candidate_index]
            abs(candidate_x) >= target_magnitude || break
            target_index = candidate_index
        end
        target_x = path.nodes[target_index]
        endpoint = _scaled_horizon_endpoint(
            coefficients.horizon, propagation_scale)
        state1_z = DirectComplexRationalState(
            state1.X, ComplexF64(propagation_scale * state1.dXdx))
        state2_z = DirectComplexRationalState(
            state2.X, ComplexF64(propagation_scale * state2.dXdx))
        segment = _propagate_scaled_x_pair!(
            scratch,
            endpoint,
            propagation_scale,
            source_x / propagation_scale,
            state1_z,
            state2_z,
            target_x / propagation_scale,
            settings,
        )
        x = target_x
        state1 = DirectComplexRationalState(
            segment.state1.X,
            ComplexF64(segment.state1.dXdx / propagation_scale),
        )
        state2 = DirectComplexRationalState(
            segment.state2.X,
            ComplexF64(segment.state2.dXdx / propagation_scale),
        )
        source_index = target_index
        patches += segment.patches
        scaled_x_patches += segment.patches
        rejected += segment.rejected
        max_score = max(max_score, segment.max_score)
    end
    return (;
        x,
        state1,
        state2,
        patches,
        rejected,
        max_score,
        scaled_x_patches,
    )
end

@inline function _path_segment(path, first_index::Int, last_index::Int)
    nodes = path.nodes[first_index:last_index]
    return merge(path, (; x=last(nodes), nodes))
end

function _infinity_overlap_match(
    coefficients,
    seed,
    target_state,
    settings;
    variable_scale=1.0 + 0.0im,
)
    node_count = length(seed.path.nodes)
    node_count >= 3 ||
        error("infinity overlap matching requires at least three path nodes.")
    indices = Int[]
    for fraction in (0.5, 0.35, 0.65, 0.2, 0.8, 0.9)
        index = clamp(
            round(Int, 1 + fraction * (node_count - 1)),
            2,
            node_count - 1,
        )
        index in indices || push!(indices, index)
    end

    best = nothing
    failures = String[]
    for index in indices
        candidate = try
            target_path = _path_segment(seed.path, 1, index)
            basis_path = _path_segment(seed.path, index, node_count)
            target = _propagate_forward_on_path(
                coefficients,
                target_path,
                target_state,
                target_state,
                settings;
                variable_scale,
            )
            basis = _propagate_on_path(
                coefficients,
                basis_path,
                seed.checks[1].state,
                seed.checks[2].state,
                settings;
                variable_scale,
            )
            denominator = direct_abel_denominator(
                coefficients, :infinity, target.x)
            denominator.status == "OK" ||
                error("infinity Abel denominator failed: $(denominator.status).")
            normalization = _abel_normalize_second(
                basis.state1,
                basis.state2,
                denominator.denominator,
            )
            diagnostics = _matching_diagnostics(
                target.state1,
                basis.state1,
                normalization.corrected,
            )
            score = max(
                target.max_score,
                basis.max_score,
                normalization.corrected_error,
                eps(Float64) * diagnostics.condition,
                eps(Float64) * diagnostics.coefficient1_cancellation,
                eps(Float64) * diagnostics.coefficient2_cancellation,
            )
            (
                x=target.x,
                state1=target.state1,
                state2=target.state2,
                basis1=basis.state1,
                basis2=normalization.corrected,
                normalization,
                denominator,
                diagnostics,
                score,
                patches=target.patches + basis.patches,
                rejected=target.rejected + basis.rejected,
                max_score=max(target.max_score, basis.max_score),
                scaled_y_patches=
                    target.scaled_y_patches + basis.scaled_y_patches,
            )
        catch exception
            push!(failures, "index=$index: $(sprint(showerror, exception))")
            nothing
        end
        candidate === nothing && continue
        candidate.score <= settings.infinity_target && return candidate
        if best === nothing || candidate.score < best.score
            best = candidate
        end
    end
    best !== nothing && return best
    error("no certified infinity overlap match; " * join(failures, " | "))
end

function _scaled_abel_ratio(state1, state2, exact_denominator)
    _finite_complex(exact_denominator) && !iszero(exact_denominator) ||
        error("nonfinite Abel denominator.")
    scale1 = max(abs(state1.X), abs(state1.dXdx))
    scale2 = max(abs(state2.X), abs(state2.dXdx))
    isfinite(scale1) && scale1 > 0 && isfinite(scale2) && scale2 > 0 ||
        error("nonfinite Abel basis state.")
    numerical = (state1.X / scale1) * (state2.dXdx / scale2) -
        (state2.X / scale2) * (state1.dXdx / scale1)
    _finite_complex(numerical) && !iszero(numerical) ||
        error("singular Abel basis pair.")
    logabs_ratio = log(abs(numerical)) + log(scale1) + log(scale2) -
        log(abs(exact_denominator))
    log(nextfloat(0.0)) <= logabs_ratio <= log(floatmax(Float64)) ||
        error("unrepresentable Abel pair-normalization ratio.")
    ratio = ComplexF64(
        exp(logabs_ratio) *
        cis(angle(numerical) - angle(exact_denominator)))
    _finite_complex(ratio) && !iszero(ratio) ||
        error("nonfinite Abel pair-normalization ratio.")
    return ratio
end

function _abel_normalize_second(state1, state2, exact_denominator)
    numerical = state1.X * state2.dXdx - state2.X * state1.dXdx
    ratio = ComplexF64(numerical / exact_denominator)
    if !_finite_complex(ratio) || iszero(ratio)
        ratio = _scaled_abel_ratio(state1, state2, exact_denominator)
    end
    corrected = DirectComplexRationalState(
        ComplexF64(state2.X / ratio),
        ComplexF64(state2.dXdx / ratio),
    )
    _finite_complex(corrected.X) && _finite_complex(corrected.dXdx) ||
        error("nonfinite Abel-normalized basis state.")
    corrected_ratio =
        _scaled_abel_ratio(state1, corrected, exact_denominator)
    return (;
        corrected,
        ratio,
        raw_error=abs(ratio - 1),
        corrected_error=abs(corrected_ratio - 1),
    )
end

@inline function _abel_range_error(exception)
    exception isa ErrorException || return false
    message = exception.msg
    return message == "singular Abel basis pair." ||
        message == "unrepresentable Abel pair-normalization ratio." ||
        message == "nonfinite Abel pair-normalization ratio." ||
        message == "nonfinite Abel-normalized basis state."
end

@inline function _connection_coefficients(target, basis1, basis2, denominator)
    incidence = (target.X * basis2.dXdx - basis2.X * target.dXdx) /
        denominator
    reflection = (basis1.X * target.dXdx - target.X * basis1.dXdx) /
        denominator
    return ComplexF64(incidence), ComplexF64(reflection)
end

function _matching_diagnostics(target, basis1, basis2)
    xscale = max(abs(target.X), abs(basis1.X), abs(basis2.X),
        floatmin(Float64))
    dscale = max(abs(target.dXdx), abs(basis1.dXdx),
        abs(basis2.dXdx), floatmin(Float64))
    m11 = basis1.X / xscale
    m12 = basis2.X / xscale
    m21 = basis1.dXdx / dscale
    m22 = basis2.dXdx / dscale
    determinant = m11 * m22 - m12 * m21
    trace = abs2(m11) + abs2(m12) + abs2(m21) + abs2(m22)
    discriminant = sqrt(max(0.0, trace^2 - 4abs2(determinant)))
    sigma_max_squared = 0.5 * (trace + discriminant)
    condition = sigma_max_squared /
        max(abs(determinant), floatmin(Float64))
    c1a = target.X * basis2.dXdx
    c1b = basis2.X * target.dXdx
    c2a = basis1.X * target.dXdx
    c2b = target.X * basis1.dXdx
    return (;
        condition,
        coefficient1_cancellation=(abs(c1a) + abs(c1b)) /
            max(abs(c1a - c1b), floatmin(Float64)),
        coefficient2_cancellation=(abs(c2a) + abs(c2b)) /
            max(abs(c2a - c2b), floatmin(Float64)),
    )
end

@inline function _amplitude_pair_distance(first, second)
    scale = max(
        abs(first.incidence),
        abs(first.reflection),
        abs(second.incidence),
        abs(second.reflection),
        floatmin(Float64),
    )
    return max(
        abs(first.incidence - second.incidence),
        abs(first.reflection - second.reflection),
    ) / scale
end

@inline function _horizon_stokes_pretrigger(result, branch, params)
    result.amplitude_match_policy in (
        :scaled_infinity_endpoint, :scaled_horizon_endpoint) && return false
    branch == :IN || return false
    params.kappa <= _HORIZON_STOKES_KAPPA_MAX || return false
    iszero(result.match_rstar) || return false
    result.horizon_seed.path.kind == :rotated || return false
    return iszero(result.horizon_seed.path.angle_offset)
end

function _solve_horizon_angle_at_match(
    coefficients,
    locals,
    selected,
    p,
    settings,
    angle_offset,
)
    horizon_seed = nothing
    endpoint_seconds = @elapsed horizon_seed =
        _select_rotated_horizon_path(
            coefficients,
            (locals.horizon_in,),
            (:horizon_in,),
            p,
            selected.match_x,
            settings,
            angle_offset,
        )
    horizon = nothing
    propagation_seconds = @elapsed horizon = _propagate_on_path(
        coefficients,
        horizon_seed.path,
        horizon_seed.checks[1].state,
        horizon_seed.checks[1].state,
        settings,
    )
    denominator_result = nothing
    diagnostics = nothing
    incidence = zero(ComplexF64)
    reflection = zero(ComplexF64)
    matching_seconds = @elapsed begin
        denominator_result = direct_abel_denominator(
            coefficients, :infinity, ComplexF64(selected.match_x))
        denominator_result.status == "OK" ||
            error("infinity Abel denominator failed: " *
                "$(denominator_result.status).")
        infinity_in = selected.endpoint_states.infinity_in
        infinity_out = selected.endpoint_states.infinity_out
        incidence, reflection = _connection_coefficients(
            horizon.state1,
            infinity_in,
            infinity_out,
            denominator_result.denominator,
        )
        diagnostics = _matching_diagnostics(
            horizon.state1, infinity_in, infinity_out)
    end
    endpoint_states = (;
        horizon_in=horizon.state1,
        horizon_out=missing,
        infinity_in=selected.endpoint_states.infinity_in,
        infinity_out=selected.endpoint_states.infinity_out,
    )
    return merge(selected, (;
        incidence,
        reflection,
        match_state=horizon.state1,
        endpoint_states,
        horizon_seed,
        horizon,
        patch_count=horizon.patches + selected.infinity.patches,
        matching_condition=diagnostics.condition,
        coefficient1_cancellation=diagnostics.coefficient1_cancellation,
        coefficient2_cancellation=diagnostics.coefficient2_cancellation,
        endpoint_us=1.0e6 * endpoint_seconds,
        propagation_us=1.0e6 * propagation_seconds,
        matching_us=1.0e6 * matching_seconds,
    ))
end

function _horizon_stokes_consensus_retry(
    coefficients,
    locals,
    branch,
    selected,
    p,
    settings,
)
    _horizon_stokes_pretrigger(selected, branch, coefficients.params) ||
        return (;
            selected,
            accepted=false,
            triggered=false,
            candidate_count=0,
            agreement=Inf,
            separation=0.0,
            angle_offset=0.0,
        )

    candidate_count = 0
    previous = nothing
    previous_step = -2
    best_agreement = Inf
    best_separation = 0.0
    for step in 0:_HORIZON_STOKES_MAX_ANGLE_STEPS
        angle_offset = _HORIZON_STOKES_INITIAL_ANGLE +
            step * _HORIZON_STOKES_ANGLE_STEP
        candidate = try
            _solve_horizon_angle_at_match(
                coefficients,
                locals,
                selected,
                p,
                settings,
                angle_offset,
            )
        catch
            nothing
        end
        candidate === nothing && continue
        candidate_count += 1
        separation = _amplitude_pair_distance(selected, candidate)
        if previous === nothing
            if step == 0 && separation < _HORIZON_STOKES_SEPARATION_MIN
                return (;
                    selected,
                    accepted=false,
                    triggered=true,
                    candidate_count,
                    agreement=Inf,
                    separation,
                    angle_offset=0.0,
                )
            end
        elseif step == previous_step + 1
            agreement = _amplitude_pair_distance(previous, candidate)
            pair_separation = min(
                _amplitude_pair_distance(selected, previous),
                separation,
            )
            if agreement < best_agreement
                best_agreement = agreement
                best_separation = pair_separation
            end
            accepted = agreement <= _HORIZON_STOKES_AGREEMENT_MAX &&
                pair_separation >= _HORIZON_STOKES_SEPARATION_MIN &&
                pair_separation >=
                    _HORIZON_STOKES_SEPARATION_RATIO_MIN * agreement
            accepted && return (;
                selected=candidate,
                accepted=true,
                triggered=true,
                candidate_count,
                agreement,
                separation=pair_separation,
                angle_offset,
            )
        end
        previous = candidate
        previous_step = step
    end
    return (;
        selected,
        accepted=false,
        triggered=true,
        candidate_count,
        agreement=best_agreement,
        separation=best_separation,
        angle_offset=0.0,
    )
end

@inline function _matching_consensus_pretrigger(result, branch, params)
    result.amplitude_match_policy in (
        :scaled_infinity_endpoint, :scaled_horizon_endpoint) && return false
    params.kappa <= _MATCH_CONSENSUS_KAPPA_MAX || return false
    imag_scale = abs(imag(params.omega))
    imag_scale > 100eps(Float64) * max(1.0, abs(params.omega)) || return false
    if branch == :IN
        condition_trigger = result.coefficient2_cancellation >=
                   _MATCH_CONSENSUS_IN_CANCELLATION_MIN &&
            result.matching_condition >= _MATCH_CONSENSUS_IN_CONDITION_MIN
        scale = max(abs(result.incidence), abs(result.reflection),
            floatmin(Float64))
        pole_trigger = params.kappa <=
                _MATCH_CONSENSUS_IN_POLE_KAPPA_MAX &&
            result.coefficient1_cancellation >=
                _MATCH_CONSENSUS_IN_POLE_IN_CANCELLATION_MIN &&
            result.coefficient2_cancellation >=
                _MATCH_CONSENSUS_IN_POLE_REF_CANCELLATION_MIN &&
            result.matching_condition >=
                _MATCH_CONSENSUS_IN_POLE_CONDITION_MIN &&
            abs(result.incidence) / scale <=
                _MATCH_CONSENSUS_IN_POLE_RESIDUAL_MAX
        return condition_trigger || pole_trigger
    end
    return result.coefficient2_cancellation >=
               _MATCH_CONSENSUS_UP_CANCELLATION_MIN &&
        result.matching_condition >= _MATCH_CONSENSUS_UP_CONDITION_MIN
end

function _matching_consensus_retry(
    coefficients,
    locals,
    branch,
    selected,
    settings,
)
    params = coefficients.params
    candidate_records =
        NamedTuple{(:fraction, :candidate),Tuple{Float64,typeof(selected)}}[]
    _matching_consensus_pretrigger(selected, branch, params) || return (;
        selected,
        accepted=false,
        triggered=false,
        candidate_count=0,
        agreement=Inf,
        separation=0.0,
        candidate_records,
    )
    direction = branch == :IN ? 1.0 : -1.0
    rstar_scale = log(25) / abs(imag(params.omega))
    candidates = typeof(selected)[]
    for fraction in (0.25, 0.5, 0.75)
        candidate = try
            _solve_at_match(
                coefficients,
                locals,
                branch,
                direction * fraction * rstar_scale,
                settings,
            )
        catch
            nothing
        end
        if candidate !== nothing
            push!(candidates, candidate)
            push!(candidate_records, (; fraction, candidate))
        end
    end
    length(candidates) >= 2 || return (;
        selected,
        accepted=false,
        triggered=true,
        candidate_count=length(candidates),
        agreement=Inf,
        separation=0.0,
        candidate_records,
    )

    best_first = candidates[1]
    best_second = candidates[2]
    agreement = _amplitude_pair_distance(best_first, best_second)
    for first_index in eachindex(candidates)
        for second_index in (first_index + 1):length(candidates)
            candidate_agreement = _amplitude_pair_distance(
                candidates[first_index], candidates[second_index])
            if candidate_agreement < agreement
                best_first = candidates[first_index]
                best_second = candidates[second_index]
                agreement = candidate_agreement
            end
        end
    end
    separation = min(
        _amplitude_pair_distance(selected, best_first),
        _amplitude_pair_distance(selected, best_second),
    )
    condition_trigger = branch == :IN &&
        selected.coefficient2_cancellation >=
            _MATCH_CONSENSUS_IN_CANCELLATION_MIN &&
        selected.matching_condition >= _MATCH_CONSENSUS_IN_CONDITION_MIN
    pole_only = branch == :IN && !condition_trigger
    agreement_max = pole_only ?
        _MATCH_CONSENSUS_IN_POLE_AGREEMENT_MAX :
        _MATCH_CONSENSUS_AGREEMENT_MAX
    retry = best_first.matching_condition <= best_second.matching_condition ?
        best_first : best_second
    condition_ok = !pole_only || retry.matching_condition <=
        _MATCH_CONSENSUS_IN_POLE_CONDITION_RATIO_MAX *
            selected.matching_condition
    accepted = agreement <= agreement_max &&
        separation >= _MATCH_CONSENSUS_SEPARATION_MIN &&
        separation >= _MATCH_CONSENSUS_SEPARATION_RATIO_MIN * agreement &&
        condition_ok
    return (;
        selected=accepted ? retry : selected,
        accepted,
        triggered=true,
        candidate_count=length(candidates),
        agreement,
        separation,
        candidate_records,
    )
end

function direct_complex_offpole_match_plateau_candidate(
    evaluator,
    branch,
    retry_cache=nothing,
    ; early_stability_max=_OFFPOLE_PLATEAU_EARLY_STABILITY_MAX,
)
    branch in (:IN, :UP) ||
        throw(ArgumentError("off-pole matching plateau requires :IN or :UP."))
    params = evaluator.params
    coefficients = evaluator.coefficients
    settings = evaluator.settings
    locals = retry_cache === nothing ?
        _local_solutions(coefficients, branch, settings.order) :
        retry_cache.locals
    rstar_scale = log(25) / abs(imag(params.omega))
    candidates = Any[]
    fractions = branch == :UP ? (0.0, -0.25, -0.5, -0.75) :
        (0.25, 0.5, 0.75, 1.0)
    reused_candidate_count = 0
    early_stability = Inf
    for fraction in fractions
        candidate = nothing
        if retry_cache !== nothing
            if branch == :UP && iszero(fraction)
                candidate = retry_cache.initial
            else
                magnitude = abs(fraction)
                for record in retry_cache.matching_candidates
                    if record.fraction == magnitude
                        candidate = record.candidate
                        break
                    end
                end
            end
            candidate === nothing || (reused_candidate_count += 1)
        end
        if candidate === nothing
            candidate = try
                _solve_at_match(
                    coefficients,
                    locals,
                    branch,
                    fraction * rstar_scale,
                    settings,
                )
            catch
                nothing
            end
        end
        if candidate !== nothing
            push!(candidates, candidate)
            if branch == :UP && length(candidates) == 2 &&
                    early_stability_max !== nothing
                early_stability = _amplitude_pair_distance(
                    candidates[1], candidates[2])
                early_stability <= early_stability_max && break
            end
        end
    end
    length(candidates) >= 2 ||
        error("fewer than two off-pole matching candidates constructed.")

    best_first = candidates[1]
    best_second = candidates[2]
    agreement = _amplitude_pair_distance(best_first, best_second)
    for first_index in eachindex(candidates)
        for second_index in (first_index + 1):length(candidates)
            candidate_agreement = _amplitude_pair_distance(
                candidates[first_index], candidates[second_index])
            if candidate_agreement < agreement
                best_first = candidates[first_index]
                best_second = candidates[second_index]
                agreement = candidate_agreement
            end
        end
    end
    retry = best_first.matching_condition <= best_second.matching_condition ?
        best_first : best_second
    split_mismatch = maximum(candidate.abel_error for candidate in candidates)
    return (;
        selected=retry,
        all_candidates=candidates,
        candidate_count=length(candidates),
        reused_candidate_count,
        early_stability,
        agreement,
        split_mismatch,
        first_rstar=best_first.match_rstar,
        second_rstar=best_second.match_rstar,
    )
end

function _match_x_from_rstar(params, match_rstar)
    r = r_from_rstar(params.a, Float64(match_rstar))
    x = Float64(real(_compact_x(params, r)))
    0 < x < 1 || error("real-axis matching coordinate lies outside (0,1).")
    return x
end

function _match_rstar_from_x(params, match_x)
    x = Float64(match_x)
    0 < x < 1 || throw(ArgumentError("complex rational xm must lie in (0,1)."))
    return Float64(rstar_from_r(params.a, real(_compact_r(params, x))))
end

function _local_solutions(coefficients, branch::Symbol, order::Int)
    if branch == :IN
        return (;
            horizon_in=_complex_horizon_solution(
                coefficients, :in, order),
            horizon_out=nothing,
            infinity_in=_complex_infinity_solution(
                coefficients, :in, order),
            infinity_out=_complex_infinity_solution(
                coefficients, :out, order),
        )
    elseif branch == :UP
        return (;
            horizon_in=_complex_horizon_solution(
                coefficients, :in, order),
            horizon_out=_complex_horizon_solution(
                coefficients, :out, order),
            infinity_in=nothing,
            infinity_out=_complex_infinity_solution(
                coefficients, :out, order),
        )
    end
    throw(ArgumentError("complex rational branch must be :IN or :UP."))
end

@inline function _horizon_representation(solution)
    solution === nothing && return :not_built
    return hasproperty(solution, :representation) ?
        solution.representation : :power_frobenius
end

@inline function _horizon_resonance_order(solution)
    solution === nothing && return 0
    return hasproperty(solution, :resonance_order) ?
        Int(solution.resonance_order) : 0
end

function _solve_at_match(
    coefficients,
    locals,
    branch::Symbol,
    match_rstar,
    settings,
)
    params = coefficients.params
    match_x = _match_x_from_rstar(params, match_rstar)
    omega_horizon = params.a / (2 * _rplus(params))
    p = params.omega - params.m * omega_horizon

    horizon_seed = nothing
    infinity_seed = nothing
    endpoint_seconds = @elapsed begin
        if branch == :IN
            horizon_seed = _select_horizon_path(
                coefficients, (locals.horizon_in,), (:horizon_in,), p,
                match_x, settings)
            infinity_seed = try
                _select_infinity_path(
                    coefficients, (locals.infinity_in, locals.infinity_out),
                    (:infinity_in, :infinity_out), params.omega, match_x,
                    settings)
            catch exception
                _principal_mst_axis(params.omega) || rethrow()
                message = sprint(showerror, exception)
                occursin("no certified two-ray infinity path", message) ||
                    rethrow()
                _select_principal_mst_infinity_path(
                    coefficients,
                    (locals.infinity_in, locals.infinity_out),
                    (:infinity_in, :infinity_out),
                    params.omega,
                    match_x,
                    settings,
                )
            end
        else
            horizon_seed = _select_horizon_path(
                coefficients, (locals.horizon_in, locals.horizon_out),
                (:horizon_in, :horizon_out), p, match_x, settings;
                pair_abel_target=isone(locals.infinity_out.variable_scale) ?
                    nothing : settings.infinity_target,
                fallback_pair_abel_target=
                    isone(locals.infinity_out.variable_scale) ?
                    nothing : settings.horizon_target)
            infinity_seed = _select_infinity_path(
                coefficients, (locals.infinity_out,), (:infinity_out,),
                params.omega, match_x, settings)
        end
    end

    horizon = nothing
    infinity = nothing
    infinity_endpoint_match = nothing
    horizon_endpoint_match = nothing
    propagation_seconds = @elapsed begin
        if branch == :IN
            horizon = try
                _propagate_on_path(
                    coefficients, horizon_seed.path,
                    horizon_seed.checks[1].state,
                    horizon_seed.checks[1].state,
                    settings)
            catch exception
                error("horizon path propagation failed: " *
                    sprint(showerror, exception))
            end
            if infinity_seed.path.kind == :principal_mst ||
                    !isone(locals.infinity_in.variable_scale)
                infinity = _seed_bundle(infinity_seed)
                infinity_endpoint_match = try
                    _propagate_forward_on_path(
                        coefficients,
                        infinity_seed.path,
                        horizon.state1,
                        horizon.state1,
                        settings;
                        variable_scale=locals.infinity_in.variable_scale,
                    )
                catch exception
                    error("infinity endpoint matching propagation failed: " *
                        sprint(showerror, exception))
                end
            else
                infinity = try
                    _propagate_on_path(
                        coefficients, infinity_seed.path,
                        infinity_seed.checks[1].state,
                        infinity_seed.checks[2].state,
                        settings)
                catch exception
                    error("infinity path propagation failed: " *
                        sprint(showerror, exception))
                end
            end
        else
            endpoint_match = !isone(locals.infinity_out.variable_scale) &&
                infinity_seed.path.kind != :principal_mst
            if endpoint_match
                horizon = _seed_bundle(horizon_seed)
            else
                horizon = try
                    _propagate_on_path(
                        coefficients, horizon_seed.path,
                        horizon_seed.checks[1].state,
                        horizon_seed.checks[2].state,
                        settings)
                catch exception
                    error("horizon path propagation failed: " *
                        sprint(showerror, exception))
                end
            end
            infinity = try
                if infinity_seed.path.kind == :principal_mst
                    _propagate_on_path(
                        coefficients, infinity_seed.path,
                        infinity_seed.checks[1].state,
                        infinity_seed.checks[1].state,
                        settings;
                        variable_scale=locals.infinity_out.variable_scale)
                else
                    _propagate_on_path(
                        coefficients, infinity_seed.path,
                        infinity_seed.checks[1].state,
                        infinity_seed.checks[1].state,
                        settings;
                        variable_scale=locals.infinity_out.variable_scale)
                end
            catch exception
                error("infinity path propagation failed: " *
                    sprint(showerror, exception))
            end
            if endpoint_match
                horizon_endpoint_match = try
                    _propagate_forward_horizon_path(
                        coefficients,
                        horizon_seed.path,
                        infinity.state1,
                        infinity.state1,
                        settings,
                    )
                catch exception
                    error("horizon endpoint matching propagation failed: " *
                        sprint(showerror, exception))
                end
            end
        end
    end

    denominator_result = nothing
    normalization = nothing
    diagnostics = nothing
    incidence = zero(ComplexF64)
    reflection = zero(ComplexF64)
    endpoint_states = nothing
    match_state = nothing
    amplitude_match_policy = :interior
    amplitude_match_x = match_x
    matching_seconds = @elapsed begin
        endpoint = branch == :IN ? :infinity : :horizon
        denominator_x = infinity_endpoint_match !== nothing ?
            infinity_endpoint_match.x :
            (horizon_endpoint_match !== nothing ?
                horizon_endpoint_match.x : ComplexF64(match_x))
        denominator_result = direct_abel_denominator(
            coefficients, endpoint, denominator_x)
        denominator_result.status == "OK" ||
            error("$(endpoint) Abel denominator failed: " *
                "$(denominator_result.status).")
        if branch == :IN
            basis1 = infinity_endpoint_match === nothing ?
                infinity.state1 : infinity_seed.checks[1].state
            basis2 = infinity_endpoint_match === nothing ?
                infinity.state2 : infinity_seed.checks[2].state
            target_state = infinity_endpoint_match === nothing ?
                horizon.state1 : infinity_endpoint_match.state1
            normalization = try
                _abel_normalize_second(
                    basis1, basis2, denominator_result.denominator)
            catch exception
                infinity_endpoint_match === nothing &&
                    _abel_range_error(exception) || rethrow()
                infinity_endpoint_match = try
                    _infinity_overlap_match(
                        coefficients,
                        infinity_seed,
                        horizon.state1,
                        settings;
                        variable_scale=locals.infinity_in.variable_scale,
                    )
                catch fallback_exception
                    error("infinity overlap matching fallback failed: " *
                        sprint(showerror, fallback_exception))
                end
                denominator_result = infinity_endpoint_match.denominator
                basis1 = infinity_endpoint_match.basis1
                basis2 = infinity_endpoint_match.basis2
                target_state = infinity_endpoint_match.state1
                infinity_endpoint_match.normalization
            end
            incidence, reflection = _connection_coefficients(
                target_state, basis1, normalization.corrected,
                denominator_result.denominator)
            diagnostics = _matching_diagnostics(
                target_state, basis1, normalization.corrected)
            if infinity_endpoint_match !== nothing
                amplitude_match_policy = :scaled_infinity_endpoint
                amplitude_match_x = real(infinity_endpoint_match.x)
            end
            match_state = horizon.state1
            endpoint_states = (;
                horizon_in=target_state,
                horizon_out=missing,
                infinity_in=basis1,
                infinity_out=normalization.corrected,
            )
        else
            basis1 = horizon_endpoint_match === nothing ?
                horizon.state1 : horizon_seed.checks[1].state
            basis2 = horizon_endpoint_match === nothing ?
                horizon.state2 : horizon_seed.checks[2].state
            target_state = horizon_endpoint_match === nothing ?
                infinity.state1 : horizon_endpoint_match.state1
            normalization = try
                _abel_normalize_second(
                    basis1, basis2, denominator_result.denominator)
            catch exception
                horizon_endpoint_match === nothing &&
                    _abel_range_error(exception) || rethrow()
                horizon_endpoint_match = try
                    _propagate_forward_horizon_path(
                        coefficients,
                        horizon_seed.path,
                        infinity.state1,
                        infinity.state1,
                        settings,
                    )
                catch fallback_exception
                    error("horizon endpoint matching fallback failed: " *
                        sprint(showerror, fallback_exception))
                end
                denominator_result = direct_abel_denominator(
                    coefficients, :horizon, horizon_endpoint_match.x)
                denominator_result.status == "OK" ||
                    error("horizon Abel denominator failed: " *
                        "$(denominator_result.status).")
                basis1 = horizon_seed.checks[1].state
                basis2 = horizon_seed.checks[2].state
                target_state = horizon_endpoint_match.state1
                _abel_normalize_second(
                    basis1, basis2, denominator_result.denominator)
            end
            reflection, incidence = _connection_coefficients(
                target_state, basis1, normalization.corrected,
                denominator_result.denominator)
            diagnostics = _matching_diagnostics(
                target_state, basis1, normalization.corrected)
            if horizon_endpoint_match !== nothing
                amplitude_match_policy = :scaled_horizon_endpoint
                amplitude_match_x = real(horizon_endpoint_match.x)
            end
            match_state = infinity.state1
            endpoint_states = (;
                horizon_in=basis1,
                horizon_out=normalization.corrected,
                infinity_in=missing,
                infinity_out=target_state,
            )
        end
    end
    patch_count = horizon.patches + infinity.patches +
        (infinity_endpoint_match === nothing ?
            0 : infinity_endpoint_match.patches) +
        (horizon_endpoint_match === nothing ?
            0 : horizon_endpoint_match.patches)
    return (;
        incidence,
        reflection,
        match_rstar=Float64(match_rstar),
        match_x,
        amplitude_match_policy,
        amplitude_match_x,
        match_state,
        endpoint_states,
        horizon_seed,
        infinity_seed,
        horizon,
        infinity,
        infinity_endpoint_match,
        horizon_endpoint_match,
        patch_count,
        matching_condition=diagnostics.condition,
        coefficient1_cancellation=diagnostics.coefficient1_cancellation,
        coefficient2_cancellation=diagnostics.coefficient2_cancellation,
        abel_ratio=normalization.ratio,
        abel_raw_error=normalization.raw_error,
        abel_error=normalization.corrected_error,
        endpoint_us=1.0e6 * endpoint_seconds,
        propagation_us=1.0e6 * propagation_seconds,
        matching_us=1.0e6 * matching_seconds,
    )
end

function direct_complex_rational_build(
    s::Integer,
    l::Integer,
    m::Integer,
    a,
    omega,
    branch::Symbol;
    lambda=nothing,
    nu=nothing,
    N=nothing,
    tol=nothing,
    xm=nothing,
)
    branch in (:IN, :UP) ||
        throw(ArgumentError("complex rational branch must be :IN or :UP."))
    order = Int(N === nothing ? _DEFAULT_ORDER : N)
    tolerance = Float64(tol === nothing ? 1.0e-14 : tol)
    settings = DirectComplexRationalSettings(order, tolerance)

    params = nothing
    parameter_seconds = @elapsed params = direct_gsn_parameters(
        s, l, m, a, omega; lambda, nu)
    auto_near_real_match = xm === nothing &&
        _use_near_real_match(params.omega)
    initial_match_rstar = if xm !== nothing
        _match_rstar_from_x(params, xm)
    elseif auto_near_real_match
        _match_rstar_from_x(params, _NEAR_REAL_MATCH_X)
    else
        0.0
    end
    initial_match_x = _match_x_from_rstar(params, initial_match_rstar)
    initial_controls = direct_gsn_controls(
        params; N=order, xm=initial_match_x, tol=tolerance,
        sfe=false, lfe=false)
    coefficients = nothing
    coefficient_seconds = @elapsed coefficients = direct_gsn_coefficients(
        params; controls=initial_controls)
    locals = nothing
    local_seconds = @elapsed locals = _local_solutions(
        coefficients, branch, order)

    initial = _solve_at_match(
        coefficients, locals, branch, initial_match_rstar, settings)
    selected = initial
    match_policy = xm !== nothing ? :explicit_x :
        auto_near_real_match ? :near_real_x07 : :rstar_zero
    retry_seconds = 0.0
    if xm === nothing && branch == :UP &&
            params.kappa <= _MATCH_CONSENSUS_KAPPA_MAX &&
            initial.matching_condition > _MATCH_CONDITION_RETRY
        shifted_rstar = -log(25) / abs(imag(params.omega))
        shifted = nothing
        retry_seconds = @elapsed shifted = try
            _solve_at_match(
                coefficients, locals, branch, shifted_rstar, settings)
        catch
            nothing
        end
        if shifted !== nothing &&
                shifted.matching_condition < initial.matching_condition
            selected = shifted
            match_policy = :up_condition_retry
        end
    end
    matching_consensus = nothing
    consensus_seconds = @elapsed matching_consensus = xm === nothing ?
        _matching_consensus_retry(
            coefficients, locals, branch, selected, settings) :
        (;
            selected,
            accepted=false,
            triggered=false,
            candidate_count=0,
            agreement=Inf,
            separation=0.0,
            candidate_records=
                NamedTuple{(:fraction, :candidate),
                    Tuple{Float64,typeof(selected)}}[],
        )
    if matching_consensus.accepted
        selected = matching_consensus.selected
        match_policy = :branch_shift_consensus
    end
    horizon_stokes = nothing
    stokes_seconds = @elapsed horizon_stokes = xm === nothing ?
        _horizon_stokes_consensus_retry(
            coefficients, locals, branch, selected, params.omega -
                params.m * params.a / (2 * _rplus(params)), settings) :
        (;
            selected,
            accepted=false,
            triggered=false,
            candidate_count=0,
            agreement=Inf,
            separation=0.0,
            angle_offset=0.0,
        )
    if horizon_stokes.accepted
        selected = horizon_stokes.selected
        match_policy = :horizon_stokes_consensus
    end
    controls = direct_gsn_controls(
        params; N=order, xm=selected.match_x, tol=tolerance,
        sfe=false, lfe=false)
    evaluator = DirectComplexRationalEvaluator(
        params, coefficients, settings, selected.match_x,
        selected.match_state)
    metadata = (;
        backend=:direct_gsn_two_ray_rational,
        horizon_in_representation=
            _horizon_representation(locals.horizon_in),
        horizon_out_representation=
            _horizon_representation(locals.horizon_out),
        horizon_resonance_order=max(
            _horizon_resonance_order(locals.horizon_in),
            _horizon_resonance_order(locals.horizon_out),
        ),
        amplitude_observable=(
            _horizon_representation(locals.horizon_in) ==
                :log_frobenius ||
            _horizon_representation(locals.horizon_out) ==
                :log_frobenius
        ) ? :resonant_horizon_connection_coefficients :
            :plane_wave_amplitudes,
        match_policy,
        match_rstar=selected.match_rstar,
        match_x=selected.match_x,
        amplitude_match_policy=selected.amplitude_match_policy,
        amplitude_match_x=selected.amplitude_match_x,
        initial_matching_condition=initial.matching_condition,
        matching_condition=selected.matching_condition,
        coefficient1_cancellation=selected.coefficient1_cancellation,
        coefficient2_cancellation=selected.coefficient2_cancellation,
        horizon_path_kind=selected.horizon_seed.path.kind,
        horizon_direction=selected.horizon_seed.path.direction,
        horizon_angle_offset=selected.horizon_seed.path.angle_offset,
        horizon_rho=selected.horizon_seed.path.rho,
        infinity_rho=selected.infinity_seed.path.rho,
        horizon_coordinate_steps=selected.horizon_seed.path.accepted_steps,
        infinity_coordinate_steps=selected.infinity_seed.path.accepted_steps,
        horizon_endpoint_score=selected.horizon_seed.score,
        infinity_endpoint_score=selected.infinity_seed.score,
        horizon_patches=selected.horizon.patches,
        infinity_patches=selected.infinity.patches,
        infinity_scaled_y_patches=selected.infinity.scaled_y_patches,
        horizon_scaled_x_patches=
            selected.horizon_endpoint_match === nothing ?
                0 : selected.horizon_endpoint_match.scaled_x_patches,
        patch_count=selected.patch_count,
        propagation_score=max(
            selected.horizon.max_score,
            selected.infinity.max_score,
        ),
        abel_ratio=selected.abel_ratio,
        abel_raw_error=selected.abel_raw_error,
        abel_error=selected.abel_error,
        parameter_us=1.0e6 * parameter_seconds,
        coefficient_us=1.0e6 * coefficient_seconds,
        local_us=1.0e6 * local_seconds,
        endpoint_us=selected.endpoint_us,
        propagation_us=selected.propagation_us,
        matching_us=selected.matching_us,
        retry_us=1.0e6 * retry_seconds,
        horizon_stokes_triggered=horizon_stokes.triggered,
        horizon_stokes_accepted=horizon_stokes.accepted,
        horizon_stokes_candidate_count=horizon_stokes.candidate_count,
        horizon_stokes_agreement=horizon_stokes.agreement,
        horizon_stokes_separation=horizon_stokes.separation,
        horizon_stokes_angle_offset=horizon_stokes.angle_offset,
        horizon_stokes_us=1.0e6 * stokes_seconds,
        matching_consensus_triggered=matching_consensus.triggered,
        matching_consensus_accepted=matching_consensus.accepted,
        matching_consensus_candidate_count=matching_consensus.candidate_count,
        matching_consensus_agreement=matching_consensus.agreement,
        matching_consensus_separation=matching_consensus.separation,
        matching_consensus_us=1.0e6 * consensus_seconds,
        endpoint_states=selected.endpoint_states,
    )
    return (;
        params,
        controls,
        coefficients,
        evaluator,
        incidence=selected.incidence,
        reflection=selected.reflection,
        transmission=ComplexF64(1),
        metadata,
        retry_cache=(;
            locals,
            initial,
            matching_candidates=matching_consensus.candidate_records,
        ),
    )
end

const _UP_HORIZON_IN_ENDPOINT_SCORE_MAX = 1.0e-11

function direct_complex_up_horizon_in_candidate(evaluator, metadata)
    params = evaluator.params
    coefficients = evaluator.coefficients
    settings = evaluator.settings
    match_x = evaluator.match_x
    horizon_in = direct_zero_local_solution(
        coefficients, :in, settings.order)
    p = params.omega - params.m * params.a / (2 * _rplus(params))
    seed = _select_rotated_horizon_path(
        coefficients,
        (horizon_in,),
        (:horizon_in,),
        p,
        match_x,
        settings;
        max_best_score=_UP_HORIZON_IN_ENDPOINT_SCORE_MAX,
    )
    propagated = _propagate_on_path(
        coefficients,
        seed.path,
        seed.checks[1].state,
        seed.checks[1].state,
        settings,
    )
    denominator = direct_abel_denominator(
        coefficients, :horizon, ComplexF64(match_x))
    denominator.status == "OK" ||
        error("horizon Abel denominator failed: $(denominator.status).")
    endpoint_states = metadata.endpoint_states
    normalization = _abel_normalize_second(
        propagated.state1,
        endpoint_states.horizon_out,
        denominator.denominator,
    )
    reflection, incidence = _connection_coefficients(
        endpoint_states.infinity_out,
        propagated.state1,
        normalization.corrected,
        denominator.denominator,
    )
    diagnostics = _matching_diagnostics(
        endpoint_states.infinity_out,
        propagated.state1,
        normalization.corrected,
    )
    return (;
        incidence,
        reflection,
        horizon_in=propagated.state1,
        horizon_out=normalization.corrected,
        infinity_out=endpoint_states.infinity_out,
        seed,
        propagated,
        normalization,
        matching_condition=diagnostics.condition,
        coefficient1_cancellation=diagnostics.coefficient1_cancellation,
        coefficient2_cancellation=diagnostics.coefficient2_cancellation,
        split_mismatch=normalization.corrected_error,
        patch_count=propagated.patches + metadata.infinity_patches,
    )
end

function direct_complex_up_initial_match_candidates(evaluator)
    params = evaluator.params
    coefficients = evaluator.coefficients
    settings = evaluator.settings
    locals = _local_solutions(coefficients, :UP, settings.order)
    scale = log(25) / abs(imag(params.omega))
    far_rstar = -0.5 * scale
    mid_rstar = -0.25 * scale
    far = _solve_at_match(
        coefficients, locals, :UP, far_rstar, settings)
    mid = _solve_at_match(
        coefficients, locals, :UP, mid_rstar, settings)
    initial = _solve_at_match(
        coefficients, locals, :UP, 0.0, settings)
    return (; scale, far_rstar, mid_rstar, far, mid, initial)
end

function _state_x(evaluator::DirectComplexRationalEvaluator, x::Real)
    target = Float64(x)
    0 < target < 1 ||
        throw(DomainError(x, "real-axis compact coordinate must lie in (0,1)."))
    if abs(target - evaluator.match_x) <=
            100eps(Float64) * max(1.0, abs(evaluator.match_x))
        return (;
            X=evaluator.match_state.X,
            dXdx=evaluator.match_state.dXdx,
            error=0.0,
            patches=0,
        )
    end
    propagated = _propagate_pair(
        evaluator.coefficients,
        evaluator.match_x,
        evaluator.match_state,
        evaluator.match_state,
        target,
        evaluator.settings,
    )
    return (;
        X=propagated.state1.X,
        dXdx=propagated.state1.dXdx,
        error=propagated.max_score,
        patches=propagated.patches,
    )
end

function direct_complex_rational_state_r(
    evaluator::DirectComplexRationalEvaluator,
    r::Real,
)
    rf = Float64(r)
    rf > _rplus(evaluator.params) ||
        throw(DomainError(r, "real-axis Direct complex evaluation requires r > r_plus."))
    x = Float64(_compact_x(evaluator.params, rf))
    state = _state_x(evaluator, x)
    dxdrstar = 2 * evaluator.params.kappa * x /
        (rf * rf + evaluator.params.a * evaluator.params.a)
    return (;
        X=state.X,
        dXdrstar=ComplexF64(state.dXdx * dxdrstar),
        error=state.error,
        patches=state.patches,
    )
end

function _states_x(
    evaluator::DirectComplexRationalEvaluator,
    targets::AbstractVector{<:Real},
    chain_max::Int=_EVALUATION_CHAIN_MAX,
)
    count = length(targets)
    states = Vector{DirectComplexRationalState}(undef, count)
    errors = Vector{Float64}(undef, count)
    count == 0 && return states, errors

    xs = Float64.(targets)
    all(x -> 0 < x < 1, xs) ||
        throw(DomainError(
            targets, "real-axis compact coordinates must lie in (0,1)."))
    order = sortperm(xs)
    split = searchsortedlast(xs[order], evaluator.match_x)

    function normalize_state(state)
        scale = max(abs(state.X), abs(state.dXdx), floatmin(Float64))
        return DirectComplexRationalState(
            state.X / scale,
            state.dXdx / scale,
        ), scale
    end

    match_state, match_scale = normalize_state(evaluator.match_state)

    function advance(
        current_x,
        current_state,
        current_scale,
        current_error,
        target,
    )
        base_scale = current_scale
        propagated = try
            _propagate_pair(
                evaluator.coefficients,
                current_x,
                current_state,
                current_state,
                target,
                evaluator.settings,
            )
        catch error
            if !(error isa ErrorException &&
                    error.msg == "complex rational patch limit reached.")
                rethrow()
            end
            base_scale = match_scale
            _propagate_pair(
                evaluator.coefficients,
                evaluator.match_x,
                match_state,
                match_state,
                target,
                evaluator.settings,
            )
        end
        next_state, next_scale = normalize_state(propagated.state1)
        cumulative_scale = base_scale * next_scale
        actual_state = DirectComplexRationalState(
            next_state.X * cumulative_scale,
            next_state.dXdx * cumulative_scale,
        )
        return (
            target,
            next_state,
            cumulative_scale,
            max(current_error, propagated.max_score),
            actual_state,
        )
    end

    current_x = evaluator.match_x
    current_state = match_state
    current_scale = match_scale
    current_actual = evaluator.match_state
    current_error = 0.0
    chain_length = 0
    for position in split:-1:1
        index = order[position]
        target = xs[index]
        if abs(target - current_x) >
                100eps(Float64) * max(1.0, abs(current_x))
            if chain_length >= chain_max
                current_x = evaluator.match_x
                current_state = match_state
                current_scale = match_scale
                current_actual = evaluator.match_state
                current_error = 0.0
                chain_length = 0
            end
            current_x, current_state, current_scale, current_error,
                current_actual = advance(
                    current_x,
                    current_state,
                    current_scale,
                    current_error,
                    target,
                )
            chain_length += 1
        end
        states[index] = current_actual
        errors[index] = current_error
    end

    current_x = evaluator.match_x
    current_state = match_state
    current_scale = match_scale
    current_actual = evaluator.match_state
    current_error = 0.0
    chain_length = 0
    for position in (split + 1):count
        index = order[position]
        target = xs[index]
        if abs(target - current_x) >
                100eps(Float64) * max(1.0, abs(current_x))
            if chain_length >= chain_max
                current_x = evaluator.match_x
                current_state = match_state
                current_scale = match_scale
                current_actual = evaluator.match_state
                current_error = 0.0
                chain_length = 0
            end
            current_x, current_state, current_scale, current_error,
                current_actual = advance(
                    current_x,
                    current_state,
                    current_scale,
                    current_error,
                    target,
                )
            chain_length += 1
        end
        states[index] = current_actual
        errors[index] = current_error
    end
    return states, errors
end

function direct_complex_rational_states_r(
    evaluator::DirectComplexRationalEvaluator,
    radii::AbstractVector{<:Real},
)
    rs = Float64.(radii)
    isempty(rs) && return NamedTuple[]
    all(r -> r > _rplus(evaluator.params), rs) ||
        throw(DomainError(
            radii, "real-axis Direct complex evaluation requires r > r_plus."))
    xs = _compact_x.(Ref(evaluator.params), rs)
    function convert_states(states, errors)
        return map(eachindex(rs)) do index
            x = xs[index]
            r = rs[index]
            dxdrstar = 2 * evaluator.params.kappa * x /
                (r * r + evaluator.params.a * evaluator.params.a)
            state = states[index]
            (
                X=state.X,
                dXdrstar=ComplexF64(state.dXdx * dxdrstar),
                error=errors[index],
            )
        end
    end
    function certificate(states)
        count = length(rs)
        indices = unique(round.(Int, range(1, count; length=min(9, count))))
        error = 0.0
        for index in indices
            reference = direct_complex_rational_state_r(
                evaluator, rs[index])
            candidate = states[index]
            scale = max(
                abs(reference.X),
                abs(reference.dXdrstar),
                abs(candidate.X),
                abs(candidate.dXdrstar),
                floatmin(Float64),
            )
            error = max(
                error,
                abs(reference.X - candidate.X) / scale,
                abs(reference.dXdrstar - candidate.dXdrstar) / scale,
            )
        end
        return error
    end

    states_x, errors = _states_x(
        evaluator, xs, _EVALUATION_CHAIN_MAX)
    states = convert_states(states_x, errors)
    certificate(states) <= 1.0e-10 && return states
    return map(r -> direct_complex_rational_state_r(evaluator, r), rs)
end

function (evaluator::DirectComplexRationalEvaluator)(rstar::Real)
    r = r_from_rstar(evaluator.params.a, Float64(rstar))
    state = direct_complex_rational_state_r(evaluator, r)
    return state.X, state.dXdrstar, state.error
end

function (evaluator::DirectComplexRationalEvaluator)(
    rstars::AbstractVector{<:Real},
)
    radii = r_from_rstar.(Ref(evaluator.params.a), Float64.(rstars))
    states = direct_complex_rational_states_r(evaluator, radii)
    return (
        getproperty.(states, :X),
        getproperty.(states, :dXdrstar),
        getproperty.(states, :error),
    )
end

end
