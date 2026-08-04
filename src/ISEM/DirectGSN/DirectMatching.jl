module DirectMatching

using ..DirectCoefficientTables: DirectCoefficientSet, direct_gsn_coefficients
using ..DirectIteration:
    DirectBasis,
    DirectPatch,
    DirectScaledPatch,
    DirectLogScaledPatch,
    DirectLogScaledState,
    DirectLogScaledBasis,
    DirectIterationScratch,
    DirectTruncation,
    direct_basis_patch_count,
    direct_basis_state,
    direct_basis_value,
    direct_endpoint_scale,
    direct_infinity_exponents,
    direct_iterate_scaled_y,
    direct_iterate_logscaled_from_state,
    direct_iterate_from_state,
    direct_iterate_from_infinity,
    direct_iterate_pair_from_zero,
    direct_iterate_from_zero,
    direct_iterate_pair_from_infinity,
    direct_infinity_seed_wronskian,
    direct_select_infinity_endpoint,
    direct_logscaled_state,
    direct_logscaled_basis_state,
    direct_materialize_logscaled_state,
    _SFE_BRIDGE_TARGET_Y,
    _endpoint_state
using ..DirectEikonal: DirectEikonalPatch, eikonal_preflight
using ..DirectNearExtreme: near_extreme_prepare, near_extreme_selected
using ..DirectMSTInfinity: MSTCertificateError, direct_abel_denominator
using ..DirectMSTInfinity: direct_mst_eval_plan, direct_mst_infinity_basis
using ..DirectMSTInfinity: direct_mst_infinity_pair, direct_mst_plan, direct_mst_state
using ..DirectMSTInfinity: direct_mst_logscaled_seed, direct_mst_pin_state
using ..DirectMSTInfinity: direct_mst_scaled_state
using ..DirectMSTInfinity: mst_principal_amplitudes
using ..DirectParameters:
    DirectGSNControls,
    DirectGSNParameters,
    SFE_FREQUENCY_LIMIT,
    SFE_TRANSITION_LIMIT,
    _ordinary_fallback,
    direct_gsn_controls,
    direct_gsn_parameters,
    direct_horizon_tail,
    direct_spin_regime

export DirectRoute, DirectConjugatedRoute, DirectRoutePlan
export direct_match, direct_gsn_radial
export direct_evaluate, direct_state, direct_route_patch_count, direct_route_plan, direct_route_truncations

const DIRECT_DENOMINATOR_REL_MIN = 1.0e-10
const LOW_FREQUENCY_MIN_ORDER = 40
const LOW_FREQUENCY_TAIL_HORIZON_ORDER = 80
const LOW_FREQUENCY_MATCH_X = 0.9
const LOW_FREQUENCY_AXISYMMETRIC_MATCH_X = 0.99
const ORDINARY_IN_MATCH_X = 0.7
const ORDINARY_UP_MATCH_X = 0.3
const BRIDGE_CONDITION_LIMIT = 1.0e12
const PHYSICAL_BRIDGE = get(ENV, "DIRECT_GSN_PHYSICAL_BRIDGE", "1") != "0"
const PUBLIC_EVALUATION_ENDPOINT_X = 0.999995
const MATCHING_SCALE_LOG_MARGIN = 32.0
const SFE_ENDPOINT_HANDOFF_TOLERANCE = 1.0e-10
const SFE_UP_TRIAL_STEPS = 128
const REAL_MST_CONDITION_MAX = 1.0e3
const REAL_MST_NEAR_STATIC_MAX = 1.0e-5

struct DirectEndpointPlan{S}
    selection::S
    scale::ComplexF64
    seed_x::Float64
end

struct DirectScaledBasis{B}
    basis::B
    scale::ComplexF64
end

struct DirectRoute{P,C,HI,HO,IO,II,B,EB,EP,M}
    branch::Symbol
    params::P
    controls::DirectGSNControls
    coefficients::C
    match_x::Float64
    infinity_endpoint_y::Float64
    infinity_out_order::Int
    infinity_in_order::Int
    incidence::ComplexF64
    reflection::ComplexF64
    solution_scale::Float64
    horizon_in::HI
    horizon_out::HO
    infinity_out::IO
    infinity_in::II
    bridge::B
    endpoint_bridge::EB
    endpoint_plan::EP
    mst_plan::M
end

struct DirectConjugatedRoute{R,P}
    route::R
    params::P
end

function Base.getproperty(route::DirectConjugatedRoute, name::Symbol)
    name === :route && return getfield(route, :route)
    name === :params && return getfield(route, :params)
    return getproperty(getfield(route, :route), name)
end

DirectRoute(
    branch,
    params,
    controls,
    coefficients,
    match_x,
    infinity_endpoint_y,
    infinity_out_order,
    infinity_in_order,
    incidence,
    reflection,
    horizon_in,
    horizon_out,
    infinity_out,
    infinity_in,
) = DirectRoute(
    branch,
    params,
    controls,
    coefficients,
    match_x,
    infinity_endpoint_y,
    infinity_out_order,
    infinity_in_order,
    incidence,
    reflection,
    1.0,
    horizon_in,
    horizon_out,
    infinity_out,
    infinity_in,
    nothing,
    nothing,
    nothing,
    nothing,
)

DirectRoute(
    branch,
    params,
    controls,
    coefficients,
    match_x,
    infinity_endpoint_y,
    infinity_out_order,
    infinity_in_order,
    incidence,
    reflection,
    horizon_in,
    horizon_out,
    infinity_out,
    infinity_in,
    bridge,
) = DirectRoute(
    branch,
    params,
    controls,
    coefficients,
    match_x,
    infinity_endpoint_y,
    infinity_out_order,
    infinity_in_order,
    incidence,
    reflection,
    1.0,
    horizon_in,
    horizon_out,
    infinity_out,
    infinity_in,
    bridge,
    nothing,
    nothing,
    nothing,
)

struct DirectRoutePlan
    frequency::Symbol
    frequency_reason::Symbol
    spin::Symbol
    spin_reason::Symbol
    endpoint::Symbol
    propagation::Symbol
    matching::Symbol
    match_x::Float64
    horizon_order::Int
    ordinary_order::Int
    infinity_order::Int
    eikonal_candidate::Bool
    ordinary_patches::Int
    eikonal_patches::Int
    endpoint_patches::Int
    horizon_patches::Int
    infinity_patches::Int
    angular_compression::Float64
    phase_compression::Float64
end

function _normalized_route_branch(branch)
    text = uppercase(String(Symbol(branch)))
    text == "IN" && return :IN
    text == "UP" && return :UP
    throw(ArgumentError("direct GSN route branch must be :IN or :UP."))
end

_basis_tuple(basis::DirectBasis) = (X=basis.match_X, dXdx=basis.match_dXdx)
_logscaled_match_state(basis::DirectLogScaledBasis) = basis.match_state
_logscaled_match_state(basis::DirectBasis) = direct_logscaled_state(
    basis.match_X, basis.match_dXdx)
_logscaled_match_state(state::DirectLogScaledState) = state
_logscaled_match_state(state) = direct_logscaled_state(state.X, state.dXdx)
_scaled_state(state, scale) = (
    X=ComplexF64(scale * state.X),
    dXdx=ComplexF64(scale * state.dXdx),
)

function _required_basis(basis, name)
    basis === nothing && error("missing direct GSN basis $name")
    return basis
end

function _with_low_frequency_defaults(ctrl::DirectGSNControls, params, route_branch::Symbol)
    ctrl.sfe || return ctrl
    omega = abs(getproperty(params, :omega))
    0 < omega <= SFE_TRANSITION_LIMIT || return ctrl
    if omega > SFE_FREQUENCY_LIMIT
        route_branch == :IN || return ctrl
        getproperty(ctrl, :source) == :default || return ctrl
        return DirectGSNControls(
            ctrl.horizon_order,
            ctrl.ordinary_order,
            ctrl.infinity_order,
            LOW_FREQUENCY_MATCH_X,
            ctrl.tolerance,
            ctrl.basis,
            ctrl.endpoint_basis,
            ctrl.ordinary_basis,
            ctrl.sfe,
            ctrl.lfe,
            ctrl.frequency_selection,
            :sfe_transition_match,
        )
    end
    match_x = if getproperty(ctrl, :source) == :default
        spin = direct_spin_regime(params)
        spin.regime == :near_extremal && iszero(getproperty(params, :m)) ?
            LOW_FREQUENCY_AXISYMMETRIC_MATCH_X : LOW_FREQUENCY_MATCH_X
    else
        ctrl.match_x
    end
    0 < match_x < 1 || return ctrl
    horizon_order = direct_horizon_tail(params) ?
        LOW_FREQUENCY_TAIL_HORIZON_ORDER : LOW_FREQUENCY_MIN_ORDER
    return DirectGSNControls(
        max(ctrl.horizon_order, horizon_order),
        max(ctrl.ordinary_order, LOW_FREQUENCY_MIN_ORDER),
        ctrl.infinity_order,
        match_x,
        ctrl.tolerance,
        :low_frequency_scaled_y,
        ctrl.endpoint_basis,
        ctrl.ordinary_basis,
        ctrl.sfe,
        ctrl.lfe,
        ctrl.frequency_selection,
        :low_frequency_default,
    )
end

function _with_ordinary_defaults(
    ctrl::DirectGSNControls,
    params,
    route_branch::Symbol,
)
    getproperty(ctrl, :source) == :default || return ctrl
    ctrl.frequency_selection.regime == :ordinary || return ctrl
    isreal(getproperty(params, :omega)) || return ctrl
    match_x = route_branch == :IN ?
        ORDINARY_IN_MATCH_X : ORDINARY_UP_MATCH_X
    return DirectGSNControls(
        ctrl.horizon_order,
        ctrl.ordinary_order,
        ctrl.infinity_order,
        match_x,
        ctrl.tolerance,
        ctrl.basis,
        ctrl.endpoint_basis,
        ctrl.ordinary_basis,
        ctrl.sfe,
        ctrl.lfe,
        ctrl.frequency_selection,
        :ordinary_branch_match,
    )
end

function _connection_coefficients(target, basis1, basis2)
    det = basis1.X * basis2.dXdx - basis2.X * basis1.dXdx
    iszero(det) && error("singular direct GSN matching system")
    c1 = (target.X * basis2.dXdx - basis2.X * target.dXdx) / det
    c2 = (basis1.X * target.dXdx - target.X * basis1.dXdx) / det
    return c1, c2
end

@inline _finite_complex(value) =
    isfinite(real(value)) && isfinite(imag(value))

function _logabs(value)
    re = abs(real(value))
    im = abs(imag(value))
    scale = max(re, im)
    iszero(scale) && return -Inf
    isfinite(scale) || return Inf
    return log(scale) + 0.5 * log((re / scale)^2 + (im / scale)^2)
end

function _logadd(a, b)
    top = max(a, b)
    top == -Inf && return -Inf
    isfinite(top) || return top
    return top + log(exp(a - top) + exp(b - top))
end

function _eval_condition(target, coefficient1, basis1, coefficient2, basis2)
    value_terms = _logadd(
        _logabs(coefficient1) + _logabs(basis1.X),
        _logabs(coefficient2) + _logabs(basis2.X),
    )
    derivative_terms = _logadd(
        _logabs(coefficient1) + _logabs(basis1.dXdx),
        _logabs(coefficient2) + _logabs(basis2.dXdx),
    )
    value_scale = _logabs(target.X)
    derivative_scale = _logabs(target.dXdx)
    value_condition = value_scale == -Inf ? Inf : value_terms - value_scale
    derivative_condition = derivative_scale == -Inf ? Inf :
        derivative_terms - derivative_scale
    return max(value_condition, derivative_condition)
end

function _needs_bridge(target, coefficient1, basis1, coefficient2, basis2)
    PHYSICAL_BRIDGE || return false
    condition = _eval_condition(
        target, coefficient1, basis1, coefficient2, basis2)
    return condition > log(BRIDGE_CONDITION_LIMIT)
end

function _build_bridge(
    coefficients,
    ctrl,
    scratch,
    branch,
    match_x,
    target_x,
    target,
    coefficient1,
    basis1,
    coefficient2,
    basis2,
)
    _needs_bridge(target, coefficient1, basis1, coefficient2, basis2) ||
        return nothing
    abs(target_x - match_x) > 100eps(Float64) || return nothing
    kind = branch == :IN ? :horizon_in : :infinity_out
    try
        return direct_iterate_from_state(
            coefficients,
            kind,
            match_x,
            target.X,
            target.dXdx,
            target_x;
            controls=ctrl,
            scratch=scratch,
            allow_eikonal=true,
            allow_partial=true,
        )
    catch err
        message = sprint(showerror, err)
        startswith(message, "direct GSN hybrid recurrence failed") || rethrow()
        return nothing
    end
end

function _endpoint_branch_state(coefficients, selection, branch, x)
    kind = branch == :in ? :infinity_in : :infinity_out
    coeffs = branch == :in ? selection.in_coeffs : selection.out_coeffs
    order = branch == :in ? selection.in_order : selection.out_order
    rho, sigma = direct_infinity_exponents(coefficients.params, branch)
    value, derivative = _endpoint_state(
        kind,
        @view(coeffs[1:(order + 1)]),
        rho,
        sigma / selection.variable_scale,
        x,
        selection.variable_scale,
    )
    scale = direct_endpoint_scale(kind, coefficients.params)
    return (X=scale * value, dXdx=scale * derivative)
end

function _physical_endpoint_state(
    coefficients,
    route_branch,
    selection,
    x,
    incidence,
    reflection,
    solution_scale,
    scale=1.0 + 0.0im,
)
    outgoing = _endpoint_branch_state(coefficients, selection, :out, x)
    if route_branch == :UP
        return (
            X=scale * solution_scale * outgoing.X,
            dXdx=scale * solution_scale * outgoing.dXdx,
        )
    end
    incoming = _endpoint_branch_state(coefficients, selection, :in, x)
    return (
        X=scale * (incidence * incoming.X + reflection * outgoing.X),
        dXdx=scale * (incidence * incoming.dXdx +
            reflection * outgoing.dXdx),
    )
end

function _fit_state_scale(reference, candidate)
    reference_scale = max(abs(reference.X), abs(reference.dXdx))
    candidate_scale = max(abs(candidate.X), abs(candidate.dXdx))
    reference_scale > 0 && candidate_scale > 0 || return nothing
    isfinite(reference_scale) && isfinite(candidate_scale) || return nothing
    ref_x = reference.X / reference_scale
    ref_dx = reference.dXdx / reference_scale
    candidate_x = candidate.X / candidate_scale
    candidate_dx = candidate.dXdx / candidate_scale
    denominator = abs2(candidate_x) + abs2(candidate_dx)
    denominator > 0 && isfinite(denominator) || return nothing
    unit_scale = (conj(candidate_x) * ref_x +
        conj(candidate_dx) * ref_dx) / denominator
    scale = (reference_scale / candidate_scale) * unit_scale
    return _finite_complex(scale) ? ComplexF64(scale) : nothing
end

function _state_error(left, right)
    scale = max(abs(left.X), abs(left.dXdx), abs(right.X), abs(right.dXdx),
        floatmin(Float64))
    isfinite(scale) || return Inf
    return max(abs(left.X - right.X), abs(left.dXdx - right.dXdx)) / scale
end

function _sfe_endpoint_selection(coefficients, ctrl, match_x)
    try
        return direct_select_infinity_endpoint(
            coefficients, match_x; controls=ctrl)
    catch err
        message = sprint(showerror, err)
        startswith(message, "direct GSN infinity") || rethrow()
        return nothing
    end
end

function _build_sfe_endpoint_bridge(
    coefficients,
    ctrl,
    kind,
    route_branch,
    match_x,
    target,
    incidence,
    reflection,
    solution_scale,
)
    PHYSICAL_BRIDGE && ctrl.sfe || return nothing, nothing
    match_x < PUBLIC_EVALUATION_ENDPOINT_X || return nothing, nothing
    omega = abs(getproperty(coefficients.params, :omega))
    omega > 0 || return nothing, nothing
    target_y = (1.0 - PUBLIC_EVALUATION_ENDPOINT_X) / omega
    selection = _sfe_endpoint_selection(coefficients, ctrl, match_x)
    plan = Ref{Any}(nothing)

    function stop_check(x, X, dXdx)
        selection === nothing && return false
        1.0 - x <= selection.endpoint_y * (1 + 20eps(Float64)) ||
            return false
        bridge_state = (X=X, dXdx=dXdx)
        endpoint = _physical_endpoint_state(
            coefficients,
            route_branch,
            selection,
            x,
            incidence,
            reflection,
            solution_scale,
        )
        endpoint_scale = _fit_state_scale(bridge_state, endpoint)
        endpoint_scale === nothing && return false
        aligned = _physical_endpoint_state(
            coefficients,
            route_branch,
            selection,
            x,
            incidence,
            reflection,
            solution_scale,
            endpoint_scale,
        )
        _state_error(bridge_state, aligned) <=
            SFE_ENDPOINT_HANDOFF_TOLERANCE || return false
        plan[] = DirectEndpointPlan(selection, endpoint_scale, Float64(x))
        return true
    end

    bridge = direct_iterate_scaled_y(
        coefficients,
        kind,
        match_x,
        target.X,
        target.dXdx;
        target_y,
        stop_check,
    )
    return bridge, plan[]
end

function _sfe_up_basis(coefficients, ctrl, scratch, match_x, target)
    omega = abs(getproperty(coefficients.params, :omega))
    omega > 0 || return nothing
    target_y = (1.0 - PUBLIC_EVALUATION_ENDPOINT_X) / omega
    extended = target_y > _SFE_BRIDGE_TARGET_Y
    extended && eikonal_preflight(coefficients.params, :sfe).candidate &&
        return nothing
    selection = _sfe_endpoint_selection(coefficients, ctrl, match_x)
    selection === nothing && return nothing
    basis = try
        direct_iterate_from_infinity(
            coefficients,
            :out;
            controls=ctrl,
            match_x,
            selection,
            scratch,
            certified_steps=extended,
            step_budget=SFE_UP_TRIAL_STEPS,
        )
    catch err
        startswith(sprint(showerror, err), "direct GSN") || rethrow()
        return nothing
    end
    basis === nothing && return nothing
    scale = _fit_state_scale(target, _basis_tuple(basis))
    scale === nothing && return nothing
    aligned = _scaled_state(_basis_tuple(basis), scale)
    _state_error(target, aligned) <= SFE_ENDPOINT_HANDOFF_TOLERANCE ||
        return nothing
    return DirectScaledBasis(basis, scale)
end

function _scaled_det_ratio(a, b, c, d, det)
    direct = (a * b - c * d) / det
    _finite_complex(direct) && return direct

    left_scale = max(abs(a), abs(c))
    right_scale = max(abs(b), abs(d))
    (iszero(left_scale) || iszero(right_scale)) && return 0.0 + 0.0im

    mantissa = (a / left_scale) * (b / right_scale) -
        (c / left_scale) * (d / right_scale)
    iszero(mantissa) && return 0.0 + 0.0im

    det_scale = abs(det)
    (!isfinite(det_scale) || iszero(det_scale)) &&
        error("singular direct GSN scaled matching denominator")
    log_magnitude = log(abs(mantissa)) + log(left_scale) +
        log(right_scale) - log(det_scale)
    log_magnitude <= log(floatmax(Float64)) ||
        error("direct GSN matching coefficient exceeds Float64 range")
    phase = angle(mantissa) - angle(det)
    value = exp(log_magnitude) * cis(phase)
    _finite_complex(value) ||
        error("nonfinite direct GSN scaled matching coefficient")
    return ComplexF64(value)
end

function _scaled_det_ratio_polar(a, b, c, d, det)
    left_scale = max(abs(a), abs(c))
    right_scale = max(abs(b), abs(d))
    (iszero(left_scale) || iszero(right_scale)) && return (-Inf, 0.0)
    isfinite(left_scale) && isfinite(right_scale) ||
        error("nonfinite direct GSN matching numerator scale")

    mantissa = (a / left_scale) * (b / right_scale) -
        (c / left_scale) * (d / right_scale)
    iszero(mantissa) && return (-Inf, 0.0)
    det_scale = abs(det)
    isfinite(det_scale) && !iszero(det_scale) ||
        error("singular direct GSN scaled matching denominator")
    log_magnitude = log(abs(mantissa)) + log(left_scale) +
        log(right_scale) - log(det_scale)
    phase = angle(mantissa) - angle(det)
    return log_magnitude, phase
end

function _scaled_polar_value(log_magnitude, phase, log_scale)
    log_magnitude == -Inf && return 0.0 + 0.0im
    value = exp(log_magnitude + log_scale) * cis(phase)
    _finite_complex(value) ||
        error("nonfinite direct GSN jointly scaled matching coefficient")
    return ComplexF64(value)
end

function _connection_coefficients(target, basis1, basis2, det)
    (!isfinite(real(det)) || !isfinite(imag(det)) || iszero(det)) &&
        error("singular direct GSN Abel matching denominator")
    target_log = log(floatmax(Float64)) - MATCHING_SCALE_LOG_MARGIN
    try
        c1 = _scaled_det_ratio(
            target.X, basis2.dXdx, basis2.X, target.dXdx, det)
        c2 = _scaled_det_ratio(
            basis1.X, target.dXdx, target.X, basis1.dXdx, det)
        max(_logabs(c1), _logabs(c2)) <= target_log &&
            return c1, c2, 1.0
    catch err
        sprint(showerror, err) ==
            "direct GSN matching coefficient exceeds Float64 range" || rethrow()
    end

    log1, phase1 = _scaled_det_ratio_polar(
        target.X, basis2.dXdx, basis2.X, target.dXdx, det)
    log2, phase2 = _scaled_det_ratio_polar(
        basis1.X, target.dXdx, target.X, basis1.dXdx, det)
    max_log = max(log1, log2)
    isfinite(max_log) || error("nonfinite direct GSN matching scale")
    log_scale = min(0.0, target_log - max_log)
    solution_scale = exp(log_scale)
    isfinite(solution_scale) && solution_scale > 0 ||
        error("direct GSN matching normalization is outside Float64 range")
    c1 = _scaled_polar_value(log1, phase1, log_scale)
    c2 = _scaled_polar_value(log2, phase2, log_scale)
    return c1, c2, solution_scale
end

function _connection_coefficients_logscaled(
    target::DirectLogScaledState,
    basis1::DirectLogScaledState,
    basis2::DirectLogScaledState,
    det,
)
    _finite_complex(det) && !iszero(det) ||
        error("singular direct GSN Abel matching denominator")
    numerator1 = target.X * basis2.dXdx - basis2.X * target.dXdx
    numerator2 = basis1.X * target.dXdx - target.X * basis1.dXdx
    _finite_complex(numerator1) && _finite_complex(numerator2) ||
        error("nonfinite direct GSN log-scaled matching numerator")
    iszero(numerator1) && iszero(numerator2) &&
        error("singular direct GSN log-scaled matching numerators")

    log_det = log(abs(det))
    phase_det = angle(det)
    log1 = iszero(numerator1) ? -Inf :
        target.log_scale + basis2.log_scale +
        log(abs(numerator1)) - log_det
    log2 = iszero(numerator2) ? -Inf :
        basis1.log_scale + target.log_scale +
        log(abs(numerator2)) - log_det
    phase1 = iszero(numerator1) ? 0.0 : angle(numerator1) - phase_det
    phase2 = iszero(numerator2) ? 0.0 : angle(numerator2) - phase_det
    max_log = max(log1, log2)
    isfinite(max_log) || error("nonfinite direct GSN log-scaled matching scale")

    target_log = log(floatmax(Float64)) - MATCHING_SCALE_LOG_MARGIN
    joint_log_scale = min(0.0, target_log - max_log)
    solution_scale = exp(joint_log_scale)
    isfinite(solution_scale) && solution_scale > 0 ||
        error("direct GSN log-scaled matching normalization is outside Float64 range")
    coefficient1 = _scaled_polar_value(log1, phase1, joint_log_scale)
    coefficient2 = _scaled_polar_value(log2, phase2, joint_log_scale)
    return coefficient1, coefficient2, solution_scale
end

function _direct_denom_ok(fallback, scale)
    _finite = isfinite(real(fallback)) && isfinite(imag(fallback)) && !iszero(fallback)
    _finite || return false
    return abs(fallback) / max(scale * scale, eps(Float64)) >= DIRECT_DENOMINATOR_REL_MIN
end

function _basis_scale(basis1, basis2)
    return max(
        abs(basis1.X),
        abs(basis1.dXdx),
        abs(basis2.X),
        abs(basis2.dXdx),
        eps(Float64),
    )
end

@inline function _sfe_auto_fallback(ctrl::DirectGSNControls, err)
    return ctrl.frequency_selection.sfe_request == :auto &&
        err isa MSTCertificateError
end

function _match_denominator(
    coefficients,
    endpoint::Symbol,
    match_x::Float64,
    ctrl::DirectGSNControls,
    fallback;
    fallback_scale=NaN,
)
    denom = direct_abel_denominator(coefficients, endpoint, match_x)
    direct_ok = endpoint == :horizon && _direct_denom_ok(fallback, Float64(fallback_scale))
    if direct_ok && denom.method == :rational_quadrature_depth
        direct_rel = abs(fallback) /
            max(Float64(fallback_scale)^2, eps(Float64))
        direct_error = 1024eps(Float64) / max(direct_rel, eps(Float64))
        direct_error < denom.tail && return fallback
    end
    if denom.status != "OK"
        direct_ok && return fallback
        error("direct GSN $(endpoint) Abel denominator failed: $(denom.status)")
    end
    return denom.denominator
end

function _infinity_abel_certificate(
    coefficients,
    endpoint_y,
    out_coeffs,
    out_order,
    in_coeffs,
    in_order,
    variable_scale,
)
    closed = direct_abel_denominator(
        coefficients, :infinity, 1.0 - endpoint_y)
    closed.status == "OK" || return Inf
    exact = closed.denominator
    iszero(exact) && return Inf
    numerical = direct_infinity_seed_wronskian(
        coefficients,
        endpoint_y,
        out_coeffs,
        out_order,
        in_coeffs,
        in_order,
        variable_scale,
    )
    ratio = numerical / exact
    isfinite(real(ratio)) && isfinite(imag(ratio)) || return Inf
    return abs(ratio - 1)
end

function _infinity_basis(
    coefficients,
    branch::Symbol,
    selection,
    match_x::Float64,
    ctrl::DirectGSNControls,
    scratch,
    mst_plan=nothing,
)
    if ctrl.sfe
        return direct_mst_infinity_basis(
            coefficients,
            branch,
            match_x;
            controls=ctrl,
            scratch=scratch,
            plan=mst_plan,
        )
    end
    return direct_iterate_from_infinity(coefficients, branch; controls=ctrl,
        match_x=match_x, selection=selection, scratch=scratch)
end

function _logscaled_mst_basis(
    coefficients,
    branch::Symbol,
    match_x::Float64,
    ctrl,
    scratch,
    mst_plan,
)
    normalized = branch == :in ? :in : :out
    kind = normalized == :in ? :infinity_in : :infinity_out
    seed, _ = direct_mst_logscaled_seed(
        coefficients, mst_plan, normalized)
    return direct_iterate_logscaled_from_state(
        coefficients,
        kind,
        mst_plan.seed_x,
        seed,
        match_x;
        controls=ctrl,
        scratch=scratch,
    )
end

function _logscaled_mst_pair(
    coefficients,
    match_x::Float64,
    ctrl,
    scratch,
    mst_plan,
)
    incoming = _logscaled_mst_basis(
        coefficients, :in, match_x, ctrl, scratch, mst_plan)
    outgoing = _logscaled_mst_basis(
        coefficients, :out, match_x, ctrl, scratch, mst_plan)
    return incoming, outgoing
end

function _physical_handoff_state(
    coefficients,
    route_branch,
    bridge,
    endpoint_plan,
    x,
    incidence,
    reflection,
    solution_scale,
)
    if bridge !== nothing && _bridge_covers(bridge, x)
        value, derivative = _bridge_state(bridge, x)
        return (X=value, dXdx=derivative)
    end
    if endpoint_plan !== nothing && _endpoint_plan_covers(endpoint_plan, x)
        return _physical_endpoint_state(
            coefficients,
            route_branch,
            endpoint_plan.selection,
            x,
            incidence,
            reflection,
            solution_scale,
            endpoint_plan.scale,
        )
    end
    error("direct GSN log-scaled route lacks a physical MST handoff state")
end

function _logscaled_mst_eval_plan(
    coefficients,
    route_branch,
    mst_plan,
    bridge,
    endpoint_plan,
    incidence,
    reflection,
    solution_scale,
    logscaled_basis=nothing,
)
    use_logscaled_basis = route_branch == :UP &&
        logscaled_basis isa DirectLogScaledBasis
    desired_x = min(Float64(mst_plan.seed_x), PUBLIC_EVALUATION_ENDPOINT_X)
    handoff_x = if use_logscaled_basis
        Float64(mst_plan.seed_x)
    elseif bridge !== nothing && _bridge_covers(bridge, desired_x)
        desired_x
    elseif endpoint_plan !== nothing &&
            _endpoint_plan_covers(endpoint_plan, desired_x)
        desired_x
    elseif bridge !== nothing && !isempty(_route_basis(bridge).patches)
        Float64(_route_basis(bridge).patches[end].next_x)
    else
        desired_x
    end
    physical = if use_logscaled_basis
        direct_materialize_logscaled_state(
            direct_logscaled_basis_state(logscaled_basis, handoff_x);
            scale=solution_scale,
        )
    else
        _physical_handoff_state(
            coefficients,
            route_branch,
            bridge,
            endpoint_plan,
            handoff_x,
            incidence,
            reflection,
            solution_scale,
        )
    end
    if route_branch == :IN
        candidate = direct_mst_pin_state(
            coefficients, mst_plan, handoff_x; scale=1)
        pin_scale = _fit_state_scale(physical, candidate)
        pin_scale === nothing &&
            error("direct GSN physical-in MST handoff scale is unavailable")
        aligned = direct_mst_pin_state(
            coefficients, mst_plan, handoff_x; scale=pin_scale)
        _state_error(physical, aligned) <= SFE_ENDPOINT_HANDOFF_TOLERANCE ||
            error("direct GSN physical-in MST handoff failed validation")
        return direct_mst_eval_plan(
            mst_plan;
            pin_scale=pin_scale,
            logscaled=true,
            handoff_x=handoff_x,
        )
    end

    aligned = if use_logscaled_basis
        seed, estimated_relerr = direct_mst_logscaled_seed(
            coefficients, mst_plan, :out)
        materialized = direct_materialize_logscaled_state(
            seed; scale=solution_scale)
        (X=materialized.X, dXdx=materialized.dXdx,
            estimated_relerr=estimated_relerr)
    else
        direct_mst_scaled_state(
            coefficients,
            mst_plan,
            :out,
            handoff_x;
            scale=solution_scale,
        )
    end
    _state_error(physical, aligned) <= SFE_ENDPOINT_HANDOFF_TOLERANCE ||
        error("direct GSN outgoing MST handoff failed validation")
    return direct_mst_eval_plan(
        mst_plan;
        out_scale=solution_scale,
        logscaled=true,
        handoff_x=handoff_x,
    )
end

function _public_exterior_covered(bridge, endpoint_plan)
    bridge !== nothing &&
        _bridge_covers(bridge, PUBLIC_EVALUATION_ENDPOINT_X) && return true
    return endpoint_plan !== nothing &&
        _endpoint_plan_covers(endpoint_plan, PUBLIC_EVALUATION_ENDPOINT_X)
end

function direct_match(
    coefficients::DirectCoefficientSet,
    branch;
    controls=nothing,
)
    return _direct_match_impl(coefficients, branch, controls)
end

function _direct_match_impl(
    coefficients::DirectCoefficientSet,
    branch,
    controls,
)
    base_ctrl = controls === nothing ? direct_gsn_controls(coefficients.params) : controls
    route_branch = _normalized_route_branch(branch)
    ctrl = _with_ordinary_defaults(
        _with_low_frequency_defaults(
            base_ctrl, coefficients.params, route_branch),
        coefficients.params,
        route_branch,
    )
    coefficients = near_extreme_prepare(coefficients, ctrl)
    match_x = ctrl.match_x
    selection = ctrl.sfe ? nothing : direct_select_infinity_endpoint(
        coefficients,
        match_x;
        controls=ctrl,
        certificate=_infinity_abel_certificate,
    )
    scratch = DirectIterationScratch(coefficients,
        getproperty(ctrl, :ordinary_order);
        force_eikonal=ctrl.lfe, lfe=ctrl.lfe, sfe=ctrl.sfe,
        route_branch=route_branch)
    mst_branches = route_branch == :IN ? (:in, :out) : (:out,)
    mst_plan = if ctrl.sfe
        try
            direct_mst_plan(coefficients, match_x; branches=mst_branches)
        catch err
            _sfe_auto_fallback(base_ctrl, err) || rethrow()
            fallback = _ordinary_fallback(
                base_ctrl, :sfe_infinity_in_tail_fallback)
            return _direct_match_impl(coefficients, route_branch, fallback)
        end
    else
        nothing
    end
    if mst_plan !== nothing &&
            mst_plan.selection.status == "NO_SAFE_SEED" &&
            base_ctrl.frequency_selection.sfe_request == :auto
        fallback = _ordinary_fallback(
            base_ctrl, :sfe_no_safe_seed_fallback)
        return _direct_match_impl(coefficients, route_branch, fallback)
    end
    endpoint_y = mst_plan === nothing ? selection.endpoint_y :
        1.0 - mst_plan.seed_x

    if route_branch == :IN
        h_in = direct_iterate_from_zero(coefficients, :in; controls=ctrl, match_x=match_x, scratch=scratch)
        i_in, i_out, logscaled_mst = if ctrl.sfe
            try
                incoming, outgoing = direct_mst_infinity_pair(
                    coefficients,
                    :in,
                    :out,
                    match_x;
                    controls=ctrl,
                    scratch=scratch,
                    plan=mst_plan,
                )
                (incoming, outgoing, false)
            catch err
                if err isa MSTCertificateError
                    _sfe_auto_fallback(base_ctrl, err) || rethrow()
                    fallback = _ordinary_fallback(
                        base_ctrl, :sfe_infinity_in_tail_fallback)
                    return _direct_match_impl(
                        coefficients, route_branch, fallback)
                end
                try
                    incoming, outgoing = _logscaled_mst_pair(
                        coefficients, match_x, ctrl, scratch, mst_plan)
                    (incoming, outgoing, true)
                catch robust_err
                    _sfe_auto_fallback(base_ctrl, robust_err) ||
                        throw(robust_err)
                    fallback = _ordinary_fallback(
                        base_ctrl, :sfe_infinity_in_tail_fallback)
                    return _direct_match_impl(
                        coefficients, route_branch, fallback)
                end
            end
        else
            incoming, outgoing = direct_iterate_pair_from_infinity(
                coefficients, :in, :out;
                controls=ctrl, selection, match_x, scratch)
            (incoming, outgoing, false)
        end
        fallback = logscaled_mst ? 0.0 + 0.0im :
            _basis_tuple(i_in).X * _basis_tuple(i_out).dXdx -
            _basis_tuple(i_out).X * _basis_tuple(i_in).dXdx
        denom = _match_denominator(coefficients, :infinity, match_x, ctrl, fallback)
        incidence, reflection, solution_scale = if logscaled_mst
            _connection_coefficients_logscaled(
                _logscaled_match_state(h_in),
                _logscaled_match_state(i_in),
                _logscaled_match_state(i_out),
                denom,
            )
        else
            _connection_coefficients(
                _basis_tuple(h_in),
                _basis_tuple(i_in),
                _basis_tuple(i_out),
                denom,
            )
        end
        target = _scaled_state(_basis_tuple(h_in), solution_scale)
        bridge, endpoint_plan = if ctrl.sfe
            _build_sfe_endpoint_bridge(
                coefficients,
                ctrl,
                :horizon_in,
                :IN,
                match_x,
                target,
                incidence,
                reflection,
                solution_scale,
            )
        else
            (_build_bridge(
                coefficients,
                ctrl,
                scratch,
                :IN,
                match_x,
                1.0 - endpoint_y,
                target,
                incidence,
                _basis_tuple(i_in),
                reflection,
                _basis_tuple(i_out),
            ), nothing)
        end
        mst_eval_plan = if mst_plan === nothing
            nothing
        else
            try
                logscaled_mst ? _logscaled_mst_eval_plan(
                    coefficients,
                    :IN,
                    mst_plan,
                    bridge,
                    endpoint_plan,
                    incidence,
                    reflection,
                    solution_scale,
                ) : direct_mst_eval_plan(mst_plan)
            catch err
                if logscaled_mst &&
                        _public_exterior_covered(bridge, endpoint_plan)
                    nothing
                else
                    _sfe_auto_fallback(base_ctrl, err) || rethrow()
                    fallback = _ordinary_fallback(
                        base_ctrl, :sfe_evaluation_plan_fallback)
                    return _direct_match_impl(
                        coefficients, route_branch, fallback)
                end
            end
        end
        return DirectRoute(
            :IN,
            coefficients.params,
            ctrl,
            coefficients,
            match_x,
            endpoint_y,
            selection === nothing ?
                (endpoint_plan === nothing ? 0 : endpoint_plan.selection.out_order) :
                selection.out_order,
            selection === nothing ?
                (endpoint_plan === nothing ? 0 : endpoint_plan.selection.in_order) :
                selection.in_order,
            ComplexF64(incidence),
            ComplexF64(reflection),
            solution_scale,
            h_in,
            nothing,
            i_out,
            i_in,
            bridge,
            nothing,
            endpoint_plan,
            mst_eval_plan,
        )
    end

    i_out, logscaled_mst = if ctrl.sfe
        try
            (_infinity_basis(
                coefficients, :out, selection, match_x, ctrl, scratch, mst_plan),
                false,
            )
        catch err
            if err isa MSTCertificateError
                _sfe_auto_fallback(base_ctrl, err) || rethrow()
                fallback = _ordinary_fallback(
                    base_ctrl, :sfe_infinity_in_tail_fallback)
                return _direct_match_impl(
                    coefficients, route_branch, fallback)
            end
            try
                (_logscaled_mst_basis(
                    coefficients, :out, match_x, ctrl, scratch, mst_plan),
                    true,
                )
            catch robust_err
                _sfe_auto_fallback(base_ctrl, robust_err) ||
                    throw(robust_err)
                fallback = _ordinary_fallback(
                    base_ctrl, :sfe_infinity_in_tail_fallback)
                return _direct_match_impl(
                    coefficients, route_branch, fallback)
            end
        end
    else
        (_infinity_basis(
            coefficients, :out, selection, match_x, ctrl, scratch, mst_plan),
            false,
        )
    end
    h_in, h_out = direct_iterate_pair_from_zero(
        coefficients,
        :in,
        :out;
        controls=ctrl,
        match_x=match_x,
        scratch=scratch,
    )
    fallback = _basis_tuple(h_in).X * _basis_tuple(h_out).dXdx -
        _basis_tuple(h_out).X * _basis_tuple(h_in).dXdx
    denom = _match_denominator(
        coefficients,
        :horizon,
        match_x,
        ctrl,
        fallback;
        fallback_scale=_basis_scale(_basis_tuple(h_in), _basis_tuple(h_out)),
    )
    reflection, incidence, solution_scale = if logscaled_mst
        _connection_coefficients_logscaled(
            _logscaled_match_state(i_out),
            _logscaled_match_state(h_in),
            _logscaled_match_state(h_out),
            denom,
        )
    else
        _connection_coefficients(
            _basis_tuple(i_out),
            _basis_tuple(h_in),
            _basis_tuple(h_out),
            denom,
        )
    end
    target = logscaled_mst ? direct_materialize_logscaled_state(
        _logscaled_match_state(i_out); scale=solution_scale) :
        _scaled_state(_basis_tuple(i_out), solution_scale)
    bridge = _build_bridge(
        coefficients,
        ctrl,
        scratch,
        :UP,
        match_x,
        min(h_in.seed_x, h_out.seed_x),
        target,
        reflection,
        _basis_tuple(h_in),
        incidence,
        _basis_tuple(h_out),
    )
    endpoint_basis = ctrl.sfe && !logscaled_mst ?
        _sfe_up_basis(coefficients, ctrl, scratch, match_x, target) : nothing
    endpoint_bridge, endpoint_plan = if endpoint_basis === nothing
        _build_sfe_endpoint_bridge(
            coefficients,
            ctrl,
            :infinity_out,
            :UP,
            match_x,
            target,
            incidence,
            reflection,
            solution_scale,
        )
    else
        (endpoint_basis, nothing)
    end
    mst_eval_plan = if mst_plan === nothing
        nothing
    else
        try
            logscaled_mst ? _logscaled_mst_eval_plan(
                coefficients,
                :UP,
                mst_plan,
                endpoint_bridge,
                endpoint_plan,
                incidence,
                reflection,
                solution_scale,
                i_out,
            ) : direct_mst_eval_plan(mst_plan)
        catch err
            if logscaled_mst
                _public_exterior_covered(endpoint_bridge, endpoint_plan) ?
                    nothing : rethrow()
            else
                _sfe_auto_fallback(base_ctrl, err) || rethrow()
                fallback = _ordinary_fallback(
                    base_ctrl, :sfe_evaluation_plan_fallback)
                return _direct_match_impl(coefficients, route_branch, fallback)
            end
        end
    end
    return DirectRoute(
        :UP,
        coefficients.params,
        ctrl,
        coefficients,
        match_x,
        endpoint_y,
        selection === nothing ?
            (endpoint_plan === nothing ? 0 : endpoint_plan.selection.out_order) :
            selection.out_order,
        selection === nothing ?
            (endpoint_plan === nothing ? 0 : endpoint_plan.selection.in_order) :
            selection.in_order,
        ComplexF64(incidence),
        ComplexF64(reflection),
        solution_scale,
        h_in,
        h_out,
        i_out,
        nothing,
        bridge,
        endpoint_bridge,
        endpoint_plan,
        mst_eval_plan,
    )
end

function _certified_mst_pair(
    route::DirectRoute,
    _target_params::DirectGSNParameters,
    _controls::DirectGSNControls,
)
    result = try
        mst_principal_amplitudes(route.params, route.branch)
    catch
        return route
    end
    result.max_condition <= REAL_MST_CONDITION_MAX || return route
    target_pair = result.gsn
    all(value -> isfinite(real(value)) && isfinite(imag(value)), target_pair) ||
        return route
    scale = route.solution_scale
    incidence = ComplexF64(scale * target_pair[1])
    reflection = ComplexF64(scale * target_pair[2])
    all(value -> isfinite(real(value)) && isfinite(imag(value)),
        (incidence, reflection)) || return route
    return DirectRoute(
        route.branch,
        route.params,
        route.controls,
        route.coefficients,
        route.match_x,
        route.infinity_endpoint_y,
        route.infinity_out_order,
        route.infinity_in_order,
        incidence,
        reflection,
        route.solution_scale,
        route.horizon_in,
        route.horizon_out,
        route.infinity_out,
        route.infinity_in,
        route.bridge,
        route.endpoint_bridge,
        route.endpoint_plan,
        route.mst_plan,
    )
end

function direct_gsn_radial(
    s::Integer,
    l::Integer,
    m::Integer,
    a,
    omega,
    branch=:IN;
    lambda=nothing,
    nu=nothing,
    controls=nothing,
    kwargs...,
)
    default_request = controls === nothing && lambda === nothing &&
        nu === nothing && all(
            value -> value === nothing || value === :auto,
            values(kwargs),
        )
    params = direct_gsn_parameters(s, l, m, a, omega; lambda=lambda, nu=nu)
    ctrl = controls === nothing ? direct_gsn_controls(params; kwargs...) : controls
    if isreal(params.omega) && params.omega < 0
        positive_params = DirectGSNParameters(
            params.s,
            params.l,
            -params.m,
            params.a,
            -params.omega,
            params.lambda,
            params.nu,
            params.kappa,
        )
        positive_route = direct_gsn_radial(
            positive_params,
            branch;
            controls=ctrl,
        )
        use_mst_pair = default_request &&
            positive_route.branch == :IN &&
            (
                abs(params.omega) <= REAL_MST_NEAR_STATIC_MAX ||
                direct_horizon_tail(params)
            )
        use_mst_pair && (positive_route =
            _certified_mst_pair(positive_route, params, ctrl))
        return DirectConjugatedRoute(positive_route, params)
    end
    coefficients = direct_gsn_coefficients(params; controls=ctrl)
    return direct_match(coefficients, branch; controls=ctrl)
end

function direct_gsn_radial(
    params::DirectGSNParameters,
    branch=:IN;
    controls=nothing,
    coefficients=nothing,
    kwargs...,
)
    ctrl = controls === nothing ? direct_gsn_controls(params; kwargs...) : controls
    coeffs = coefficients === nothing ? direct_gsn_coefficients(params; controls=ctrl) : coefficients
    return direct_match(coeffs, branch; controls=ctrl)
end

function _bridge_covers(basis::DirectBasis, x)
    if basis.endpoint_valid
        horizon = basis.kind in (:horizon_in, :horizon_out)
        ((horizon && x <= basis.seed_x) || (!horizon && x >= basis.seed_x)) &&
            return true
    end
    isempty(basis.patches) && return x == basis.seed_x
    target_x = basis.patches[end].next_x
    lo = min(basis.seed_x, target_x) - 20eps(Float64)
    hi = max(basis.seed_x, target_x) + 20eps(Float64)
    return lo <= x <= hi
end

_bridge_covers(basis::DirectScaledBasis, x) =
    _bridge_covers(basis.basis, x)

function _bridge_covers(basis::DirectLogScaledBasis, x)
    isempty(basis.patches) && return x == basis.seed_x
    target_x = basis.patches[end].next_x
    lo = min(basis.seed_x, target_x) - 20eps(Float64)
    hi = max(basis.seed_x, target_x) + 20eps(Float64)
    return lo <= x <= hi
end

_bridge_value(basis::DirectBasis, x) = direct_basis_value(basis, x)
_bridge_value(basis::DirectScaledBasis, x) =
    basis.scale * direct_basis_value(basis.basis, x)

_bridge_state(basis::DirectBasis, x) = direct_basis_state(basis, x)
function _bridge_state(basis::DirectScaledBasis, x)
    value, derivative = direct_basis_state(basis.basis, x)
    return basis.scale * value, basis.scale * derivative
end

function _physical_logscaled_basis_state(basis, x, scale)
    return direct_materialize_logscaled_state(
        direct_logscaled_basis_state(basis, x); scale=scale)
end

_route_basis(basis::DirectBasis) = basis
_route_basis(basis::DirectScaledBasis) = basis.basis
_route_basis(basis::DirectLogScaledBasis) = basis

_endpoint_plan_covers(plan::DirectEndpointPlan, x) =
    x >= plan.seed_x - 20eps(Float64)

function _endpoint_plan_state(route::DirectRoute, x)
    plan = route.endpoint_plan
    plan === nothing && error("missing direct GSN endpoint evaluation plan")
    return _physical_endpoint_state(
        route.coefficients,
        route.branch,
        plan.selection,
        x,
        route.incidence,
        route.reflection,
        route.solution_scale,
        plan.scale,
    )
end

@inline _logscaled_mst_eval(plan) = plan !== nothing &&
    hasproperty(plan, :logscaled) && plan.logscaled

function _physical_mst_out_state(route::DirectRoute, x)
    if _logscaled_mst_eval(route.mst_plan)
        route.mst_plan.out_scale === nothing &&
            error("missing direct GSN log-scaled outgoing normalization")
        return direct_mst_scaled_state(
            route.coefficients,
            route.mst_plan,
            :out,
            x;
            scale=route.mst_plan.out_scale,
        )
    end
    state = direct_mst_state(
        route.coefficients, route.mst_plan, :out, x)
    return (
        X=route.solution_scale * state.X,
        dXdx=route.solution_scale * state.dXdx,
        estimated_relerr=state.estimated_relerr,
    )
end

function direct_evaluate(route::DirectRoute, x)
    x = Float64(x)
    if route.branch == :IN
        if x <= route.match_x
            return route.solution_scale *
                direct_basis_value(_required_basis(route.horizon_in, "horizon_in"), x)
        end
        route.bridge !== nothing && _bridge_covers(route.bridge, x) &&
            return _bridge_value(route.bridge, x)
        if route.endpoint_plan !== nothing &&
                _endpoint_plan_covers(route.endpoint_plan, x)
            return _endpoint_plan_state(route, x).X
        end
        if route.mst_plan !== nothing && x > route.mst_plan.seed_x
            if _logscaled_mst_eval(route.mst_plan)
                return direct_mst_pin_state(
                    route.coefficients, route.mst_plan, x).X
            end
            incoming = direct_mst_state(
                route.coefficients, route.mst_plan, :in, x)
            outgoing = direct_mst_state(
                route.coefficients, route.mst_plan, :out, x)
            return route.incidence * incoming.X + route.reflection * outgoing.X
        end
        return route.incidence * direct_basis_value(_required_basis(route.infinity_in, "infinity_in"), x) +
            route.reflection * direct_basis_value(_required_basis(route.infinity_out, "infinity_out"), x)
    end
    if x <= route.match_x
        route.bridge !== nothing && _bridge_covers(route.bridge, x) &&
            return _bridge_value(route.bridge, x)
        return route.reflection * direct_basis_value(_required_basis(route.horizon_in, "horizon_in"), x) +
            route.incidence * direct_basis_value(_required_basis(route.horizon_out, "horizon_out"), x)
    end
    route.endpoint_bridge !== nothing &&
        _bridge_covers(route.endpoint_bridge, x) &&
        return _bridge_value(route.endpoint_bridge, x)
    if route.endpoint_plan !== nothing &&
            _endpoint_plan_covers(route.endpoint_plan, x)
        return _endpoint_plan_state(route, x).X
    end
    if route.infinity_out isa DirectLogScaledBasis &&
            _bridge_covers(route.infinity_out, x)
        return _physical_logscaled_basis_state(
            route.infinity_out, x, route.solution_scale).X
    end
    if route.mst_plan !== nothing && x > route.mst_plan.seed_x
        return _physical_mst_out_state(route, x).X
    end
    return route.solution_scale *
        direct_basis_value(_required_basis(route.infinity_out, "infinity_out"), x)
end

direct_evaluate(route::DirectConjugatedRoute, x) =
    conj(direct_evaluate(route.route, x))

function direct_state(route::DirectRoute, x)
    x = Float64(x)
    if route.branch == :IN
        if x <= route.match_x
            value, derivative = direct_basis_state(_required_basis(route.horizon_in, "horizon_in"), x)
            return (
                X=route.solution_scale * value,
                dXdx=route.solution_scale * derivative,
            )
        end
        if route.bridge !== nothing && _bridge_covers(route.bridge, x)
            value, derivative = _bridge_state(route.bridge, x)
            return (X=value, dXdx=derivative)
        end
        if route.endpoint_plan !== nothing &&
                _endpoint_plan_covers(route.endpoint_plan, x)
            return _endpoint_plan_state(route, x)
        end
        if route.mst_plan !== nothing && x > route.mst_plan.seed_x
            if _logscaled_mst_eval(route.mst_plan)
                state = direct_mst_pin_state(
                    route.coefficients, route.mst_plan, x)
                return (X=state.X, dXdx=state.dXdx)
            end
            incoming = direct_mst_state(
                route.coefficients, route.mst_plan, :in, x)
            outgoing = direct_mst_state(
                route.coefficients, route.mst_plan, :out, x)
            return (
                X=route.incidence * incoming.X + route.reflection * outgoing.X,
                dXdx=route.incidence * incoming.dXdx +
                    route.reflection * outgoing.dXdx,
            )
        end
        vin, din = direct_basis_state(_required_basis(route.infinity_in, "infinity_in"), x)
        vout, dout = direct_basis_state(_required_basis(route.infinity_out, "infinity_out"), x)
        return (
            X=route.incidence * vin + route.reflection * vout,
            dXdx=route.incidence * din + route.reflection * dout,
        )
    end
    if x <= route.match_x
        if route.bridge !== nothing && _bridge_covers(route.bridge, x)
            value, derivative = _bridge_state(route.bridge, x)
            return (X=value, dXdx=derivative)
        end
        vin, din = direct_basis_state(_required_basis(route.horizon_in, "horizon_in"), x)
        vout, dout = direct_basis_state(_required_basis(route.horizon_out, "horizon_out"), x)
        return (
            X=route.reflection * vin + route.incidence * vout,
            dXdx=route.reflection * din + route.incidence * dout,
        )
    end
    if route.endpoint_bridge !== nothing &&
            _bridge_covers(route.endpoint_bridge, x)
        value, derivative = _bridge_state(route.endpoint_bridge, x)
        return (X=value, dXdx=derivative)
    end
    if route.endpoint_plan !== nothing &&
            _endpoint_plan_covers(route.endpoint_plan, x)
        return _endpoint_plan_state(route, x)
    end
    if route.infinity_out isa DirectLogScaledBasis &&
            _bridge_covers(route.infinity_out, x)
        return _physical_logscaled_basis_state(
            route.infinity_out, x, route.solution_scale)
    end
    if route.mst_plan !== nothing && x > route.mst_plan.seed_x
        state = _physical_mst_out_state(route, x)
        return (X=state.X, dXdx=state.dXdx)
    end
    value, derivative = direct_basis_state(_required_basis(route.infinity_out, "infinity_out"), x)
    return (
        X=route.solution_scale * value,
        dXdx=route.solution_scale * derivative,
    )
end

function direct_state(route::DirectConjugatedRoute, x)
    state = direct_state(route.route, x)
    return (X=conj(state.X), dXdx=conj(state.dXdx))
end

(route::DirectRoute)(x) = direct_evaluate(route, x)
(route::DirectConjugatedRoute)(x) = direct_evaluate(route, x)

function direct_route_patch_count(route::DirectRoute)
    count = 0
    for basis in (route.horizon_in, route.horizon_out, route.infinity_out,
            route.infinity_in, route.bridge, route.endpoint_bridge)
        count += basis === nothing ? 0 :
            direct_basis_patch_count(_route_basis(basis))
    end
    return count
end

direct_route_patch_count(route::DirectConjugatedRoute) =
    direct_route_patch_count(route.route)

function _patch_counts(route::DirectRoute)
    ordinary = 0
    eikonal = 0
    endpoint = 0
    horizon = 0
    infinity = 0
    for (location, basis) in ((:horizon, route.horizon_in),
            (:horizon, route.horizon_out),
            (:infinity, route.infinity_out),
            (:infinity, route.infinity_in),
            (route.branch == :IN ? :infinity : :horizon, route.bridge),
            (:infinity, route.endpoint_bridge))
        basis === nothing && continue
        basis = _route_basis(basis)
        count = length(basis.patches)
        location == :horizon ? (horizon += count) : (infinity += count)
        for patch in basis.patches
            if patch isa DirectPatch || patch isa DirectScaledPatch ||
                    patch isa DirectLogScaledPatch
                ordinary += 1
            elseif patch isa DirectEikonalPatch
                eikonal += 1
            else
                endpoint += 1
            end
        end
    end
    return ordinary, eikonal, endpoint, horizon, infinity
end

function direct_route_plan(route::DirectRoute)
    controls = route.controls
    frequency = controls.frequency_selection
    spin = direct_spin_regime(route.params)
    eikonal = eikonal_preflight(
        route.params, frequency.regime, route.branch)
    ordinary_patches, eikonal_patches, endpoint_patches,
        horizon_patches, infinity_patches = _patch_counts(route)
    logscaled_mst = route.infinity_out isa DirectLogScaledBasis ||
        route.infinity_in isa DirectLogScaledBasis
    endpoint = route.endpoint_bridge isa DirectScaledBasis ?
        :direct_endpoint_basis :
        logscaled_mst ? :mst_logscaled :
        route.endpoint_plan !== nothing ? :mst_scaled_y_handoff :
        controls.sfe ? :mst : controls.lfe ? :lfe_carrier :
        near_extreme_selected(route.coefficients) ? :near_extremal : :taylor
    propagation = eikonal_patches > 0 ? :eikonal : :taylor
    return DirectRoutePlan(
        frequency.regime,
        frequency.reason,
        spin.regime,
        spin.reason,
        endpoint,
        propagation,
        :abel,
        route.match_x,
        controls.horizon_order,
        controls.ordinary_order,
        controls.infinity_order,
        eikonal.candidate,
        ordinary_patches,
        eikonal_patches,
        endpoint_patches,
        horizon_patches,
        infinity_patches,
        spin.angular_compression,
        spin.phase_compression,
    )
end

direct_route_plan(route::DirectConjugatedRoute) =
    direct_route_plan(route.route)

function direct_route_truncations(route::DirectRoute)
    records = DirectTruncation[]
    for basis in (route.horizon_in, route.horizon_out, route.infinity_out,
            route.infinity_in, route.bridge, route.endpoint_bridge)
        basis === nothing || append!(records, _route_basis(basis).truncations)
    end
    return records
end

direct_route_truncations(route::DirectConjugatedRoute) =
    direct_route_truncations(route.route)

end
