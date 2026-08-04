module DirectNIAAmplitudes

import ..DirectMSTInfinity
using ..DirectMSTInfinity:
    direct_mst_anchor,
    mst_nia_amplitudes,
    mst_nu_complex,
    mst_principal_amplitudes
using ..DirectCoefficientTables: direct_gsn_coefficients
using ..DirectParameters:
    direct_gsn_controls,
    direct_gsn_parameters
using ..DirectIteration:
    direct_iterate_from_zero
import ..DirectComplexFrequency
import ..DirectComplexRational
using ..DirectComplexFrequency:
    DirectComplexRoute,
    DirectComplexRoutePlan,
    DirectComplexSFEvaluator,
    DirectComplexStateEvaluator
using ..DirectComplexRational:
    DirectComplexRationalEvaluator,
    DirectComplexRationalState

const _REAL_RATIO_MAX = 1.0e-8
const _NONROTATING_MAX_ABS_A = 64eps(Float64)
const _MIN_ABS_IMAG_OMEGA = 1.8
const _MAX_ABS_IMAG_OMEGA = 2.2
const _NIA_LATERAL_REAL_RATIO_MAX = 1.0e-6
const _STATE_ANCHOR_X = 0.5
const _STATE_ANCHOR_MAX_ERROR = 1.0e-10
const _REAL_AXIS_ANCHOR_MIN_DEPARTURE = 1.0e-8
const _STATE_ANCHOR_SUM_TOLERANCE = 1.0e-11
const _STATE_ANCHOR_RELAXED_TOLERANCE = 1.0e-9
const _STATE_ANCHOR_RELAXED_REAL_RATIO_MAX = 1.0e-3
const _AMPLITUDE_AGREEMENT_MAX = 1.0e-10
const _MST_STATE_NU_DELTA_MIN = 1.0e-10
const _FAST_ANCHOR_MIN_OMEGA = 0.025
const _FAST_ANCHOR_MAX_OMEGA = 2.0
const _FAST_ANCHOR_RATIO_LOW = 0.2
const _FAST_ANCHOR_RATIO_HIGH = 5.0
const _FAST_ANCHOR_ORDER = 40
const _FAST_ANCHOR_TOLERANCE = 1.0e-14

struct ScaledEvaluator{E}
    evaluator::E
    scale::ComplexF64
end

@inline _finite(value) =
    isfinite(real(value)) && isfinite(imag(value))

function DirectComplexFrequency._state_at_r(
    evaluator::ScaledEvaluator,
    r::Real,
)
    state = DirectComplexFrequency._state_at_r(evaluator.evaluator, r)
    return (
        X=ComplexF64(evaluator.scale * state.X),
        dXdrstar=ComplexF64(evaluator.scale * state.dXdrstar),
        error=state.error,
    )
end

function (evaluator::ScaledEvaluator)(rstar::Real)
    X, dXdrstar, error = evaluator.evaluator(rstar)
    return (
        ComplexF64(evaluator.scale * X),
        ComplexF64(evaluator.scale * dXdrstar),
        error,
    )
end

function (evaluator::ScaledEvaluator)(rstars::AbstractVector{<:Real})
    states = map(evaluator, rstars)
    return (
        getindex.(states, 1),
        getindex.(states, 2),
        getindex.(states, 3),
    )
end

@inline _rational_core(evaluator::DirectComplexRationalEvaluator) =
    (evaluator, ComplexF64(1))

function _rational_core(evaluator::ScaledEvaluator)
    core = _rational_core(evaluator.evaluator)
    core === nothing && return nothing
    return (core[1], ComplexF64(evaluator.scale * core[2]))
end

@inline _rational_core(evaluator) = nothing

@inline function _pair_error(first, second)
    scale = max(
        abs(first[1]), abs(first[2]),
        abs(second[1]), abs(second[2]),
        floatmin(Float64),
    )
    return max(
        abs(first[1] - second[1]),
        abs(first[2] - second[2]),
    ) / scale
end

function _fit_pair_scale(target, candidate)
    target_scale = max(abs(target[1]), abs(target[2]))
    candidate_scale = max(abs(candidate[1]), abs(candidate[2]))
    target_scale > 0 && candidate_scale > 0 || return nothing
    isfinite(target_scale) && isfinite(candidate_scale) || return nothing
    target_value = target[1] / target_scale
    target_derivative = target[2] / target_scale
    candidate_value = candidate[1] / candidate_scale
    candidate_derivative = candidate[2] / candidate_scale
    denominator =
        abs2(candidate_value) + abs2(candidate_derivative)
    denominator > 0 && isfinite(denominator) || return nothing
    unit_scale = (
        conj(candidate_value) * target_value +
        conj(candidate_derivative) * target_derivative
    ) / denominator
    scale = (target_scale / candidate_scale) * unit_scale
    return _finite(scale) ? ComplexF64(scale) : nothing
end

function _in_real_axis_anchor(route, evaluator)
    rejected(status; projective_error=Inf, scale=ComplexF64(NaN, NaN),
            departure=Inf, steps=-1) = (
        evaluator,
        metadata=(;
            real_axis_state_anchor_status=status,
            real_axis_state_anchor_projective_error=
                Float64(projective_error),
            real_axis_state_anchor_scale=ComplexF64(scale),
            real_axis_state_anchor_departure=Float64(departure),
            real_axis_state_anchor_steps=Int(steps),
        ),
    )
    route.branch == :IN || return rejected(:not_applicable)
    core = _rational_core(evaluator)
    core === nothing && return rejected(:not_rational)
    rational, existing_scale = core
    omega = ComplexF64(rational.params.omega)
    abs(real(omega)) >=
        _FAST_ANCHOR_RATIO_LOW * abs(imag(omega)) ||
        return rejected(:outside_real_axis_sector)
    controls = direct_gsn_controls(
        rational.params;
        N=rational.settings.order,
        xm=rational.match_x,
        tol=route.plan.controls.tolerance,
        sfe=false,
        lfe=false,
    )
    basis = try
        direct_iterate_from_zero(
            rational.coefficients,
            :in;
            controls,
            match_x=rational.match_x,
        )
    catch
        return rejected(:construction_error)
    end
    basis.endpoint_valid ||
        return rejected(:invalid_endpoint; steps=basis.step_count)
    target = (basis.match_X, basis.match_dXdx)
    current = (
        existing_scale * rational.match_state.X,
        existing_scale * rational.match_state.dXdx,
    )
    all(_finite, (target..., current...)) ||
        return rejected(:nonfinite_state; steps=basis.step_count)
    correction = _fit_pair_scale(target, current)
    correction === nothing &&
        return rejected(:rejected_scale; steps=basis.step_count)
    corrected = (
        correction * current[1],
        correction * current[2],
    )
    projective_error = Float64(_pair_error(corrected, target))
    departure = Float64(abs(correction - 1))
    projective_error <= _STATE_ANCHOR_MAX_ERROR || return rejected(
        :rejected_projective;
        projective_error,
        scale=correction,
        departure,
        steps=basis.step_count,
    )
    departure > _REAL_AXIS_ANCHOR_MIN_DEPARTURE || return rejected(
        :retained;
        projective_error,
        scale=correction,
        departure,
        steps=basis.step_count,
    )
    return (
        evaluator=ScaledEvaluator(evaluator, correction),
        metadata=(;
            real_axis_state_anchor_status=:rescaled,
            real_axis_state_anchor_projective_error=projective_error,
            real_axis_state_anchor_scale=correction,
            real_axis_state_anchor_departure=departure,
            real_axis_state_anchor_steps=basis.step_count,
        ),
    )
end

@inline function _pretrigger(
    s,
    a,
    omega;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
)
    backend == :direct_rational || return false
    controls === nothing || return false
    all(value === nothing for value in (
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
    )) || return false
    pole_normalization && return false
    omegac = ComplexF64(omega)
    abs(s) == 2 || return false
    abs(a) > _NONROTATING_MAX_ABS_A || return false
    imag(omegac) < 0 || return false
    !iszero(real(omegac)) || return false
    abs(real(omegac)) <= _REAL_RATIO_MAX * abs(imag(omegac)) ||
        return false
    return _MIN_ABS_IMAG_OMEGA <= abs(imag(omegac)) <=
        _MAX_ABS_IMAG_OMEGA
end

function _overlay(
    route::DirectComplexRoute,
    result;
    amplitude_backend::Symbol=:nia_mst_analytic,
    state_evaluator=route.state_evaluator,
    state_metadata=(;),
)
    values = (
        route.transmission,
        route.teukolsky_transmission,
        result.gsn...,
        result.teuk...,
    )
    all(_finite, values) || return route
    iszero(route.transmission) && return route
    iszero(route.teukolsky_transmission) && return route

    incidence = ComplexF64(route.transmission * result.gsn[1])
    reflection = ComplexF64(route.transmission * result.gsn[2])
    teukolsky_incidence =
        ComplexF64(route.teukolsky_transmission * result.teuk[1])
    teukolsky_reflection =
        ComplexF64(route.teukolsky_transmission * result.teuk[2])
    all(_finite, (
        incidence,
        reflection,
        teukolsky_incidence,
        teukolsky_reflection,
    )) || return route
    rebuilt = _rebuild_up_state(
        route,
        state_evaluator,
        incidence,
        reflection,
    )
    state_evaluator = rebuilt.evaluator
    state_metadata = merge(state_metadata, rebuilt.metadata)

    metadata = merge(route.metadata, (;
        amplitude_backend,
        mst_amplitude_certificate=result.certificate_kind,
        mst_amplitude_representation_spread=
            result.representation_spread,
        mst_amplitude_nearest_agreement=result.nearest_agreement,
        mst_amplitude_truncation_agreement=
            result.truncation_agreement,
        mst_amplitude_max_condition=result.max_condition,
    ), state_metadata)
    amplitudes = merge(route.plan.amplitudes, (;
        teukolsky=merge(route.plan.amplitudes.teukolsky, (;
            incidence=teukolsky_incidence,
            reflection=teukolsky_reflection,
        )),
        gsn=merge(route.plan.amplitudes.gsn, (;
            incidence,
            reflection,
        )),
    ))
    plan = DirectComplexRoutePlan(
        route.plan.contour,
        route.plan.matching,
        route.plan.controls,
        amplitudes,
        route.plan.patch_count,
    )
    return DirectComplexRoute(
        route.branch,
        route.params,
        route.controls,
        route.p_solution,
        state_evaluator,
        metadata,
        plan,
        route.transmission,
        incidence,
        reflection,
        route.teukolsky_transmission,
        teukolsky_incidence,
        teukolsky_reflection,
    )
end

function _state_overlay(route::DirectComplexRoute, anchor)
    anchor.evaluator === route.state_evaluator &&
        isempty(anchor.metadata) && return route
    return DirectComplexRoute(
        route.branch,
        route.params,
        route.controls,
        route.p_solution,
        anchor.evaluator,
        merge(route.metadata, anchor.metadata),
        route.plan,
        route.transmission,
        route.incidence,
        route.reflection,
        route.teukolsky_transmission,
        route.teukolsky_incidence,
        route.teukolsky_reflection,
    )
end

function _retain_analytic_agreement(route, result, agreement)
    metadata = merge(route.metadata, (;
        amplitude_backend=:complex_rational_certified,
        mst_amplitude_certificate=result.certificate_kind,
        mst_amplitude_agreement=Float64(agreement),
        mst_amplitude_representation_spread=
            result.representation_spread,
        mst_amplitude_nearest_agreement=result.nearest_agreement,
        mst_amplitude_truncation_agreement=
            result.truncation_agreement,
        mst_amplitude_max_condition=result.max_condition,
    ))
    return DirectComplexRoute(
        route.branch,
        route.params,
        route.controls,
        route.p_solution,
        route.state_evaluator,
        metadata,
        route.plan,
        route.transmission,
        route.incidence,
        route.reflection,
        route.teukolsky_transmission,
        route.teukolsky_incidence,
        route.teukolsky_reflection,
    )
end

function _rebuild_up_state(
    route,
    evaluator,
    incidence,
    reflection,
)
    rejected(status; projective_error=Inf) = (
        evaluator,
        metadata=(;
            amplitude_state_status=status,
            amplitude_state_projective_error=Float64(projective_error),
        ),
    )
    route.branch == :UP || return rejected(:not_applicable)
    evaluator isa DirectComplexRationalEvaluator ||
        return rejected(:not_rational)
    metadata = route.metadata
    amplitude_match_policy =
        hasproperty(metadata, :amplitude_match_policy) ?
        metadata.amplitude_match_policy : :unknown
    amplitude_match_policy == :interior ||
        return rejected(:noninterior_amplitudes)
    hasproperty(metadata, :endpoint_states) ||
        return rejected(:missing_endpoint_states)
    endpoints = metadata.endpoint_states
    endpoints === nothing && return rejected(:missing_endpoint_states)
    horizon_in = endpoints.horizon_in
    horizon_out = endpoints.horizon_out
    reconstructed = DirectComplexRationalState(
        ComplexF64(
            reflection * horizon_in.X +
            incidence * horizon_out.X,
        ),
        ComplexF64(
            reflection * horizon_in.dXdx +
            incidence * horizon_out.dXdx,
        ),
    )
    current = evaluator.match_state
    all(_finite, (
        reconstructed.X,
        reconstructed.dXdx,
        current.X,
        current.dXdx,
    )) || return rejected(:nonfinite_state)
    determinant = reconstructed.X * current.dXdx -
        current.X * reconstructed.dXdx
    scale = max(
        abs(reconstructed.X * current.dXdx),
        abs(current.X * reconstructed.dXdx),
        floatmin(Float64),
    )
    projective_error = Float64(abs(determinant) / scale)
    projective_error <= _STATE_ANCHOR_MAX_ERROR ||
        return rejected(
            :rejected_projective;
            projective_error,
        )
    rebuilt = DirectComplexRationalEvaluator(
        evaluator.params,
        evaluator.coefficients,
        evaluator.settings,
        evaluator.match_x,
        reconstructed,
    )
    return (
        evaluator=rebuilt,
        metadata=(;
            amplitude_state_status=:reconstructed,
            amplitude_state_projective_error=projective_error,
        ),
    )
end

@inline function _nu_delta(nu)
    value = real(ComplexF64(nu))
    return Float64(value - round(value))
end

@inline function _allows_uncertified_anchor(omega)
    value = ComplexF64(omega)
    denominator = abs(imag(value))
    denominator > 0 && isfinite(denominator) || return false
    ratio = abs(real(value)) / denominator
    return isfinite(ratio) && ratio < _FAST_ANCHOR_RATIO_HIGH
end

@inline function _anchor_accepted(anchor, omega, require_certificate)
    anchor.certified && return true
    require_certificate && return false
    _allows_uncertified_anchor(omega) && return true
    return isfinite(anchor.condition) &&
        anchor.condition <= DirectMSTInfinity.MST_COEFF_CONDITION_MAX
end

@inline function _allows_relaxed_state_anchor(omega)
    value = ComplexF64(omega)
    imag(value) < 0 || return false
    !iszero(real(value)) || return false
    return abs(real(value)) <=
        _STATE_ANCHOR_RELAXED_REAL_RATIO_MAX *
        max(abs(imag(value)), 1.0)
end

function _mst_out_anchor(
    source;
    nu=nothing,
    allow_relaxed::Bool=false,
)
    params = hasproperty(source, :params) ? source.params : source
    selected_nu = if nu === nothing
        try
            mst_nu_complex(
                params.s,
                params.l,
                params.m,
                params.a,
                params.omega,
                params.lambda,
            )
        catch
            nothing
        end
    else
        ComplexF64(nu)
    end
    if selected_nu !== nothing
        selected_nu = ComplexF64(selected_nu)
        nu_delta = _nu_delta(selected_nu)
        if nu_delta > _MST_STATE_NU_DELTA_MIN
            selected = try
                DirectMSTInfinity.direct_mst_monodromy_state(
                    params,
                    _STATE_ANCHOR_X;
                    tolerance=_STATE_ANCHOR_SUM_TOLERANCE,
                    nu=selected_nu,
                )
            catch
                nothing
            end
            if selected !== nothing &&
                    selected.estimated_relerr <= _STATE_ANCHOR_MAX_ERROR &&
                    all(_finite, (selected.X, selected.dXdx))
                return merge(selected, (
                    source=:complex_nu,
                    nu_delta,
                    selector_nu=selected_nu,
                    certified=true,
                    relaxed_sum=false,
                ))
            end
        end
    end

    fallback_tolerance =
        allow_relaxed && _allows_relaxed_state_anchor(params.omega) ?
        _STATE_ANCHOR_RELAXED_TOLERANCE : nothing
    principal = direct_mst_anchor(
        source,
        :out,
        _STATE_ANCHOR_X;
        tolerance=_STATE_ANCHOR_SUM_TOLERANCE,
        fallback_tolerance,
    )
    relaxed_sum = hasproperty(principal, :relaxed_sum) &&
        principal.relaxed_sum
    principal_nu = ComplexF64(principal.nu)
    selected_nu === nothing && return merge(principal, (
        source=relaxed_sum ?
            :principal_relaxed_fallback : :principal_fallback,
        nu_delta=_nu_delta(principal_nu),
        selector_nu=principal_nu,
        certified=false,
        relaxed_sum,
    ))
    nu_delta = _nu_delta(selected_nu)
    nu_delta <= _MST_STATE_NU_DELTA_MIN &&
        return merge(principal, (
            source=relaxed_sum ? :principal_relaxed : :principal,
            nu_delta,
            selector_nu=selected_nu,
            certified=!relaxed_sum,
            relaxed_sum,
        ))
    return merge(principal, (
        source=relaxed_sum ?
            :principal_relaxed_fallback : :principal_fallback,
        nu_delta,
        selector_nu=selected_nu,
        certified=false,
        relaxed_sum,
    ))
end

function _p_state_anchor(
    route,
    evaluator;
    nu=nothing,
    require_certificate::Bool=false,
)
    params = evaluator.params
    mst_params = direct_gsn_parameters(
        params.s,
        params.l,
        params.m,
        params.a,
        params.omega;
        lambda=params.lambda,
    )
    anchor = _mst_out_anchor(mst_params; nu)
    source = anchor.source
    !_anchor_accepted(anchor, params.omega, require_certificate) && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:uncertified,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    anchor.estimated_relerr <= _STATE_ANCHOR_MAX_ERROR || return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_error,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    r = DirectMSTInfinity._direct_x_to_r(params, _STATE_ANCHOR_X)
    current_state = DirectComplexFrequency._state_at_r(evaluator, r)
    current = (current_state.X, current_state.dXdrstar)
    target = (
        anchor.X,
        anchor.dXdx * DirectMSTInfinity._direct_dx_drstar(
            params, _STATE_ANCHOR_X),
    )
    disagreement = all(_finite, current) ?
        Float64(_pair_error(current, target)) : Inf
    disagreement <= _STATE_ANCHOR_MAX_ERROR && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:retained,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=disagreement,
            mst_state_anchor_scale=ComplexF64(1),
        ),
    )
    correction = _fit_pair_scale(target, current)
    correction === nothing && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_scale,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    corrected = (
        correction * current[1],
        correction * current[2],
    )
    projective_error = Float64(_pair_error(corrected, target))
    projective_error <= _STATE_ANCHOR_MAX_ERROR || return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_projective,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=projective_error,
            mst_state_anchor_scale=correction,
        ),
    )
    anchored = DirectComplexStateEvaluator(
        evaluator.params,
        evaluator.p_solution,
        evaluator.coefficients,
        ComplexF64(evaluator.scale * correction),
    )
    return (
        evaluator=anchored,
        metadata=(;
            mst_state_anchor_status=:rescaled,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=projective_error,
            mst_state_anchor_scale=correction,
        ),
    )
end

function _sfe_state_anchor(
    route,
    evaluator;
    nu=nothing,
    require_certificate::Bool=false,
)
    coefficients = evaluator.route.coefficients
    anchor = _mst_out_anchor(coefficients; nu)
    source = anchor.source
    !_anchor_accepted(
        anchor,
        coefficients.params.omega,
        require_certificate,
    ) && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:uncertified,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    anchor.estimated_relerr <= _STATE_ANCHOR_MAX_ERROR || return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_error,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    params = evaluator.route.params
    r = DirectMSTInfinity._direct_x_to_r(params, _STATE_ANCHOR_X)
    current_state = DirectComplexFrequency._state_at_r(evaluator, r)
    dxdrstar =
        DirectMSTInfinity._direct_dx_drstar(params, _STATE_ANCHOR_X)
    current = (current_state.X, current_state.dXdrstar)
    target = (anchor.X, anchor.dXdx * dxdrstar)
    disagreement = all(_finite, current) ?
        Float64(_pair_error(current, target)) : Inf
    disagreement <= _STATE_ANCHOR_MAX_ERROR && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:retained,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=disagreement,
            mst_state_anchor_scale=ComplexF64(1),
        ),
    )
    correction = _fit_pair_scale(target, current)
    correction === nothing && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_scale,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=Inf,
            mst_state_anchor_scale=ComplexF64(NaN, NaN),
        ),
    )
    corrected = (
        correction * current[1],
        correction * current[2],
    )
    projective_error = Float64(_pair_error(corrected, target))
    projective_error <= _STATE_ANCHOR_MAX_ERROR || return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_projective,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=projective_error,
            mst_state_anchor_scale=correction,
        ),
    )
    return (
        evaluator=ScaledEvaluator(evaluator, correction),
        metadata=(;
            mst_state_anchor_status=:rescaled,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
            mst_state_anchor_projective_error=projective_error,
            mst_state_anchor_scale=correction,
        ),
    )
end

function _state_anchor(
    route;
    nu=nothing,
    require_certificate::Bool=false,
)
    route.branch == :IN &&
        return _in_real_axis_anchor(route, route.state_evaluator)
    route.branch == :UP || return (
        evaluator=route.state_evaluator,
        metadata=(;),
    )
    evaluator = route.state_evaluator
    evaluator isa DirectComplexSFEvaluator &&
        return _sfe_state_anchor(
            route,
            evaluator;
            nu,
            require_certificate,
        )
    evaluator isa DirectComplexStateEvaluator &&
        return _p_state_anchor(
            route,
            evaluator;
            nu,
            require_certificate,
        )
    evaluator isa DirectComplexRationalEvaluator || return (
        evaluator,
        metadata=(;),
    )
    anchor = _mst_out_anchor(
        evaluator.coefficients;
        nu,
        allow_relaxed=true,
    )
    source = anchor.source
    relaxed_sum = hasproperty(anchor, :relaxed_sum) &&
        anchor.relaxed_sum
    !_anchor_accepted(
        anchor,
        evaluator.coefficients.params.omega,
        require_certificate,
    ) && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:uncertified,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
        ),
    )
    anchor_error_limit = relaxed_sum ?
        _STATE_ANCHOR_RELAXED_TOLERANCE :
        _STATE_ANCHOR_MAX_ERROR
    anchor.estimated_relerr <= anchor_error_limit || return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:rejected_error,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=Inf,
        ),
    )
    current = try
        state = DirectComplexRational._state_x(
            evaluator, _STATE_ANCHOR_X)
        (state.X, state.dXdx)
    catch
        (ComplexF64(NaN, NaN), ComplexF64(NaN, NaN))
    end
    target = (anchor.X, anchor.dXdx)
    disagreement = all(_finite, current) ?
        Float64(_pair_error(current, target)) : Inf
    if relaxed_sum
        correction = _fit_pair_scale(target, current)
        correction === nothing && return (
            evaluator,
            metadata=(;
                mst_state_anchor_status=:rejected_scale,
                mst_state_anchor_source=source,
                mst_state_anchor_x=_STATE_ANCHOR_X,
                mst_state_anchor_error=anchor.estimated_relerr,
                mst_state_anchor_disagreement=disagreement,
                mst_state_anchor_projective_error=Inf,
                mst_state_anchor_scale=ComplexF64(NaN, NaN),
                mst_state_anchor_relaxed_sum=true,
            ),
        )
        corrected = (
            correction * current[1],
            correction * current[2],
        )
        projective_error =
            Float64(_pair_error(corrected, target))
        projective_error <= _STATE_ANCHOR_MAX_ERROR || return (
            evaluator,
            metadata=(;
                mst_state_anchor_status=:rejected_projective,
                mst_state_anchor_source=source,
                mst_state_anchor_x=_STATE_ANCHOR_X,
                mst_state_anchor_error=anchor.estimated_relerr,
                mst_state_anchor_disagreement=disagreement,
                mst_state_anchor_projective_error=projective_error,
                mst_state_anchor_scale=correction,
                mst_state_anchor_relaxed_sum=true,
            ),
        )
        return (
            evaluator=ScaledEvaluator(evaluator, correction),
            metadata=(;
                mst_state_anchor_status=:rescaled_relaxed,
                mst_state_anchor_source=source,
                mst_state_anchor_x=_STATE_ANCHOR_X,
                mst_state_anchor_error=anchor.estimated_relerr,
                mst_state_anchor_disagreement=disagreement,
                mst_state_anchor_projective_error=projective_error,
                mst_state_anchor_scale=correction,
                mst_state_anchor_relaxed_sum=true,
            ),
        )
    end
    disagreement <= _STATE_ANCHOR_MAX_ERROR && return (
        evaluator,
        metadata=(;
            mst_state_anchor_status=:retained,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
        ),
    )
    anchored = DirectComplexRationalEvaluator(
        evaluator.params,
        evaluator.coefficients,
        evaluator.settings,
        _STATE_ANCHOR_X,
        DirectComplexRationalState(anchor.X, anchor.dXdx),
    )
    return (
        evaluator=anchored,
        metadata=(;
            mst_state_anchor_status=:replaced,
            mst_state_anchor_source=source,
            mst_state_anchor_x=_STATE_ANCHOR_X,
            mst_state_anchor_error=anchor.estimated_relerr,
            mst_state_anchor_disagreement=disagreement,
        ),
    )
end

function _analytic_result(route)
    p = route.params
    evaluator = route.state_evaluator
    if evaluator isa DirectComplexRationalEvaluator
        principal = try
            mst_principal_amplitudes(
                evaluator.coefficients, route.branch)
        catch
            nothing
        end
        principal !== nothing &&
            return principal, nothing, :principal
    end
    refined_nu = try
        mst_nu_complex(
            p.s, p.l, p.m, p.a, p.omega, p.lambda)
    catch
        nothing
    end
    if refined_nu !== nothing
        refined = try
            mst_nia_amplitudes(
                p.s,
                p.l,
                p.m,
                p.a,
                p.omega,
                p.lambda,
                route.branch;
                nu=refined_nu,
            )
        catch
            nothing
        end
        refined !== nothing && refined.certificate_accepted &&
            return refined, refined_nu, :complex_refined
    end
    return nothing
end

@inline function _fast_anchor_pretrigger(
    a,
    omega,
    branch;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
    max_omega::Float64=_FAST_ANCHOR_MAX_OMEGA,
)
    backend == :direct_rational || return false
    controls === nothing || return false
    all(value === nothing for value in (
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
    )) || return false
    pole_normalization && return false
    branch_symbol = try
        DirectComplexFrequency._normalize_branch(branch)
    catch
        return false
    end
    branch_symbol in (:IN, :UP) || return false
    af = try
        Float64(a)
    catch
        return false
    end
    kappa_squared = (1.0 - af) * (1.0 + af)
    kappa_squared > 0.0 || return false
    sqrt(kappa_squared) >
        DirectComplexFrequency._P_CONSENSUS_KAPPA_MAX || return false
    omegac = ComplexF64(omega)
    _finite(omegac) && imag(omegac) < 0 &&
        !iszero(real(omegac)) || return false
    omega_abs = abs(omegac)
    _FAST_ANCHOR_MIN_OMEGA < omega_abs <=
        max_omega || return false
    ratio = abs(real(omegac)) / abs(imag(omegac))
    return ratio <= _FAST_ANCHOR_RATIO_LOW ||
        ratio >= _FAST_ANCHOR_RATIO_HIGH
end

function _fast_route(
    s,
    l,
    m,
    a,
    omega,
    branch;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
    max_omega::Float64=_FAST_ANCHOR_MAX_OMEGA,
    rescue_source::Symbol=:prebuild,
)
    _fast_anchor_pretrigger(
        a,
        omega,
        branch;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
        max_omega,
    ) || return nothing
    branch_symbol = DirectComplexFrequency._normalize_branch(branch)

    params = try
        direct_gsn_parameters(s, l, m, a, omega)
    catch
        return nothing
    end
    data = try
        DirectMSTInfinity._mst_data(params)
    catch
        return nothing
    end
    result = try
        DirectMSTInfinity._principal_amplitudes(data, branch_symbol)
    catch error
        error isa DirectMSTInfinity.MSTCertificateError &&
            error.kind == :nia_amplitude_certificate ||
            return nothing
        try
            DirectMSTInfinity._refined_amplitudes(
                data, branch_symbol)
        catch
            return nothing
        end
    end
    result.certificate_accepted || return nothing
    state_data = if hasproperty(result, :nu)
        refined_nu = ComplexF64(result.nu)
        DirectMSTInfinity._nia_series_data(
            DirectMSTInfinity.MSTParams(
                data.params.s,
                data.params.l,
                data.params.m,
                data.params.a,
                data.params.omega,
                data.params.lambda,
                refined_nu,
            ),
            DirectMSTInfinity.MST_NIA_TAIL,
        )
    else
        data
    end
    direct_controls = direct_gsn_controls(
        params;
        N=_FAST_ANCHOR_ORDER,
        xm=_STATE_ANCHOR_X,
        tol=_FAST_ANCHOR_TOLERANCE,
        sfe=false,
        lfe=false,
    )
    coefficients = try
        direct_gsn_coefficients(params; controls=direct_controls)
    catch
        return nothing
    end
    state = try
        if branch_symbol == :UP
            anchor = _mst_out_anchor(
                coefficients;
                nu=hasproperty(result, :nu) ? result.nu : nothing,
                allow_relaxed=true,
            )
            (
                X=ComplexF64(anchor.X),
                dXdx=ComplexF64(anchor.dXdx),
                estimated_relerr=Float64(anchor.estimated_relerr),
                nu=ComplexF64(anchor.nu),
                condition=Float64(anchor.condition),
                source=anchor.source,
                certified=Bool(anchor.certified),
            )
        else
            pin = if hasproperty(result, :nu_offset_dd)
                DirectMSTInfinity._dd_pin_state(
                    coefficients,
                    state_data,
                    result.nu_offset_dd,
                    _STATE_ANCHOR_X;
                    scale=DirectMSTInfinity._pin_unit_scale(params),
                )
            else
                plan = (
                    data=state_data,
                    converter=DirectMSTInfinity._p_converter(params),
                    pin_norm=DirectMSTInfinity._incoming_raw_factor(
                        state_data),
                    pin_scale=nothing,
                )
                DirectMSTInfinity.direct_mst_pin_state(
                    coefficients,
                    plan,
                    _STATE_ANCHOR_X;
                    scale=DirectMSTInfinity._pin_unit_scale(params),
                )
            end
            (
                X=ComplexF64(pin.X),
                dXdx=ComplexF64(pin.dXdx),
                estimated_relerr=Float64(max(
                    pin.estimated_relerr,
                    pin.residual,
                )),
                nu=ComplexF64(NaN, NaN),
                condition=hasproperty(pin, :condition) ?
                    Float64(pin.condition) : NaN,
                source=:physical_in,
                representation=hasproperty(pin, :representation) ?
                    pin.representation : :native_type1,
                certified=true,
            )
        end
    catch
        return nothing
    end
    all(_finite, (state.X, state.dXdx)) || return nothing
    state.estimated_relerr <= _STATE_ANCHOR_MAX_ERROR ||
        return nothing
    _anchor_accepted(
        state,
        params.omega,
        false,
    ) || return nothing

    settings = DirectComplexRational.DirectComplexRationalSettings(
        _FAST_ANCHOR_ORDER, _FAST_ANCHOR_TOLERANCE)
    evaluator = DirectComplexRationalEvaluator(
        params,
        coefficients,
        settings,
        _STATE_ANCHOR_X,
        DirectComplexRationalState(state.X, state.dXdx),
    )

    complex_params = try
        DirectComplexFrequency.direct_complex_parameters(
            params.s,
            params.l,
            params.m,
            params.a,
            params.omega,
            params.lambda,
        )
    catch
        return nothing
    end
    factors =
        DirectComplexFrequency._amplitude_factors(
            complex_params, branch_symbol)
    transmission = ComplexF64(1)
    incidence = ComplexF64(result.gsn[1])
    reflection = ComplexF64(result.gsn[2])
    teukolsky_transmission = ComplexF64(factors.transmission)
    teukolsky_incidence =
        ComplexF64(teukolsky_transmission * result.teuk[1])
    teukolsky_reflection =
        ComplexF64(teukolsky_transmission * result.teuk[2])
    all(_finite, (
        transmission,
        incidence,
        reflection,
        teukolsky_transmission,
        teukolsky_incidence,
        teukolsky_reflection,
    )) || return nothing

    requested = DirectComplexFrequency.direct_complex_controls()
    resolved = (
        xm=_STATE_ANCHOR_X,
        rhom=nothing,
        N=_FAST_ANCHOR_ORDER,
        tolerance=_FAST_ANCHOR_TOLERANCE,
        sfe=false,
        lfe=false,
        TSinInf=nothing,
        TSoutInf=nothing,
        TSinHor=nothing,
        TSoutHor=nothing,
        rescue_source=:none,
    )
    contour = DirectComplexFrequency.complex_contour_parameters(
        complex_params; rho_match=1.0)
    matching = (
        requested_x=nothing,
        allowed_min_x=0.0,
        allowed_max_x=1.0,
        candidate_x=_STATE_ANCHOR_X,
        selected_x=_STATE_ANCHOR_X,
        split_mismatch=state.estimated_relerr,
        condition=1.0,
        policy=:mst_anchor,
    )
    amplitude_summary = (
        teukolsky=(
            transmission=teukolsky_transmission,
            incidence=teukolsky_incidence,
            reflection=teukolsky_reflection,
        ),
        conversion_factors=factors,
        raw_gsn_transmission=transmission,
        gsn=(;
            transmission,
            incidence,
            reflection,
        ),
    )
    plan = DirectComplexRoutePlan(
        contour,
        matching,
        resolved,
        amplitude_summary,
        0,
    )
    metadata = (
        backend=:direct_gsn_mst_anchor,
        match_policy=:mst_anchor,
        match_x=_STATE_ANCHOR_X,
        xsplit=_STATE_ANCHOR_X,
        xm_match=_STATE_ANCHOR_X,
        matching_condition=1.0,
        split_mismatch=state.estimated_relerr,
        abel_error=state.estimated_relerr,
        patch_count=0,
        control_xm=_STATE_ANCHOR_X,
        control_rhom=nothing,
        control_N=_FAST_ANCHOR_ORDER,
        control_tol=_FAST_ANCHOR_TOLERANCE,
        control_sfe=false,
        control_lfe=false,
        amplitude_backend=:principal_mst_anchor,
        amplitude_patch_count=0,
        mst_amplitude_certificate=result.certificate_kind,
        mst_amplitude_representation_spread=
            result.representation_spread,
        mst_amplitude_nearest_agreement=result.nearest_agreement,
        mst_amplitude_truncation_agreement=
            result.truncation_agreement,
        mst_amplitude_max_condition=result.max_condition,
        mst_amplitude_nu_residual=
            hasproperty(result, :nu_residual) ?
            result.nu_residual : NaN,
        mst_state_anchor_status=:accepted,
        mst_state_anchor_source=state.source,
        mst_state_anchor_x=_STATE_ANCHOR_X,
        mst_state_anchor_error=state.estimated_relerr,
        mst_state_anchor_nu=state.nu,
        mst_state_anchor_condition=state.condition,
        mst_state_anchor_certified=state.certified,
        mst_state_anchor_representation=hasproperty(
            state, :representation) ?
            state.representation : :unknown,
        mst_fast_rescue_source=rescue_source,
    )
    return DirectComplexRoute(
        branch_symbol,
        complex_params,
        requested,
        nothing,
        evaluator,
        metadata,
        plan,
        transmission,
        incidence,
        reflection,
        teukolsky_transmission,
        teukolsky_incidence,
        teukolsky_reflection,
    )
end

@inline function _fast_p_pretrigger(
    route::DirectComplexRoute;
    max_omega::Float64=_MAX_ABS_IMAG_OMEGA,
)
    route.branch == :IN || return false
    route_backend = hasproperty(route.metadata, :backend) ?
        route.metadata.backend : :unknown
    route_backend in (
        :direct_gsn_p_build_fallback,
        :direct_gsn_p_consensus,
    ) || return false
    omega_abs = abs(ComplexF64(route.params.omega))
    return _FAST_ANCHOR_MAX_OMEGA < omega_abs <= max_omega
end

function _fast_p_rescue(
    route::DirectComplexRoute,
    s,
    l,
    m,
    a,
    omega,
    branch;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
    max_omega::Float64=_MAX_ABS_IMAG_OMEGA,
)
    _fast_p_pretrigger(route; max_omega) || return route
    route_backend = route.metadata.backend
    candidate = _fast_route(
        s,
        l,
        m,
        a,
        omega,
        branch;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
        max_omega,
        rescue_source=route_backend,
    )
    return candidate === nothing ? route : candidate
end

@inline function _offaxis_pretrigger(
    route::DirectComplexRoute,
    ;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
)
    backend == :direct_rational || return false
    controls === nothing || return false
    all(value === nothing for value in (
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
    )) || return false
    pole_normalization && return false
    omega = ComplexF64(route.params.omega)
    imag(omega) < 0 || return false
    !iszero(real(omega)) || return false
    near_nia = abs(real(omega)) <=
        _NIA_LATERAL_REAL_RATIO_MAX * max(abs(imag(omega)), 1.0)
    near_nia && return true
    amplitude_backend = hasproperty(route.metadata, :amplitude_backend) ?
        route.metadata.amplitude_backend : :unknown
    match_policy = hasproperty(route.metadata, :match_policy) ?
        route.metadata.match_policy : :unknown
    match_policy == :branch_shift_consensus && return false
    return amplitude_backend in (
        :complex_rational,
        :principal_mst_logscaled,
    )
end

@inline function _state_pretrigger(
    route::DirectComplexRoute,
    ;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
)
    backend == :direct_rational || return false
    controls === nothing || return false
    all(value === nothing for value in (
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
    )) || return false
    pole_normalization && return false
    route.branch == :UP || return false
    route.state_evaluator isa DirectComplexStateEvaluator || return false
    omega = ComplexF64(route.params.omega)
    imag(omega) < 0 || return false
    !iszero(real(omega)) || return false
    route_backend = hasproperty(route.metadata, :backend) ?
        route.metadata.backend : :unknown
    return route_backend == :direct_gsn_p_build_fallback
end

@noinline function _post_overlay(
    route::DirectComplexRoute,
    s,
    l,
    m,
    a,
    omega,
    branch;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
)
    Base.@nospecialize route
    rescued = _fast_p_rescue(
        route,
        s,
        l,
        m,
        a,
        omega,
        branch;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
        max_omega=_MAX_ABS_IMAG_OMEGA,
    )
    rescued === route || return rescued
    amplitude_trigger = _offaxis_pretrigger(
        route;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
    )
    state_trigger = _state_pretrigger(
        route;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
    )
    (amplitude_trigger || state_trigger) || return route
    analytic = amplitude_trigger ? _analytic_result(route) : nothing
    p = route.params
    near_nia = abs(real(p.omega)) <=
        _NIA_LATERAL_REAL_RATIO_MAX * max(abs(imag(p.omega)), 1.0)
    if analytic === nothing
        (near_nia || state_trigger) || return route
        anchor = try
            _state_anchor(route)
        catch
            (evaluator=route.state_evaluator, metadata=(;))
        end
        return _state_overlay(route, anchor)
    end
    result, nu_value, _ = analytic
    amplitude_agreement = if _finite(route.transmission) &&
            !iszero(route.transmission)
        current = (
            route.incidence / route.transmission,
            route.reflection / route.transmission,
        )
        _pair_error(current, result.gsn)
    else
        Inf
    end
    if amplitude_agreement <= _AMPLITUDE_AGREEMENT_MAX
        anchor = try
            _state_anchor(
                route;
                nu=nu_value,
                require_certificate=true,
            )
        catch
            (evaluator=route.state_evaluator, metadata=(;))
        end
        return _retain_analytic_agreement(
            _state_overlay(route, anchor),
            result,
            amplitude_agreement,
        )
    end
    anchor = try
        _state_anchor(route; nu=nu_value)
    catch
        (evaluator=route.state_evaluator, metadata=(;))
    end
    anchored_route = _state_overlay(route, anchor)
    near_nia && !(result.certificate_kind in (
        :global_medoid,
        :principal_single_nu,
    )) &&
        return anchored_route
    return _overlay(
        anchored_route,
        result;
        amplitude_backend=:complex_mst_analytic,
        state_evaluator=anchored_route.state_evaluator,
        state_metadata=anchor.metadata,
    )
end

@noinline function build(
    s,
    l,
    m,
    a,
    omega,
    branch;
    backend,
    controls,
    lambda,
    nu,
    xm,
    rhom,
    N,
    tol,
    sfe,
    lfe,
    TSinInf,
    TSoutInf,
    TSinHor,
    TSoutHor,
    pole_normalization,
)
    fast = _fast_route(
        s,
        l,
        m,
        a,
        omega,
        branch;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
        max_omega=_MAX_ABS_IMAG_OMEGA,
    )
    fast === nothing || return fast
    _pretrigger(
        s,
        a,
        omega;
        backend,
        controls,
        lambda,
        nu,
        xm,
        rhom,
        N,
        tol,
        sfe,
        lfe,
        TSinInf,
        TSoutInf,
        TSinHor,
        TSoutHor,
        pole_normalization,
    ) || return nothing
    route = try
        DirectComplexFrequency._direct_complex_p_route(
            s, l, m, a, omega, branch)
    catch
        return nothing
    end
    result = try
        mst_nia_amplitudes(
            route.params.s,
            route.params.l,
            route.params.m,
            route.params.a,
            route.params.omega,
            route.params.lambda,
            route.branch,
        )
    catch
        return nothing
    end
    result.certificate_accepted || return nothing
    anchor = try
        _state_anchor(route)
    catch
        (evaluator=route.state_evaluator, metadata=(;))
    end
    return _overlay(
        route,
        result;
        state_evaluator=anchor.evaluator,
        state_metadata=anchor.metadata,
    )
end

DirectComplexFrequency._register_special_route_builder!(build)
DirectComplexFrequency._register_post_route_builder!(_post_overlay)

end
