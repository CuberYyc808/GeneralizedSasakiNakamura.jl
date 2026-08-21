function _amplitude_pair(result::LeaverResult, omega;
        gsn_solver=GSN_radial, teukolsky_solver=Teukolsky_radial)
    mode = result.mode
    if _use_wronskian_incidence(result, omega)
        pair = _wronskian_incidence(
            result, omega; radial_solver=gsn_solver)
        inputs = pair.inputs
        incidence_conversion = ConversionFactors.Binc(
            mode.s, mode.m, inputs.a, inputs.omega, pair.lambda)
        reflection_conversion = ConversionFactors.Bref(
            mode.s, mode.m, inputs.a, inputs.omega, pair.lambda)
        transmission_conversion = ConversionFactors.Btrans(
            mode.s, mode.m, inputs.a, inputs.omega, pair.lambda)
        return (
            omega=omega,
            evaluation_a=inputs.a,
            evaluation_omega=inputs.omega,
            evaluation_policy=inputs.policy,
            frequency_projection_drift=inputs.projection_drift,
            gsn_incidence=pair.incidence,
            gsn_reflection=pair.reflection,
            teukolsky_incidence=
                incidence_conversion / transmission_conversion *
                pair.incidence,
            teukolsky_reflection=
                reflection_conversion / transmission_conversion *
                pair.reflection,
            gsn_method=pair.method,
            gsn_lambda=pair.lambda,
            teukolsky_lambda=pair.lambda,
            gsn_normalization=pair.normalization,
            teukolsky_normalization=:analytic_conversion_from_gsn,
            incidence_backend=:direct_real_axis_wronskian,
            wronskian_drift=pair.wronskian_drift,
        )
    end
    inputs = _radial_evaluation_inputs(result, omega)
    gsn = gsn_solver(
        mode.s, mode.l, mode.m, inputs.a, inputs.omega, IN)
    teukolsky = teukolsky_solver(
        mode.s, mode.l, mode.m, inputs.a, inputs.omega, IN)
    return (
        omega=omega,
        evaluation_a=inputs.a,
        evaluation_omega=inputs.omega,
        evaluation_policy=inputs.policy,
        frequency_projection_drift=inputs.projection_drift,
        gsn_incidence=gsn.incidence_amplitude,
        gsn_reflection=gsn.reflection_amplitude,
        teukolsky_incidence=teukolsky.incidence_amplitude,
        teukolsky_reflection=teukolsky.reflection_amplitude,
        gsn_method=gsn.method,
        gsn_lambda=gsn.mode.lambda,
        teukolsky_lambda=teukolsky.mode.lambda,
        gsn_normalization=gsn.normalization_convention,
        teukolsky_normalization=teukolsky.normalization_convention,
        incidence_backend=:direct_asymptotic_amplitude,
        wronskian_drift=zero(real(abs(gsn.incidence_amplitude))),
    )
end

function _richardson_derivative(values, order::Int=2)
    factor = 2^order
    first = (factor * values[2] - values[1]) / (factor - 1)
    second = (factor * values[3] - values[2]) / (factor - 1)
    next_factor = 4factor
    extrapolated = (next_factor * second - first) / (next_factor - 1)
    scale = max(one(abs(extrapolated)), abs(first), abs(second),
        abs(extrapolated))
    drift = abs(second - first) / scale
    return (value=extrapolated, first=first, second=second, drift=drift)
end

function _gsn_amplitude_sample(result::LeaverResult, omega;
        radial_solver=GSN_radial)
    mode = result.mode
    if _use_wronskian_incidence(result, omega)
        pair = _wronskian_incidence(
            result, omega; radial_solver)
        return (
            omega=omega,
            incidence=pair.incidence,
            reflection=pair.reflection,
            wronskian_drift=pair.wronskian_drift,
            incidence_backend=:direct_real_axis_wronskian,
            evaluation_policy=pair.inputs.policy,
            frequency_projection_drift=pair.inputs.projection_drift,
        )
    end

    inputs = _radial_evaluation_inputs(result, omega)
    solution = radial_solver(
        mode.s, mode.l, mode.m, inputs.a, inputs.omega, IN)
    return (
        omega=omega,
        incidence=solution.incidence_amplitude,
        reflection=solution.reflection_amplitude,
        wronskian_drift=zero(real(abs(solution.incidence_amplitude))),
        incidence_backend=:direct_asymptotic_amplitude,
        evaluation_policy=inputs.policy,
        frequency_projection_drift=inputs.projection_drift,
    )
end

function _cached_gsn_amplitude_sample!(cache, request_count,
        result::LeaverResult, omega, radial_solver)
    request_count[] += 1
    return get!(cache, omega) do
        _gsn_amplitude_sample(result, omega; radial_solver)
    end
end

function _gsn_real_derivative_attempt(
        result::LeaverResult, base_h, radial_solver, cache, request_count)
    C = typeof(result.omega)
    T = typeof(real(result.omega))
    steps = T[base_h, base_h / 2, base_h / 4]
    derivatives = C[]
    stencil = NamedTuple[]
    for step in steps
        plus = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega + step,
            radial_solver)
        minus = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega - step,
            radial_solver)
        push!(derivatives,
            (plus.incidence - minus.incidence) / (2 * step))
        push!(stencil, merge(plus,
            (step=step, direction=:real, sign=:plus)))
        push!(stencil, merge(minus,
            (step=step, direction=:real, sign=:minus)))
    end
    return (
        base_h=T(base_h),
        steps,
        derivatives,
        stencil,
        richardson=_richardson_derivative(derivatives),
    )
end

function _gsn_five_point_real_derivative_attempt(
        result::LeaverResult, base_h, radial_solver, cache, request_count)
    C = typeof(result.omega)
    T = typeof(real(result.omega))
    steps = T[base_h, base_h / 2, base_h / 4]
    derivatives = C[]
    stencil = NamedTuple[]
    for step in steps
        minus2 = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega - 2step,
            radial_solver)
        minus1 = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega - step,
            radial_solver)
        plus1 = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega + step,
            radial_solver)
        plus2 = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega + 2step,
            radial_solver)
        push!(derivatives,
            (minus2.incidence - 8minus1.incidence + 8plus1.incidence -
                plus2.incidence) / (12step))
        for (offset, row) in ((-2, minus2), (-1, minus1),
                (1, plus1), (2, plus2))
            push!(stencil, merge(row, (
                step=step,
                direction=:real,
                sign=offset < 0 ? :minus : :plus,
                offset=offset,
            )))
        end
    end
    return (
        base_h=T(base_h),
        steps,
        derivatives,
        stencil,
        richardson=_richardson_derivative(derivatives, 4),
    )
end

function _near_as_conditioned_validation_candidate(
        result::LeaverResult, validation::ISEMValidationResult)
    provenance = result.provenance
    near_as_role = hasproperty(provenance, :near_as_role) ?
        provenance.near_as_role :
        hasproperty(provenance, :endpoint_kind) ?
            provenance.endpoint_kind : :none
    return validation.status == :failed &&
        validation.stop_reason == :scaled_incidence_gate &&
        _near_as_complete_scope(result.mode, near_as_role) &&
        near_as_role == :unconventional && result.mode.n == 8 &&
        hasproperty(provenance, :acceptance_basis) &&
        provenance.acceptance_basis == :near_as_root_position_stability
end

function _condition_incidence_validation(
        result::LeaverResult, validation::ISEMValidationResult, alpha,
        failure_reason::Symbol, conditioned_flag::NamedTuple,
        conditioned_basis::Symbol;
        inferred_root_shift_tolerance=1.0e-8)
    T = typeof(validation.scaled_incidence)
    shift = abs(validation.incidence / alpha) /
        max(one(T), abs(result.omega))
    tolerance = T(inferred_root_shift_tolerance)
    accepted = isfinite(shift) && shift <= tolerance
    return ISEMValidationResult(
        validation.omega,
        validation.incidence,
        validation.reflection,
        validation.transmission,
        validation.absolute_incidence,
        validation.scaled_incidence,
        validation.selected_method,
        accepted ? :accepted : :failed,
        accepted ? :accepted : failure_reason,
        merge(validation.metadata, (
            standard_incidence_status=validation.status,
            standard_incidence_stop_reason=validation.stop_reason,
            standard_incidence_gate_passed=false,
        ), conditioned_flag, (
            inferred_root_shift=T(shift),
            inferred_root_shift_tolerance=tolerance,
            conditioned_validation_basis=conditioned_basis,
        )),
    )
end

function _condition_near_as_validation(
        result::LeaverResult, validation::ISEMValidationResult, alpha;
        inferred_root_shift_tolerance=1.0e-8)
    return _condition_incidence_validation(
        result, validation, alpha, :near_as_inferred_root_shift_gate,
        (conditioned_near_as_validation=true,),
        :independent_root_and_local_incidence_condition_number;
        inferred_root_shift_tolerance)
end

function _near_extremal_conditioned_validation_candidate(
        result::LeaverResult, validation::ISEMValidationResult)
    validation.status == :failed || return false
    validation.stop_reason == :scaled_incidence_gate || return false
    root_conditioned = result.status == :estimated &&
        abs(result.a) >= oftype(result.a, 0.9999)
    low_overtone_zdm = result.mode.n <= 4 &&
        _near_extremal_zdm_seed_scope(result.mode, result.a)
    root_conditioned || low_overtone_zdm || return false
    tolerance = validation.metadata.incidence_tolerance
    return isfinite(validation.scaled_incidence) &&
        validation.scaled_incidence <=
            (root_conditioned ? 1000 : 100) * tolerance
end

function _finite_high_damping_diagnostic_candidate(
        result::LeaverResult, validation::ISEMValidationResult)
    result.status in (:accepted, :estimated) || return false
    validation.status == :failed || return false
    validation.stop_reason == :scaled_incidence_gate || return false
    imag(result.omega) < oftype(imag(result.omega), -0.5) || return false
    return all(isfinite, (
        validation.incidence,
        validation.reflection,
        validation.transmission,
        validation.scaled_incidence,
    ))
end

function _condition_near_extremal_validation(
        result::LeaverResult, validation::ISEMValidationResult, alpha;
        inferred_root_shift_tolerance=1.0e-8)
    return _condition_incidence_validation(
        result, validation, alpha,
        :near_extremal_inferred_root_shift_gate,
        (conditioned_near_extremal_validation=true,),
        :independent_leaver_root_and_local_gsn_condition_number;
        inferred_root_shift_tolerance)
end

function _qnm_gsn_excitation_factor(result::LeaverResult;
        h=nothing,
        incidence_tolerance=1.0e-8,
        step_tolerance=1.0e-7,
        wronskian_tolerance=1.0e-8,
        radial_solver=GSN_radial,
        direction_tolerance=nothing,
        bridge_tolerance=nothing,
        teukolsky_solver=nothing)
    validation = validate_qnm_with_isem(result;
        incidence_tolerance, wronskian_tolerance, radial_solver)
    C = typeof(result.omega)
    T = typeof(real(result.omega))
    conditioned_candidate = _near_as_conditioned_validation_candidate(
        result, validation)
    near_extremal_candidate =
        _near_extremal_conditioned_validation_candidate(result, validation)
    high_damping_diagnostic =
        _finite_high_damping_diagnostic_candidate(result, validation)
    any_conditioned_candidate = conditioned_candidate ||
        near_extremal_candidate || high_damping_diagnostic
    if validation.status != :accepted && !any_conditioned_candidate
        nan_complex = complex(T(NaN), T(NaN))
        return (
            omega=result.omega,
            alpha=nan_complex,
            reflection=nan_complex,
            B_gsn=nan_complex,
            step_drift=T(Inf),
            status=:failed,
            stop_reason=:isem_incidence_not_accepted,
            metadata=(validation=validation, stencil=NamedTuple[],
                derivative_direction=:real_frequency),
        )
    end

    base_h = h === nothing ?
        conditioned_candidate ? T(3.2e-3) :
        T(1.0e-4) * max(one(T), abs(result.omega)) : T(h)
    base_h > zero(T) || throw(ArgumentError("h must be positive."))
    step_factors = h === nothing && high_damping_diagnostic ? T[one(T)] :
        h === nothing ?
        conditioned_candidate ? T[one(T), T(0.5), T(2), T(0.25), T(4)] :
        T[one(T), T(2), T(0.5), T(4), T(0.25)] : T[one(T)]
    attempts = NamedTuple[]
    selected = nothing
    selected_score = T(Inf)
    sample_cache = Dict{C,Any}()
    sample_requests = Ref(0)
    validation_wronskian_drift = T(validation.metadata.wronskian_drift)
    for factor in step_factors
        attempt = conditioned_candidate ?
            _gsn_five_point_real_derivative_attempt(
                result, base_h * factor, radial_solver, sample_cache,
                sample_requests) :
            _gsn_real_derivative_attempt(
                result, base_h * factor, radial_solver, sample_cache,
                sample_requests)
        richardson_attempt = attempt.richardson
        maximum_wronskian_drift = T(max(
            validation_wronskian_drift,
            maximum(row.wronskian_drift for row in attempt.stencil),
        ))
        finite_attempt = isfinite(richardson_attempt.value)
        accepted_attempt = finite_attempt &&
            richardson_attempt.drift <= T(step_tolerance) &&
            maximum_wronskian_drift <= T(wronskian_tolerance)
        score = finite_attempt ? max(
            richardson_attempt.drift / T(step_tolerance),
            maximum_wronskian_drift / T(wronskian_tolerance),
        ) : T(Inf)
        summary = (;
            factor,
            base_h=attempt.base_h,
            step_drift=T(richardson_attempt.drift),
            maximum_wronskian_drift,
            accepted=accepted_attempt,
        )
        push!(attempts, summary)
        if selected === nothing || score < selected_score
            selected = merge(attempt, (; maximum_wronskian_drift))
            selected_score = score
        end
        if accepted_attempt
            selected = merge(attempt, (; maximum_wronskian_drift))
            break
        end
    end

    richardson = selected.richardson
    steps = selected.steps
    stencil = selected.stencil
    alpha = richardson.value
    if conditioned_candidate
        validation = _condition_near_as_validation(
            result, validation, alpha)
    elseif near_extremal_candidate
        validation = _condition_near_extremal_validation(
            result, validation, alpha)
    end
    evaluation_omega = validation.metadata.evaluation_omega
    reflection = validation.reflection
    B_gsn = reflection / (2 * evaluation_omega * alpha)
    maximum_wronskian_drift = selected.maximum_wronskian_drift
    finite = isfinite(alpha) && isfinite(reflection) && isfinite(B_gsn)
    accepted = validation.status == :accepted && finite &&
        richardson.drift <= T(step_tolerance) &&
        maximum_wronskian_drift <= T(wronskian_tolerance)
    stop_reason = validation.status != :accepted ? validation.stop_reason :
        !finite ? :nonfinite_gsn_excitation :
        richardson.drift > T(step_tolerance) ? :richardson_step_gate :
        maximum_wronskian_drift > T(wronskian_tolerance) ?
            :wronskian_constancy_gate : :accepted
    return (
        omega=result.omega,
        alpha=alpha,
        reflection=reflection,
        B_gsn=B_gsn,
        step_drift=T(richardson.drift),
        status=accepted ? :accepted : :failed,
        stop_reason=stop_reason,
        metadata=(
            validation=validation,
            base_step=base_h,
            selected_base_step=selected.base_h,
            step_policy=h === nothing ?
                high_damping_diagnostic ?
                    :single_high_damping_diagnostic :
                    :bounded_default_fallback : :explicit,
            step_attempts=attempts,
            radial_sample_requests=sample_requests[],
            radial_unique_samples=length(sample_cache),
            radial_sample_reuses=sample_requests[] - length(sample_cache),
            steps=steps,
            stencil=stencil,
            gsn_real=richardson,
            derivative_direction=:real_frequency,
            derivative_stencil=conditioned_candidate ?
                :five_point_fourth_order : :three_point_second_order,
            convention_gsn=:Aout_over_2omega_alpha,
            step_tolerance=T(step_tolerance),
            incidence_backend=validation.metadata.incidence_backend,
            maximum_wronskian_drift=maximum_wronskian_drift,
            wronskian_tolerance=T(wronskian_tolerance),
            evaluation_policy=validation.metadata.evaluation_policy,
            frequency_projection_drift=
                validation.metadata.frequency_projection_drift,
            evaluation_a=validation.metadata.evaluation_a,
            evaluation_omega=evaluation_omega,
            high_damping_diagnostic,
        ),
    )
end

"""
    qnm_excitation_factor(result::LeaverResult; kwargs...)

Compute the incidence-amplitude derivative with central differences along the
real and imaginary frequency directions and Richardson extrapolation over
`h`, `h/2`, and `h/4`. The returned GSN and Teukolsky factors use their distinct
package conventions and include an explicit conversion-bridge residual.
"""
function qnm_excitation_factor(result::LeaverResult;
        h=nothing,
        incidence_tolerance=1.0e-8,
        direction_tolerance=1.0e-7,
        step_tolerance=1.0e-7,
        bridge_tolerance=1.0e-10,
        wronskian_tolerance=1.0e-8,
        radial_solver=GSN_radial,
        teukolsky_solver=Teukolsky_radial)
    validation = validate_qnm_with_isem(result;
        incidence_tolerance, wronskian_tolerance, radial_solver)
    C = typeof(result.omega)
    T = typeof(real(result.omega))
    conditioned_candidate = _near_as_conditioned_validation_candidate(
        result, validation)
    if validation.status != :accepted && !conditioned_candidate
        nan_complex = complex(T(NaN), T(NaN))
        return ExcitationFactorResult(
            result.omega, nan_complex, nan_complex, nan_complex,
            nan_complex, nan_complex, nan_complex, T(Inf), T(Inf),
            T(Inf), :failed, :isem_incidence_not_accepted,
            (validation=validation, stencil=NamedTuple[]))
    end
    base_h = h === nothing ? T(1.0e-4) *
        max(one(T), abs(result.omega)) : T(h)
    base_h > zero(T) || throw(ArgumentError("h must be positive."))
    steps = T[base_h, base_h / 2, base_h / 4]
    stencil = NamedTuple[]
    gsn_real = C[]
    gsn_imag = C[]
    teuk_real = C[]
    teuk_imag = C[]

    for step in steps
        plus = _amplitude_pair(result, result.omega + step;
            gsn_solver=radial_solver, teukolsky_solver)
        minus = _amplitude_pair(result, result.omega - step;
            gsn_solver=radial_solver, teukolsky_solver)
        plus_i = _amplitude_pair(result, result.omega + im * step;
            gsn_solver=radial_solver, teukolsky_solver)
        minus_i = _amplitude_pair(result, result.omega - im * step;
            gsn_solver=radial_solver, teukolsky_solver)
        push!(gsn_real, (plus.gsn_incidence - minus.gsn_incidence) /
            (2 * step))
        push!(gsn_imag, (plus_i.gsn_incidence - minus_i.gsn_incidence) /
            (2im * step))
        push!(teuk_real,
            (plus.teukolsky_incidence - minus.teukolsky_incidence) /
            (2 * step))
        push!(teuk_imag,
            (plus_i.teukolsky_incidence - minus_i.teukolsky_incidence) /
            (2im * step))
        for (direction, sign, row) in (
                (:real, :plus, plus), (:real, :minus, minus),
                (:imaginary, :plus, plus_i),
                (:imaginary, :minus, minus_i))
            push!(stencil, merge(row, (step=step, direction=direction,
                sign=sign)))
        end
    end

    rg = _richardson_derivative(gsn_real)
    ig = _richardson_derivative(gsn_imag)
    rt = _richardson_derivative(teuk_real)
    it = _richardson_derivative(teuk_imag)
    alpha = (rg.value + ig.value) / 2
    alpha_teukolsky = (rt.value + it.value) / 2
    if conditioned_candidate
        validation = _condition_near_as_validation(
            result, validation, alpha)
    end
    direction_scale = max(one(T), abs(rg.value), abs(ig.value),
        abs(rt.value), abs(it.value))
    direction_drift = T(max(abs(rg.value - ig.value),
        abs(rt.value - it.value)) / direction_scale)
    step_drift = T(max(rg.drift, ig.drift, rt.drift, it.drift))

    root_pair = _amplitude_pair(result, result.omega;
        gsn_solver=radial_solver, teukolsky_solver)
    reflection = root_pair.gsn_reflection
    reflection_teukolsky = root_pair.teukolsky_reflection
    evaluation_a = root_pair.evaluation_a
    evaluation_omega = root_pair.evaluation_omega
    B_gsn = reflection / (2 * evaluation_omega * alpha)
    B_teukolsky = reflection_teukolsky /
        (2im * evaluation_omega * alpha_teukolsky)
    lambda = root_pair.gsn_lambda
    incidence_conversion = ConversionFactors.Binc(
        result.mode.s, result.mode.m, evaluation_a, evaluation_omega, lambda)
    reflection_conversion = ConversionFactors.Bref(
        result.mode.s, result.mode.m, evaluation_a, evaluation_omega, lambda)
    transmission_conversion = ConversionFactors.Btrans(
        result.mode.s, result.mode.m, evaluation_a, evaluation_omega, lambda)
    expected_alpha_teukolsky =
        incidence_conversion / transmission_conversion * alpha
    expected_reflection_teukolsky =
        reflection_conversion / transmission_conversion * reflection
    expected_B_teukolsky =
        reflection_conversion / (im * incidence_conversion) * B_gsn
    bridge_scale = max(one(T), abs(alpha_teukolsky),
        abs(expected_alpha_teukolsky), abs(reflection_teukolsky),
        abs(expected_reflection_teukolsky), abs(B_teukolsky),
        abs(expected_B_teukolsky))
    bridge_residual = T(max(
        abs(alpha_teukolsky - expected_alpha_teukolsky),
        abs(reflection_teukolsky - expected_reflection_teukolsky),
        abs(B_teukolsky - expected_B_teukolsky)) / bridge_scale)
    maximum_wronskian_drift = T(maximum(
        row.wronskian_drift for row in stencil))
    accepted = validation.status == :accepted &&
        direction_drift <= T(direction_tolerance) &&
        step_drift <= T(step_tolerance) &&
        bridge_residual <= T(bridge_tolerance) &&
        maximum_wronskian_drift <= T(wronskian_tolerance)
    stop_reason = validation.status != :accepted ? validation.stop_reason :
        direction_drift > T(direction_tolerance) ?
        :derivative_direction_gate :
        step_drift > T(step_tolerance) ? :richardson_step_gate :
        bridge_residual > T(bridge_tolerance) ? :normalization_bridge_gate :
        maximum_wronskian_drift > T(wronskian_tolerance) ?
            :wronskian_constancy_gate :
        :accepted
    return ExcitationFactorResult(
        result.omega, alpha, alpha_teukolsky, reflection,
        reflection_teukolsky, B_gsn, B_teukolsky,
        direction_drift, step_drift, bridge_residual,
        accepted ? :accepted : :failed, stop_reason,
        (
            validation=validation,
            steps=steps,
            stencil=stencil,
            gsn_real=rg,
            gsn_imag=ig,
            teukolsky_real=rt,
            teukolsky_imag=it,
            incidence_conversion=incidence_conversion,
            reflection_conversion=reflection_conversion,
            transmission_conversion=transmission_conversion,
            expected_alpha_teukolsky=expected_alpha_teukolsky,
            expected_reflection_teukolsky=expected_reflection_teukolsky,
            expected_B_teukolsky=expected_B_teukolsky,
            convention_gsn=:Aout_over_2omega_alpha,
            convention_teukolsky=:Aout_over_2iomega_alpha,
            direction_tolerance=T(direction_tolerance),
            step_tolerance=T(step_tolerance),
            bridge_tolerance=T(bridge_tolerance),
            incidence_backend=root_pair.incidence_backend,
            maximum_wronskian_drift=maximum_wronskian_drift,
            wronskian_tolerance=T(wronskian_tolerance),
            evaluation_policy=root_pair.evaluation_policy,
            frequency_projection_drift=
                root_pair.frequency_projection_drift,
            evaluation_a=evaluation_a,
            evaluation_omega=evaluation_omega,
        ),
    )
end
