function _gsn_amplitude_data(result::LeaverResult, omega;
        radial_solver=GSN_radial)
    mode = result.mode
    if _use_wronskian_incidence(result, omega)
        pair = _wronskian_incidence(result, omega; radial_solver)
        inputs = pair.inputs
        incidence, reflection = pair.incidence, pair.reflection
        lambda, method, normalization = pair.lambda, pair.method, pair.normalization
        backend, drift = pair.backend, pair.wronskian_drift
    else
        inputs = _radial_evaluation_inputs(result, omega)
        solution = radial_solver(
            mode.s, mode.l, mode.m, inputs.a, inputs.omega, IN)
        incidence, reflection = solution.incidence_amplitude, solution.reflection_amplitude
        lambda, method = solution.mode.lambda, solution.method
        normalization = solution.normalization_convention
        backend, drift = :direct_asymptotic_amplitude, zero(real(abs(incidence)))
    end
    return (;
        omega, incidence, reflection, lambda, method, normalization,
        evaluation_a=inputs.a,
        evaluation_omega=inputs.omega,
        evaluation_policy=inputs.policy,
        frequency_projection_drift=inputs.projection_drift,
        incidence_backend=backend, wronskian_drift=drift,
    )
end

# Two-level Richardson extrapolation of derivatives at steps h, h/2, h/4 with
# leading error O(h^order); `drift` compares the two first-level values.
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
    data = _gsn_amplitude_data(result, omega; radial_solver)
    return (;
        omega, incidence=data.incidence, reflection=data.reflection,
        wronskian_drift=data.wronskian_drift,
        incidence_backend=data.incidence_backend,
        evaluation_policy=data.evaluation_policy,
        frequency_projection_drift=data.frequency_projection_drift,
    )
end

function _cached_gsn_amplitude_sample!(cache, request_count,
        result::LeaverResult, omega, radial_solver)
    request_count[] += 1
    return get!(cache, omega) do
        _gsn_amplitude_sample(result, omega; radial_solver)
    end
end

# Three-point central differences of Ain at steps h, h/2, h/4 along real (direction = 1) or imaginary (direction = im)
# frequency, Richardson-extrapolated.
function _gsn_derivative_attempt(
        result::LeaverResult, base_h, radial_solver, cache, request_count;
        normalization_pole=nothing, root_incidence=zero(result.omega), direction=1)
    C = typeof(result.omega)
    T = typeof(real(result.omega))
    steps = T[base_h, base_h / 2, base_h / 4]
    derivatives = C[]
    stencil = NamedTuple[]
    for step in steps
        dstep = direction * step
        plus = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega + dstep,
            radial_solver)
        minus = _cached_gsn_amplitude_sample!(
            cache, request_count, result, result.omega - dstep,
            radial_solver)
        derivative = if normalization_pole === nothing
            (plus.incidence - minus.incidence) / (2 * dstep)
        else
            # Differentiate f(z)=(z-zH)*Ain(z), then undo the product rule.
            # This removes the nearby Frobenius normalization pole, not a QNM.
            distance = result.omega - normalization_pole
            ((distance + dstep)*plus.incidence -
                (distance - dstep)*minus.incidence) / (2dstep) - root_incidence
        end
        normalization_pole === nothing ||
            (derivative /= result.omega - normalization_pole)
        push!(derivatives, derivative)
        label = direction == 1 ? :real : :imaginary
        push!(stencil, merge(plus, (step=step, direction=label, sign=:plus)))
        push!(stencil, merge(minus, (step=step, direction=label, sign=:minus)))
    end
    return (base_h=T(base_h), steps, derivatives, stencil, richardson=_richardson_derivative(derivatives),)
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
            push!(stencil, merge(row, (step=step, direction=:real, sign=offset < 0 ? :minus : :plus, offset=offset,)))
        end
    end
    return (base_h=T(base_h), steps, derivatives, stencil, richardson=_richardson_derivative(derivatives, 4),)
end

# True when the only failure is the scaled-incidence gate on the n = 8 unconventional near-algebraically-special
# root accepted on root-position stability; such roots are re-gated on the inferred root shift instead
# (_condition_near_as_validation).
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
            standard_incidence_status=validation.status, standard_incidence_stop_reason=validation.stop_reason,
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
        validation.incidence, validation.reflection, validation.transmission, validation.scaled_incidence,
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

# Exact-extremal complex-path excitation factor: alpha = dAin/domega from the variational
# solve made during validation, so no finite-difference stencil is needed. `step_drift` carries
# the root offset |Ain/alpha| relative to max(1, |omega|), which bounds how far the validated
# frequency is from the zero of Ain on this path.
function _complex_path_excitation_factor(result::LeaverResult, validation, step_tolerance)
    T = typeof(step_tolerance)
    alpha = validation.metadata.incidence_derivative
    evaluation_omega = validation.metadata.evaluation_omega
    reflection = validation.reflection
    B_gsn = reflection / (2 * evaluation_omega * alpha)
    drift = T(abs(validation.incidence / alpha) / max(one(T), abs(evaluation_omega)))
    finite = isfinite(alpha) && isfinite(reflection) && isfinite(B_gsn)
    accepted = finite && drift <= step_tolerance
    return (
        omega=result.omega,
        alpha=alpha,
        reflection=reflection,
        B_gsn=B_gsn,
        step_drift=drift,
        status=accepted ? :accepted : :failed,
        stop_reason=!finite ? :nonfinite_gsn_excitation :
            accepted ? :accepted : :richardson_step_gate,
        metadata=(
            validation=validation,
            base_step=zero(T),
            selected_base_step=zero(T),
            step_policy=:variational_frequency_derivative,
            step_attempts=NamedTuple[],
            radial_sample_requests=0,
            radial_unique_samples=0,
            radial_sample_reuses=0,
            steps=T[],
            stencil=NamedTuple[],
            gsn_real=(value=alpha, first=alpha, second=alpha, drift=drift),
            derivative_direction=:variational,
            normalization_pole=nothing,
            derivative_stencil=:variational_equation,
            convention_gsn=:Aout_over_2omega_alpha,
            step_tolerance=step_tolerance,
            incidence_backend=validation.metadata.incidence_backend,
            maximum_wronskian_drift=T(validation.metadata.wronskian_drift),
            wronskian_tolerance=T(validation.metadata.wronskian_tolerance),
            evaluation_policy=validation.metadata.evaluation_policy,
            frequency_projection_drift=
                validation.metadata.frequency_projection_drift,
            evaluation_a=validation.metadata.evaluation_a,
            evaluation_omega=evaluation_omega,
            high_damping_diagnostic=false,
        ),
    )
end

# GSN-only excitation factor used by the compact `qnm` result: real-direction
# Richardson derivative, no Teukolsky values. `direction_tolerance` and
# `bridge_tolerance` are accepted so that the same `excitation_options` serve
# `qnm_excitation_factor`; they are not used here.
function _qnm_richardson_excitation_factor(result::LeaverResult;
        h=nothing,
        incidence_tolerance=1.0e-8,
        step_tolerance=1.0e-7,
        wronskian_tolerance=1.0e-8,
        radial_solver=GSN_radial,
        direction_tolerance=nothing,
        bridge_tolerance=nothing)
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
    # The exact-extremal complex-path amplitudes carry a Float64 floor near 1e-8 of the
    # reflection; a wider fourth-order stencil keeps the difference quotient above it.
    complex_path = validation.metadata.incidence_backend == :extremal_complex_path
    if validation.status != :accepted && !any_conditioned_candidate
        nan_complex = complex(T(NaN), T(NaN))
        return (
            omega=result.omega, alpha=nan_complex, reflection=nan_complex, B_gsn=nan_complex, step_drift=T(Inf),
            status=:failed, stop_reason=:isem_incidence_not_accepted,
            metadata=(validation=validation, stencil=NamedTuple[], derivative_direction=:real_frequency),
        )
    end

    if complex_path && h === nothing && validation.status == :accepted &&
            hasproperty(validation.metadata, :incidence_derivative) &&
            !ismissing(validation.metadata.incidence_derivative)
        return _complex_path_excitation_factor(result, validation, T(step_tolerance))
    end

    normalization_pole = nothing
    surface_gravity = zero(T)
    if abs(result.a) < 1 && imag(result.omega) < -1 && !any_conditioned_candidate
        q = sqrt(1-result.a^2)
        surface_gravity = q / (2*(1+q))
        j = round(Int, -imag(result.omega)/surface_gravity)
        pole = result.mode.m*result.a/(2*(1+q)) - im*j*surface_gravity
        abs(result.omega-pole) < surface_gravity/8 && (normalization_pole = pole)
    end
    base_h = h === nothing ?
        normalization_pole !== nothing ? T(0.02)*surface_gravity :
        conditioned_candidate ? T(3.2e-3) :
        complex_path ? T(3.0e-3) * max(one(T), abs(result.omega)) :
        T(1.0e-4) * max(one(T), abs(result.omega)) : T(h)
    base_h > zero(T) || throw(ArgumentError("h must be positive."))
    step_factors = h === nothing && high_damping_diagnostic ? T[one(T)] :
        h === nothing ?
        conditioned_candidate ? T[one(T), T(0.5), T(2), T(0.25), T(4)] :
        complex_path ? T[one(T), T(0.5), T(2)] :
        T[one(T), T(2), T(0.5), T(4), T(0.25)] : T[one(T)]
    attempts = NamedTuple[]
    selected = nothing
    selected_score = T(Inf)
    sample_cache = Dict{C,Any}()
    sample_requests = Ref(0)
    validation_wronskian_drift = T(validation.metadata.wronskian_drift)
    stencils = [(factor=factor, five_point=conditioned_candidate || complex_path)
        for factor in step_factors]
    if h === nothing && !any_conditioned_candidate && !complex_path &&
            normalization_pole === nothing
        # A wider fourth-order stencil reduces cancellation when the small
        # central differences reach the radial evaluator's roundoff floor.
        push!(stencils, (factor=T(8), five_point=true))
    end
    for (factor, five_point) in stencils
        attempt = five_point ?
            _gsn_five_point_real_derivative_attempt(
                result, base_h * factor, radial_solver, sample_cache,
                sample_requests) :
            _gsn_derivative_attempt(
                result, base_h * factor, radial_solver, sample_cache,
                sample_requests; normalization_pole,
                root_incidence=validation.incidence)
        richardson_attempt = attempt.richardson
        maximum_wronskian_drift = T(max(
            validation_wronskian_drift, maximum(row.wronskian_drift for row in attempt.stencil),
        ))
        finite_attempt = isfinite(richardson_attempt.value)
        accepted_attempt = finite_attempt &&
            richardson_attempt.drift <= T(step_tolerance) &&
            maximum_wronskian_drift <= T(wronskian_tolerance)
        score = finite_attempt ? max(
            richardson_attempt.drift / T(step_tolerance), maximum_wronskian_drift / T(wronskian_tolerance),
        ) : T(Inf)
        summary = (;
            factor,
            five_point,
            base_h=attempt.base_h,
            step_drift=T(richardson_attempt.drift),
            maximum_wronskian_drift,
            accepted=accepted_attempt,
        )
        push!(attempts, summary)
        if accepted_attempt || selected === nothing || score < selected_score
            selected = merge(attempt, (; maximum_wronskian_drift, five_point))
            selected_score = score
        end
        accepted_attempt && break
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
            normalization_pole=normalization_pole,
            derivative_stencil=selected.five_point ?
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
    _qnm_richardson_full_excitation_factor(result::LeaverResult; direction_tolerance=1e-7, bridge_tolerance=nothing, kwargs...)

Finite-difference route of `qnm_excitation_factor` and of `qnm(...)`, used when the direct route (the GSN equation
integrated together with its frequency derivative at the root) is out of scope or fails its checks (except with
`backend=:variational`), and always with `backend=:richardson`, an explicit `h` or `direction_tolerance`, or a custom
radial solver. `alpha = dAin/domega` is the real-direction Richardson derivative of the compact route
(`_qnm_richardson_excitation_factor`, which takes the other keywords); the Teukolsky values follow from the analytic
conversion with the frequency tangent of the conversion factors, as on the direct route. The derivative is also
taken along imaginary frequency at the selected step, and `direction_drift` is the relative mismatch with `alpha`; it
must not exceed `direction_tolerance` (`nothing` skips this stencil). Returns an `ExcitationFactorResult`.
"""
function _qnm_richardson_full_excitation_factor(result::LeaverResult;
        direction_tolerance=1.0e-7, bridge_tolerance=nothing, kwargs...)
    gsn = _qnm_richardson_excitation_factor(result; kwargs...)
    md = gsn.metadata
    T = typeof(real(result.omega))
    if !hasproperty(md, :evaluation_omega)
        nan = complex(T(NaN), T(NaN))
        return ExcitationFactorResult(result.omega, nan, nan, nan, nan, nan, nan, T(Inf), T(Inf), T(Inf), :failed,
            gsn.stop_reason, md)
    end
    lambda, dlambda = _qnm_lambda_tangent(result.mode, md.evaluation_a, md.evaluation_omega)
    factor = _qnm_variational_teukolsky(result, merge(gsn, (metadata=merge(md, (; lambda, dlambda)),));
        bridge_tolerance)
    (direction_tolerance === nothing || iszero(md.selected_base_step)) && return factor
    radial_solver = get(kwargs, :radial_solver, GSN_radial)
    across = _gsn_derivative_attempt(result, md.selected_base_step, radial_solver, Dict{typeof(result.omega),Any}(),
        Ref(0); normalization_pole=md.normalization_pole, root_incidence=md.validation.incidence, direction=im)
    alpha = gsn.alpha
    drift = T(abs(across.richardson.value - alpha) / max(one(T), abs(alpha), abs(across.richardson.value)))
    gate = factor.status === :accepted && drift > T(direction_tolerance)
    return ISEM._with_fields(factor, (direction_drift=drift, status=gate ? :failed : factor.status,
        stop_reason=gate ? :derivative_direction_gate : factor.stop_reason,
        metadata=merge(factor.metadata, (direction_tolerance=T(direction_tolerance), gsn_imaginary=across.richardson,
            imaginary_stencil=across.stencil))))
end
