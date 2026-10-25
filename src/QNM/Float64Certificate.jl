# Second layer of a Float64 Leaver root: entered only where a Float64 solve would otherwise fail (or, with
# automatic_precision_fallback=true, go to BigFloat). The continued fraction is evaluated with the closed-form D
# coefficients, and a Wilkinson-type bound eta_F on its binary64 rounding error is carried along both recurrences;
# the root position is then certified as delta = (|F| + eta_F) / |F'|. The ordinary evaluation
# (leaver_cf_truncated, _leaver_cf_asymptotic, leaver_radial_residual) is not changed.

# D coefficients of leaver_D_coefficients without the cancellation of alpha (4p - delta) against sigma (about
# 26 eps u^2): kappa = sqrt(1 - a^2), u = (2 omega - m a) / kappa. Also returns the tail constant
# c1 = sqrt(-4 i kappa omega) = sqrt(-(D1 + D2 + D3)).
function _leaver_D_closed_form(omega::ComplexF64, a::Float64, s::Int, m::Int, angular_A)
    kappa = sqrt(1 - a^2)
    u = (2omega - m * a) / kappa
    alpha = 1 + s - im * u - 2im * omega
    D = (1 - s - im * u - 2im * omega,
        -4 + 2im * u + 8im * omega + 4im * kappa * omega,
        3 + s - im * u - 6im * omega,
        -(1 + s) - angular_A + 2im * omega * (2 + kappa) + (11 + 8kappa + kappa^2) * omega^2 +
            u * (im + 2omega * (2 + kappa)),
        alpha * (1 - 4im * omega))
    return D, sqrt(-4im * kappa * omega)
end

# Magnitude sums of the terms of alpha_k, beta_k, gamma_k (leaver_recurrence_coefficients): their rounding.
function _leaver_coefficient_magnitudes(k::Int, D)
    n = Float64(k)
    return (alpha=n^2 + abs(D[1] + 1) * n + abs(D[1]), beta=2n^2 + abs(D[2] + 2) * n + abs(D[4]),
        gamma=n^2 + abs(D[3] - 3) * n + abs(D[5] - D[3] + 2))
end

# F at depth N with its rounding bound: e_k = |alpha_k U_k^2 / gamma_k| e_{k+1} + l_k down the infinite side
# (e_{N+1} the last tail term kept), the same up the terminating side, and
# eta_F = |alpha_p| e_{p+1} + |gamma_p| e_L + 3 eps S with S the largest of the three pivot terms. The local error
# l_k = eps |U_k| (2 + |gamma_k|~ / |gamma_k| + (|beta_k|~ + 2 |alpha_k|~ |U_{k+1}|) / |beta_k - alpha_k U_{k+1}|)
# (|x|~ the magnitude sum of the terms of x) counts the rounding of the coefficients and the cancellation in the
# denominator, which 5 eps |U_k| misses (up to 9x at (-2,2,-2,19), a = 0.5).
function _leaver_floor_value(D, tail_coefficients, n_inv::Int, N::Int)
    epsilon = eps(Float64)
    lower = zero(ComplexF64)
    lower_error = 0.0
    for k in 0:(n_inv - 1)
        c = leaver_recurrence_coefficients(k, D)
        size = _leaver_coefficient_magnitudes(k, D)
        denominator = c.beta - c.gamma * lower
        local_error = epsilon * (2 + size.alpha / abs(c.alpha) +
            (size.beta + 2size.gamma * abs(lower)) / abs(denominator))
        lower = c.alpha / denominator
        gain = iszero(c.alpha) ? 0.0 : abs(c.gamma) * abs2(lower) / abs(c.alpha)
        lower_error = gain * lower_error + local_error * abs(lower)
    end
    t = inv(sqrt(Float64(N + 1)))
    upper = evalpoly(t, tail_coefficients)
    upper_error = abs(tail_coefficients[end]) * t^(length(tail_coefficients) - 1)
    for k in N:-1:(n_inv + 1)
        c = leaver_recurrence_coefficients(k, D)
        size = _leaver_coefficient_magnitudes(k, D)
        denominator = c.beta - c.alpha * upper
        local_error = epsilon * (2 + size.gamma / abs(c.gamma) +
            (size.beta + 2size.alpha * abs(upper)) / abs(denominator))
        upper = c.gamma / denominator
        gain = iszero(c.gamma) ? 0.0 : abs(c.alpha) * abs2(upper) / abs(c.gamma)
        upper_error = gain * upper_error + local_error * abs(upper)
    end
    pivot = leaver_recurrence_coefficients(n_inv, D)
    scale = max(abs(pivot.beta), abs(pivot.gamma * lower), abs(pivot.alpha * upper))
    value = pivot.beta - pivot.gamma * lower - pivot.alpha * upper
    eta = abs(pivot.alpha) * upper_error + abs(pivot.gamma) * lower_error + 3epsilon * scale
    return (value=value, eta=eta, scale=scale)
end

# Depth by doubling from `minimum_depth` until |F_N - F_2N| <= eta_F(2N); the smaller depth is kept (near the
# NIA the rounding error grows with N), with that difference added to its bound.
function _leaver_floor_evaluation(mode::QNMMode, a::Float64, omega::ComplexF64; angular_order::Int,
        sheet_id::Symbol, inversion_index::Int, minimum_depth::Int, maximum_depth::Int)
    angular = angular_branch(mode, a, omega; truncation_order=angular_order, sheet_id)
    angular.status in (:continued, :predictor_corrected, :spherical_anchor, :high_precision_refined) ||
        return nothing
    D, c1 = _leaver_D_closed_form(omega, a, mode.s, mode.m, ComplexF64(angular.angular_A))
    coefficients = _leaver_tail_coefficients(D; leading=c1)
    depth = max(minimum_depth, inversion_index + 32)
    current = _leaver_floor_value(D, coefficients, inversion_index, depth)
    while 2depth <= maximum_depth
        next = _leaver_floor_value(D, coefficients, inversion_index, 2depth)
        change = abs(current.value - next.value)
        change <= next.eta && return (value=current.value, eta=current.eta + change, scale=current.scale,
            depth=depth, converged=true, angular=angular)
        depth *= 2
        current = next
    end
    return (value=current.value, eta=current.eta, scale=current.scale, depth=depth, converged=false,
        angular=angular)
end

# Depth ceiling of the second layer: near extremality (|a| >= 0.99, away from the NIA) the 12-term tail needs
# N >> u = |2 omega - m a| / kappa, so max(baseline, 100 u); otherwise the baseline.
function _floor_maximum_depth(mode::QNMMode, a::Float64, omega::ComplexF64, baseline::Int, near_axis::Bool)
    (abs(a) >= 0.99 && !near_axis) || return baseline
    u = abs(2omega - mode.m * a) / sqrt(1 - a^2)
    return max(baseline, min(1_000_000, ceil(Int, 100u)))
end

"""
    _float64_floor_certificate(mode, a, omega; ...)

Second-layer Float64 certificate of the Leaver root next to `omega`: a few Newton steps on the closed-form F
until |F| <= eta_F or no further decrease, then the four-point derivative at h and h/2 (agreeing to
max(1e-4, 2 eta_F / (h |F'|))), and delta = (|F| + eta_F) / |F'_{h/2}|. Status :accepted when
delta <= min(root_tolerance, stability_tolerance) max(1, |omega|), :estimated when delta <= 1e-8 max(1, |omega|),
otherwise :failed.
"""
function _float64_floor_certificate(mode::QNMMode, a::Float64, omega0::ComplexF64; angular_order::Int,
        sheet_id::Symbol, inversion_index::Int, cf_minimum_iterations::Int, cf_maximum_iterations::Int,
        root_tolerance::Float64, stability_tolerance::Float64, derivative_step_scale::Float64=0.5,
        near_axis::Bool=false, maximum_newton_iterations::Int=8)
    maximum_depth = _floor_maximum_depth(mode, a, omega0, cf_maximum_iterations, near_axis)
    clamp_branch = !_negative_imaginary_axis_crossing_exempt(mode)
    evaluate(frequency) = (clamp_branch && _branch_sign(mode) * real(frequency) <= 0) ? nothing :
        _leaver_floor_evaluation(mode, a, frequency; angular_order, sheet_id, inversion_index,
            minimum_depth=cf_minimum_iterations, maximum_depth)
    function stencil(frequency, h)
        values = [evaluate(frequency + k * h) for k in (2, 1, -1, -2)]
        any(isnothing, values) && return complex(NaN, NaN)
        return (-values[1].value + 8values[2].value - 8values[3].value + values[4].value) / (12h)
    end
    failed(reason; kwargs...) = (status=:failed, stop_reason=reason, omega=omega0, kwargs...)
    omega = omega0
    evaluation = evaluate(omega)
    evaluation === nothing && return failed(:float64_certificate_evaluation)
    newton_iterations = 0
    for _ in 1:maximum_newton_iterations
        abs(evaluation.value) <= evaluation.eta && break
        h = _stencil_step(mode, omega, derivative_step_scale)
        derivative = stencil(omega, h)
        (isfinite(derivative) && !iszero(derivative)) || break
        trial = omega - evaluation.value / derivative
        trial_evaluation = evaluate(trial)
        (trial_evaluation !== nothing && abs(trial_evaluation.value) < abs(evaluation.value)) || break
        omega = trial
        evaluation = trial_evaluation
        newton_iterations += 1
    end
    bound = max(1.0, abs(omega))
    drift = abs(omega - omega0) / bound
    # F' from the four-point stencil at h and h/2; where they disagree (F varying on a scale below h, as for the
    # near-extremal zero-damping roots) the step is halved, at most three times.
    h = _stencil_step(mode, omega, derivative_step_scale)
    eta = evaluation.eta
    derivative_h = stencil(omega, h)
    derivative = stencil(omega, h / 2)
    for _ in 1:3
        agreed = isfinite(derivative_h) && isfinite(derivative) && !iszero(derivative) &&
            abs(derivative_h - derivative) <= max(1.0e-4, 2eta / (h * abs(derivative))) * abs(derivative)
        agreed && break
        h /= 2
        derivative_h = derivative
        derivative = stencil(omega, h / 2)
    end
    details = (omega=omega, value=evaluation.value, eta_F=eta, scale=evaluation.scale,
        c_eff=eta / (eps(Float64) * evaluation.scale), depth=evaluation.depth,
        depth_converged=evaluation.converged, newton_iterations, drift, angular=evaluation.angular)
    (isfinite(derivative_h) && isfinite(derivative) && !iszero(derivative)) ||
        return failed(:float64_certificate_derivative; details..., delta=Inf)
    delta = (abs(evaluation.value) + eta) / abs(derivative)
    details = merge(details, (delta=delta, derivative=derivative))
    agreement = max(1.0e-4, 2eta / (h * abs(derivative)))
    abs(derivative_h - derivative) <= agreement * abs(derivative) ||
        return failed(:float64_certificate_derivative; details...)
    abs(evaluation.value) <= eta || return failed(:float64_certificate_above_floor; details...)
    delta <= 1.0e-6 * bound && drift <= 1.0e-6 ||
        return failed(:float64_certificate_position; details...)
    # Next to the NIA the certified disc must stay on the physical side of the tail cut.
    near_axis && clamp_branch && _branch_sign(mode) * real(omega) < 4delta &&
        return failed(:float64_certificate_axis_distance; details...)
    delta <= min(root_tolerance, stability_tolerance) * bound &&
        return (status=:accepted, stop_reason=:float64_residual_floor_position_certificate, details...)
    delta <= 1.0e-8 * bound && return (status=:estimated, stop_reason=:float64_position_estimate, details...)
    return failed(:float64_certificate_position; details...)
end

# The certificate at one of the solver's failure sites (_qnm_frequency_labeled), with the settings of the polish
# that produced omega; next to the NIA (a re-emerged segment within 2h of the axis) the depth is not raised.
function _site_float64_certificate(mode::QNMMode, a, omega, sheet_id::Symbol, angular_order::Int,
        inversion_index::Int, cf_minimum_iterations::Int, cf_maximum_iterations::Int, root_tolerance,
        stability_tolerance, segment::Int)
    a64 = Float64(a)
    omega64 = ComplexF64(omega)
    scale = Float64(_near_extremal_derivative_scale(mode, a64))
    near_axis = segment >= 1 && _branch_sign(mode) * real(omega64) < 2 * _root_derivative_step(Float64, omega64) * scale
    return _float64_floor_certificate(mode, a64, omega64; angular_order, sheet_id, inversion_index,
        cf_minimum_iterations, cf_maximum_iterations, root_tolerance=Float64(root_tolerance),
        stability_tolerance=Float64(stability_tolerance), derivative_step_scale=0.5scale, near_axis)
end

# LeaverResult of a second-layer certificate with status :accepted or :estimated.
function _float64_certificate_result(mode::QNMMode, certificate, a::Float64, convention, overtone_index,
        inversion_index, provenance)
    angular = certificate.angular
    return LeaverResult(mode, convention, overtone_index, inversion_index, a, certificate.omega,
        angular.angular_A, angular.lambda, angular.mixing, certificate.value, certificate.eta_F,
        certificate.depth, abs(certificate.value), Float64(angular.residual), 53, certificate.status,
        certificate.stop_reason, merge(provenance, (acceptance_basis=:float64_position_certificate,
            float64_certificate=_certificate_provenance(certificate))))
end

# A root certified at the final gate of _qnm_frequency_labeled. As there, a root reached by the jump from the
# sequence anchor is only an estimate when the jump exceeds a quarter of the near-extremal level spacing.
function _gate_certificate_result(mode::QNMMode, certificate, a, convention, overtone_index, inversion_index,
        terminal_rows, accepted_roots, provenance)
    a64 = Float64(a)
    if certificate.status == :accepted && !isempty(terminal_rows) && length(accepted_roots) >= 2
        kappa = sqrt(max(0.0, 1 - a64^2)) / (2 * (1 + sqrt(max(0.0, 1 - a64^2))))
        if abs(last(terminal_rows).candidate - accepted_roots[end-1]) > kappa / 4
            certificate = merge(certificate, (status=:estimated,
                stop_reason=:terminal_extrapolation_label_uncertain))
        end
    end
    return _float64_certificate_result(mode, certificate, a64, convention, overtone_index, inversion_index,
        provenance)
end

_certificate_provenance(certificate) = (status=certificate.status, stop_reason=certificate.stop_reason,
    delta=get(certificate, :delta, missing), eta_F=get(certificate, :eta_F, missing),
    c_eff=get(certificate, :c_eff, missing), depth=get(certificate, :depth, missing),
    drift=get(certificate, :drift, missing))
