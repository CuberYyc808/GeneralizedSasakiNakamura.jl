module InfinityLargeFrequencyExpansion

using ..CoefficientTables: CoefficientSet
using ..OrdinaryPointExpansion:
    endpoint_ab_series,
    ordinary_ab_series,
    poly_triple,
    poly_value
using ..LocalSolutionAtInfinity: infinity_exponents

export InfinityLargeFrequencyExpansionBranch, InfinityLargeFrequencyExpansionPatch
export large_frequency_expansion_infinity_single, large_frequency_expansion_infinity_pair, large_frequency_expansion_infinity_reach, large_frequency_expansion_infinity_reach_pair
export large_frequency_expansion_infinity_anchor, large_frequency_expansion_infinity_anchor_pair
export large_frequency_expansion_infinity_value, large_frequency_expansion_infinity_state

# Infinity patches for the large-frequency route. Each branch is represented as
# X = exp(sigma / y) y^rho x^alpha sum c_n y^n (y = 1 - x), where the x^alpha factor
# carries the horizon phase, and is used from the infinity seed toward the matching
# point as far as it passes the tail, ODE-residual and nested-order checks.

const LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER = 24
const LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP = 8
const LARGE_FREQUENCY_EXPANSION_INFINITY_MID_ORDER = 64
const LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER = 96
const LARGE_FREQUENCY_EXPANSION_INFINITY_TAIL = 1e-10
const LARGE_FREQUENCY_EXPANSION_INFINITY_RESIDUAL = 1e-14
const LARGE_FREQUENCY_EXPANSION_INFINITY_CHECK = 1e-10
const LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_ORDER = 48
const LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_START = 16
const LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_REJECT = 1e-2

# Asymptotic series of one branch (:out or :in) at infinity: exponents rho, sigma,
# alpha and the coefficients c_n in y = 1 - x.
struct InfinityLargeFrequencyExpansionBranch
    branch::Symbol
    rho::ComplexF64
    sigma::ComplexF64
    alpha::ComplexF64
    coeffs::Vector{ComplexF64}
end

# A branch used on [center_x, next_x] with its matched scale; evaluate with
# `large_frequency_expansion_infinity_state` / `large_frequency_expansion_infinity_value`.
struct InfinityLargeFrequencyExpansionPatch
    center_x::Float64
    next_x::Float64
    plan::InfinityLargeFrequencyExpansionBranch
    scale::ComplexF64
end

# Laurent series of the infinity A and B (endpoint_ab_series, group :I, with order + 5 and order + 8 terms), or
# `nothing` if a denominator vanishes identically.
function _endpoint_ab(coefficients, order)
    endpoint = coefficients.infinity
    (all(iszero, endpoint.A.denominator) || all(iszero, endpoint.B.denominator)) && return nothing
    return endpoint_ab_series(coefficients, :I, order + 3)
end

@inline function _complex_laurent(values, first_power, power)
    index = power - first_power + 1
    return 1 <= index <= length(values) ? values[index] : ComplexF64(0)
end

# Recurrence for the c_n of one branch in binary64 (it agrees with the former double-double recurrence to
# rounding; residual floor 2e-16).
function _fast_branch(avec, bvec, params, branch, order)
    rho, sigma = infinity_exponents(params, branch)
    sign = branch == :out ? 1.0 : -1.0
    horizon = (2 * params.omega * (1 + params.kappa) -
        params.a * params.m) / (2 * params.kappa)
    alpha = ComplexF64(1im * sign * horizon)
    a(power) = _complex_laurent(avec, -1, power)
    b(power) = _complex_laurent(bvec, -4, power)
    p(j) = j == 0 ? -2sigma :
        (j == 1 ? 2rho - a(-1) : -a(j - 2) - 2alpha)
    q = zeros(ComplexF64, order + 1)
    @inbounds for j in 0:order
        value = b(j - 2) + sigma * a(j) - rho * a(j - 1) +
            (j == 0 ? rho^2 - rho : ComplexF64(0)) +
            2sigma * alpha
        if j >= 1
            value -= 2rho * alpha
            for power in -1:(j - 2)
                value += alpha * a(power)
            end
        end
        j >= 2 && (value += (alpha^2 - alpha) * (j - 1))
        q[j + 1] = value
    end
    coeffs = zeros(ComplexF64, order + 1)
    coeffs[1] = 1
    p0 = p(0)
    @inbounds for n in 0:(order - 1)
        numerator = n >= 1 ? (n - 1) * n * coeffs[n + 1] :
            ComplexF64(0)
        for j in 1:(n + 1)
            numerator += (n + 1 - j) * p(j) * coeffs[n - j + 2]
        end
        for j in 0:n
            numerator += q[j + 1] * coeffs[n - j + 1]
        end
        coeffs[n + 2] = -numerator / ((n + 1) * p0)
    end
    return InfinityLargeFrequencyExpansionBranch(branch, rho, sigma, alpha, coeffs)
end

# X, dX/dx and d2X/dx2 from the series truncated at `degree`.
function _state(plan::InfinityLargeFrequencyExpansionBranch, y, degree)
    count = min(Int(degree) + 1, length(plan.coeffs))
    coeffs = @view plan.coeffs[1:count]
    series, derivative_y, second_y = poly_triple(coeffs, y)
    x = 1 - y
    prefactor = exp(plan.sigma / y) * y^plan.rho * x^plan.alpha
    log_derivative = plan.sigma / y^2 - plan.rho / y +
        plan.alpha / x
    log_second = 2 * plan.sigma / y^3 - plan.rho / y^2 -
        plan.alpha / x^2
    return (
        X=prefactor * series,
        dXdx=prefactor * (log_derivative * series - derivative_y),
        d2Xdx2=prefactor * (
            second_y - 2 * log_derivative * derivative_y +
            (log_derivative^2 + log_second) * series),
    )
end

@inline function _error(left, right)
    scale = max(abs(left.X), abs(left.dXdx),
        abs(right.X), abs(right.dXdx), floatmin(Float64))
    return max(abs(left.X - right.X),
        abs(left.dXdx - right.dXdx)) / scale
end

function _tail(plan, y, order=length(plan.coeffs) - 1; width=4)
    tail = 0.0
    @inbounds for degree in max(0, order - width + 1):order
        tail += abs(plan.coeffs[degree + 1] * y^degree)
    end
    return tail /
        max(abs(poly_value(
            @view(plan.coeffs[1:(order + 1)]), y)), eps(Float64))
end

function _ab_value(coefficients, x)
    avec, bvec = ordinary_ab_series(coefficients, x, 2)
    return avec[1], bvec[1]
end

function _residual(state, A, B)
    second = state.d2Xdx2
    first = A * state.dXdx
    zeroth = B * state.X
    return abs(second + first + zeroth) /
        max(abs(second) + abs(first) + abs(zeroth), eps(Float64))
end

# ODE residual in the y variable, from the series and the Laurent A, B at y.
function _y_residual(plan, avec, bvec, y, order)
    count = min(Int(order) + 1, length(plan.coeffs))
    z, dz, ddz = poly_triple(@view(plan.coeffs[1:count]), y)
    inv_y = inv(y)
    inv_x = inv(1 - y)
    log_derivative = -plan.sigma * inv_y^2 + plan.rho * inv_y - plan.alpha * inv_x
    log_second = 2plan.sigma * inv_y^3 - plan.rho * inv_y^2 - plan.alpha * inv_x^2
    A = _laurent_value(avec, -1, y)
    B = _laurent_value(bvec, -4, y)
    second = ddz + 2log_derivative * dz + (log_second + log_derivative^2) * z
    first = -A * (dz + log_derivative * z)
    zeroth = B * z
    scale = abs(second) + abs(first) + abs(zeroth)
    return abs(second + first + zeroth) / max(scale, eps(Float64))
end

function _laurent_value(values, first_power, y)
    value = zero(ComplexF64)
    @inbounds for index in length(values):-1:1
        value = value * y + values[index]
    end
    return first_power < 0 ? value / y^(-first_power) : value * y^first_power
end

function _reach_check(plan, ab, y)
    order = length(plan.coeffs) - 1
    full = _state(plan, y, order)
    nested = _state(plan, y, max(8, order - LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP))
    tail = _tail(plan, y, order)
    residual = _y_residual(plan, ab[1], ab[2], y, order)
    check = _error(full, nested)
    accepted = isfinite(residual) && tail <= LARGE_FREQUENCY_EXPANSION_INFINITY_TAIL &&
        residual <= LARGE_FREQUENCY_EXPANSION_INFINITY_RESIDUAL &&
        check <= LARGE_FREQUENCY_EXPANSION_INFINITY_CHECK
    return (; accepted, full, tail, residual, check)
end

# Largest y up to 1 - match_x at which every plan passes _reach_check: grow
# geometrically from the seed, then refine the first rejection by three bisections.
function _reach(plans, ab, seed_y, match_x)
    target_y = 1 - Float64(match_x)
    target_y > seed_y * (1 + 100eps(Float64)) || return nothing

    # Grow outward from the certified endpoint seed.  This finds the
    # contiguous carrier interval without spending dozens of DD checks in
    # the distant, manifestly invalid part of the asymptotic expansion.
    y = min(target_y, 1.25 * seed_y)
    accepted = nothing
    rejected_y = NaN
    while true
        checks = map(plan -> _reach_check(plan, ab, y), plans)
        if all(check -> check.accepted, checks)
            accepted = (; y, checks)
            y == target_y && return accepted
            next_y = min(target_y, 1.6 * y)
            next_y > y * (1 + 100eps(Float64)) || return accepted
            y = next_y
        else
            rejected_y = y
            break
        end
    end

    accepted === nothing && return nothing
    low = accepted.y
    high = rejected_y
    for _ in 1:3
        y = sqrt(low * high)
        checks = map(plan -> _reach_check(plan, ab, y), plans)
        if all(check -> check.accepted, checks)
            accepted = (; y, checks)
            low = y
        else
            high = y
        end
    end
    return accepted
end

# Least-squares complex factor matching (raw.X, raw.dXdx) to (value, derivative).
function _fit_scale(raw, value, derivative)
    scale = max(abs(raw.X), abs(raw.dXdx), floatmin(Float64))
    x = raw.X / scale
    dx = raw.dXdx / scale
    target = ComplexF64(value) / scale
    target_dx = ComplexF64(derivative) / scale
    denominator = abs2(x) + abs2(dx)
    denominator > floatmin(Float64) || return 1.0 + 0.0im
    return (conj(x) * target + conj(dx) * target_dx) / denominator
end

function _certify(plan, y, order, A, B)
    nested_order = max(8, order - LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP)
    full = _state(plan, y, order)
    nested = _state(plan, y, nested_order)
    tail = _tail(plan, y, order)
    residual = _residual(full, A, B)
    check = _error(full, nested)
    accepted = tail <= LARGE_FREQUENCY_EXPANSION_INFINITY_TAIL && residual <= LARGE_FREQUENCY_EXPANSION_INFINITY_RESIDUAL &&
        check <= LARGE_FREQUENCY_EXPANSION_INFINITY_CHECK
    return (; accepted, full, tail, residual, check)
end

# Cheap order-24 test at the matching point: returns order 24 if accepted, 0 to
# reject, or -1 to defer to _preflight_high.
function _preflight_low(avec, bvec, params, branch, match_x, A, B)
    order = LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER
    y = 1 - Float64(match_x)
    plan = _fast_branch(avec, bvec, params, branch, order)
    nested = _state(plan, y, LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_START)
    state = _state(plan, y, order)
    tail = _tail(plan, y, order)
    residual = _residual(state, A, B)
    check = _error(state, nested)
    score = max(tail, residual, check)
    score = isfinite(score) ? score : Inf
    tail <= LARGE_FREQUENCY_EXPANSION_INFINITY_TAIL && residual <= LARGE_FREQUENCY_EXPANSION_INFINITY_RESIDUAL &&
        check <= LARGE_FREQUENCY_EXPANSION_INFINITY_CHECK && return (; order=LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER, score)
    score > LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_REJECT && return (; order=0, score)
    return (; order=-1, score)
end

# Order-48 score sequence: returns LARGE_FREQUENCY_EXPANSION_INFINITY_MID_ORDER or LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER when the
# scores are small or still falling, and 0 to reject.
function _preflight_high(avec, bvec, params, branch, match_x, A, B,
        score24)
    order = LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_ORDER
    y = 1 - Float64(match_x)
    plan = _fast_branch(avec, bvec, params, branch, order)
    scores = Float64[score24]
    for degree in 32:LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP:order
        state = _state(plan, y, degree)
        nested = _state(plan, y, degree - LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP)
        score = max(_tail(plan, y, degree), _residual(state, A, B),
            _error(state, nested))
        push!(scores, isfinite(score) ? score : Inf)
    end
    minimum(scores) <= 1e-6 && return LARGE_FREQUENCY_EXPANSION_INFINITY_MID_ORDER
    last = scores[end]
    earlier = scores[2]
    last <= 1e-2 && last <= 0.1 * earlier && return LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER
    return 0
end

# Lowest certified truncation of `plan` at y, copied into a new branch.
function _select(plan, y, A, B)
    for order in LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER:LARGE_FREQUENCY_EXPANSION_INFINITY_ORDER_STEP:(length(plan.coeffs) - 1)
        check = _certify(plan, y, order, A, B)
        if check.accepted
            branch = InfinityLargeFrequencyExpansionBranch(
                plan.branch, plan.rho, plan.sigma, plan.alpha,
                copy(@view(plan.coeffs[1:(order + 1)])))
            return (; branch, check.tail, check.residual, check.check)
        end
    end
    return nothing
end

function large_frequency_expansion_infinity_state(patch::InfinityLargeFrequencyExpansionPatch, x)
    state = _state(
        patch.plan, 1 - x, length(patch.plan.coeffs) - 1)
    return (
        X=patch.scale * state.X,
        dXdx=patch.scale * state.dXdx,
        d2Xdx2=patch.scale * state.d2Xdx2,
    )
end

large_frequency_expansion_infinity_value(patch::InfinityLargeFrequencyExpansionPatch, x) = large_frequency_expansion_infinity_state(patch, x).X

# Patch from the infinity seed directly to match_x for one branch, or `nothing`.
function large_frequency_expansion_infinity_single(coefficients::CoefficientSet, branch,
        seed_y, match_x, scale)
    y = 1 - Float64(match_x)
    y > seed_y * (1 + 100eps(Float64)) || return nothing
    A, B = _ab_value(coefficients, match_x)
    low_ab = endpoint_ab_series(
        coefficients, :I, LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER + 2)
    low = _preflight_low(low_ab[1], low_ab[2], coefficients.params,
        branch, match_x, A, B)
    order = low.order
    if order < 0
        high_ab = endpoint_ab_series(
            coefficients, :I, LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_ORDER + 2)
        order = _preflight_high(high_ab[1], high_ab[2],
            coefficients.params, branch, match_x, A, B, low.score)
    end
    order > 0 || return nothing
    ab = _endpoint_ab(coefficients, order)
    ab === nothing && return nothing
    plan = _fast_branch(ab[1], ab[2], coefficients.params, branch, order)
    selected = _select(plan, y, A, B)
    selected === nothing && return nothing
    patch = InfinityLargeFrequencyExpansionPatch(
        1 - Float64(seed_y), Float64(match_x), selected.branch,
        ComplexF64(scale))
    state = large_frequency_expansion_infinity_state(patch, match_x)
    return (; patch, state, selected.tail,
        selected.residual, selected.check)
end

function large_frequency_expansion_infinity_pair(coefficients::CoefficientSet,
        branch1, branch2, seed_y, match_x, scale1, scale2)
    y = 1 - Float64(match_x)
    y > seed_y * (1 + 100eps(Float64)) || return nothing
    A, B = _ab_value(coefficients, match_x)
    low_ab = endpoint_ab_series(
        coefficients, :I, LARGE_FREQUENCY_EXPANSION_INFINITY_MIN_ORDER + 2)
    low1 = _preflight_low(low_ab[1], low_ab[2], coefficients.params,
        branch1, match_x, A, B)
    low2 = _preflight_low(low_ab[1], low_ab[2], coefficients.params,
        branch2, match_x, A, B)
    order1 = low1.order
    order2 = low2.order
    if order1 < 0 || order2 < 0
        high_ab = endpoint_ab_series(
            coefficients, :I, LARGE_FREQUENCY_EXPANSION_INFINITY_PREFLIGHT_ORDER + 2)
        order1 < 0 && (order1 = _preflight_high(
            high_ab[1], high_ab[2], coefficients.params, branch1,
            match_x, A, B, low1.score))
        order2 < 0 && (order2 = _preflight_high(
            high_ab[1], high_ab[2], coefficients.params, branch2,
            match_x, A, B, low2.score))
    end
    (order1 > 0 && order2 > 0) || return nothing
    ab = _endpoint_ab(coefficients, max(order1, order2))
    ab === nothing && return nothing
    plan1 = _fast_branch(ab[1], ab[2], coefficients.params, branch1, order1)
    plan2 = _fast_branch(ab[1], ab[2], coefficients.params, branch2, order2)
    selected1 = _select(plan1, y, A, B)
    selected2 = _select(plan2, y, A, B)
    (selected1 !== nothing && selected2 !== nothing) || return nothing
    patch1 = InfinityLargeFrequencyExpansionPatch(
        1 - Float64(seed_y), Float64(match_x), selected1.branch,
        ComplexF64(scale1))
    patch2 = InfinityLargeFrequencyExpansionPatch(
        1 - Float64(seed_y), Float64(match_x), selected2.branch,
        ComplexF64(scale2))
    return (
        patch1,
        patch2,
        state1=large_frequency_expansion_infinity_state(patch1, match_x),
        state2=large_frequency_expansion_infinity_state(patch2, match_x),
        tail=max(selected1.tail, selected2.tail),
        residual=max(selected1.residual, selected2.residual),
        check=max(selected1.check, selected2.check),
    )
end

# Patch from the seed as far toward match_x as the series is certified, scaled to
# match the given seed value and derivative.
function large_frequency_expansion_infinity_reach(coefficients::CoefficientSet, branch,
        seed_y, match_x, seed_X, seed_dXdx)
    y = 1 - Float64(match_x)
    y > seed_y * (1 + 100eps(Float64)) || return nothing
    ab = _endpoint_ab(coefficients, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    ab === nothing && return nothing
    plan = _fast_branch(ab[1], ab[2], coefficients.params,
        branch, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    selected = _reach((plan,), ab, seed_y, match_x)
    selected === nothing && return nothing
    raw_seed = _state(plan, seed_y, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    scale = _fit_scale(raw_seed, seed_X, seed_dXdx)
    patch = InfinityLargeFrequencyExpansionPatch(1 - Float64(seed_y),
        1 - selected.y, plan, scale)
    check = selected.checks[1]
    state = large_frequency_expansion_infinity_state(patch, 1 - selected.y)
    return (; patch, state, check.tail, check.residual, check.check)
end

function large_frequency_expansion_infinity_reach_pair(coefficients::CoefficientSet,
        branch1, branch2, seed_y, match_x,
        seed_X1, seed_dXdx1, seed_X2, seed_dXdx2)
    y = 1 - Float64(match_x)
    y > seed_y * (1 + 100eps(Float64)) || return nothing
    ab = _endpoint_ab(coefficients, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    ab === nothing && return nothing
    plan1 = _fast_branch(ab[1], ab[2], coefficients.params,
        branch1, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    plan2 = _fast_branch(ab[1], ab[2], coefficients.params,
        branch2, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    selected = _reach((plan1, plan2), ab, seed_y, match_x)
    selected === nothing && return nothing
    raw1 = _state(plan1, seed_y, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    raw2 = _state(plan2, seed_y, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    scale1 = _fit_scale(raw1, seed_X1, seed_dXdx1)
    scale2 = _fit_scale(raw2, seed_X2, seed_dXdx2)
    patch1 = InfinityLargeFrequencyExpansionPatch(1 - Float64(seed_y),
        1 - selected.y, plan1, scale1)
    patch2 = InfinityLargeFrequencyExpansionPatch(1 - Float64(seed_y),
        1 - selected.y, plan2, scale2)
    check1, check2 = selected.checks
    return (
        patch1,
        patch2,
        state1=large_frequency_expansion_infinity_state(patch1, 1 - selected.y),
        state2=large_frequency_expansion_infinity_state(patch2, 1 - selected.y),
        tail=max(check1.tail, check2.tail),
        residual=max(check1.residual, check2.residual),
        check=max(check1.check, check2.check),
    )
end

# As large_frequency_expansion_infinity_reach with a given scale, re-checking the hand-off point after the
# round trip y -> x -> y.
function large_frequency_expansion_infinity_anchor(coefficients::CoefficientSet, branch,
        seed_y, match_x, scale)
    # Round the seed through x = 1 - y once, so that the y used for the checks
    # and 1 - patch.center_x are the same floating-point number.
    canonical_seed = 1.0 - (1.0 - Float64(seed_y))
    target_y = 1.0 - Float64(match_x)
    target_y > canonical_seed * (1 + 100eps(Float64)) || return nothing
    ab = _endpoint_ab(coefficients, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    ab === nothing && return nothing
    plan = _fast_branch(ab[1], ab[2], coefficients.params,
        branch, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    selected = _reach((plan,), ab, canonical_seed, match_x)
    selected === nothing && return nothing
    next_x = 1.0 - selected.y
    handoff_y = 1.0 - next_x
    handoff_y > canonical_seed * (1 + 100eps(Float64)) || return nothing
    check = _reach_check(plan, ab, handoff_y)
    check.accepted || return nothing
    patch = InfinityLargeFrequencyExpansionPatch(
        1.0 - canonical_seed, next_x, plan, ComplexF64(scale))
    seed = large_frequency_expansion_infinity_state(patch, patch.center_x)
    state = large_frequency_expansion_infinity_state(patch, patch.next_x)
    return (; patch, seed, state, check.tail,
        check.residual, check.check)
end

function large_frequency_expansion_infinity_anchor_pair(coefficients::CoefficientSet,
        branch1, branch2, seed_y, match_x, scale1, scale2)
    # Round the seed through x = 1 - y once, so that the y used for the checks
    # and 1 - patch.center_x are the same floating-point number.
    canonical_seed = 1.0 - (1.0 - Float64(seed_y))
    target_y = 1.0 - Float64(match_x)
    target_y > canonical_seed * (1 + 100eps(Float64)) || return nothing
    ab = _endpoint_ab(coefficients, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    ab === nothing && return nothing
    plan1 = _fast_branch(ab[1], ab[2], coefficients.params,
        branch1, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    plan2 = _fast_branch(ab[1], ab[2], coefficients.params,
        branch2, LARGE_FREQUENCY_EXPANSION_INFINITY_MAX_ORDER)
    selected = _reach((plan1, plan2), ab, canonical_seed, match_x)
    selected === nothing && return nothing
    next_x = 1.0 - selected.y
    handoff_y = 1.0 - next_x
    handoff_y > canonical_seed * (1 + 100eps(Float64)) || return nothing
    check1 = _reach_check(plan1, ab, handoff_y)
    check2 = _reach_check(plan2, ab, handoff_y)
    (check1.accepted && check2.accepted) || return nothing
    patch1 = InfinityLargeFrequencyExpansionPatch(
        1.0 - canonical_seed, next_x, plan1, ComplexF64(scale1))
    patch2 = InfinityLargeFrequencyExpansionPatch(
        1.0 - canonical_seed, next_x, plan2, ComplexF64(scale2))
    return (
        patch1,
        patch2,
        seed1=large_frequency_expansion_infinity_state(patch1, patch1.center_x),
        seed2=large_frequency_expansion_infinity_state(patch2, patch2.center_x),
        state1=large_frequency_expansion_infinity_state(patch1, patch1.next_x),
        state2=large_frequency_expansion_infinity_state(patch2, patch2.next_x),
        tail=max(check1.tail, check2.tail),
        residual=max(check1.residual, check2.residual),
        check=max(check1.check, check2.check),
    )
end

end
