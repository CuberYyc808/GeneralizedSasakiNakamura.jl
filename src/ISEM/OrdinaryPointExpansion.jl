module OrdinaryPointExpansion

using ..CoefficientTables:
    CoefficientSet,
    EndpointCoefficientSet,
    RationalCoefficients,
    evaluate_ordinary_coefficients!,
    evaluate_ordinary_variable!
using ...Potentials: sF, sU, sF_delta, sW
using TaylorSeries: Taylor1, getcoeff, derivative

export PQLayout
export pq_layout, split_starts, ab_term_counts, horizon_kplus
export rational_series_from_coefficients!
export endpoint_ab_series!, endpoint_ab_series
export ordinary_ab_series!, ordinary_ab_series
export ordinary_scaled_y_ab_series!
export shifted_ab!, shifted_infinity_ab!
export ordinary_point_coeffs!
export direct_ab_series!, direct_ab_value
export horizon_closed_form_ab_series!
export endpoint_power_lookup
export poly_value, poly_pair, poly_triple
export series_issue, denom_ok

include("TaylorArena.jl")
include("RationalArena.jl")

const _SERIES_MAG_LIMIT = sqrt(floatmax(Float64))
const _SERIES_RATIO_LIMIT = inv(sqrt(eps(Float64)))
const _DENOM_REL_LIMIT = 64eps(Float64)
const _SHIFT_EPS_LIMIT = 1e-13

# The same guard limits in the working floating type; Float64 uses the constants above.
@inline _series_mag_limit(::Type{Float64}) = _SERIES_MAG_LIMIT
@inline _series_mag_limit(::Type{T}) where {T<:AbstractFloat} = sqrt(floatmax(T))
@inline _series_ratio_limit(::Type{Float64}) = _SERIES_RATIO_LIMIT
@inline _series_ratio_limit(::Type{T}) where {T<:AbstractFloat} = inv(sqrt(eps(T)))
@inline _denom_rel_limit(::Type{Float64}) = _DENOM_REL_LIMIT
@inline _denom_rel_limit(::Type{T}) where {T<:AbstractFloat} = 64eps(T)
@inline _working_float(value) = typeof(float(real(value)))

@inline _finite_complex(z) = isfinite(real(z)) && isfinite(imag(z))

# Check the newest series coefficient for non-finite values, over/underflow, or
# two consecutive coefficient ratios above 1/sqrt(eps); returns :none if it is usable.
function series_issue(coeffs, index::Integer, scale::Real=1.0)
    coeff = coeffs[index]
    _finite_complex(coeff) || return :nonfinite_coefficient
    T = _working_float(coeff)

    magnitude = abs(coeff) / max(abs(scale), floatmin(T))
    !isfinite(magnitude) && return :coefficient_too_large
    magnitude > _series_mag_limit(T) && return :coefficient_too_large
    !iszero(coeff) && iszero(magnitude) && return :coefficient_too_small
    0 < magnitude < floatmin(T) && return :coefficient_too_small

    if index >= 3
        previous2 = abs(coeffs[index - 2])
        previous1 = abs(coeffs[index - 1])
        current = abs(coeff)
        if previous2 > 0 && previous1 > 0 && current > 0
            ratio1 = previous1 / previous2
            ratio2 = current / previous1
            growth1 = !isfinite(ratio1) || ratio1 > _series_ratio_limit(T)
            growth2 = !isfinite(ratio2) || ratio2 > _series_ratio_limit(T)
            if growth1 && growth2
                return :consecutive_ratio_growth
            end
        end
    end
    return :none
end

# A recurrence denominator is usable if finite and not small relative to `scale`.
function denom_ok(denominator, scale)
    _finite_complex(denominator) || return false
    T = _working_float(denominator)
    denom_scale = max(abs(scale), floatmin(T))
    return abs(denominator) > _denom_rel_limit(T) * denom_scale
end

function _q0_ok(q0::Union{Real,Complex})
    _finite_complex(q0) && !iszero(q0) || return false
    return _finite_complex(inv(complex(float(q0))))
end

# Other number types (e.g. the Double64 of the arbitrary-precision route): the binary64 inverse is checked.
function _q0_ok(q0)
    _finite_complex(q0) && !iszero(q0) || return false
    return _finite_complex(inv(ComplexF64(q0)))
end

# Lengths of the A and B numerator/denominator polynomials (ap, aq, bp, bq) in a packed P/Q vector.
struct PQLayout
    ap_len::Int
    aq_len::Int
    bp_len::Int
    bq_len::Int
end

# Block lengths of the packed A/B numerator and denominator coefficients for the
# horizon (:H), infinity (:I) and ordinary-point (:O) tables.
function pq_layout(s::Integer, group::Symbol)
    if s == 0
        group == :H && return PQLayout(11, 11, 11, 13)
        group == :I && return PQLayout(10, 10, 9, 9)
        group == :O && return PQLayout(10, 11, 9, 13)
    else
        group == :H && return PQLayout(11, 11, 17, 19)
        group == :I && return PQLayout(10, 10, 15, 15)
        group == :O && return PQLayout(10, 11, 15, 19)
    end
    throw(ArgumentError("unsupported ISEM GSN coefficient group $group."))
end

# 1-based start offsets of the four polynomials in a packed P/Q vector.
function split_starts(layout::PQLayout)
    ap = 1
    aq = ap + layout.ap_len
    bp = aq + layout.aq_len
    bq = bp + layout.bp_len
    return ap, aq, bp, bq
end

# Number of Laurent/Taylor terms of A and B needed for a solution series of
# `order`; endpoint series start at x^-1 and x^-2 (horizon) or y^-1 and y^-4 (infinity).
function ab_term_counts(group::Symbol, order::Integer)
    order >= 0 || throw(ArgumentError("series order must be non-negative."))
    group == :H && return order + 2, order + 3
    group == :I && return order + 2, order + 5
    return order + 1, order + 1
end

function _denominator_leading_shift(values)
    @inbounds for k in 0:(length(values) - 1)
        !iszero(values[k + 1]) && return k
    end
    error("all-zero denominator in ISEM GSN rational P/Q recurrence")
end

function _check_regular_leading_terms(numerator, shift::Integer)
    @inbounds for k in 0:(shift - 1)
        if k < length(numerator) && !iszero(numerator[k + 1])
            error("non-regular ISEM GSN rational P/Q leading terms")
        end
    end
    return nothing
end

function _ordinary_rational_series_from_pq_len!(
    out::AbstractVector{<:Complex},
    out_len::Integer,
    pq::AbstractVector{<:Complex},
    p_start::Integer,
    p_len::Integer,
    q_start::Integer,
    q_len::Integer,
)
    out_len <= length(out) || throw(ArgumentError("output buffer is too short."))
    q0 = pq[q_start]
    _q0_ok(q0) || error("noninvertible q0 in ISEM GSN ordinary P/Q recurrence")
    inv_q0 = inv(q0)
    order = out_len - 1
    @inbounds for n in 0:order
        acc = n < p_len ? pq[p_start + n] : zero(eltype(out))
        for k in 1:min(n, q_len - 1)
            acc -= pq[q_start + k] * out[n - k + 1]
        end
        out[n + 1] = acc * inv_q0
    end
    return out
end

@inline function _shift_source(source, pole_order, index)
    index <= pole_order && return 0.0 + 0.0im
    return ComplexF64(source[index - pole_order])
end

# Taylor shift of a polynomial (times x^pole_order) to the point `center`,
# written into `work` starting at `start`; returns the number of coefficients.
function _shift_poly!(
    work::Vector{<:Number},
    start::Integer,
    source,
    pole_order::Integer,
    center,
)
    count = length(source) + pole_order
    start + count - 1 <= length(work) ||
        throw(ArgumentError("shift workspace is too short."))
    work[start] = _shift_source(source, pole_order, count)
    active = 1
    @inbounds for source_index in (count - 1):-1:1
        work[start + active] = work[start + active - 1]
        for index in active:-1:2
            slot = start + index - 1
            work[slot] = work[slot - 1] + center * work[slot]
        end
        work[start] = _shift_source(source, pole_order, source_index) +
            center * work[start]
        active += 1
    end
    return count
end

# Condition number of the Taylor shift: sum of absolute contributions over the
# magnitude of each shifted coefficient.
function _shift_condition(
    work::Vector{<:Number},
    start::Integer,
    source,
    pole_order::Integer,
    center,
)
    count = length(source) + pole_order
    degree = count - 1
    maximum_condition = 1.0
    @inbounds for n in 0:degree
        scale = 0.0
        weight = 1.0
        for k in n:degree
            scale += abs(_shift_source(source, pole_order, k + 1)) * weight
            k < degree &&
                (weight *= abs(center) * (k + 1) / (k + 1 - n))
        end
        shifted = work[start + n]
        condition = scale / max(abs(shifted), floatmin(Float64))
        isfinite(condition) || return Inf
        maximum_condition = max(maximum_condition, condition)
    end
    return maximum_condition
end

# Series division of the shifted numerator by the shifted denominator; returns
# the largest cancellation ratio met in the recurrence.
function _shift_series!(
    out::Vector{<:Number},
    out_len::Integer,
    work::Vector{<:Number},
    p_start::Integer,
    p_len::Integer,
    q_start::Integer,
    q_len::Integer,
)
    q0 = work[q_start]
    _q0_ok(q0) || return Inf
    inv_q0 = inv(q0)
    maximum_condition = 1.0
    @inbounds for n in 0:(out_len - 1)
        acc = n < p_len ? work[p_start + n] : 0.0 + 0.0im
        scale = abs(acc)
        for k in 1:min(n, q_len - 1)
            term = work[q_start + k] * out[n - k + 1]
            acc -= term
            scale += abs(term)
        end
        out[n + 1] = acc * inv_q0
        condition = scale / max(abs(acc), floatmin(Float64))
        isfinite(condition) || return Inf
        maximum_condition = max(maximum_condition, condition)
    end
    return maximum_condition
end

function _shift_rational!(
    out::Vector{<:Number},
    out_len::Integer,
    work::Vector{<:Number},
    rational::RationalCoefficients,
    pole_order::Integer,
    center,
)
    p_start = 1
    p_len = _shift_poly!(work, p_start, rational.numerator, 0, center)
    p_condition = _shift_condition(
        work, p_start, rational.numerator, 0, center)
    q_start = p_start + p_len
    q_len = _shift_poly!(work, q_start, rational.denominator,
        pole_order, center)
    q_condition = _shift_condition(
        work, q_start, rational.denominator, pole_order, center)
    recurrence_condition = _shift_series!(
        out, out_len, work, p_start, p_len, q_start, q_len)
    return max(p_condition, q_condition, recurrence_condition)
end

# Shift in binary64; returns the estimated relative error eps * condition. Above `eps_limit` the callers
# reject the shift (shifted_ab! / shifted_infinity_ab! return accepted = false) and take the ordinary-point
# tables or their own fallback, so no wider arithmetic is used for binary64 inputs.
function _accurate_shift_rational!(out, terms, work, rational, pole_order, center, eps_limit)
    condition = _shift_rational!(out, terms, work, rational, pole_order, center)
    return eps(Float64) * condition
end

# Second gate of a shift that fails the per-coefficient bound: the same rounding contributions weighted by
# h^n, h the step radius, i.e. the relative error of the series as it is used over one step rather than of
# its worst single coefficient (whose relative error is large where the coefficient itself is small). The
# shifted polynomials and the series are in `work` and `out` from _shift_rational!.
function _weighted_shift_condition(out, terms, work, rational, pole_order, center, h)
    p_len = length(rational.numerator)
    q_len = length(rational.denominator) + pole_order
    q_start = 1 + p_len
    weighted(start, source, order) = begin
        degree = length(source) + order - 1
        numerator = 0.0
        denominator = 0.0
        hn = 1.0
        @inbounds for n in 0:degree
            scale = 0.0
            weight = 1.0
            for k in n:degree
                scale += abs(_shift_source(source, order, k + 1)) * weight
                k < degree && (weight *= abs(center) * (k + 1) / (k + 1 - n))
            end
            numerator += scale * hn
            denominator += abs(work[start + n]) * hn
            hn *= h
        end
        numerator / max(denominator, floatmin(Float64))
    end
    p_condition = weighted(1, rational.numerator, 0)
    q_condition = weighted(q_start, rational.denominator, pole_order)
    q0 = abs(work[q_start])
    numerator = 0.0
    denominator = 0.0
    hn = 1.0
    @inbounds for n in 0:(terms - 1)
        scale = n < p_len ? abs(work[1 + n]) : 0.0
        for k in 1:min(n, q_len - 1)
            scale += abs(work[q_start + k] * out[n - k + 1])
        end
        numerator += scale / q0 * hn
        denominator += abs(out[n + 1]) * hn
        hn *= h
    end
    recurrence_condition = numerator / max(denominator, floatmin(Float64))
    condition = max(p_condition, q_condition, recurrence_condition)
    return isfinite(condition) ? condition : Inf
end

# Step radius at which a shifted series is used (the weight of the second gate): 0.3 min(|c|, |1 - c|).
_shift_weight_radius(center) = 0.3 * min(abs(center), abs(1 - center))

@inline function _valid_shift_center(center)
    finite = isfinite(real(center)) && isfinite(imag(center))
    finite || return false
    if center isa Real
        return 0 < center < 1
    end
    # A rational Taylor shift is not restricted to the unit disk. Its
    # denominator and conditioning checks below determine local validity.
    return !iszero(center) && !isone(center)
end

# Taylor coefficients of A and B about x0 obtained by shifting the endpoint
# rational functions. Returns (accepted, A terms, B terms, error estimate).
function shifted_ab!(
    a_out::Vector{ComplexF64},
    b_out::Vector{ComplexF64},
    work::Vector{ComplexF64},
    endpoint::EndpointCoefficientSet,
    x0,
    order::Integer;
    eps_limit::Real=_SHIFT_EPS_LIMIT,
    radius::Real=_shift_weight_radius(x0),
)
    center = x0 isa Real ? Float64(x0) : ComplexF64(x0)
    _valid_shift_center(center) ||
        return false, order + 1, order + 1, Inf
    a_terms, b_terms = ab_term_counts(:O, order)
    length(a_out) >= a_terms ||
        throw(ArgumentError("A-series output buffer is too short."))
    length(b_out) >= b_terms ||
        throw(ArgumentError("B-series output buffer is too short."))
    a_error = _accurate_shift_rational!(
        a_out, a_terms, work, endpoint.A, 1, center, eps_limit)
    (a_error <= eps_limit || !isfinite(a_error)) || (a_error = min(a_error, eps(Float64) * _weighted_shift_condition(
        a_out, a_terms, work, endpoint.A, 1, center, radius)))
    b_error = _accurate_shift_rational!(
        b_out, b_terms, work, endpoint.B, 2, center, eps_limit)
    (b_error <= eps_limit || !isfinite(b_error)) || (b_error = min(b_error, eps(Float64) * _weighted_shift_condition(
        b_out, b_terms, work, endpoint.B, 2, center, radius)))
    eps_condition = max(a_error, b_error)
    accepted = isfinite(eps_condition) && eps_condition <= eps_limit
    return accepted, a_terms, b_terms, eps_condition
end

# As shifted_ab!, for the infinity rational functions about y0; A is negated
# so that the result is for derivatives in y.
function shifted_infinity_ab!(
    a_out::Vector{ComplexF64},
    b_out::Vector{ComplexF64},
    work::Vector{ComplexF64},
    endpoint::EndpointCoefficientSet,
    y0,
    order::Integer;
    eps_limit::Real=_SHIFT_EPS_LIMIT,
    radius::Real=_shift_weight_radius(y0),
)
    center = y0 isa Real ? Float64(y0) : ComplexF64(y0)
    _valid_shift_center(center) ||
        return false, order + 1, order + 1, Inf
    a_terms, b_terms = ab_term_counts(:O, order)
    length(a_out) >= a_terms ||
        throw(ArgumentError("A-series output buffer is too short."))
    length(b_out) >= b_terms ||
        throw(ArgumentError("B-series output buffer is too short."))
    a_error = _accurate_shift_rational!(
        a_out, a_terms, work, endpoint.A, 1, center, eps_limit)
    (a_error <= eps_limit || !isfinite(a_error)) || (a_error = min(a_error, eps(Float64) * _weighted_shift_condition(
        a_out, a_terms, work, endpoint.A, 1, center, radius)))
    b_error = _accurate_shift_rational!(
        b_out, b_terms, work, endpoint.B, 4, center, eps_limit)
    (b_error <= eps_limit || !isfinite(b_error)) || (b_error = min(b_error, eps(Float64) * _weighted_shift_condition(
        b_out, b_terms, work, endpoint.B, 4, center, radius)))
    eps_condition = max(a_error, b_error)
    accepted = isfinite(eps_condition) && eps_condition <= eps_limit
    if accepted
        @inbounds for index in 1:a_terms
            a_out[index] = -a_out[index]
        end
    end
    return accepted, a_terms, b_terms, eps_condition
end

# Power series of numerator/denominator by recursive division into the first
# `out_len` entries of `out`, after removing a common leading power of the
# argument from the denominator.
function rational_series_from_coefficients!(
    out::AbstractVector{<:Complex},
    out_len::Integer,
    numerator,
    denominator,
)
    out_len <= length(out) || throw(ArgumentError("output buffer is too short."))
    shift = _denominator_leading_shift(denominator)
    _check_regular_leading_terms(numerator, shift)
    q0 = denominator[shift + 1]
    _q0_ok(q0) || error("noninvertible q0 in ISEM GSN rational P/Q recurrence")
    inv_q0 = inv(q0)
    order = out_len - 1
    @inbounds for n in 0:order
        p_index = n + shift
        acc = p_index < length(numerator) ? numerator[p_index + 1] : zero(eltype(out))
        for k in 1:min(n, length(denominator) - shift - 1)
            acc -= denominator[shift + k + 1] * out[n - k + 1]
        end
        out[n + 1] = acc * inv_q0
    end
    return out
end

# Laurent series of A and B about the horizon (:H, in x) or infinity (:I, in y),
# written into `a_out` and `b_out`; returns the numbers of terms.
function endpoint_ab_series!(
    a_out::AbstractVector{<:Complex},
    b_out::AbstractVector{<:Complex},
    endpoint::EndpointCoefficientSet,
    group::Symbol,
    order::Integer,
)
    a_terms, b_terms = ab_term_counts(group, order)
    length(a_out) >= a_terms || throw(ArgumentError("A-series output buffer is too short."))
    length(b_out) >= b_terms || throw(ArgumentError("B-series output buffer is too short."))
    if group == :H && endpoint.laurent !== nothing
        laurent = endpoint.laurent
        if order <= laurent.order
            copyto!(a_out, 1, laurent.a, 1, a_terms)
            copyto!(b_out, 1, laurent.b, 1, b_terms)
        else
            # A longer request is computed again, never padded.
            horizon_closed_form_ab_series!(a_out, b_out, laurent.params, 0.0, order)
        end
        return a_terms, b_terms
    end
    rational_series_from_coefficients!(a_out, a_terms, endpoint.A.numerator, endpoint.A.denominator)
    rational_series_from_coefficients!(b_out, b_terms, endpoint.B.numerator, endpoint.B.denominator)
    return a_terms, b_terms
end

# Allocating form of endpoint_ab_series!; returns (A series, B series).
function endpoint_ab_series(coefficients::CoefficientSet, group::Symbol, order::Integer)
    group in (:H, :I) || throw(ArgumentError("endpoint group must be :H or :I."))
    endpoint = group == :H ? coefficients.horizon : coefficients.infinity
    a_terms, b_terms = ab_term_counts(group, order)
    C = eltype(endpoint.A.numerator)
    a_coeffs = Vector{C}(undef, a_terms)
    b_coeffs = Vector{C}(undef, b_terms)
    endpoint_ab_series!(a_coeffs, b_coeffs, endpoint, group, order)
    return a_coeffs, b_coeffs
end

# Taylor series of A and B from P/Q values already evaluated at the expansion point.
function ordinary_ab_series!(
    a_out::AbstractVector{<:Complex},
    b_out::AbstractVector{<:Complex},
    pq::AbstractVector{<:Complex},
    s::Integer,
    order::Integer,
)
    layout = pq_layout(s, :O)
    ap, aq, bp, bq = split_starts(layout)
    a_terms, b_terms = ab_term_counts(:O, order)
    length(a_out) >= a_terms || throw(ArgumentError("A-series output buffer is too short."))
    length(b_out) >= b_terms || throw(ArgumentError("B-series output buffer is too short."))
    _ordinary_rational_series_from_pq_len!(a_out, a_terms, pq, ap, layout.ap_len, aq, layout.aq_len)
    _ordinary_rational_series_from_pq_len!(b_out, b_terms, pq, bp, layout.bp_len, bq, layout.bq_len)
    return a_terms, b_terms
end

# A and B series in the scaled variable u = (1 - x) / omega_scale about u = y0
# (y0 is given in scaled units); the P/Q blocks are rescaled before division.
function ordinary_scaled_y_ab_series!(a_out, b_out, pq,
        coefficients::CoefficientSet, y0, omega_scale, order::Integer)
    variable = coefficients.ordinary.variable == :y ?
        omega_scale * y0 : 1.0 - omega_scale * y0
    evaluate_ordinary_variable!(pq, coefficients.ordinary, variable)
    layout = pq_layout(coefficients.params.s, :O)
    starts = split_starts(layout)
    lengths = (layout.ap_len, layout.aq_len, layout.bp_len, layout.bq_len)
    # Rescale the rational polynomials before division; the unscaled Taylor
    # coefficients can overflow even when the scaled equation is regular.
    for (start, count) in zip(starts, lengths)
        power = 1.0
        for j in 0:(count-1)
            pq[start+j] *= power
            power *= -omega_scale
        end
    end
    na, nb = ordinary_ab_series!(
        a_out, b_out, pq, coefficients.params.s, order)
    for j in 1:na
        a_out[j] *= -omega_scale
    end
    for j in 1:nb
        b_out[j] *= omega_scale^2
    end
    return na, nb
end

# Below this x the ordinary-point tables lose accuracy: they form B from O(1)
# terms that cancel to O(x^2) near the horizon, so B carries a relative error
# of about 1e-12 / x^2 (4e-8 at x = 0.0044, 9e-3 at x = 1e-5), while the Taylor
# shift of the horizon rational functions stays at about 2e-13 there. The
# horizon series of Y starts at a small x when a root of eta lies near the
# horizon or the spin is near extremal, and its first patches then took the
# table values: Y(r) of (s, l, m) = (-2, 2, 2) was off by 1.9e-5 at
# (a, omega) = (0.8, 0.9) and 1.0e-6 at (0.99, 0.76) without any warning. The
# shift loses accuracy towards x = 1 (relative B error 2e-10 at 0.5, 1e-6 at
# 0.9), so it is used only below this point; up to 0.25 it made Y of
# (0.95, 0.9) jump by 4e-9 between neighbouring omega (1e-10 with the tables).
const HORIZON_SHIFT_MAX_X = 0.1

# Real binary64 points between HORIZON_SHIFT_MAX_X and DIRECT_AB_MAX_X take A and B from direct_ab_series!:
# there the tables put an error of about 1e-10 into B_inc / B_trans of (2, 2) at a = 0.95, omega = 0.74 - 0.81
# (near m Omega_H), which scattered between neighbouring omega by 4e-12 (Chebyshev tail over 129 points; 4e-13
# with the closed form, 3e-12 largest error against a 128-bit reference).
const DIRECT_AB_MAX_X = 0.7

# r as a truncated series with its powers r^2 ... r^n computed once: the closed-form potentials take
# powers of r dozens of times (sU of s = -2, order 41: 135 us per call with the powers taken each time,
# 62 us cached).
struct _PowerCachedSeries{T}
    powers::Vector{T}
end

function _PowerCachedSeries(r, n::Integer)
    powers = [r]
    for _ in 2:n
        push!(powers, powers[end] * r)
    end
    return _PowerCachedSeries(powers)
end

Base.literal_pow(::typeof(^), c::_PowerCachedSeries, ::Val{k}) where {k} =
    k <= length(c.powers) ? c.powers[k] : c.powers[1]^k
Base.:-(c::_PowerCachedSeries) = -c.powers[1]
for op in (:+, :-, :*, :/)
    @eval Base.$op(c::_PowerCachedSeries, y) = $op(c.powers[1], y)
    @eval Base.$op(y, c::_PowerCachedSeries) = $op(y, c.powers[1])
    @eval Base.$op(c::_PowerCachedSeries, d::_PowerCachedSeries) = $op(c.powers[1], d.powers[1])
end

# Taylor series of A and B about a real x0 from the closed-form GSN potentials:
# with g = dx/dr*, X'' + A X' + B X = 0 in x has A = (dg/dx - F) / g and
# B = -U / g^2, for X_{r*r*} - F X_{r*} - U X = 0. Its B is good to about 2e-13
# where the tables are off by 1e-12 to 2e-12 (x = 0.2 - 0.4 for s = -2,
# l = 3, a = 0.9, omega = 2.35), at about 0.1 ms per call.
function direct_ab_series!(a_out, b_out, params, x0::Real, order::Integer)
    _arena_eligible(params, x0) || return _direct_ab_series!(a_out, b_out, params, x0,
        Taylor1(ComplexF64, order + 1), order)
    # binary64: the rational form (RationalArena.jl, ~2.5x faster), the Taylor form if it exceeds its fixed sizes
    terms = _rational_ab_series!(a_out, b_out, params, x0, order)
    terms === nothing || return terms
    return _direct_ab_series!(a_out, b_out, params, x0, _arena_variable(_reset!(_task_arena()), order + 1), order)
end

# Binary64 parameters and point: the series run in _ArenaSeries (bitwise the Taylor1 values, see TaylorArena.jl).
_arena_eligible(params, x0) = x0 isa Union{Float64,ComplexF64} && params.a isa Float64 &&
    params.omega isa Union{Float64,ComplexF64} && params.lambda isa Union{Float64,ComplexF64}

# `t` is the series variable of order `order + 1` (Taylor1 or _ArenaSeries).
function _direct_ab_series!(a_out, b_out, params, x0, t, order::Integer)
    s, m, a, omega, lambda = params.s, params.m, params.a, params.omega, params.lambda
    kappa = sqrt(1 - a^2)
    rplus, rminus = 1 + kappa, 1 - kappa
    x = x0 + t
    r = (rplus - rminus * x) / (1 - x)
    delta = r^2 - 2r + a^2
    g = (rplus - rminus) * delta / ((r - rminus)^2 * (r^2 + a^2))
    cached = _PowerCachedSeries(r, 7)
    A = (derivative(g) - sF(s, m, a, omega, lambda, cached)) / g
    B = -sU(s, m, a, omega, lambda, cached) / g^2
    a_terms, b_terms = ab_term_counts(:O, order)
    for k in 1:a_terms
        a_out[k] = getcoeff(A, k - 1)
    end
    for k in 1:b_terms
        b_out[k] = getcoeff(B, k - 1)
    end
    return a_terms, b_terms
end

# K+ = 2 r+ omega - a m with kappa = sqrt((1 - a)(1 + a)) in fma form, error <= 0.44 eps (n + m) at the horizon
# resonances, where 2 r+ omega - a m and omega - m a / (2 r+) lose eps a m / |K+|.
function horizon_kplus(a::Real, m, omega::Number)
    kappa = sqrt((1 - a) * (1 + a))
    return complex(fma(2kappa, real(omega), fma(-a, Float64(m), 2real(omega))), fma(2kappa, imag(omega), 2imag(omega)))
end

# A and B in the horizon coordinate x from the closed-form potentials with every horizon-vanishing factor kept
# in factored form: r = r+ + 2 kappa x / (1 - x), Delta = 4 kappa^2 x / (1 - x)^2 (sF_delta),
# S = r^2 + a^2, g = dx/dr* = 2 kappa x / S, so that
#     A x = 1 - x S'/S - F S / (2 kappa),    B x^2 = -U S^2 / (4 kappa^2) = (K^2 - Delta W) / (4 kappa^2)
# with K = S omega - a m and U S^2 = -K^2 + Delta W (sW): sU itself cancels internally by ~1e2 near the horizon
# with no cancellation as x -> 0 or kappa -> 0. At x0 = 0 (exactly) the Laurent coefficients of the horizon
# group :H are written (a_{-1} = 1 exactly), otherwise the Taylor coefficients about the ordinary point x0
# (group :O). x0 and omega may be complex. With `kplus` (horizon_kplus), K = K+ + (r - r+)(r + r+) omega carries
# K+ = 2 r+ omega - a m without the cancellation of S omega - a m (the complex-rational horizon); returns the
# numbers of terms.
function horizon_closed_form_ab_series!(a_out, b_out, params, x0::Number, order::Integer; kplus=nothing)
    t = _arena_eligible(params, x0) && (kplus === nothing || kplus isa ComplexF64) ?
        _arena_variable(_reset!(_task_arena()), order + 4) : Taylor1(ComplexF64, order + 4)
    return _horizon_closed_form_ab_series!(a_out, b_out, params, x0, t, order, kplus)
end

# `t` is the series variable of order `order + 4` (Taylor1 or _ArenaSeries).
function _horizon_closed_form_ab_series!(a_out, b_out, params, x0, t, order::Integer, kplus)
    s, m, a, omega, lambda = params.s, params.m, params.a, params.omega, params.lambda
    kappa = sqrt((1 - a) * (1 + a))
    x = x0 + t
    w = 1 - x
    r = (1 + kappa) + 2kappa * x / w
    S = r^2 + a^2
    Delta = 4kappa^2 * x / w^2
    cached = _PowerCachedSeries(r, 7)
    F = sF_delta(s, m, a, omega, lambda, cached, Delta)
    K = kplus === nothing ? S * omega - a * m : kplus + (2kappa * x / w) * (r + (1 + kappa)) * omega
    Ax = 1 - x * derivative(S) / S - F * S / (2kappa)
    Bx2 = (K^2 - Delta * sW(s, m, a, omega, lambda, cached)) / (4kappa^2)
    if iszero(x0)
        a_terms, b_terms = ab_term_counts(:H, order)
        for k in 1:a_terms
            a_out[k] = getcoeff(Ax, k - 1)
        end
        for k in 1:b_terms
            b_out[k] = getcoeff(Bx2, k - 1)
        end
        a_out[1] == 1 || error("horizon closed form: a_{-1} = $(a_out[1]) is not exactly 1")
        return a_terms, b_terms
    end
    A = Ax / x
    B = Bx2 / x^2
    a_terms, b_terms = ab_term_counts(:O, order)
    for k in 1:a_terms
        a_out[k] = getcoeff(A, k - 1)
    end
    for k in 1:b_terms
        b_out[k] = getcoeff(B, k - 1)
    end
    return a_terms, b_terms
end

# A and B at a (complex) point x from the closed-form potentials, as in direct_ab_series!.
function direct_ab_value(params, x)
    s, m, a, omega, lambda = params.s, params.m, params.a, params.omega, params.lambda
    kappa = sqrt(1 - a^2)
    rplus, rminus = 1 + kappa, 1 - kappa
    r = (rplus - rminus * x) / (1 - x)
    drdx = (rplus - rminus) / (1 - x)^2
    delta = r^2 - 2r + a^2
    den = (r - rminus)^2 * (r^2 + a^2)
    g = (rplus - rminus) * delta / den
    dgdr = (rplus - rminus) * ((2r - 2) - delta * (2 / (r - rminus) + 2r / (r^2 + a^2))) / den
    return (dgdr * drdx - sF(s, m, a, omega, lambda, r)) / g, -sU(s, m, a, omega, lambda, r) / g^2
end

# Taylor series of A and B about the ordinary point x0. Real binary64 points below DIRECT_AB_MAX_X at real omega take
# the closed-form potentials: horizon_closed_form_ab_series! (horizon coordinate, factored Delta) below
# HORIZON_SHIFT_MAX_X and direct_ab_series! above. Otherwise: the shift of the horizon rational functions when
# `metadata.horizon_shift` is set or x0 is a real point below HORIZON_SHIFT_MAX_X, and the shift is accepted; the
# ordinary-point tables. Complex omega takes the horizon closed form below HORIZON_SHIFT_MAX_X only with
# `closed_form_march` (near-extremal marches, Iteration). (Below 0.1 the shift instead of the closed form left
# (2, 2, 0.8, 0.8925) Y off by 3.7e-9 at r = 20 - 50 and B_inc / B_trans of (2, 2), a = 0.99, omega = 0.80 - 0.87
# off by up to 2.1e-11 with a 1.5e-12 Chebyshev tail over 129 points; 9.9e-12 and 2.5e-13 with the closed form.
# A horizon-coordinate closed form up to 0.7 put 2e-9 into a = 0.95, omega = 0.74 - 0.81.)
function ordinary_ab_series!(
    a_out::AbstractVector{<:Complex},
    b_out::AbstractVector{<:Complex},
    pq::AbstractVector{<:Complex},
    coefficients::CoefficientSet,
    x0,
    order::Integer;
    shift_eps_limit::Real=_SHIFT_EPS_LIMIT,
    closed_form_march::Bool=false,
)
    if x0 isa Real && 0 < x0 < HORIZON_SHIFT_MAX_X && eltype(a_out) === ComplexF64 &&
            (coefficients.params.omega isa Real || closed_form_march)
        return horizon_closed_form_ab_series!(a_out, b_out, coefficients.params, x0, order)
    end
    if x0 isa Real && HORIZON_SHIFT_MAX_X <= x0 < DIRECT_AB_MAX_X &&
            coefficients.params.omega isa Real && eltype(a_out) === ComplexF64
        return direct_ab_series!(a_out, b_out, coefficients.params, x0, order)
    end
    # The binary64 shift needs binary64 buffers (_shift_source converts to ComplexF64).
    use_shift = eltype(a_out) === ComplexF64 && (
        (hasproperty(coefficients.metadata, :horizon_shift) &&
            getproperty(coefficients.metadata, :horizon_shift)) ||
        (x0 isa Real && 0 < x0 < HORIZON_SHIFT_MAX_X))
    if use_shift
        accepted, a_terms, b_terms, _ = shifted_ab!(
            a_out, b_out, pq, coefficients.horizon, x0, order; eps_limit=shift_eps_limit)
        accepted && return a_terms, b_terms
        # Below HORIZON_SHIFT_MAX_X the tables are off by ~1e-12 / x^2 in B: a rejected shift there takes the
        # closed form in the horizon coordinate (horizon_closed_form_ab_series!), not the tables.
        x0 isa Real && 0 < x0 < HORIZON_SHIFT_MAX_X &&
            return horizon_closed_form_ab_series!(a_out, b_out, coefficients.params, x0, order)
    end
    evaluate_ordinary_coefficients!(pq, coefficients.ordinary, x0)
    return ordinary_ab_series!(a_out, b_out, pq, coefficients.params.s, order)
end

# Allocating form of ordinary_ab_series!; returns (A series, B series, P/Q work vector).
function ordinary_ab_series(coefficients::CoefficientSet, x0, order::Integer)
    a_terms, b_terms = ab_term_counts(:O, order)
    C = eltype(coefficients.ordinary.coefficients)
    pq = Vector{C}(undef, coefficients.ordinary.value_count)
    a_coeffs = Vector{C}(undef, a_terms)
    b_coeffs = Vector{C}(undef, b_terms)
    ordinary_ab_series!(a_coeffs, b_coeffs, pq, coefficients, x0, order)
    return a_coeffs, b_coeffs, pq
end

# Taylor coefficients of a solution about an ordinary point from the A and B
# series and the initial value c0 and slope c1.
function ordinary_point_coeffs!(
    coeffs::AbstractVector{<:Complex},
    a_coeffs,
    b_coeffs,
    c0,
    c1,
    order::Integer,
)
    order >= 0 || throw(ArgumentError("series order must be non-negative."))
    length(coeffs) >= order + 1 || throw(ArgumentError("solution coefficient buffer is too short."))
    C = eltype(coeffs)
    coeffs[1] = convert(C, c0)
    order >= 1 && (coeffs[2] = convert(C, c1))
    order < 2 && return coeffs
    @inbounds for n in 0:(order - 2)
        numerator = zero(C)
        for j in 0:n
            numerator += a_coeffs[j + 1] * (n - j + 1) * coeffs[n - j + 2]
            numerator += b_coeffs[j + 1] * coeffs[n - j + 1]
        end
        coeffs[n + 3] = -numerator / ((n + 2) * (n + 1))
    end
    return coeffs
end

# Closure p -> coefficient of power p, for a series whose first stored power is
# `first_power`; zero outside the stored range.
function endpoint_power_lookup(coeffs, first_power::Integer)
    return p -> begin
        idx = p - first_power + 1
        1 <= idx <= length(coeffs) ? coeffs[idx] : zero(eltype(coeffs))
    end
end

# Horner evaluation of a polynomial, with first (and second) derivatives in the
# _pair and _triple versions.
function poly_value(coeffs, z)
    isempty(coeffs) && return zero(eltype(coeffs))
    value = coeffs[end]
    @inbounds for i in (length(coeffs) - 1):-1:1
        value = value * z + coeffs[i]
    end
    return value
end

function poly_pair(coeffs, z)
    n = length(coeffs) - 1
    C = eltype(coeffs)
    n < 0 && return zero(C), zero(C)
    n == 0 && return coeffs[1], zero(C)
    value = coeffs[n + 1]
    derivative = zero(C)
    @inbounds for i in n:-1:1
        derivative = derivative * z + value
        value = value * z + coeffs[i]
    end
    return value, derivative
end

function poly_triple(coeffs, z)
    n = length(coeffs) - 1
    C = eltype(coeffs)
    n < 0 && return zero(C), zero(C), zero(C)
    n == 0 && return coeffs[1], zero(C), zero(C)
    value = coeffs[n + 1]
    derivative = zero(C)
    second = zero(C)
    @inbounds for i in n:-1:1
        second = second * z + 2 * derivative
        derivative = derivative * z + value
        value = value * z + coeffs[i]
    end
    return value, derivative, second
end

end
