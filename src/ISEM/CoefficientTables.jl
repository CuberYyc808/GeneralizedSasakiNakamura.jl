module CoefficientTables

using ..Parameters: GSNParameters

export RationalCoefficients, EndpointCoefficientSet
export OrdinaryCoefficientEvaluator, CoefficientSet
export evaluate_ordinary_coefficients!
export evaluate_ordinary_variable!
export gsn_coefficients

# Numerator and denominator polynomial coefficients of one rational ODE coefficient; the element
# type is ComplexF64 for binary64 inputs and the promoted complex type of the inputs otherwise.
struct RationalCoefficients{N<:AbstractVector{<:Complex},D<:AbstractVector{<:Complex}}
    numerator::N
    denominator::D
end

# Rational A(x) and B(x) of X'' + A X' + B X = 0 in one endpoint basis (:hc or :ic).
# `laurent`: optional closed-form Laurent coefficients of A and B at the endpoint (near-extremal horizon, see
# NearExtreme), used by endpoint_ab_series! instead of the division of the rational functions; the rational
# functions stay for the consumers that shift them.
struct EndpointCoefficientSet{A<:RationalCoefficients,B<:RationalCoefficients,L}
    A::A
    B::B
    basis::Symbol
    laurent::L
end

EndpointCoefficientSet(A, B, basis::Symbol) = EndpointCoefficientSet(A, B, basis, nothing)

# The packed P/Q Taylor coefficients of A and B about an ordinary point x0, as polynomials in t = 1 - x0: block j of
# each of the four polynomials q = (A_P, y A_Q, B_P, y^4 B_Q) of the infinity endpoint set (y = 1 - x) holds
# (-1)^j binomial(j + i, j) q_{j+i}, i = 0, 1, ... (the Taylor shift of q to y = t; GeneratedBackend builds them).
struct OrdinaryCoefficientEvaluator{C,R}
    basis::Symbol
    variable::Symbol
    coefficients::C
    value_count::Int
    ranges::R
end

# Horner evaluation at t of each coefficient block in `ranges`, writing one
# value per block into `out` (zero for an empty block).
@noinline function _evaluate_polynomial_blocks!(out, coefficients, t, ranges)
    @inbounds for j in eachindex(ranges)
        block = ranges[j]
        if isempty(block)
            out[j] = zero(eltype(out))
        else
            value = coefficients[last(block)]
            for k in (last(block)-1):-1:first(block)
                value = value*t + coefficients[k]
            end
            out[j] = value
        end
    end
    return out
end

# Fill the first `e.value_count` entries of `out` with the ordinary-point coefficients
# at expansion point `x0`; returns `out`.
function evaluate_ordinary_coefficients!(out::AbstractVector{<:Complex}, e::OrdinaryCoefficientEvaluator, x0)
    length(out) >= e.value_count || throw(ArgumentError("ordinary output buffer is too short."))
    x0 = complex(x0)
    return _evaluate_polynomial_blocks!(out, e.coefficients, one(x0) - x0, e.ranges)
end

# As `evaluate_ordinary_coefficients!`, at the already transformed expansion variable t = `value`.
function evaluate_ordinary_variable!(out::AbstractVector{<:Complex}, e::OrdinaryCoefficientEvaluator, value)
    length(out) >= e.value_count || throw(ArgumentError("ordinary output buffer is too short."))
    return _evaluate_polynomial_blocks!(out, e.coefficients, complex(value), e.ranges)
end

function (e::OrdinaryCoefficientEvaluator)(x0)
    out = Vector{eltype(e.coefficients)}(undef, e.value_count)
    evaluate_ordinary_coefficients!(out, e, x0)
    return out
end

# All ODE coefficient data for one mode: horizon and infinity endpoint sets,
# the ordinary-point evaluator, and metadata about how they were built.
struct CoefficientSet{P,H,I,O,M}
    params::P
    horizon::H
    infinity::I
    ordinary::O
    metadata::M
end

"""
    gsn_coefficients(params::GSNParameters; controls=nothing)

The CoefficientSet for `params` from the generated tables (method defined in GeneratedBackend, which is included
after this module because it builds these types); `controls` selects the ordinary-point basis.
"""
function gsn_coefficients end

end
