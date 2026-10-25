module LargeFrequencyExpansion

using ..CoefficientTables: CoefficientSet, evaluate_ordinary_coefficients!
using ..OrdinaryPointExpansion: shifted_ab!, horizon_closed_form_ab_series!, ordinary_ab_series!

export LargeFrequencyExpansionRouting, large_frequency_expansion_routing, large_frequency_expansion_ab!
export large_frequency_expansion_ab_value

# Binary64 A, B coefficients of the large-frequency route (instead of the double-double table): below the
# hand-off x0 (0.25; 0.15 for kappa < 0.05) the Taylor shift of the horizon rational functions where it passes
# its gates (per coefficient, else step weighted, 1e-13; shifted_ab!); otherwise up to x = 0.7 the closed form in
# the horizon coordinate (horizon_closed_form_ab_series!), and beyond it the ordinary-point tables. The tables
# are off by 1e-14 - 5e-13 at x0 = 0.1 - 0.3 for large omega (6.7e-14 in B at (-2,2,2,0.3,300), x0 = 0.25; their
# own construction error is 1.5e-14 - 4.3e-14 at omega = 300 - 1000), the closed form by ~1e-15. The shift's own
# per-coefficient estimate is 30 - 300 times pessimistic, so handing over on it alone took the tables from
# x0 = 0.05 - 0.15 on.
const LARGE_FREQUENCY_EXPANSION_CLOSED_FORM_MAX_X = 0.7
struct LargeFrequencyExpansionRouting{C<:CoefficientSet}
    coefficients::C
    pq::Vector{ComplexF64}
    handoff::Float64
end

function large_frequency_expansion_routing(coefficients::CoefficientSet)
    kappa = Float64(coefficients.params.kappa)
    return LargeFrequencyExpansionRouting(coefficients,
        Vector{ComplexF64}(undef, max(coefficients.ordinary.value_count, 256)),
        kappa < 0.05 ? 0.15 : 0.25)
end

function large_frequency_expansion_ab!(a_out, b_out, routing::LargeFrequencyExpansionRouting, x0, order::Integer)
    coefficients = routing.coefficients
    if x0 isa Real && 0 < x0 < routing.handoff
        accepted, a_terms, b_terms, _ = shifted_ab!(
            a_out, b_out, routing.pq, coefficients.horizon, x0, order)
        accepted && return a_terms, b_terms
    end
    x0 isa Real && 0 < x0 < LARGE_FREQUENCY_EXPANSION_CLOSED_FORM_MAX_X &&
        return horizon_closed_form_ab_series!(a_out, b_out, coefficients.params, x0, order)
    evaluate_ordinary_coefficients!(routing.pq, coefficients.ordinary, x0)
    return ordinary_ab_series!(a_out, b_out, routing.pq, coefficients.params.s, order)
end

# A and B at x by the same routing.
function large_frequency_expansion_ab_value(coefficients::CoefficientSet, x)
    a = Vector{ComplexF64}(undef, 3)
    b = Vector{ComplexF64}(undef, 3)
    large_frequency_expansion_ab!(a, b, large_frequency_expansion_routing(coefficients), x, 2)
    return a[1], b[1]
end

end
