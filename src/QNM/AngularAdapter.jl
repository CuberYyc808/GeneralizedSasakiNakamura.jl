# Angular eigen-branch at one c = a*omega: separation constant `angular_A`, `lambda`, the spherical-harmonic
# `mixing` vector, residual, truncation order, precision bits, sheet id, status and route-specific `metadata`.
struct AngularBranchResult{T,C,V,M}
    c::C
    angular_A::C
    lambda::C
    mixing::V
    residual::T
    truncation_order::Int
    precision_bits::Int
    sheet_id::Symbol
    status::Symbol
    metadata::M
end

# The angular spectral matrix M(c) (quadratic in c) of the near-extremal expansion. `construct_spectral_matrix` is
# not exported by SpinWeightedSpheroidalHarmonics 1.5; its export is requested (refactor prompt S6).
_big_angular_spectral_matrix(c, s::Int, m::Int, size::Int) =
    SpinWeightedSpheroidalHarmonics.construct_spectral_matrix(c, s, m, size)

"""
    angular_A_to_lambda(angular_A, c, m)

Convert the angular eigenvalue `A` (Leaver's convention, `c = a*omega`) to the separation constant of the radial Teukolsky equation, `lambda = A + c^2 - 2 m c`.
"""
@inline angular_A_to_lambda(angular_A, c, m::Int) =
    angular_A + c^2 - 2m * c

"""
    lambda_to_angular_A(lambda, c, m)

Inverse of `angular_A_to_lambda`: `A = lambda - c^2 + 2 m c`.
"""
@inline lambda_to_angular_A(lambda, c, m::Int) =
    lambda - c^2 + 2m * c

# Float64 angular eigenpair at c, continued on `sheet_id` by SpinWeightedSpheroidalHarmonics.
function _angular_branch_float64(mode::QNMMode, c;
        truncation_order::Int=32, sheet_id::Symbol=:qnm_straight_from_spherical)
    pair = SpinWeightedSpheroidalHarmonics.continue_angular_mode(
        mode.s, mode.l, mode.m, ComplexF64(c);
        sheet_id, truncation_order, cache=nothing)
    return AngularBranchResult(
        pair.c,
        pair.angular_sep,
        pair.lambda,
        pair.coefficients,
        pair.residual,
        pair.truncation_order,
        pair.precision_bits,
        pair.sheet_id,
        pair.status,
        (
            route=:fast_eigenvalue_continuation, previous_overlap=pair.previous_overlap,
            overlap_margin=pair.overlap_margin, spectral_gap=pair.spectral_gap,
            eigenvector_condition=pair.eigenvector_condition, matrix_size=pair.matrix_size,
        ),
    )
end

# Angular separation constant, eigenvalue and coefficients at c = a*omega on `sheet_id`.
function angular_branch(mode::QNMMode, a, omega;
        truncation_order::Int=32, sheet_id::Symbol=:qnm_straight_from_spherical)
    c = a * omega
    real(c) isa Union{Float16,Float32,Float64} || throw(ArgumentError(
        "QNM angular continuation supports Float16/32/64 inputs."))
    return _angular_branch_float64(mode, c; truncation_order, sheet_id)
end
