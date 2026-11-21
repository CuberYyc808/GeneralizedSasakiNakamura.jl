module Iteration

include("LocalSolutionAtZero.jl")
using .LocalSolutionAtZero
include("LocalSolutionAtInfinity.jl")
using .LocalSolutionAtInfinity
include("OrdinaryPointExpansion.jl")
using .OrdinaryPointExpansion

export iterate_inf_in, iterate_inf_out, iterate_zero_in, iterate_zero_out,
       iterate_inf_in_rho, iterate_inf_out_rho, iterate_zero_in_rho, iterate_zero_out_rho

_TruncatioN = 40
_TOLERANCE = 1e-13
# Patch budget of the real-axis walks (as ComplexRational._MAX_PATCHES). Converging walks take
# <= ~1190 patches (a >= 1e-3, |omega| >= 1e-12); at a ~ 0 and tiny |omega| the infinity walk
# starts near |x| ~ 1e21 with steps ~ 1e5 and would otherwise append patches without bound.
const _MAX_PATCHES = 2048

# The contour horizon series is started inside its estimated convergence radius,
# closer in for large |lambda|; for |lambda| < 1 dividing by sqrt|lambda| would
# start it outside the radius (s = +2, l = 2 at small a).
@inline _seed_scale(lambda) = max(one(abs(lambda)), sqrt(abs(lambda)))

# Real-axis walk of the P equation from the seed point x0 (value P, P') with order-N ordinary-point Taylor patches,
# towards the horizon (dir = +1, from the infinity series) or away from it (dir = -1, from the horizon series), until
# x passes xm. Returns the patch start points and coefficients and x -> (P, dP/dx, d2P/dx2, residual), with the seed
# series (`value`, `residual`) on its own side of x0.
function _walk_x(value, residual, x0, xm, dir, s, epsilon, tau, kappa, lambda, N, tol, ::Type{C}) where {C}
    x = x0
    P, P_prime, _ = value(x)
    coeffs_list = C[]
    x_list = [x]
    while dir > 0 ? x < xm : x > xm
        length(coeffs_list) < _MAX_PATCHES ||
            Base.error("P-equation real-axis patch limit $_MAX_PATCHES reached at x = $x, target $xm.")
        coeffs_ord = ordinary_point_coeffs(s, epsilon, tau, kappa, lambda, x, P, P_prime, N)
        Δx = min(convergence_radius_x0_right(x, s, epsilon, tau, kappa, lambda, coeffs_ord, tol),
            convergence_radius_x0_left(x, s, epsilon, tau, kappa, lambda, coeffs_ord, tol))
        (isfinite(Δx) && Δx > 0) || Base.error("Step size Δx invalid: $Δx")
        P, P_prime, _ = evaluate_ordinary_point_series(x + dir * Δx, x, coeffs_ord)
        x += dir * Δx
        push!(coeffs_list, coeffs_ord)
        push!(x_list, x)
    end
    function evaluate_P(x)
        x >= 0.0 && Base.error("x = $x should be a negative real number.")
        if dir > 0 ? x <= x_list[1] : x_list[1] <= x
            vals = value(x)
            return (vals..., residual(x, vals))
        end
        (dir > 0 ? x <= x_list[end] : x >= x_list[end]) || Base.error("x = $x is outside the computed domain.")
        i = searchsortedlast(x_list, x; rev=dir < 0)
        i > length(coeffs_list) && Base.error("x = $x is outside computed ordinary-point domain.")
        Px, Px_prime, Px_primeprime = evaluate_ordinary_point_series(x, x_list[i], coeffs_list[i])
        err = error_ordinary((x, Px, Px_prime, Px_primeprime), s, epsilon, tau, kappa, lambda)
        return (Px, Px_prime, Px_primeprime, err)
    end
    return x_list, coeffs_list, evaluate_P
end

# Infinity IN and OUT solutions on the real x axis: from the asymptotic series towards the horizon until x >= xm.
function iterate_inf_in(xm, s, epsilon, tau, kappa, lambda, N = _TruncatioN, tol = _TOLERANCE)
    c, _, R = an_inf(s, epsilon, tau, kappa, lambda, tol)
    x0 = - convergence_radius_an_inf(R, s, epsilon, tau, kappa, lambda, c, tol)
    return _walk_x(x -> evaluate_inf_in(x, s, epsilon, tau, c),
        (x, _) -> error_inf_in(x, s, epsilon, tau, kappa, lambda, c), x0, xm, 1, s, epsilon, tau, kappa, lambda, N, tol,
        typeof(c))
end

function iterate_inf_out(xm, s, epsilon, tau, kappa, lambda, N = _TruncatioN, tol = _TOLERANCE)
    c, _, R = bn_inf(s, epsilon, tau, kappa, lambda, tol)
    x0 = - convergence_radius_bn_inf(R, s, epsilon, tau, kappa, lambda, c, tol)
    return _walk_x(x -> evaluate_inf_out(x, s, epsilon, tau, kappa, c),
        (x, _) -> error_inf_out(x, s, epsilon, tau, kappa, lambda, c), x0, xm, 1, s, epsilon, tau, kappa, lambda, N,
        tol, typeof(c))
end

# Horizon IN and OUT solutions: from the series at x = 0 away from the horizon until x <= xm.
function iterate_zero_in(xm, s, epsilon, tau, kappa, lambda, N = _TruncatioN, tol = _TOLERANCE)
    c = an_zero(s, epsilon, tau, kappa, lambda, N)
    x0 = - convergence_radius_an_zero(c, tol) / sqrt(abs(lambda))
    return _walk_x(x -> evaluate_zero_in(x, c), (x, v) -> error_zero_in((x, v...), s, epsilon, tau, kappa, lambda),
        x0, xm, -1, s, epsilon, tau, kappa, lambda, N, tol, typeof(c))
end

function iterate_zero_out(xm, s, epsilon, tau, kappa, lambda, N = _TruncatioN, tol = _TOLERANCE)
    c = bn_zero(s, epsilon, tau, kappa, lambda, N)
    x0 = - convergence_radius_bn_zero(c, tol) / sqrt(abs(lambda))
    return _walk_x(x -> evaluate_zero_out(x, s, epsilon, tau, c),
        (x, v) -> error_zero_out((x, v...), s, epsilon, tau, kappa, lambda), x0, xm, -1, s, epsilon, tau, kappa, lambda,
        N, tol, typeof(c))
end

# Rotated-contour walk (x = -rho/z) from the seed series (coefficients `seed`, stored as the first patch) at rho0:
# inward to rho_m from infinity (dir = -1) or outward from the horizon (dir = +1). The seed series is used beyond
# rho0 = R0 (infinity) or for rho <= R0 (horizon, and at rho0 itself).
function _walk_rho(value, residual, seed, rho0, R0, rho_m, dir, s, epsilon, tau, kappa, lambda, z, N, tol)
    rho = rho0
    P, Pp, _ = value(rho)
    coeffs_list = [seed]
    rho_list = [rho]
    while dir > 0 ? rho < rho_m : rho > rho_m
        coeffs_ord = ordinary_point_coeffs(s, epsilon, tau, kappa, lambda, z, rho, P, Pp, N)
        Δρ = min(
            convergence_radius_rho_left(rho, s, epsilon, tau, kappa, lambda, z, coeffs_ord, tol),
            convergence_radius_rho_right(rho, s, epsilon, tau, kappa, lambda, z, coeffs_ord, tol),
            dir > 0 ? rho_m - rho : rho - rho_m,
        )
        Δρ > 0 || error("Step size Δρ invalid: $Δρ")
        rho += dir * Δρ
        P, Pp, _ = evaluate_ordinary_point_series(rho, rho - dir * Δρ, coeffs_ord)
        push!(rho_list, rho)
        push!(coeffs_list, coeffs_ord)
    end
    function evaluate_P(rho_eval)
        lo, hi = dir > 0 ? (rho_list[1], rho_list[end]) : (rho_list[end], rho_list[1])
        seed_side = dir > 0 ? (rho_eval == rho_list[1] || lo <= rho_eval <= min(R0, hi)) : rho_eval > rho_list[1]
        if seed_side
            vals = value(rho_eval)
            return (vals..., residual(rho_eval))
        end
        lo <= rho_eval <= hi || error("rho = $rho_eval is outside the computed domain [$lo, $hi].")
        i = searchsortedlast(rho_list, rho_eval; rev = dir < 0)
        i == 0 && error("rho = $rho_eval is outside the computed patch list.")
        rho0, coeffs = i == length(rho_list) ? (rho_list[end - 1], coeffs_list[end]) :
            (rho_list[i], coeffs_list[i + 1])
        Pr, Prp, Prpp = evaluate_ordinary_point_series(rho_eval, rho0, coeffs)
        err = error_ordinary(rho_eval, rho0, s, epsilon, tau, kappa, lambda, z, coeffs)
        return (Pr, Prp, Prpp, err)
    end
    return rho_list, coeffs_list, evaluate_P
end

function iterate_inf_in_rho(rho_m, s, epsilon, tau, kappa, lambda, z, N = _TruncatioN, tol = _TOLERANCE)
    c, _, R0 = inf_in_coeffs(s, epsilon, tau, kappa, lambda, z, N; tol = tol)
    return _walk_rho(rho -> evaluate_inf_in(rho, s, epsilon, tau, c, z),
        rho -> error_inf_in(rho, s, epsilon, tau, kappa, lambda, c, z), c, R0, R0, rho_m, -1, s, epsilon, tau, kappa,
        lambda, z, N, tol)
end

function iterate_inf_out_rho(rho_m, s, epsilon, tau, kappa, lambda, z, N = _TruncatioN, tol = _TOLERANCE)
    c, _, canonical_R0 = inf_out_coeffs(s, epsilon, tau, kappa, lambda, z, N; tol = tol)
    # Start the outward series where its residual meets seed_target: at the
    # canonical radius, or at the first of 1.05, 1.10, ..., 1.50 times it that does.
    seed_target = max(0.05 * Float64(tol), 2.5 * eps(Float64))
    residual(rho) = error_inf_out(rho, s, epsilon, tau, kappa, lambda, c, z)
    seed_error = residual(canonical_R0)
    R0 = canonical_R0
    if !isfinite(seed_error) || seed_error > seed_target
        for step in 1:10
            candidate_R0 = (Float64(100 + 5 * step) / 100.0) * canonical_R0
            candidate_error = residual(candidate_R0)
            if isfinite(candidate_error) && candidate_error <= seed_target
                R0 = candidate_R0
                break
            end
        end
    end
    return _walk_rho(rho -> evaluate_inf_out(rho, s, epsilon, tau, kappa, c, z), residual, c, R0, R0, rho_m, -1, s,
        epsilon, tau, kappa, lambda, z, N, tol)
end

function iterate_zero_in_rho(rho_m, s, epsilon, tau, kappa, lambda, z, N = _TruncatioN, tol = _TOLERANCE)
    c, R0 = zero_in_coeffs(s, epsilon, tau, kappa, lambda, z, N; tol = tol)
    return _walk_rho(rho -> evaluate_zero_in_rho(rho, c),
        rho -> error_zero_in_rho(rho, s, epsilon, tau, kappa, lambda, z, c), c, R0 / _seed_scale(lambda), R0, rho_m, 1,
        s, epsilon, tau, kappa, lambda, z, N, tol)
end

function iterate_zero_out_rho(rho_m, s, epsilon, tau, kappa, lambda, z, N = _TruncatioN, tol = _TOLERANCE)
    c, R0 = zero_out_coeffs(s, epsilon, tau, kappa, lambda, z, N; tol = tol)
    return _walk_rho(rho -> evaluate_zero_out_rho(rho, s, epsilon, tau, c, z),
        rho -> error_zero_out_rho(rho, s, epsilon, tau, kappa, lambda, z, c), c, R0 / _seed_scale(lambda), R0, rho_m, 1,
        s, epsilon, tau, kappa, lambda, z, N, tol)
end

end
