# Arbitrary-precision ISEM route of the GSN equation.
#
# For mode parameters beyond binary64 (a BigFloat spin, frequency or separation constant) the IN and
# UP solutions are built from the same ingredients as the Float64 routes -- the exact generated
# coefficient tables, the horizon Frobenius series, the asymptotic series at infinity and Taylor
# patches between them, matched at an interior point of the compact coordinate x -- with every number
# in the floating type R of the inputs. The series orders, the seed radii and every truncation test
# follow the requested tolerance (default: 64 ulps of the inputs' precision), and the BigFloat
# working precision is raised internally by a guard and by the documented continuation loss.
#
# The regimes of the Float64 routes that rest on binary64 data (large-frequency tables, near-extremal
# endpoint factors, the eikonal patches and the rotated complex contour) are not used here. Complex
# frequencies are seeded by the asymptotic series at infinity, whose error law is recorded in the
# diagnostics (`complex_loss_digits`); the MST representation of the infinity side replaces it in
# ManoSuzukiTakasugiGeneric.
module ArbitraryPrecisionRoute

using ...Coordinates: r_from_rstar, rstar_from_r
using ...ConversionFactors
using ...Solutions: Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix
using ..Parameters: GSNParameters, GSNControls, gsn_controls, is_binary64_parameters
using ..Parameters: DEFAULT_TOLERANCE, DEFAULT_MATCH_X
using ..CoefficientTables: CoefficientSet, gsn_coefficients
using ..OrdinaryPointExpansion: ab_term_counts, ordinary_ab_series!, ordinary_point_coeffs!, poly_pair
using ..LocalSolutionAtZero: zero_local_solution, is_horizon_threshold
using ..LocalSolutionAtInfinity: infinity_local_solution
using ..Matching: ConjugatedRoute
using ..RadialFunctions: TeukolskySolution, TeukolskyRadialEvaluator, TeukolskyTupleEvaluator,
    _teukolsky_converter
import ..RadialFunctions: gsn_radial_function, teukolsky_radial_function, y_radial_function,
    _teukolsky_pair_with_error
using ..TeukolskyTransformation: d2R, GSN_to_Y_solution_from_matrix
using ..ManoSuzukiTakasugiInfinity: mano_suzuki_takasugi_nu, mano_suzuki_takasugi_nu_complex
using ..ManoSuzukiTakasugiGeneric: MSTGenericData, mst_generic_data, mst_nu_generic,
    mst_up_state, mst_down_state
using StaticArrays: SA

export ArbitraryPrecisionRadialRoute, arbitrary_precision_route
export route_state, route_state_r

# Default series tolerance in ulps of the inputs' precision, Taylor order per decimal digit, guard
# bits added to the working precision, and the largest seed coordinates at the two ends.
const TOLERANCE_ULPS = 64
const ORDER_PER_DIGIT = 2.2
const MINIMUM_ORDER = 24
const GUARD_BITS = 24
const HORIZON_SEED_XMAX = 1 // 2
const INFINITY_SEED_YMAX = 1 // 2
const MAX_PATCHES = 2000

_root_module() = parentmodule(parentmodule(@__MODULE__))
_isem_module() = parentmodule(@__MODULE__)

_real_type(::GSNParameters{R}) where {R} = R
_complex_type(::GSNParameters{R}) where {R} = Complex{R}

# Compact coordinate x = (r - r_+) / (r - r_-), r_+- = 1 +- kappa; dx/dr* = 2 kappa x / (r^2 + a^2).
r_plus(p::GSNParameters) = one(p.kappa) + p.kappa
r_minus(p::GSNParameters) = one(p.kappa) - p.kappa
x_from_r(p::GSNParameters, r) = (r - r_plus(p)) / (r - r_minus(p))
r_from_x(p::GSNParameters, x) = (r_plus(p) - r_minus(p) * x) / (one(x) - x)
dx_drstar(p::GSNParameters, x, r) = 2 * p.kappa * x / (r^2 + p.a^2)
horizon_wavenumber(p::GSNParameters) = p.omega - p.m * p.a / (2 * r_plus(p))

# Constant phases of the unit-transmission normalization (Iteration.endpoint_seed_scale in R):
# horizon in/out = exp(-/+ i k (r_+ + 2 log kappa)) with k = omega - m a / (2 r_+),
# infinity up/down = exp(+/- i omega (r_- + 2 log kappa)).
function _horizon_scale(p::GSNParameters, branch::Symbol)
    phase = horizon_wavenumber(p) * (r_plus(p) + 2 * log(p.kappa))
    return branch == :in ? exp(-im * phase) : exp(im * phase)
end
function _infinity_scale(p::GSNParameters, branch::Symbol)
    phase = p.omega * (r_minus(p) + 2 * log(p.kappa))
    return branch == :up ? exp(im * phase) : exp(-im * phase)
end

# Taylor patch about x0 valid between x0 and xend.
struct SeriesPatch{R,C}
    x0::R
    xend::R
    coeffs::Vector{C}
end

# Horizon Frobenius series X = scale x^mu sum c_n x^n, used for x <= x_seed.
struct HorizonExpansion{R,C}
    exponent::C
    scale::C
    coeffs::Vector{C}
    x_seed::R
end

# Representations of the two infinity branches (:up outgoing, :down ingoing).
abstract type InfinityBranch end

# Asymptotic series at infinity X = scale exp(sigma / y) y^rho sum d_n y^n, y = 1 - x, used for
# y <= y_seed (optimally truncated at the seed); Taylor patches continue it inward.
struct InfinityExpansion{R,C} <: InfinityBranch
    rho::C
    sigma::C
    scale::C
    coeffs::Vector{C}
    y_seed::R
end

# MST representation (ManoSuzukiTakasugiGeneric): the Teukolsky UP or DOWN solution, convergent at
# every r > r_+, converted to the GSN state at the requested point. `gsn_scale` is the Teukolsky
# amplitude of the unit-GSN-transmission branch (ConversionFactors.Ctrans for UP, Binc for DOWN).
struct MSTInfinityBranch{D<:MSTGenericData,C} <: InfinityBranch
    data::D
    branch::Symbol
    gsn_scale::C
end

"""
    ArbitraryPrecisionRadialRoute

IN or UP solution of the GSN equation in the floating type of the mode parameters, in the GSN
unit-transmission normalization. Stores the endpoint expansions, the Taylor patches to the matching
point and the connection coefficients: `incidence`, `reflection` are B_inc, B_ref for IN
(X_in = B_inc X_down + B_ref X_up beyond the matching point) and C_inc, C_ref for UP
(X_up = C_inc X_out + C_ref X_in inside it). `route_state(route, x)` evaluates (X, dX/dx).
"""
struct ArbitraryPrecisionRadialRoute{P<:GSNParameters,R,C}
    branch::Symbol
    params::P
    controls::GSNControls
    order::Int
    tolerance::R
    working_bits::Int
    match_x::R
    horizon_in::HorizonExpansion{R,C}
    horizon_in_patches::Vector{SeriesPatch{R,C}}
    horizon_out::Union{Nothing,HorizonExpansion{R,C}}
    horizon_out_patches::Vector{SeriesPatch{R,C}}
    infinity_up::InfinityBranch
    infinity_up_patches::Vector{SeriesPatch{R,C}}
    infinity_down::Union{Nothing,InfinityBranch}
    infinity_down_patches::Vector{SeriesPatch{R,C}}
    incidence::C
    reflection::C
    diagnostics::NamedTuple
end

function _normalized_branch(branch)
    branch in (:IN, :in, :ingoing) && return :IN
    branch in (:UP, :up, :outgoing) && return :UP
    throw(ArgumentError("the arbitrary-precision ISEM route builds :IN or :UP; DOWN and OUT are combined from them."))
end

# Precision of the inputs in bits (BigFloat values carry their own; other types their fixed one).
_input_bits(p::GSNParameters{BigFloat}) = max(precision(p.a), precision(real(p.omega)), precision(imag(p.omega)))
_input_bits(p::GSNParameters{R}) where {R} = precision(R)

# Smallest relative spacing at `bits` bits in the type R.
_ulp(::Type{BigFloat}, bits::Int) = ldexp(one(BigFloat), 1 - bits)
_ulp(::Type{R}, bits::Int) where {R} = eps(R)

# Bits of working precision lost by continuing the inward-decaying branch from the seed radius r0 to
# the matching radius r_m at complex frequency: exp(2 |Im omega| (r0 - r_m)); the seed radius of the
# asymptotic series is about ln(1 / tol) / (2 |omega|) (Sec. 5.3 of the design note).
function _complex_seed_radius(p::GSNParameters, tol)
    return log(1 / Float64(tol)) / (2 * Float64(abs(p.omega)))
end

"""
    arbitrary_precision_route(params, branch; N=nothing, xm=nothing, tol=nothing, infinity=:auto)

IN or UP route in the floating type of `params` (see `ArbitraryPrecisionRadialRoute`). `tol` is the
relative truncation tolerance of every series (default: $(TOLERANCE_ULPS) ulps of the inputs'
precision), `N` the Taylor order of the patches (default: $(ORDER_PER_DIGIT) per requested decimal
digit, at least $(MINIMUM_ORDER); the endpoint series use 2N and 3N terms), `xm` the matching point in
x (default 1/2). `infinity` selects the representation of the infinity side: `:series` (asymptotic
series seeds continued by Taylor patches; exact only for real frequencies), `:mst` (the convergent
MST series of ManoSuzukiTakasugiGeneric, evaluated at every requested point), or `:auto` (`:series`
for real, `:mst` for complex frequencies). Negative real frequencies are built from the mirror mode
(m -> -m, omega -> -omega) and returned as a `ConjugatedRoute`.
"""
function arbitrary_precision_route(params::GSNParameters, branch=:IN; N=nothing, xm=nothing, tol=nothing, controls=nothing, infinity::Symbol=:auto)
    route_branch = _normalized_branch(branch)
    iszero(params.omega) && throw(ArgumentError("the arbitrary-precision ISEM route requires a nonzero frequency."))
    infinity in (:auto, :series, :mst) || throw(ArgumentError("infinity must be :auto, :series or :mst."))
    infinity_mode = infinity === :auto ? (isreal(params.omega) ? :series : :mst) : infinity
    # Re omega < 0 (real or complex): the principal-branch MST formulas are not on the physical sheet
    # there, so the route is built for the mirror mode (m, omega) -> (-m, -conj(omega)), whose
    # separation constant is conj(lambda), and complex-conjugated on evaluation.
    if real(params.omega) < 0
        positive = GSNParameters(params.s, params.l, -params.m, params.a, -conj(params.omega),
            conj(params.lambda), params.nu, params.kappa)
        return ConjugatedRoute(
            arbitrary_precision_route(positive, route_branch; N, xm, tol, controls, infinity), params)
    end
    if controls !== nothing
        # Controls built by gsn_controls carry binary64 defaults; only explicit values are taken.
        tol === nothing && controls.tolerance != DEFAULT_TOLERANCE && (tol = controls.tolerance)
        xm === nothing && controls.match_x != DEFAULT_MATCH_X && (xm = controls.match_x)
    end
    R = _real_type(params)
    bits = _input_bits(params)
    tolerance = tol === nothing ? TOLERANCE_ULPS * _ulp(R, bits) : R(tol)
    tolerance > 0 || throw(ArgumentError("tol must be positive."))
    digits = -log10(Float64(tolerance))
    order = N === nothing ? max(MINIMUM_ORDER, ceil(Int, ORDER_PER_DIGIT * digits)) : Int(N)
    order > 0 || throw(ArgumentError("N must be positive."))
    loss_bits = 0
    if !isreal(params.omega) && infinity_mode == :series
        r0 = _complex_seed_radius(params, tolerance)
        loss_bits = ceil(Int, Float64(abs(imag(params.omega))) * r0 / log(2))
    end
    needed = ceil(Int, -log2(Float64(tolerance))) + GUARD_BITS + loss_bits
    working_bits = R === BigFloat ? max(bits, needed) : precision(R)
    if R === BigFloat
        return setprecision(BigFloat, working_bits) do
            _build_route(params, route_branch, order, tolerance, xm, working_bits, infinity_mode)
        end
    end
    return _build_route(params, route_branch, order, tolerance, xm, working_bits, infinity_mode)
end

# Target |2 omega (r_m - r_-)| of the MST Tricomi arguments at the matching radius: larger arguments
# need fewer terms of the n-sums (and more Kummer cancellation, which the working precision absorbs).
const MST_MATCH_ARGUMENT = 18

function _default_match_x(p::GSNParameters, tolerance, infinity_mode::Symbol)
    R = _real_type(p)
    if infinity_mode == :mst
        r_target = r_minus(p) + R(MST_MATCH_ARGUMENT) / (2 * abs(p.omega))
        lo = r_from_x(p, R(1 // 2))
        hi = r_from_x(p, R(19 // 20))
        return x_from_r(p, clamp(r_target, lo, hi))
    end
    isreal(p.omega) && return R(1 // 2)
    # Asymptotic seed at complex frequency: match at half the seed radius so that both sides share
    # the continuation loss.
    r0 = _complex_seed_radius(p, tolerance)
    return max(R(1 // 2), x_from_r(p, R(max(r0 / 2, Float64(r_plus(p)) + 2))))
end

# Binary64 seed of the renormalized angular momentum from the Float64 MST module; `l` when it fails.
function _nu_seed(p::GSNParameters)
    a = Float64(p.a)
    omega = ComplexF64(p.omega)
    lambda = ComplexF64(p.lambda)
    try
        if isreal(p.omega)
            return ComplexF64(mano_suzuki_takasugi_nu(p.s, p.l, p.m, a, real(omega), lambda))
        end
        return ComplexF64(mano_suzuki_takasugi_nu_complex(p.s, p.l, p.m, a, omega, lambda))
    catch
        return ComplexF64(p.l)
    end
end

# MST data and the two infinity branches in GSN unit-transmission normalization.
function _mst_branches(p::GSNParameters, tolerance)
    nu, residual, depth = mst_nu_generic(p, _nu_seed(p); tol=tolerance)
    data = mst_generic_data(p, nu; tol=tolerance, nu_residual=residual, nu_depth=depth)
    up = MSTInfinityBranch(data, :up, ConversionFactors.Ctrans(p.s, p.m, p.a, p.omega, p.lambda))
    down = MSTInfinityBranch(data, :down, ConversionFactors.Binc(p.s, p.m, p.a, p.omega, p.lambda))
    return data, up, down
end

function _build_route(p::GSNParameters, branch::Symbol, order::Int, tolerance, xm, working_bits::Int, infinity_mode::Symbol)
    R = _real_type(p)
    C = _complex_type(p)
    controls = gsn_controls(p; N=order, tol=Float64(tolerance),
        xm=xm === nothing ? nothing : Float64(xm),
        small_frequency_expansion=false, large_frequency_expansion=false)
    coefficients = gsn_coefficients(p; controls=controls)
    match_x = xm === nothing ? _default_match_x(p, tolerance, infinity_mode) : R(xm)
    0 < match_x < 1 || throw(ArgumentError("xm must lie in (0, 1)."))
    work = _Workspace(C, coefficients, order)

    horizon_in = _horizon_expansion(coefficients, :in, 2order, tolerance)
    Xh, dXh = _horizon_state(horizon_in, horizon_in.x_seed)
    horizon_in_patches, Xin, dXin = _continue(work, coefficients, horizon_in.x_seed, Xh, dXh, match_x, order, tolerance)

    mst = nothing
    infinity_up_patches = SeriesPatch{R,C}[]
    infinity_down_patches = SeriesPatch{R,C}[]
    local infinity_up, Xup, dXup, mst_down
    if infinity_mode == :mst
        mst, infinity_up, mst_down = _mst_branches(p, tolerance)
        Xup, dXup = _infinity_state(infinity_up, match_x)
    else
        infinity_up = _infinity_expansion(coefficients, :up, 3order, tolerance)
        Xu, dXu = _infinity_state(infinity_up, one(R) - infinity_up.y_seed)
        infinity_up_patches, Xup, dXup = _continue(work, coefficients, one(R) - infinity_up.y_seed, Xu, dXu, match_x, order, tolerance)
    end

    horizon_out = nothing
    horizon_out_patches = SeriesPatch{R,C}[]
    infinity_down = nothing
    if branch == :IN
        if infinity_mode == :mst
            infinity_down = mst_down
            Xdn, dXdn = _infinity_state(infinity_down, match_x)
        else
            infinity_down = _infinity_expansion(coefficients, :down, 3order, tolerance)
            Xd, dXd = _infinity_state(infinity_down, one(R) - infinity_down.y_seed)
            infinity_down_patches, Xdn, dXdn = _continue(work, coefficients, one(R) - infinity_down.y_seed, Xd, dXd, match_x, order, tolerance)
        end
        # X_in = B_inc X_down + B_ref X_up at the matching point.
        det = Xdn * dXup - Xup * dXdn
        incidence = (Xin * dXup - Xup * dXin) / det
        reflection = (Xdn * dXin - Xin * dXdn) / det
    else
        is_horizon_threshold(p) && throw(ArgumentError(
            "the arbitrary-precision UP route does not build the logarithmic horizon companion at the superradiant threshold."))
        horizon_out = _horizon_expansion(coefficients, :out, 2order, tolerance)
        Xo, dXo = _horizon_state(horizon_out, horizon_out.x_seed)
        horizon_out_patches, Xout, dXout = _continue(work, coefficients, horizon_out.x_seed, Xo, dXo, match_x, order, tolerance)
        # X_up = C_inc X_out + C_ref X_in at the matching point.
        det = Xout * dXin - Xin * dXout
        incidence = (Xup * dXin - Xin * dXup) / det
        reflection = (Xout * dXup - Xup * dXout) / det
    end

    r_m = r_from_x(p, match_x)
    series = infinity_up isa InfinityExpansion
    r0 = series ? Float64(2 * p.kappa / infinity_up.y_seed) : Float64(r_m)
    loss = (isreal(p.omega) || !series) ? 0.0 :
        Float64(2 * abs(imag(p.omega)) * max(r0 - Float64(r_m), 0) / log(10))
    diagnostics = (
        working_bits = working_bits,
        input_bits = _input_bits(p),
        tolerance = Float64(tolerance),
        order = order,
        horizon_order = 2order,
        infinity_order = series ? 3order : 0,
        match_x = Float64(match_x),
        r_m = Float64(r_m),
        horizon_seed_x = Float64(horizon_in.x_seed),
        infinity_seed_y = series ? Float64(infinity_up.y_seed) : 0.0,
        infinity_seed_r = r0,
        infinity_terms = series ? length(infinity_up.coeffs) : (mst.nmax - mst.nmin + 1),
        horizon_patches = length(horizon_in_patches) + length(horizon_out_patches),
        infinity_patches = length(infinity_up_patches) + length(infinity_down_patches),
        matching_determinant = det,
        complex_loss_digits = loss,
        frequency_regime = :arbitrary_precision_series,
        infinity_representation = series ? :asymptotic_series : :mst,
        mst_nu = mst === nothing ? missing : ComplexF64(mst.nu),
        mst_nu_residual = mst === nothing ? missing : Float64(mst.nu_residual),
        mst_depth = mst === nothing ? missing : mst.nu_depth,
    )
    return ArbitraryPrecisionRadialRoute{typeof(p),R,C}(
        branch, p, controls, order, tolerance, working_bits, match_x,
        horizon_in, horizon_in_patches, horizon_out, horizon_out_patches,
        infinity_up, infinity_up_patches, infinity_down, infinity_down_patches,
        incidence, reflection, diagnostics)
end

# Buffers of the ordinary-point expansions: packed P/Q values and the A, B Taylor coefficients.
struct _Workspace{C}
    pq::Vector{C}
    a::Vector{C}
    b::Vector{C}
end
function _Workspace(::Type{C}, coefficients::CoefficientSet, order::Int) where {C}
    a_terms, b_terms = ab_term_counts(:O, order)
    return _Workspace{C}(
        Vector{C}(undef, coefficients.ordinary.value_count),
        Vector{C}(undef, a_terms),
        Vector{C}(undef, b_terms))
end

# Relative size of the last four terms of a polynomial at distance h, against the value.
function _tail_small(coeffs, h, value, tolerance)
    R = typeof(h)
    tail = zero(R)
    hk = h^(length(coeffs) - 4)
    @inbounds for k in max(1, length(coeffs) - 3):length(coeffs)
        tail += abs(coeffs[k]) * hk
        hk *= h
    end
    return tail <= tolerance * max(abs(value), floatmin(R))
end

# Horizon Frobenius series and its seed point: the largest x <= HORIZON_SEED_XMAX at which the last
# terms are below the tolerance.
function _horizon_expansion(coefficients::CoefficientSet, branch::Symbol, order::Int, tolerance)
    p = coefficients.params
    R = _real_type(p)
    local_solution = zero_local_solution(coefficients, branch, order)
    coeffs = local_solution.coefficients
    x = R(HORIZON_SEED_XMAX)
    for _ in 1:400
        value, _ = poly_pair(coeffs, x)
        _tail_small(coeffs, x, value, tolerance) && break
        x *= R(4 // 5)
    end
    return HorizonExpansion{R,eltype(coeffs)}(local_solution.exponent, _horizon_scale(p, branch), coeffs, x)
end

# (X, dX/dx) of a horizon expansion at x.
function _horizon_state(h::HorizonExpansion, x)
    value, derivative = poly_pair(h.coeffs, x)
    prefactor = h.scale * x^h.exponent
    return prefactor * value, prefactor * (derivative + h.exponent * value / x)
end

# Asymptotic series at infinity, optimally truncated: the seed y is the largest y <= INFINITY_SEED_YMAX
# at which a term falls below the tolerance relative to the partial sum (with its predecessor within
# a factor 10 of it); the series is cut there.
function _infinity_expansion(coefficients::CoefficientSet, branch::Symbol, order::Int, tolerance)
    p = coefficients.params
    R = _real_type(p)
    local_solution = infinity_local_solution(coefficients, branch, order)
    d = local_solution.coefficients
    C = eltype(d)
    y = R(INFINITY_SEED_YMAX)
    nopt = length(d)
    for _ in 1:400
        partial = zero(C)
        previous = zero(R)
        yk = one(R)
        found = false
        @inbounds for k in 1:length(d)
            term = abs(d[k]) * yk
            partial += d[k] * yk
            scale = tolerance * max(abs(partial), floatmin(R))
            if k >= 3 && term <= scale && previous <= 10 * scale
                nopt = k
                found = true
                break
            end
            previous = term
            yk *= y
        end
        found && break
        y *= R(4 // 5)
    end
    return InfinityExpansion{R,C}(local_solution.rho, local_solution.sigma,
        _infinity_scale(p, branch), d[1:nopt], y)
end

# (X, dX/dx) of an infinity expansion at x (y = 1 - x; dx = -dy).
function _infinity_state(e::InfinityExpansion, x)
    y = one(x) - x
    value, derivative = poly_pair(e.coeffs, y)
    prefactor = e.scale * exp(e.sigma / y) * y^e.rho
    dXdy = prefactor * (derivative + (-e.sigma / y^2 + e.rho / y) * value)
    return prefactor * value, -dXdy
end

# (X, dX/dx) of an MST branch at x: the Teukolsky state at r, scaled to the GSN unit-transmission
# normalization, and converted with the inverse of the GSN -> Teukolsky conversion matrix.
function _infinity_state(e::MSTInfinityBranch, x)
    p = e.data.params
    r = r_from_x(p, x)
    state = e.branch == :up ? mst_up_state(e.data, r) : mst_down_state(e.data, r)
    matrix = Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
        p.s, p.m, p.a, p.omega, p.lambda, r)
    gsn = inv(matrix) * SA[e.gsn_scale * state.R; e.gsn_scale * state.dR]
    return gsn[1], gsn[2] / dx_drstar(p, x, r)
end

# One Taylor patch from x0 (state X, dX) towards `target`: the step is at most half the empirical
# radius of the solution series (ratio of its last coefficients), then shrunk until the last four
# terms are below the tolerance relative to the value. Returns the patch and the new state.
function _taylor_step(work::_Workspace, coefficients::CoefficientSet, x0, X, dX, target, order::Int, tolerance)
    R = typeof(x0)
    C = eltype(work.pq)
    ordinary_ab_series!(work.a, work.b, work.pq, coefficients, x0, order)
    coeffs = Vector{C}(undef, order + 1)
    ordinary_point_coeffs!(coeffs, work.a, work.b, X, dX, order)
    radius = R(Inf)
    @inbounds for k in max(1, length(coeffs) - 6):(length(coeffs) - 1)
        (iszero(coeffs[k + 1]) || iszero(coeffs[k])) && continue
        radius = min(radius, abs(coeffs[k]) / abs(coeffs[k + 1]))
    end
    distance = abs(target - x0)
    direction = target >= x0 ? one(R) : -one(R)
    h = min(distance, radius / 2)
    xh = h == distance ? target : x0 + direction * h
    for _ in 1:200
        value, _ = poly_pair(coeffs, xh - x0)
        _tail_small(coeffs, abs(xh - x0), value, tolerance) && break
        h *= R(2 // 3)
        xh = x0 + direction * h
    end
    value, derivative = poly_pair(coeffs, xh - x0)
    return SeriesPatch{R,C}(x0, xh, coeffs), xh, value, derivative
end

function _continue(work::_Workspace, coefficients::CoefficientSet, x0, X, dX, target, order::Int, tolerance)
    R = typeof(x0)
    C = eltype(work.pq)
    patches = SeriesPatch{R,C}[]
    x = x0
    for _ in 1:MAX_PATCHES
        patch, x, X, dX = _taylor_step(work, coefficients, x, X, dX, target, order, tolerance)
        push!(patches, patch)
        x == target && return patches, X, dX
        abs(x - target) <= 4 * eps(R) * max(one(R), abs(target)) && return patches, X, dX
    end
    error("arbitrary-precision ISEM: the Taylor continuation did not reach x = $target from x = $x0 in $MAX_PATCHES patches.")
end

# (X, dX/dx) from a chain of patches at x; `nothing` outside the chain.
function _patch_state(patches::Vector{<:SeriesPatch}, x)
    isempty(patches) && return nothing
    R = typeof(x)
    slack = 8 * eps(R) * max(one(R), abs(x))
    @inbounds for patch in patches
        lo, hi = minmax(patch.x0, patch.xend)
        if lo - slack <= x <= hi + slack
            value, derivative = poly_pair(patch.coeffs, x - patch.x0)
            return value, derivative
        end
    end
    return nothing
end

function _chain_range(patches::Vector{<:SeriesPatch})
    lo = minimum(min(patch.x0, patch.xend) for patch in patches)
    hi = maximum(max(patch.x0, patch.xend) for patch in patches)
    return lo, hi
end

# State of the infinity branch `which` (:up or :down) at x: patches inside their chain, the
# asymptotic series beyond the seed point.
function _infinity_branch_state(route::ArbitraryPrecisionRadialRoute, which::Symbol, x)
    expansion = which == :up ? route.infinity_up : route.infinity_down
    patches = which == :up ? route.infinity_up_patches : route.infinity_down_patches
    expansion === nothing && error("the $(route.branch) route has no $which expansion.")
    expansion isa MSTInfinityBranch && return _infinity_state(expansion, x)
    x >= one(x) - expansion.y_seed && return _infinity_state(expansion, x)
    state = _patch_state(patches, x)
    state === nothing || return state
    error("arbitrary-precision ISEM: x = $x is outside the $which chain of the $(route.branch) route.")
end

function _horizon_branch_state(route::ArbitraryPrecisionRadialRoute, which::Symbol, x)
    expansion = which == :in ? route.horizon_in : route.horizon_out
    patches = which == :in ? route.horizon_in_patches : route.horizon_out_patches
    expansion === nothing && error("the $(route.branch) route has no horizon $which expansion.")
    x <= expansion.x_seed && return _horizon_state(expansion, x)
    state = _patch_state(patches, x)
    state === nothing || return state
    error("arbitrary-precision ISEM: x = $x is outside the horizon $which chain of the $(route.branch) route.")
end

"""
    route_state(route, x)

(X, dX/dx) of the route's solution at the compact coordinate x in (0, 1).
"""
function route_state(route::ArbitraryPrecisionRadialRoute, x)
    R = _real_type(route.params)
    xx = R(x)
    0 < xx < 1 || throw(DomainError(x, "the compact coordinate must lie in (0, 1)."))
    if route.branch == :IN
        xx <= route.match_x && return _horizon_branch_state(route, :in, xx)
        down = _infinity_branch_state(route, :down, xx)
        up = _infinity_branch_state(route, :up, xx)
        return route.incidence * down[1] + route.reflection * up[1],
            route.incidence * down[2] + route.reflection * up[2]
    end
    xx >= route.match_x && return _infinity_branch_state(route, :up, xx)
    out = _horizon_branch_state(route, :out, xx)
    inn = _horizon_branch_state(route, :in, xx)
    return route.incidence * out[1] + route.reflection * inn[1],
        route.incidence * out[2] + route.reflection * inn[2]
end

"""
    route_state_r(route, r)

(X, dX/dr*) at the Boyer-Lindquist radius r.
"""
function route_state_r(route::ArbitraryPrecisionRadialRoute, r)
    p = route.params
    rr = _real_type(p)(r)
    x = x_from_r(p, rr)
    X, dXdx = route_state(route, x)
    return X, dXdx * dx_drstar(p, x, rr)
end

# ---------------------------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------------------------
function _route_mode(route::ArbitraryPrecisionRadialRoute)
    p = route.params
    return getfield(_root_module(), :Mode)(p.s, p.l, p.m, p.a, p.omega, p.lambda)
end

function _route_boundary_condition(route::ArbitraryPrecisionRadialRoute)
    root = _root_module()
    return route.branch == :IN ? getfield(root, :IN) : getfield(root, :UP)
end

# r* -> (X, dX/dr*, 0) in the floating type of the route.
struct GSNStateEvaluator{Rt}
    route::Rt
end
function (e::GSNStateEvaluator)(rs)
    p = e.route.params
    R = _real_type(p)
    r = r_from_rstar(p.a, R(rs))
    X, dXdrstar = route_state_r(e.route, r)
    return (X, dXdrstar, zero(R))
end

function gsn_radial_function(route::ArbitraryPrecisionRadialRoute)
    p = route.params
    root = _root_module()
    C = _complex_type(p)
    rsmp = rstar_from_r(p.a, r_from_x(p, route.match_x))
    metadata = merge(route.diagnostics, (
        xm = Float64(route.match_x),
        tol = Float64(route.tolerance),
        ordinary_order = route.order,
        unit_normalization_available = true,
        normalization = :unit_gsn_transmission,
    ))
    return getfield(root, :GSNRadialFunction)(
        _route_mode(route),
        _route_boundary_condition(route),
        missing,
        missing,
        rsmp,
        2 * route.order,
        3 * route.order,
        one(C),
        route.incidence,
        route.reflection,
        metadata,
        missing,
        GSNStateEvaluator(route),
        getfield(root, :UNIT_GSN_TRANS),
        "ISEM",
    )
end

function _teukolsky_amplitude_factors(route::ArbitraryPrecisionRadialRoute)
    p = route.params
    if route.branch == :IN
        return (
            trans = ConversionFactors.Btrans(p.s, p.m, p.a, p.omega, p.lambda),
            incidence = ConversionFactors.Binc(p.s, p.m, p.a, p.omega, p.lambda),
            reflection = ConversionFactors.Bref(p.s, p.m, p.a, p.omega, p.lambda),
        )
    end
    return (
        trans = ConversionFactors.Ctrans(p.s, p.m, p.a, p.omega, p.lambda),
        incidence = ConversionFactors.Cinc(p.s, p.m, p.a, p.omega, p.lambda),
        reflection = ConversionFactors.Cref(p.s, p.m, p.a, p.omega, p.lambda),
    )
end

# r -> (R, dR/dr) divided by `scale`, through the closed-form GSN-to-Teukolsky converter.
struct TeukolskyPairEvaluator{Rt,Cv,T}
    route::Rt
    converter::Cv
    inv_scale::T
end
function _teukolsky_pair_with_error(e::TeukolskyPairEvaluator, r)
    p = e.route.params
    rr = _real_type(p)(r)
    X, dXdx = route_state(e.route, x_from_r(p, rr))
    R, Rp = e.converter(rr, X, dXdx)
    return e.inv_scale * R, e.inv_scale * Rp, zero(_real_type(p))
end
function (e::TeukolskyPairEvaluator)(r)
    R, Rp, _ = _teukolsky_pair_with_error(e, r)
    return R, Rp
end

function teukolsky_radial_function(route::ArbitraryPrecisionRadialRoute; normalize=:teukolsky)
    p = route.params
    root = _root_module()
    factors = _teukolsky_amplitude_factors(route)
    scale, convention = if normalize == :teukolsky
        factors.trans, getfield(root, :UNIT_TEUKOLSKY_TRANS)
    elseif normalize == :gsn
        one(factors.trans), getfield(root, :UNIT_GSN_TRANS)
    else
        throw(ArgumentError("normalize must be :teukolsky or :gsn."))
    end
    pair = TeukolskyPairEvaluator(route, _teukolsky_converter(p), inv(scale))
    solution = TeukolskySolution(
        TeukolskyTupleEvaluator(p, pair, d2R), pair, TeukolskyRadialEvaluator(pair))
    return getfield(root, :TeukolskyRadialFunction)(
        _route_mode(route),
        _route_boundary_condition(route),
        factors.trans / scale,
        route.incidence * factors.incidence / scale,
        route.reflection * factors.reflection / scale,
        missing,
        gsn_radial_function(route),
        solution,
        convention,
    )
end

function y_radial_function(route::ArbitraryPrecisionRadialRoute)
    p = route.params
    (p.s == -2 && route.branch == :IN) || (p.s == 2 && route.branch == :UP) ||
        error("ISEM Y transformation supports only s = -2 with IN and s = +2 with UP.")
    root = _root_module()
    isem = _isem_module()
    factors = _teukolsky_amplitude_factors(route)
    amplitudes = (
        trans = factors.trans,
        incidence = factors.incidence * route.incidence,
        reflection = factors.reflection * route.reflection,
    )
    gsn_solution = GSNStateEvaluator(route)
    conversion_matrix = r -> Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(
        p.s, p.m, p.a, p.omega, p.lambda, r)
    ysoln = GSN_to_Y_solution_from_matrix(
        r -> rstar_from_r(p.a, r), conversion_matrix, gsn_solution,
        p.s, p.m, p.a, p.omega, p.lambda, amplitudes.trans)
    xsoln = r -> gsn_solution(rstar_from_r(p.a, r))[1]
    yscalar = r -> ysoln(r)[1]
    return getfield(isem, :YRadialFunction)(
        _route_mode(route),
        _route_boundary_condition(route),
        amplitudes.trans,
        amplitudes.incidence,
        amplitudes.reflection,
        missing,
        missing,
        xsoln,
        yscalar,
        ysoln,
        getfield(root, :UNIT_TEUKOLSKY_TRANS),
    )
end

end
