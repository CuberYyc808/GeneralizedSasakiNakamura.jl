# Point-particle modes in arbitrary precision.
#
# Teukolsky_pointparticle_mode with BigFloat (or another non-binary64 float) orbital parameters comes here: the
# orbit from KerrGeodesics in that type (its functions of Mino time evaluate at the working precision), the
# spin-weighted spheroidal harmonic from SpinWeightedSpheroidalHarmonics in that type, the Y function from the
# arbitrary-precision ISEM route (Y_radial with the same type), and the source integral over the radial and polar
# Mino phases q_r, q_theta in [0, pi] by the trapezoidal rule, whose error falls exponentially for the periodic,
# analytic integrand. Each phase grid is doubled until the amplitude changes by less than `tol` (default 1e3 eps(T));
# an equatorial orbit has a single polar node (theta = pi/2) and a circular one a single radial node. The source
# terms are those of GridSampling.integrand_generic_sample_m2 / _p2 (s = -2 at infinity, s = +2 at the horizon),
# written for a generic number type; the binary64 routes are not touched.
module ArbitraryPrecisionModes

using KerrGeodesics: kerr_geo_orbit
using SpinWeightedSpheroidalHarmonics: spin_weighted_spheroidal_harmonic, value_and_theta_derivative
using ..Coordinates: rstar_from_r
using ..ConversionFactors
using ..ISEM: Y_radial, IN, UP
using ..GridSampling: _phase_values, _radial_phase_rates, _polar_phase_rates, _theta_constants

export arbitrary_precision_pointparticle_mode

# Largest phase grid tried in each direction before giving up.
const AP_MODE_MAX_NODES = 2^14
const AP_MODE_START_NODES = 32

struct APModeOrbit{T}
    a::T
    p::T
    e::T
    x::T
    KG::Dict
    equatorial::Bool
    circular::Bool
    frequencies::NTuple{4,T}     # (ϒt, ϒr, ϒθ, ϒϕ)
    E::T
    Lz::T
end

function _ap_orbit(a::T, p::T, e::T, x::T) where {T}
    KG = kerr_geo_orbit(a, p, e, x)
    F = KG["Frequencies"]
    equatorial = abs(x) == 1
    circular = iszero(e)
    return APModeOrbit{T}(a, p, e, x, KG, equatorial, circular,
        (T(F["ϒt"]), T(F["ϒr"]), T(F["ϒθ"]), T(F["ϒϕ"])), T(KG["Energy"]), T(KG["AngularMomentum"]))
end

_cross(KG) = KG["CrossFunction"] === nothing ? (q -> 0, q -> 0, q -> 0, q -> 0) : Tuple(KG["CrossFunction"])

# Phase nodes and trapezoidal weights on [0, pi] (n >= 2), or the single node 0 with weight pi. (Packing the radial
# nodes towards periapsis with q = u - alpha sin(u) converged more slowly at (0.9, 8, 0.95, 1) for every alpha tried.)
function _phase_grid(::Type{T}, n::Int) where {T}
    n == 1 && return T[zero(T)], T[T(pi)]
    q = collect(range(zero(T), T(pi); length = n))
    w = fill(T(pi) / (n - 1), n)
    w[1] /= 2
    w[end] /= 2
    return q, w
end

# Radial samples: r, r*, Δt_r, Δφ_r, Σ u^r on the q_r grid (forward branch; the reverse branch is its negative).
function _radial_samples(o::APModeOrbit{T}, qr) where {T}
    KG = o.KG
    _, qr0, _, _ = Tuple(T.(KG["InitialPhases"]))
    ϒr = o.frequencies[2]
    Δtr, _, Δφr, _ = _cross(KG)
    if o.circular
        n = length(qr)
        return fill(o.p, n), fill(rstar_from_r(o.a, o.p), n), zeros(T, n), zeros(T, n), zeros(T, n)
    end
    _, r, θ, _ = KG["Trajectory"]
    _, ur, _, _ = KG["FourVelocity"]
    rv = T.(_phase_values(r, qr, qr0, ϒr))
    theta = o.equatorial ? (λ -> T(pi) / 2) : θ
    urv = T.(_radial_phase_rates(ur, theta, qr, rv, o.a, qr0, ϒr; constants = _theta_constants(KG)))
    return rv, T[rstar_from_r(o.a, r) for r in rv], T.(map(Δtr, qr)), T.(map(Δφr, qr)), urv
end

# Polar samples: θ, Δt_θ, Δφ_θ, Σ u^θ on the q_θ grid (forward branch), computed on the first half of a full grid
# and extended by the symmetries of the binary64 sampler (Δ odd, Σ u^θ even about q_θ = pi / 2).
function _polar_samples(o::APModeOrbit{T}, qθ) where {T}
    KG = o.KG
    _, _, qθ0, _ = Tuple(T.(KG["InitialPhases"]))
    ϒθ = o.frequencies[3]
    if o.equatorial
        n = length(qθ)
        return fill(T(pi) / 2, n), zeros(T, n), zeros(T, n), zeros(T, n)
    end
    _, Δtθ, _, Δφθ = _cross(KG)
    _, r, θ, _ = KG["Trajectory"]
    _, _, uθ, _ = KG["FourVelocity"]
    n = length(qθ)
    θv = T.(_phase_values(θ, qθ, qθ0, ϒθ))
    half = 1:div(n, 2)
    Δt_half = T.(map(Δtθ, qθ[half]))
    Δφ_half = T.(map(Δφθ, qθ[half]))
    u_half = T.(_polar_phase_rates(uθ, r, qθ[half], θv[half], o.a, qθ0, ϒθ; constants = _theta_constants(KG)))
    return θv, vcat(Δt_half, -reverse(Δt_half)), vcat(Δφ_half, -reverse(Δφ_half)), vcat(u_half, reverse(u_half))
end

function _swsh_samples(SH, θv, s, m, c, λ)
    S0 = similar(θv, Complex{eltype(θv)}); S1 = similar(S0); S2 = similar(S0)
    for (i, θ) in enumerate(θv)
        s0, s1 = value_and_theta_derivative(SH, θ, zero(θ))
        sθ, cθ = sin(θ), cos(θ)
        S0[i] = s0
        S1[i] = s1
        S2[i] = -(cθ / sθ) * s1 + (((m + s * cθ)^2) / sθ^2 + 2c * s * cθ + c^2 * sθ^2 - s - 2c * m - λ) * s0
    end
    return S0, S1, S2
end

# Serial on purpose: an arbitrary-precision route extends its MST chain on demand (_extend_mst_chain!, under a
# lock) while its state readers do not lock, so one Y function must not be evaluated from several threads.
function _y_samples(Ysol, rv)
    n = length(rv)
    C = Complex{eltype(rv)}
    Y = Vector{C}(undef, n); Yp = similar(Y); X = similar(Y)
    for (i, r) in enumerate(rv)
        y, yp, x, _ = Ysol.Y_solution(r)
        Y[i], Yp[i], X[i] = y, yp, x
    end
    return Y, Yp, X
end

# Trapezoidal amplitude on the (q_r, q_θ) grid: the source terms of integrand_generic_sample_m2 (s = -2) and
# integrand_generic_sample_p2 (s = +2) at each node, weighted and summed.
function _amplitude(o::APModeOrbit{T}, s, m, n, k, ω, λ, Ysol, SH, incidence, Nr, Kθ) where {T}
    a, E, Lz = o.a, o.E, o.Lz
    ϒt = o.frequencies[1]
    qr, wr = _phase_grid(T, Nr)
    qθ, wθ = _phase_grid(T, Kθ)
    rv, rsv, Δtr, Δφr, urv = _radial_samples(o, qr)
    θv, Δtθ, Δφθ, uθv = _polar_samples(o, qθ)
    S0, S1, S2 = _swsh_samples(SH, θv, s, m, a * ω, λ)
    Yv, Ypv, Xv = _y_samples(Ysol, rv)
    one_ = one(T)
    kappa = sqrt(one_ - a^2)
    rp, rm = one_ + kappa, one_ - kappa
    sgn = s == -2 ? 1 : -1                     # s = +2 flips the signs of m and a ω in L1, L2
    # Polar factors (length K).
    pol = map(1:length(θv)) do j
        θ = θv[j]
        sθ, cθ = sin(θ), cos(θ)
        L1 = sgn * (-m / sθ + a * ω * sθ) + cθ / sθ
        L2 = sgn * (-m / sθ + a * ω * sθ) + 2cθ / sθ
        L2p = sgn * (m * cθ / sθ^2 + a * ω * cθ) - 2 / sθ^2
        L2S = S1[j] + L2 * S0[j]
        L1L2S = S2[j] + L1 * S1[j] + L2p * S0[j] + L2 * S1[j] + L1 * L2 * S0[j]
        (sθ = sθ, cθ = cθ, L1 = L1, L2 = L2, L2S = L2S, L1L2S = L1L2S,
            θphase = ω * Δtθ[j] - m * Δφθ[j] + k * qθ[j])
    end
    # Radial factors (length N).
    rad = map(1:length(rv)) do i
        r = rv[i]
        logt = log((r - rp) / (r - rm))
        ph = cis(sgn * (ω * rsv[i] - a * m * logt / (2kappa)))
        (r = r, Δ = r^2 - 2r + a^2, ph = ph, rphase = ω * Δtr[i] - m * Δφr[i] + n * qr[i])
    end
    total = zero(Complex{T})
    magnitude = zero(T)
    for i in eachindex(rv), j in eachindex(θv)
        R = rad[i]; P = pol[j]
        r, sθ, cθ = R.r, P.sθ, P.cθ
        ρ = -one_ / complex(r, -a * cθ)
        ρbar = -one_ / complex(r, a * cθ)
        L1pL2pS = P.L1L2S / ρ + 3im * a * sθ * P.L1 * S0[j] + 3im * a * cθ * S0[j] + 2im * a * sθ * S1[j] -
            1im * a * sθ * P.L2 * S0[j]
        Y, Yp, X = Yv[i], Ypv[i], Xv[i]
        tA = P.L2S + im * a * sθ * (ρ - ρbar) * S0[j]
        tB = 2Y + r * Yp
        tC = P.L2S * (ρ + ρbar) * r * Y
        tD = X / (2sqrt(r^2 + a^2))
        tE = (Y + 2r * Yp) * R.ph
        tF = ρ * r * (2Y + r * Yp) * R.ph
        base = E * (r^2 + a^2) - a * Lz
        tM = im * sθ * (a * E - (iszero(Lz) ? zero(T) : Lz / sθ^2))
        if s == -2
            W1 = ρbar * L1pL2pS * r^2 * Y * R.ph / 2
            W2 = -r * ρbar * (tA * tB + tC) / ρ * R.ph
            W3 = S0[j] * ρbar * (tD + tE + tF) / ρ
        else
            W1 = 2ρbar * L1pL2pS * r^2 * Y * R.ph
            W2 = 4r * ρbar * (tA * tB + tC) / ρ * R.ph
            W3 = 4S0[j] * ρbar * (tD + tE + tF) / ρ
        end
        # N = (base + u^r) / Δ, Mbar = tM + u^θ for s = -2; L = (base - u^r) / Δ, M = -tM + u^θ for s = +2.
        J = zero(Complex{T})
        for (σr, σθ) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            v = (base + sgn * σr * urv[i]) / R.Δ
            u = sgn * tM + σθ * uθv[j]
            J += (W1 * v^2 + W2 * v * u + W3 * u^2) * cis(σr * R.rphase + σθ * P.θphase)
        end
        total += wr[i] * wθ[j] * J
        magnitude += wr[i] * wθ[j] * abs(J)
    end
    # Roundoff floor of the sum relative to its value (the terms cancel by magnitude / |total|).
    floor_ = 100 * eps(T) * magnitude / max(abs(total), floatmin(T))
    # (1 / 4 pi^2) over the full periods; folding onto [0, pi]^2 is in the four branches.
    Jlmnk = total / (4 * T(pi)^2)
    KG = o.KG
    qt0, qr0, qθ0, qφ0 = Tuple(T.(KG["InitialPhases"]))
    Δtr_f, Δtθ_f, Δφr_f, Δφθ_f = _cross(KG)
    Xi = m * (Δφr_f(qr0) + Δφθ_f(qθ0) - qφ0) - ω * (Δtr_f(qr0) + Δtθ_f(qθ0) - qt0) - k * qθ0 - n * qr0
    if s == -2
        return 4im * T(pi) * ω * Jlmnk * cis(Xi) / (ϒt * incidence), floor_
    end
    κ = ω - a * m / (2rp)
    c0 = 24 + 12im * ω + λ * (10 + λ) - 12 * a * ω * (a * ω - m)
    c1 = -32im * a * m - 8im * a * m * λ + 8im * a^2 * ω * (1 + λ)
    c2 = 12 * a^2 - 24im * a * m - 24 * a^2 * m^2 + 24im * a^2 * ω + 48 * a^3 * m * ω - 24 * a^4 * ω^2
    c3 = -24im * a^3 * (a * ω - m) - 24 * a^2
    c4 = 12 * a^4
    η = c0 + c1 / rp + c2 / rp^2 + c3 / rp^3 + c4 / rp^4
    factor = 2 * sqrt(T(2)) * rp^(T(3) / 2) * (((4ω) * (im - 4ω) - a * m * (im - 8ω) - a^2 * (m^2 + 2im * ω - 4ω^2)) * rp^2 +
        a^2 * (im - 4ω) * (a * m - 2ω) * rp) / (2rp^3 * (24 + 10λ + λ^2 + 12im * ω) - rp^2 * (8im * a * m * (11 +
        2λ + 6im * ω) + a^2 * (24 + 24m^2 + 10λ + λ^2 - 28im * ω - 16im * λ * ω + 48ω^2)) + 8im * a^3 * rp * (m *
        (7 + λ - 6im * ω) - a * ω * (4 + λ)) + 12a^5 * ω * (a * ω - 3m))
    return -im * T(pi) * factor * Jlmnk * cis(Xi) * η / (ϒt * κ * incidence), floor_
end

# Mino-time polar average of (m cot^2θ Lz - a^2 ω cos^2θ E) / π on the q_θ grid (the Carter flux factor).
function _carter_factor(o::APModeOrbit{T}, m, ω, Kθ) where {T}
    o.equatorial && return zero(T)
    qθ, wθ = _phase_grid(T, Kθ)
    θv, _, _, _ = _polar_samples(o, qθ)
    return sum(wθ[j] * (m * cot(θv[j])^2 * o.Lz - o.a^2 * ω * cos(θv[j])^2 * o.E) for j in eachindex(θv)) / T(pi)
end

_relchange(new, old) = abs(new - old) / max(abs(new), floatmin(real(typeof(new))))

"""
    arbitrary_precision_pointparticle_mode(s, l, m, n, k, a, p, e, x; tol = nothing)

Teukolsky amplitude and fluxes of one point-particle mode with every ingredient in the floating-point type of
`(a, p, e, x)` (BigFloat at the current precision, or e.g. Double64). Returns a NamedTuple with `omega`, `lambda`,
`amplitude`, `energy_flux`, `angular_momentum_flux`, `carter_constant_flux`, the grid sizes `N`, `K`, the
last relative change `change` of the amplitude, and the orbit, Y solution and harmonic.
"""
function arbitrary_precision_pointparticle_mode(s::Int, l::Int, m::Int, n::Int, k::Int, a::T, p::T, e::T, x::T;
        tol = nothing) where {T<:AbstractFloat}
    s in (-2, 2) || throw(ArgumentError("arbitrary-precision point-particle modes support s = -2 and s = 2."))
    tolerance = tol === nothing ? 1000 * eps(T) : T(tol)
    o = _ap_orbit(a, p, e, x)
    o.equatorial && k != 0 && throw(ArgumentError("an equatorial orbit has no polar harmonics: k must be 0."))
    o.circular && n != 0 && throw(ArgumentError("a circular orbit has no radial harmonics: n must be 0."))
    ϒt, ϒr, ϒθ, ϒϕ = o.frequencies
    ω = (m * ϒϕ + n * ϒr + k * ϒθ) / ϒt
    SH = spin_weighted_spheroidal_harmonic(s, l, m, a * ω)
    λ = SH.lambda
    Ysol = Y_radial(s, l, m, a, ω, s == -2 ? IN : UP)
    incidence = s == -2 ? Ysol.incidence_amplitude / ConversionFactors.Binc(s, m, a, ω, λ) :
        Ysol.incidence_amplitude / ConversionFactors.Cinc(s, m, a, ω, λ)
    Nr = o.circular ? 1 : AP_MODE_START_NODES
    Kθ = o.equatorial ? 1 : AP_MODE_START_NODES
    # A doubling is converged when the amplitude changes by less than the tolerance, or than the roundoff floor
    # of the sum where the terms cancel (at (0.9, 8, 0.95, 1), (l, m, n) = (3, 3, 40) settles at 3e-33 with 128
    # bits, above 1000 eps).
    Z, floor_ = _amplitude(o, s, m, n, k, ω, λ, Ysol, SH, incidence, Nr, Kθ)
    change = T(Inf)
    done_r, done_θ = o.circular, o.equatorial
    while !(done_r && done_θ)
        if !done_r
            Nr * 2 > AP_MODE_MAX_NODES && error("arbitrary-precision mode: radial grid did not converge by $(AP_MODE_MAX_NODES) nodes.")
            Z2, floor_ = _amplitude(o, s, m, n, k, ω, λ, Ysol, SH, incidence, 2Nr, Kθ)
            change = _relchange(Z2, Z); Nr *= 2; Z = Z2
            done_r = change <= max(tolerance, floor_)
        end
        if !done_θ
            Kθ * 2 > AP_MODE_MAX_NODES && error("arbitrary-precision mode: polar grid did not converge by $(AP_MODE_MAX_NODES) nodes.")
            Z2, floor_ = _amplitude(o, s, m, n, k, ω, λ, Ysol, SH, incidence, Nr, 2Kθ)
            change = _relchange(Z2, Z); Kθ *= 2; Z = Z2
            done_θ = change <= max(tolerance, floor_)
        end
    end
    carter = _carter_factor(o, m, ω, o.equatorial ? 1 : max(Kθ, 2))
    if s == -2
        E = abs2(Z) / (4 * T(pi) * ω^2)
        L = m * abs2(Z) / (4 * T(pi) * ω^3)
        Q = abs2(Z) * (carter + k * ϒθ) / (2 * T(pi) * ω^3)
    else
        kappa = sqrt(one(T) - a^2)
        rp = 1 + kappa
        κ = ω - a * m / (2rp)
        ϵ0 = kappa / (4rp)
        hf = ω / (κ * (2rp)^3 * (κ^2 + 4ϵ0^2) * 64 * T(pi))
        E = hf * abs2(Z)
        L = hf * m * abs2(Z) / ω
        Q = 2 * hf * abs2(Z) * (carter + k * ϒθ) / ω
    end
    return (omega = ω, lambda = λ, amplitude = Z, energy_flux = E, angular_momentum_flux = L,
        carter_constant_flux = Q, N = Nr, K = Kθ, change = change, roundoff = floor_, orbit = o.KG, Y_solution = Ysol, SWSH = SH)
end

end
