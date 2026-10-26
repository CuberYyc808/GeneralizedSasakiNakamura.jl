# T1 bitwise baseline: amplitudes and X, R, Y at five radii for the isem reference points, the outcome of
# method="ISEM" where method="auto" falls back to the legacy route, and point-particle modes and flux totals.
# Values are stored as the hex of their binary64 bits; a thrown error as its type and message.

using Logging

const T1_FLUX_ORBITS = (
    # default lmax = :auto: circular and slightly eccentric equatorial orbits
    (0.9, 10.0, 0.0, 1.0),
    (0.9, 10.0, 0.1, 1.0),
)
const T1_MODES = (
    # (s, l, m, n, k, a, p, e, x): small |omega| at a = 0.95, r = 9.5 (small-frequency UP route, the case the
    # coverage retry of ConvolutionIntegrals guards), and a high-l near-static mode at p = 1e7
    (2, 2, 2, 0, 0, 0.95, 9.5, 0.0, 1.0),
    (-2, 2, 2, 0, 0, 0.95, 9.5, 0.0, 1.0),
    (2, 2, 1, 0, 0, 0.95, 9.5, 0.0, 1.0),
    (-2, 2, 2, 0, 0, 0.5, 1e7, 0.0, 1.0),
    (-2, 12, 1, 0, 0, 0.5, 1e7, 0.0, 1.0),
)

t1_hex(x::Real) = string(reinterpret(UInt64, Float64(x)), base=16, pad=16)
t1_hex(z::Complex) = t1_hex(real(z)) * ":" * t1_hex(imag(z))
t1_hex(x) = x === missing ? "missing" : x === nothing ? "nothing" : string(x)
t1_hexs(values) = join(map(t1_hex, values), ",")
t1_error(err) = "ERR:" * string(nameof(typeof(err))) * ":" *
    first(replace(sprint(showerror, err), r"\s+" => " "), 160)

function t1_try(f)
    try
        return f()
    catch err
        err isa InterruptException && rethrow()
        return t1_error(err)
    end
end

t1_points() = vcat(
    [(string(p.id), p.s, p.l, p.m, p.a, ComplexF64(p.omega)) for p in STRONG_DAMPING_POINTS],
    [("grid" * string(g.id), g.s, g.l, g.m, g.a, ComplexF64(g.omega)) for g in GRID_POINTS],
    [("reg_" * string(k), -2, 2, 2, 0.68, w) for (k, w) in enumerate((0.3 + 0im,))],
)

# Real omega is passed as Float64 so that the real-frequency routes are taken.
t1_omega(w) = iszero(imag(w)) ? real(w) : w

function t1_radii(a)
    rp = 1 + sqrt((1 - a) * (1 + a))
    return (rp + 0.05, rp + 0.5, 6.0, 20.0, 60.0)
end

function t1_radial_rows!(rows, id, s, l, m, a, w)
    omega = t1_omega(w)
    radii = t1_radii(a)
    for (label, bc) in (("IN", IN), ("UP", UP))
        key = "radial|$id|$s,$l,$m,$a,$w|$label"
        X = t1_try(() -> GSN_radial(s, l, m, a, omega, bc))
        if X isa String
            push!(rows, key * "|GSN" => X)
        else
            push!(rows, key * "|GSN|method" => string(X.method))
            push!(rows, key * "|GSN|amp" => t1_hexs((X.transmission_amplitude, X.incidence_amplitude,
                X.reflection_amplitude)))
            push!(rows, key * "|GSN|X" => join([t1_try(() -> t1_hexs(X.GSN_solution(rstar_from_r(a, r))[1:2]))
                for r in radii], ";"))
            if X.method != "ISEM"
                Xi = t1_try(() -> GSN_radial(s, l, m, a, omega, bc; method="ISEM"))
                push!(rows, key * "|GSN_ISEM" => Xi isa String ? Xi : t1_hexs((Xi.transmission_amplitude,
                    Xi.incidence_amplitude, Xi.reflection_amplitude)))
            end
        end
        R = t1_try(() -> Teukolsky_radial(s, l, m, a, omega, bc))
        if R isa String
            push!(rows, key * "|Teukolsky" => R)
        else
            push!(rows, key * "|Teukolsky|amp" => t1_hexs((R.transmission_amplitude, R.incidence_amplitude,
                R.reflection_amplitude)))
            push!(rows, key * "|Teukolsky|R" => join([t1_try(() -> t1_hexs(R.Teukolsky_solution(r)[1:2]))
                for r in radii], ";"))
        end
        Y = t1_try(() -> Y_radial(s, l, m, a, omega, bc))
        push!(rows, key * "|Y" => Y isa String ? Y :
            join([t1_try(() -> t1_hexs(Y.Y_solution(r)[1:2])) for r in radii], ";"))
    end
    return rows
end

function t1_mode_rows!(rows, c)
    mode = t1_try(() -> Teukolsky_pointparticle_mode(c...))
    push!(rows, "mode|" * join(c, ",") => mode isa String ? mode :
        t1_hexs((mode.amplitude, mode.energy_flux, mode.angular_momentum_flux, mode.Carter_const_flux)))
    return rows
end

function t1_flux_rows!(rows, orbit)
    F = t1_try(() -> Teukolsky_pointparticle_flux(orbit...; tol=1e-6))
    push!(rows, "flux|" * join(orbit, ",") => F isa String ? F :
        t1_hexs((F.infinity_energy_flux, F.infinity_angular_momentum_flux, F.infinity_carter_constant_flux,
            F.horizon_energy_flux, F.horizon_angular_momentum_flux, F.horizon_carter_constant_flux,
            F.total_modes)))
    return rows
end

"Rows (key => value) of the T1 baseline; `flux = false` leaves out the flux totals."
function t1_rows(; flux::Bool=true)
    rows = Pair{String,String}[]
    with_logger(NullLogger()) do
        for (id, s, l, m, a, w) in t1_points()
            t1_radial_rows!(rows, id, s, l, m, a, w)
        end
        foreach(c -> t1_mode_rows!(rows, c), T1_MODES)
        flux && foreach(o -> t1_flux_rows!(rows, o), T1_FLUX_ORBITS)
    end
    return rows
end
