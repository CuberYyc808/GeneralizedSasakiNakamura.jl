module Solutions

using OrdinaryDiffEq
using HypergeometricFunctions
using StaticArrays
using ..Kerr
using ..Transformation
using ..Coordinates
using ..Potentials
using ..AsymptoticExpansionCoefficients
using ..InitialConditions
using ..ConversionFactors

const I = 1im # Mathematica being Mathematica
_DEFAULTDATATYPE = ComplexF64 # Double precision by default
_DEFAULTSOLVER = AutoVern9(Rosenbrock23(autodiff=false))
_DEFAULTTOLERANCE = 1e-12

# First-order non-linear ODE form of the GSN equation
# Step budget of the ODE solves. The historical budget, 50 steps per unit r* capped at 1e5, is kept
# unchanged for tolerances down to the default 1e-12; tighter tolerances (arbitrary-precision runs)
# need more steps, otherwise the solve stops with MaxIters and the partial solution would be returned
# as valid. The budget then grows like tol^(-1/8) (ninth-order method), capped at 1e7.
function _ode_maxiters(rsin, rsout, reltol, abstol)
    base = min(50 * floor(rsout - rsin), 1e5)
    tol = max(min(float(reltol), float(abstol)), 1e-300)
    tol >= 1e-12 && return base
    return min(base * (1e-12 / tol)^(1 / 8), 1e7)
end

function GSN_Riccati_eqn(u, p, rs)
    r = r_from_rstar(p.a, rs)
    _sF = sF(p.s, p.m, p.a, p.omega, p.lambda, r)
    _sU = sU(p.s, p.m, p.a, p.omega, p.lambda, r)
    
    #=
    We write X = exp(I*Phi)
    Substitute X in this form into the GSN equation will give
    a Riccati equation, a first-order non-linear equation

    u[1] = Phi
    u[2] = dPhidrs
    =#
    return SA[u[2], -1im*_sU + _sF*u[2] - 1im*u[2]*u[2]]
end

# Second-order linear ODE form of the GSN equation
function GSN_linear_eqn(u, p, rs)
    r = r_from_rstar(p.a, rs)
    _sF = sF(p.s, p.m, p.a, p.omega, p.lambda, r)
    _sU = sU(p.s, p.m, p.a, p.omega, p.lambda, r)

    #=
    Using the convention for OrdinaryDiffEq
    u[1] = X(rs)
    u[2] = dX/drs = X'
    therefore
    X'' - sF X' - sU X = 0 => u[2]' - sF u[2] - sU u[1] = 0 => u[2]' = sF u[2] + sU u[1]
    =#
    return SA[u[2], _sF*u[2] + _sU*u[1]]
end

function PhiPhiprime_from_XXprime(X, Xprime)
    Phi = -1im*log(X)
    Phiprime = -1im*Xprime/X

    return Phi, Phiprime
end

function XXprime_from_PhiPhiprime(Phi, Phiprime)
    X = exp(1im*Phi)
    Xprime = 1im*X*Phiprime

    return X, Xprime
end

function Xsoln_from_Phisoln(Phisoln)
    return rs -> XXprime_from_PhiPhiprime(Phisoln(rs)[1], Phisoln(rs)[2])
end

# The legacy ODE solution of `boundary` (:in from rsin outward, :up from rsout inward) in the linear form (X, dX/drs)
# or the Riccati form (Phi, dPhi/drs) with X = exp(i Phi).
function _solve_legacy(s::Int, m::Int, a, omega, lambda, rsin, rsout, boundary::Symbol, riccati::Bool;
        initialconditions_order=-1, dtype=_DEFAULTDATATYPE, odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE,
        abstol=_DEFAULTTOLERANCE)
    if rsin > rsout
        boundary == :up ? throw(DomainError(rsout, "rsout ($rsout) must be larger than rsin ($rsin)")) :
            throw(DomainError(rsin, "rsin ($rsin) must be smaller than rsout ($rsout)"))
    end
    maxiters = _ode_maxiters(rsin, rsout, reltol, abstol)
    # Initial conditions at the boundary: rs = rsout for UP, rs = rsin (close to the horizon) for IN
    X0, Xprime0 = boundary == :up ?
        Xup_initialconditions(s, m, a, omega, lambda, rsout; order=initialconditions_order, dtype=dtype) :
        Xin_initialconditions(s, m, a, omega, lambda, rsin; order=initialconditions_order, dtype=dtype)
    if riccati
        Phi, Phiprime = PhiPhiprime_from_XXprime(X0, Xprime0)
        u0 = SA[dtype(Phi); dtype(Phiprime)]
    else
        u0 = SA[dtype(X0); dtype(Xprime0)]
    end
    rsspan = boundary == :up ? (rsout, rsin) : (rsin, rsout)
    p = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    odeprob = ODEProblem(riccati ? GSN_Riccati_eqn : GSN_linear_eqn, u0, rsspan, p)
    return solve(odeprob, odealgo; reltol=reltol, abstol=abstol, maxiters=maxiters)
end

solve_Phiup(s::Int, m::Int, a, omega, lambda, rsin, rsout; kwargs...) =
    _solve_legacy(s, m, a, omega, lambda, rsin, rsout, :up, true; kwargs...)
solve_Phiin(s::Int, m::Int, a, omega, lambda, rsin, rsout; kwargs...) =
    _solve_legacy(s, m, a, omega, lambda, rsin, rsout, :in, true; kwargs...)
solve_Xup(s::Int, m::Int, a, omega, lambda, rsin, rsout; kwargs...) =
    _solve_legacy(s, m, a, omega, lambda, rsin, rsout, :up, false; kwargs...)
solve_Xin(s::Int, m::Int, a, omega, lambda, rsin, rsout; kwargs...) =
    _solve_legacy(s, m, a, omega, lambda, rsin, rsout, :in, false; kwargs...)

# Unit-amplitude asymptotic solutions (X, dX/drs) at a tortoise coordinate rs: at the horizon the ingoing (wave = :in,
# exp(-i p rs)) or outgoing (:out, exp(+i p rs)) Frobenius series with p = omega - m Omega_H, at infinity the ingoing
# (exp(-i omega rs)) or outgoing (exp(+i omega rs)) asymptotic series. These are the states the IN and UP routes start
# from (Xin_initialconditions, Xup_initialconditions) and the bases their amplitudes refer to.
function horizon_wave(s, m, a, omega, lambda, rs, wave::Symbol; order, dtype=_DEFAULTDATATYPE)
    sign = wave === :in ? -1 : 1
    coefficients = coefficient_sequence(wave === :in ? ingoing_coefficient_at_hor : outgoing_coefficient_at_hor,
        s, m, a, omega, lambda; data_type=dtype)
    r = r_from_rstar(a, rs)
    p = omega - m*omega_horizon(a)
    g = gansatz(coefficients, a, r; order=order)
    dg = dgansatz_dr(coefficients, a, r; order=order)
    phase = exp(sign*1im*p*rs)
    return g*phase, (Delta(a, r)/(r^2 + a^2)*dg + sign*1im*p*g)*phase
end

function infinity_wave(s, m, a, omega, lambda, rs, wave::Symbol; order, dtype=_DEFAULTDATATYPE)
    sign = wave === :in ? -1 : 1
    coefficients = coefficient_sequence(wave === :in ? ingoing_coefficient_at_inf : outgoing_coefficient_at_inf,
        s, m, a, omega, lambda; data_type=dtype)
    r = r_from_rstar(a, rs)
    f = fansatz(coefficients, omega, r; order=order)
    df = dfansatz_dr(coefficients, omega, r; order=order)
    phase = exp(sign*1im*omega*rs)
    return f*phase, (Delta(a, r)/(r^2 + a^2)*df + sign*1im*omega*f)*phase
end

# OUT and DOWN solved directly at real omega: OUT starts at rsin with the outgoing horizon wave and is integrated
# outward, DOWN starts at rsout with the ingoing infinity wave and is integrated inward, in the linear or the Riccati
# form. The amplitudes are the coefficients of the far pair: for OUT (outgoing, ingoing) at infinity from
# BrefBinc_SN_from_Xin, for DOWN (ingoing, outgoing) at the horizon from CrefCinc_SN_from_Xup; incidence is the first
# of each pair (the definitions of the IN + UP combination). Outside [rsin, rsout] the solution is continued with the
# asymptotic series. Returns (incidence, reflection, X(rs) -> (X, dX/drs), numerical X solution, Riccati solution or
# nothing).
function solve_down_out(s::Int, m::Int, a, omega, lambda, boundary::Symbol, rsin, rsout, horizon_order, infinity_order;
        riccati::Bool, dtype=_DEFAULTDATATYPE, odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
    rsin < rsout || throw(DomainError(rsin, "rsin ($rsin) must be smaller than rsout ($rsout)"))
    boundary in (:OUT, :DOWN) || throw(ArgumentError("boundary must be :OUT or :DOWN"))
    span = boundary === :OUT ? (rsin, rsout) : (rsout, rsin)
    X0, dX0 = boundary === :OUT ?
        horizon_wave(s, m, a, omega, lambda, rsin, :out; order=horizon_order, dtype=dtype) :
        infinity_wave(s, m, a, omega, lambda, rsout, :in; order=infinity_order, dtype=dtype)
    params = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    maxiters = _ode_maxiters(rsin, rsout, reltol, abstol)
    Phisoln, Xsoln = if riccati
        Phi, dPhi = PhiPhiprime_from_XXprime(X0, dX0)
        phi = solve(ODEProblem(GSN_Riccati_eqn, SA[dtype(Phi); dtype(dPhi)], span, params), odealgo;
            reltol=reltol, abstol=abstol, maxiters=maxiters)
        (phi, Xsoln_from_Phisoln(phi))
    else
        (nothing, solve(ODEProblem(GSN_linear_eqn, SA[dtype(X0); dtype(dX0)], span, params), odealgo;
            reltol=reltol, abstol=abstol, maxiters=maxiters))
    end
    incidence, reflection = boundary === :OUT ?
        BrefBinc_SN_from_Xin(s, m, a, omega, lambda, Xsoln, rsout; order=infinity_order, dtype=dtype) :
        CrefCinc_SN_from_Xup(s, m, a, omega, lambda, Xsoln, rsin; order=horizon_order, dtype=dtype)
    function solution(rs)
        if rs < rsin
            boundary === :OUT && return horizon_wave(s, m, a, omega, lambda, rs, :out; order=horizon_order, dtype=dtype)
            hin = horizon_wave(s, m, a, omega, lambda, rs, :in; order=horizon_order, dtype=dtype)
            hout = horizon_wave(s, m, a, omega, lambda, rs, :out; order=horizon_order, dtype=dtype)
            return (incidence*hin[1] + reflection*hout[1], incidence*hin[2] + reflection*hout[2])
        elseif rs > rsout
            boundary === :DOWN && return infinity_wave(s, m, a, omega, lambda, rs, :in; order=infinity_order, dtype=dtype)
            iout = infinity_wave(s, m, a, omega, lambda, rs, :out; order=infinity_order, dtype=dtype)
            iin = infinity_wave(s, m, a, omega, lambda, rs, :in; order=infinity_order, dtype=dtype)
            return (incidence*iout[1] + reflection*iin[1], incidence*iout[2] + reflection*iin[2])
        end
        value = Xsoln(rs)
        return (value[1], value[2])
    end
    return incidence, reflection, solution, Xsoln, Phisoln
end

function Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(s, m, a, omega, lambda, r)
    #=
    Here we use explicit form for the conversion matrix to
    faciliate cancellations
    =#
    M11 = M12 = M21 = M22 = 0.0
    if s == 0
        M11 = 1/sqrt(a^2 + r^2)
        M21 = -(r/(a^2 + r^2)^(3/2))
        M22 = sqrt(a^2 + r^2)/(a^2 + (-2 + r)*r)
    elseif s == +1
        M11 = begin
            -((I*r*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r))*
            ((-a^3)*m*r - a*m*r^3 + r^5*omega + 
            a^4*(2*I + r*omega) + 2*a^2*r*(-2*I + I*r + 
            r^2*omega)))/(sqrt(a^2 + r^2)*
            sqrt((a^2 + (-2 + r)*r)*(a^2 + r^2))*
            (-2*I*a^3*m*r - 2*I*a*m*r^3 + a^4*(1 + lambda) + 
            r^4*(2 + lambda) + a^2*r*(2 + r*(3 + 2*lambda)))))
        end
        M12 = begin
            (r^2*(a^2 + r^2)^(3/2)*
            sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r)))/
            (sqrt((a^2 + (-2 + r)*r)*(a^2 + r^2))*
            (-2*I*a^3*m*r - 2*I*a*m*r^3 + a^4*(1 + lambda) + 
            r^4*(2 + lambda) + a^2*r*(2 + r*(3 + 2*lambda))))
        end
        M21 = begin
            -((r*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r))*
            (-2*a^7*m*(I + r*omega) + a^8*omega*(2*I + r*omega) + 
            a^5*m*r*(2*I - I*r - 6*r^2*omega) + 
            a*m*r^5*(-4*I + 3*I*r - 2*r^2*omega) - 
            2*a^3*m*r^3*(I - 2*I*r + 3*r^2*omega) + 
            r^6*(2*(2 + lambda) - r*(2 + lambda) - I*r^2*omega + 
            r^3*omega^2) + a^6*(-4 + r*(1 + m^2 - lambda - 
            2*I*omega) + 5*I*r^2*omega + 4*r^3*omega^2) + 
            a^2*r^3*(4 + 4*r*lambda + r^2*(-3 + m^2 - 3*lambda - 
            2*I*omega) - I*r^3*omega + 4*r^4*omega^2) + 
            a^4*r*(8 + 2*r*(-4 + lambda) + 
            r^2*(2*m^2 - 3*lambda - 4*I*omega) + 3*I*r^3*omega + 
            6*r^4*omega^2)))/(sqrt(a^2 + r^2)*
            ((a^2 + (-2 + r)*r)*(a^2 + r^2))^(3/2)*
            (-2*I*a^3*m*r - 2*I*a*m*r^3 + a^4*(1 + lambda) + 
            r^4*(2 + lambda) + a^2*r*(2 + r*(3 + 2*lambda)))))
        end
        M22 = begin
            -((I*r^2*((a^2 + r^2)/(a^2 + (-2 + r)*r))^(3/2)*
            ((-a^3)*m - a*m*r^2 + a^4*omega + r^3*(-I + r*omega) + 
            a^2*(2*I - I*r + 2*r^2*omega)))/
            (sqrt(a^2 + r^2)*sqrt((a^2 + (-2 + r)*r)*
            (a^2 + r^2))*(-2*I*a^3*m*r - 2*I*a*m*r^3 + 
            a^4*(1 + lambda) + r^4*(2 + lambda) + 
            a^2*r*(2 + r*(3 + 2*lambda)))))
        end
    elseif s == -1
        M11 = begin
            (I*r*sqrt(a^2 + r^2)*((-a^3)*m*r - a*m*r^3 + 
            r^5*omega + a^4*(-2*I + r*omega) + 
            2*a^2*r*(2*I - I*r + r^2*omega)))/
            (sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r))*
            sqrt((a^2 + (-2 + r)*r)*(a^2 + r^2))*
            (2*I*a^3*m*r + 2*I*a*m*r^3 + a^4*(-1 + lambda) + 
            r^4*lambda + a^2*r*(2 + r*(-1 + 2*lambda))))
        end
        M12 = begin
            (r^2*sqrt(a^2 + r^2)*sqrt((a^2 + r^2)/
            (a^2 + (-2 + r)*r))*sqrt((a^2 + (-2 + r)*r)*
            (a^2 + r^2)))/(2*I*a^3*m*r + 2*I*a*m*r^3 + 
            a^4*(-1 + lambda) + r^4*lambda + 
            a^2*r*(2 + r*(-1 + 2*lambda)))
        end
        M21 = begin
            (r*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r))*
            (a^8*omega*(2*I - r*omega) + 2*a^7*m*(-I + r*omega) + 
            a*m*r^5*(-2*I + I*r + 2*r^2*omega) + 
            2*a^3*m*r^3*(I + 3*r^2*omega) + 
            a^5*m*r*(4*I - 3*I*r + 6*r^2*omega) + 
            a^6*r*(1 - m^2 + lambda - 4*I*omega + 7*I*r*omega - 
            4*r^2*omega^2) - a^4*r^2*(2*(2 + lambda) + 
            r*(-2 + 2*m^2 - 3*lambda + 10*I*omega) - 9*I*r^2*omega + 
            6*r^3*omega^2) + a^2*r^3*(4 - 4*r*(1 + lambda) - 
            r^2*(-1 + m^2 - 3*lambda + 8*I*omega) + 5*I*r^3*omega - 
            4*r^4*omega^2) + r^6*((-2 + r)*lambda - 
            r*omega*(2*I - I*r + r^2*omega))))/
            ((a^2 + r^2)^(3/2)*sqrt((a^2 + (-2 + r)*r)*
            (a^2 + r^2))*(2*I*a^3*m*r + 2*I*a*m*r^3 + 
            a^4*(-1 + lambda) + r^4*lambda + 
            a^2*r*(2 + r*(-1 + 2*lambda))))
        end
        M22 = begin
            -((r^2*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r))*
            sqrt((a^2 + (-2 + r)*r)*(a^2 + r^2))*
            (-r - (I*(a^2 + r^2)*((-a)*m + (a^2 + r^2)*omega))/
            (a^2 + (-2 + r)*r)))/(sqrt(a^2 + r^2)*
            (2*I*a^3*m*r + 2*I*a*m*r^3 + a^4*(-1 + lambda) + 
            r^4*lambda + a^2*r*(2 + r*(-1 + 2*lambda)))))
        end
    elseif s == +2
        M11 = begin
            (r^2*(a^2 + (-2 + r)*r)*(-4*a^5*m*r*(I + r*omega) - 
            2*a^3*m*r^2*(-3*I + 2*I*r + 4*r^2*omega) + 
            a*m*(2*I*r^4 - 4*r^6*omega) + 
            2*a^6*(-5 + 2*I*r*omega + r^2*omega^2) + 
            a^4*r*(32 + r*(-24 + 2*m^2 - lambda - 6*I*omega) + 
            10*I*r^2*omega + 6*r^3*omega^2) + 
            r^4*(-12 + 2*r*(9 + lambda) - 
            r^2*(6 + lambda + 6*I*omega) + 2*I*r^3*omega + 
            2*r^4*omega^2) + 2*a^2*r^2*(-12 + r*(23 + lambda) + 
            r^2*(-10 + m^2 - lambda - 6*I*omega) + 4*I*r^3*omega + 
            3*r^4*omega^2)))/
            (((a^2 + (-2 + r)*r)^2*(a^2 + r^2))^(3/2)*
            ((-r^4)*(24 + 10*lambda + lambda^2 + 12*I*omega) - 
            24*a^3*m*r*(I + 2*r*omega) + 4*a*m*r^2*
            (6*I + 2*I*r*(4 + lambda) - 3*r^2*omega) + 
            12*a^4*(-1 + 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 - 6*I*omega) - 
            2*I*r^2*(1 + lambda)*omega + 3*r^3*omega^2)))
        end
        M12 = begin
            (2*r^3*(a^2 + (-2 + r)*r)*(a^2 + r^2)^2*
            ((-I)*a*m*r + a^2*(-2 + I*r*omega) + 
            r*(3 - r + I*r^2*omega)))/
            (((a^2 + (-2 + r)*r)^2*(a^2 + r^2))^(3/2)*
            ((-r^4)*(24 + 10*lambda + lambda^2 + 12*I*omega) - 
            24*a^3*m*r*(I + 2*r*omega) + 4*a*m*r^2*
            (6*I + 2*I*r*(4 + lambda) - 3*r^2*omega) + 
            12*a^4*(-1 + 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 - 6*I*omega) - 
            2*I*r^2*(1 + lambda)*omega + 3*r^3*omega^2)))
        end
        M21 = begin
            -((I*r*(-2*a^7*m*r*(-2 + 4*I*r*omega + 3*r^2*omega^2) - 
            2*a^5*m*r^2*(-4 + r*(3 + m^2 - lambda + 2*I*omega) + 
            4*I*r^2*omega + 9*r^3*omega^2) + 
            2*a*m*r^5*(-10 + r*(3 - 2*lambda) + 
            r^2*(2 + lambda + 2*I*omega) + 4*I*r^3*omega - 
            3*r^4*omega^2) - 2*a^3*m*r^3*
            (12 + r^2*(3 + m^2 - 2*lambda) + r*(-17 + 2*lambda) - 
            4*I*r^3*omega + 9*r^4*omega^2) + 
            2*a^8*(-6*I - 2*r*omega + 2*I*r^2*omega^2 + 
            r^3*omega^3) + 2*a^6*r*(16*I + 
            I*r*(-9 + 2*m^2 - 2*lambda + 4*I*omega) + 
            r^2*(-8 + 3*m^2 - lambda + I*omega)*omega + 
            6*I*r^3*omega^2 + 4*r^4*omega^3) + 
            r^6*(-4*I*lambda - 2*r^3*(4 + lambda + 5*I*omega)*omega + 
            2*r^5*omega^3 - 12*r*(I + omega) + 
            r^2*(6*I + I*lambda + 18*omega + 4*lambda*omega)) + 
            a^4*r^2*(-16*I + 2*I*r*(10 + m^2 + 6*lambda - 
            12*I*omega) + 2*r^3*(-14 + 6*m^2 - 3*lambda - 
            3*I*omega)*omega + 12*I*r^4*omega^2 + 12*r^5*omega^3 + 
            r^2*(-2*I - 4*I*m^2 - 7*I*lambda + 22*omega + 
            4*lambda*omega)) + 2*a^2*r^4*(-4*I*lambda + 
            I*r*(-4 + 3*m^2 + 6*lambda + 6*I*omega) + 
            3*r^3*(-4 + m^2 - lambda - 3*I*omega)*omega + 
            2*I*r^4*omega^2 + 4*r^5*omega^3 + 
            r^2*(5*I - 4*I*m^2 - I*lambda + 24*omega + 
            4*lambda*omega))))/
            (((a^2 + (-2 + r)*r)^2*(a^2 + r^2))^(3/2)*
            ((-r^4)*(24 + 10*lambda + lambda^2 + 12*I*omega) - 
            24*a^3*m*r*(I + 2*r*omega) + 4*a*m*r^2*
            (6*I + 2*I*r*(4 + lambda) - 3*r^2*omega) + 
            12*a^4*(-1 + 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 - 6*I*omega) - 
            2*I*r^2*(1 + lambda)*omega + 3*r^3*omega^2))))
        end
        M22 = begin
            (r^2*(a^2 + r^2)^2*(-4*a^3*m*r*(I + r*omega) - 
            2*a*m*r^2*(I - 3*I*r + 2*r^2*omega) + 
            2*a^4*(-3 + 2*I*r*omega + r^2*omega^2) + 
            r^3*(2*lambda - r*(2 + lambda + 10*I*omega) + 2*r^3*omega^2) + 
            a^2*r*(8 + 2*m^2*r - r*lambda + 2*I*r*omega + 
            4*I*r^2*omega + 4*r^3*omega^2)))/
            (((a^2 + (-2 + r)*r)^2*(a^2 + r^2))^(3/2)*
            ((-r^4)*(24 + 10*lambda + lambda^2 + 12*I*omega) - 
            24*a^3*m*r*(I + 2*r*omega) + 4*a*m*r^2*
            (6*I + 2*I*r*(4 + lambda) - 3*r^2*omega) + 
            12*a^4*(-1 + 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 - 6*I*omega) - 
            2*I*r^2*(1 + lambda)*omega + 3*r^3*omega^2)))
        end
    elseif s == -2
        M11 = begin
            (r^2*(-4*a^5*m*r*(-I + r*omega) - 
            2*a*m*r^4*(I + 2*r^2*omega) - 2*a^3*m*r^2*
            (3*I - 2*I*r + 4*r^2*omega) + 
            2*a^6*(-5 - 2*I*r*omega + r^2*omega^2) + 
            a^4*r*(32 + r*(-20 + 2*m^2 - lambda + 6*I*omega) - 
            10*I*r^2*omega + 6*r^3*omega^2) + 
            r^4*(-12 + 2*r*(5 + lambda) - 
            r^2*(2 + lambda - 6*I*omega) - 2*I*r^3*omega + 
            2*r^4*omega^2) + 2*a^2*r^2*(-12 + r*(19 + lambda) + 
            r^2*(-6 + m^2 - lambda + 6*I*omega) - 4*I*r^3*omega + 
            3*r^4*omega^2)))/((a^2 + (-2 + r)*r)^3*
            ((a^2 + r^2)/(a^2 + (-2 + r)*r)^2)^(3/2)*
            ((-r^4)*(2*lambda + lambda^2 - 12*I*omega) + 
            24*a^3*m*r*(I - 2*r*omega) - 4*a*m*r^2*
            (6*I + 2*I*r*lambda + 3*r^2*omega) + 
            12*a^4*(-1 - 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 + 6*I*omega) + 
            2*I*r^2*(-3 + lambda)*omega + 3*r^3*omega^2)))
        end
        M12 = begin
            (2*r^3*(a^2 + (-2 + r)*r)*
            sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r)^2)*
            (I*a*m*r + a^2*(-2 - I*r*omega) + 
            r*(3 - r - I*r^2*omega)))/
            ((-r^4)*(2*lambda + lambda^2 - 12*I*omega) + 
            24*a^3*m*r*(I - 2*r*omega) - 
            4*a*m*r^2*(6*I + 2*I*r*lambda + 3*r^2*omega) + 
            12*a^4*(-1 - 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 + 6*I*omega) + 
            2*I*r^2*(-3 + lambda)*omega + 3*r^3*omega^2))
        end
        M21 = begin
            (I*r*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r)^2)*
            (2*a^7*m*r*(2 + 4*I*r*omega - 3*r^2*omega^2) - 
            2*a^5*m*r^2*(4 + r*(-1 + m^2 - lambda + 6*I*omega) - 
            12*I*r^2*omega + 9*r^3*omega^2) - 
            2*a^3*m*r^4*(-5 + 2*lambda + r*(3 + m^2 - 2*lambda + 
            16*I*omega) - 12*I*r^2*omega + 9*r^3*omega^2) - 
            2*a*m*r^5*(6 + r*(-7 + 2*lambda) + 
            r^2*(2 - lambda + 10*I*omega) - 4*I*r^3*omega + 
            3*r^4*omega^2) + 2*a^8*(6*I - 2*r*omega - 
            2*I*r^2*omega^2 + r^3*omega^3) + 
            2*a^6*r*(-36*I + r^2*(-12 + 3*m^2 - lambda + 3*I*omega)*
            omega - 10*I*r^3*omega^2 + 4*r^4*omega^3 + 
            r*(21*I - 2*I*m^2 + 2*I*lambda + 4*omega)) + 
            r^5*(-48*I + 12*I*r*(6 + lambda) - 
            12*I*r^2*(3 + lambda - 3*I*omega) - 
            2*r^4*(4 + lambda - 9*I*omega)*omega - 8*I*r^5*omega^2 + 
            2*r^6*omega^3 + r^3*(6*I + 3*I*lambda + 34*omega + 
            4*lambda*omega)) + 2*a^2*r^3*(-48*I + 
            4*I*r*(27 + 2*lambda) + I*r^2*(-72 + m^2 - 14*lambda + 
            30*I*omega) + r^4*(-16 + 3*m^2 - 3*lambda + 
            21*I*omega)*omega - 14*I*r^5*omega^2 + 4*r^6*omega^3 + 
            r^3*(15*I + 5*I*lambda + 48*omega + 4*lambda*omega)) + 
            a^4*r^2*(144*I + 2*I*r*(-90 + 3*m^2 - 8*lambda) + 
            2*r^3*(-22 + 6*m^2 - 3*lambda + 15*I*omega)*omega - 
            36*I*r^4*omega^2 + 12*r^5*omega^3 + 
            r^2*(54*I - 4*I*m^2 + 11*I*lambda + 70*omega + 
            4*lambda*omega))))/((a^2 + r^2)^2*
            ((-r^4)*(2*lambda + lambda^2 - 12*I*omega) + 
            24*a^3*m*r*(I - 2*r*omega) - 4*a*m*r^2*
            (6*I + 2*I*r*lambda + 3*r^2*omega) + 
            12*a^4*(-1 - 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 + 6*I*omega) + 
            2*I*r^2*(-3 + lambda)*omega + 3*r^3*omega^2)))
        end
        M22 = begin
            (r^2*sqrt((a^2 + r^2)/(a^2 + (-2 + r)*r)^2)*
            (-4*a^3*m*r*(-I + r*omega) - 2*a*m*r^2*
            (3*I - I*r + 2*r^2*omega) + 
            2*a^4*(-3 - 2*I*r*omega + r^2*omega^2) + 
            a^2*r*(24 + r*(-12 + 2*m^2 - lambda + 6*I*omega) - 
            12*I*r^2*omega + 4*r^3*omega^2) + 
            r^2*(-24 + 2*r*(12 + lambda) - 
            r^2*(6 + lambda - 18*I*omega) - 8*I*r^3*omega + 
            2*r^4*omega^2)))/
            ((-r^4)*(2*lambda + lambda^2 - 12*I*omega) + 
            24*a^3*m*r*(I - 2*r*omega) - 
            4*a*m*r^2*(6*I + 2*I*r*lambda + 3*r^2*omega) + 
            12*a^4*(-1 - 2*I*r*omega + 2*r^2*omega^2) + 
            4*a^2*r*(6 + r*(-3 + 6*m^2 + 6*I*omega) + 
            2*I*r^2*(-3 + lambda)*omega + 3*r^3*omega^2))
        end
    else
        # Throw an error, this spin weight is not supported
        throw(DomainError(s, "Currently only spin weight s of 0, +/-1, +/-2 are supported"))
    end 
    return SA[M11 M12; M21 M22]
end

function Teukolsky_radial_function_from_Sasaki_Nakamura_function(s::Int, m::Int, a, omega, lambda, Xsoln)
    #=
    First convert [X(rs), dX/drs(rs)] to [X(r), dX/dr(r)], this is done by
    [X(r), dX/dr(r)]^T = [1, 0; 0, drstar/dr ] * [X(rs), dX/drs(rs)]^T

    Then we convert [X(r), dX/dr(r)] to [chi(r), dchi/dr(r)] by
    [chi(r), dchi/dr(r)]^T = [chi_conversion_factor(r), 0 ; dchi_conversion_factor_dr, chi_conversion_factor] * [X(r), dX/dr(r)]^T

    After that we convert [chi(r), dchi/dr(r)] to [R(r), dR/dr(r)] by
    [R(r), dR/dr(r)]^T = 1/eta(r) * [ alpha + beta_prime*Delta^(s+1), -beta*Delta^(s+1) ; -(alpha_prime + beta*VT*Delta^s), alpha ] *  [chi(r), dchi/dr(r)]^T

    Therefore the overall conversion matrix is 'just' (one matrix for each r)
    the multiplication of each conversion matrix
    =#

    overall_conversion_matrix(r) = Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(s, m, a, omega, lambda, r)
    Rsoln = r -> begin
        matrix = overall_conversion_matrix(r)
        state = Xsoln(rstar_from_r(a, r))
        matrix * SA[state[1]; state[2]]
    end
    return Rsoln
end

function Sasaki_Nakamura_function_from_Teukolsky_radial_function(s::Int, m::Int, a, omega, lambda, Rsoln)
    #=
    Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix() returns 
    the conversion matrix for going from (X, dX/drs) to (R, dR/dr). Here we need
    the inverse of that to go from (R, dR/dr) to (X, dX/drs).
    =#
    conversion_matrix(r) = inv(Teukolsky_radial_function_from_Sasaki_Nakamura_function_conversion_matrix(s, m, a, omega, lambda, r))

    Xsoln(rs) = begin
        r = r_from_rstar(a, rs)
        conversion_matrix(r) * Rsoln(r)
    end

    return Xsoln
end

function CrefCinc_SN_from_Xup(s::Int, m::Int, a, omega, lambda, Xupsoln, rsin; order=0, dtype=_DEFAULTDATATYPE)
    p = omega - m*omega_horizon(a)

    rin = r_from_rstar(a, rsin)
    # Computing A1, A2, A3, A4
    cin = coefficient_sequence(ingoing_coefficient_at_hor, s, m, a, omega, lambda; data_type=dtype)
    cout = coefficient_sequence(outgoing_coefficient_at_hor, s, m, a, omega, lambda; data_type=dtype)
    gin(r) = gansatz(
        cin,
        a,
        r;
        order=order
    )
    dgin_dr(r) = dgansatz_dr(
        cin,
        a,
        r;
        order=order
    )
    A1 = gin(rin) * exp(-1im*p*rsin)
    gout(r) = gansatz(
        cout,
        a,
        r;
        order=order
    )
    dgout_dr(r) = dgansatz_dr(
        cout,
        a,
        r;
        order=order
    )
    A2 = gout(rin) * exp(1im*p*rsin)

    _DeltaOverr2pa2 = Delta(a, rin)/(rin^2 + a^2)
    A3 = (_DeltaOverr2pa2 * dgin_dr(rin) - 1im*p*gin(rin)) * exp(-1im*p*rsin)
    A4 = (_DeltaOverr2pa2 * dgout_dr(rin) + 1im*p*gout(rin)) * exp(1im*p*rsin)

    X(rs) = Xupsoln(rs)[1]
    Xprime(rs) = Xupsoln(rs)[2]
    C1 = X(rsin)
    C2 = Xprime(rsin)

    return -(A4*C1 - A2*C2)/(A2*A3 - A1*A4), -(-A3*C1 + A1*C2)/(A2*A3 - A1*A4)
end

function BrefBinc_SN_from_Xin(s::Int, m::Int, a, omega, lambda, Xinsoln, rsout; order=3, dtype=_DEFAULTDATATYPE)
    rout = r_from_rstar(a, rsout)
    # Computing A1, A2, A3, A4
    cin = coefficient_sequence(ingoing_coefficient_at_inf, s, m, a, omega, lambda; data_type=dtype)
    cout = coefficient_sequence(outgoing_coefficient_at_inf, s, m, a, omega, lambda; data_type=dtype)
    fin(r) = fansatz(
        cin,
        omega,
        r;
        order=order
    )
    dfin_dr(r) = dfansatz_dr(
        cin,
        omega,
        r;
        order=order
    )
    A1 = fin(rout) * exp(-1im*omega*rsout)
    fout(r) = fansatz(
        cout,
        omega,
        r;
        order=order
    )
    dfout_dr(r) = dfansatz_dr(
        cout,
        omega,
        r;
        order=order
    )
    A2 = fout(rout) * exp(1im*omega*rsout)

    _DeltaOverr2pa2 = Delta(a, rout)/(rout^2 + a^2)
    A3 = (_DeltaOverr2pa2 * dfin_dr(rout) - 1im*omega*fin(rout)) * exp(-1im*omega*rsout)
    A4 = (_DeltaOverr2pa2 * dfout_dr(rout) + 1im*omega*fout(rout)) * exp(1im*omega*rsout)

    X(rs) = Xinsoln(rs)[1]
    Xprime(rs) = Xinsoln(rs)[2]
    C1 = X(rsout)
    C2 = Xprime(rsout)

    return -(-A3*C1 + A1*C2)/(A2*A3 - A1*A4), -(A4*C1 - A2*C2)/(A2*A3 - A1*A4)
end

function semianalytical_Xin(s, m, a, omega, lambda, Xinsoln, rsin, rsout, horizon_expansionorder, infinity_expansionorder, rs; dtype=_DEFAULTDATATYPE)
    _r = r_from_rstar(a, rs) # Evaluate at this r

    if rs < rsin
        # Extend the numerical solution to the analytical ansatz from rsin to horizon

        # Construct the analytical ansatz
        p = omega - m*omega_horizon(a)
        cin = coefficient_sequence(ingoing_coefficient_at_hor, s, m, a, omega, lambda; data_type=dtype)
        gin(r) = gansatz(
            cin,
            a,
            r;
            order=horizon_expansionorder
        )
        dgin_dr(r) = dgansatz_dr(
            cin,
            a,
            r;
            order=horizon_expansionorder
        )

        _Xin = gin(_r)*exp(-1im*p*rs) # Evaluate at *this* rs

        _DeltaOverr2pa2 = Delta(a, _r)/(_r^2 + a^2)
        _dXindrs = (_DeltaOverr2pa2 * dgin_dr(_r) - 1im*p*gin(_r))*exp(-1im*p*rs)

        return (_Xin, _dXindrs)
    elseif rs > rsout
        # Extend the numerical solution to the analytical ansatz from rsout to infinity

        # Obtain the coefficients by imposing continuity in X and dX/drs
        Bref_SN, Binc_SN = BrefBinc_SN_from_Xin(s, m, a, omega, lambda, Xinsoln, rsout; order=infinity_expansionorder, dtype=dtype)

        # Construct the analytical ansatz
        cin = coefficient_sequence(ingoing_coefficient_at_inf, s, m, a, omega, lambda; data_type=dtype)
        cout = coefficient_sequence(outgoing_coefficient_at_inf, s, m, a, omega, lambda; data_type=dtype)
        fin(r) = fansatz(
            cin,
            omega,
            r;
            order=infinity_expansionorder
        )
        dfin_dr(r) = dfansatz_dr(
            cin,
            omega,
            r;
            order=infinity_expansionorder
        )
        fout(r) = fansatz(
            cout,
            omega,
            r;
            order=infinity_expansionorder
        )
        dfout_dr(r) = dfansatz_dr(
            cout,
            omega,
            r;
            order=infinity_expansionorder
        )

        _Xin = Bref_SN*fout(_r)*exp(1im*omega*rs) + Binc_SN*fin(_r)*exp(-1im*omega*rs)

        _DeltaOverr2pa2 = Delta(a, _r)/(_r^2 + a^2)
        _dXindrs = Bref_SN*(_DeltaOverr2pa2 * dfout_dr(_r) + 1im*omega*fout(_r))*exp(1im*omega*rs) + Binc_SN*(_DeltaOverr2pa2 * dfin_dr(_r) - 1im*omega*fin(_r))*exp(-1im*omega*rs)
        
        return (_Xin, _dXindrs)
    else
        # Requested rs is within the numerical solution
        return Xinsoln(rs)
    end 
end

function semianalytical_Xup(s, m, a, omega, lambda, Xupsoln, rsin, rsout, horizon_expansionorder, infinity_expansionorder, rs; dtype=_DEFAULTDATATYPE)
    _r = r_from_rstar(a, rs) # Evaluate at this r

    if rs < rsin
        # Extend the numerical solution to the analytical ansatz from rsin to horizon

        # Obtain the coefficients by imposing continuity in X and dX/drs
        Cref_SN, Cinc_SN = CrefCinc_SN_from_Xup(s, m, a, omega, lambda, Xupsoln, rsin; order=horizon_expansionorder, dtype=dtype)
        # with coefficient Cref_SN for the left-going and Cinc_SN for the right-going mode

        # Construct the analytical ansatz
        p = omega - m*omega_horizon(a)
        cin = coefficient_sequence(ingoing_coefficient_at_hor, s, m, a, omega, lambda; data_type=dtype)
        cout = coefficient_sequence(outgoing_coefficient_at_hor, s, m, a, omega, lambda; data_type=dtype)
        gin(r) = gansatz(
            cin,
            a,
            r;
            order=horizon_expansionorder
        )
        dgin_dr(r) = dgansatz_dr(
            cin,
            a,
            r;
            order=horizon_expansionorder
        )
        gout(r) = gansatz(
            cout,
            a,
            r;
            order=horizon_expansionorder
        )
        dgout_dr(r) = dgansatz_dr(
            cout,
            a,
            r;
            order=horizon_expansionorder
        )

        _Xup = Cinc_SN*gout(_r)*exp(1im*p*rs) + Cref_SN*gin(_r)*exp(-1im*p*rs) # Evaluate at *this* rs

        _DeltaOverr2pa2 = Delta(a, _r)/(_r^2 + a^2)
        _dXupdrs = Cinc_SN*(_DeltaOverr2pa2 * dgout_dr(_r) + 1im*p*gout(_r))*exp(1im*p*rs) + Cref_SN*(_DeltaOverr2pa2 * dgin_dr(_r) - 1im*p*gin(_r))*exp(-1im*p*rs)

        return (_Xup, _dXupdrs)
    elseif rs > rsout
        # Extend the numerical solution to the analytical ansatz from rsout to infinity

        cout = coefficient_sequence(outgoing_coefficient_at_inf, s, m, a, omega, lambda; data_type=dtype)
        fout(r) = fansatz(
            cout,
            omega,
            r;
            order=infinity_expansionorder
        )
        dfout_dr(r) = dfansatz_dr(
            cout,
            omega,
            r;
            order=infinity_expansionorder
        )

        _Xup = fout(_r)*exp(1im*omega*rs)

        _DeltaOverr2pa2 = Delta(a, _r)/(_r^2 + a^2)
        _dXupdrs = (_DeltaOverr2pa2 * dfout_dr(_r) + 1im*omega*fout(_r)) * exp(1im*omega*rs)
        
        return (_Xup, _dXupdrs)
    else
        # Requested rs is within the numerical solution
        return Xupsoln(rs)
    end 
end

function check_XinXup_sanity(Xin, Xup; tolerance=1e-6)
    #=
    Check if the X_in and X_up solutions are sane by checking if the identity

    Binc/Cinc = (p*c0)/(omega*eta(r_+)) [note: this is Eq. (42) in the paper]

    is satisfied within a specified tolerance
    =#

    if Xin.mode != Xup.mode
        throw(ArgumentError("The modes of Xin and Xup are different"))
    end

    # Calculate what Binc/Cinc is expected to be
    p = Xin.mode.omega - Xin.mode.m*omega_horizon(Xin.mode.a)
    c0 = eta_coefficient(Xin.mode.s, Xin.mode.m, Xin.mode.a, Xin.mode.omega, Xin.mode.lambda, 0)
    eta_at_rplus = eta(Xin.mode.s, Xin.mode.m, Xin.mode.a, Xin.mode.omega, Xin.mode.lambda, r_plus(Xin.mode.a))
    expected_value = (p*c0)/(Xin.mode.omega*eta_at_rplus)

    actual_value = Xin.incidence_amplitude/Xup.incidence_amplitude

    if abs(actual_value - expected_value) > tolerance
        return false
    else
        return true
    end
end

# Some helper functions for static modes
function x_from_r(a, r)
    #=
    x \equiv (rp - r)/(rp - rm) is a transformed coordinate that
    is used only internally in the code/static mode computation
    =#
    gamma = sqrt(1 - a^2)
    return ((1+gamma)-r)/(2*gamma)
end

function isnonpositiveinteger(x)
    isinteger(x) && x <= 0
end

# Some helper functions for manipulating hypergeometric functions
function pochhammer(x, n)
    # Wrapper for the pochhammer function implemented in HypergeometricFunctions.jl
    HypergeometricFunctions.pochhammer(x, n)
end

#=
Since there is no implementation of a "regularized" Gauss hypergeometric function in
HypergeometricFunctions.jl yet, we use the fact that when c approaches a nonpositive
integer, 2F1(a,b;c,z)/Gamma(c) approaches the value

For c = 1-m with m >= 1:
2F1(a, b; c, z)/Gamma(c) = (a)_m (b)_m / m! * z^m * 2F1(a+m, b+m; m+1; z).

which is implemented below
=#
function _2F1_over_Gamma_c(a, b, c, z)
    # Valid for nonpositive integer c.
    z *= (1 + 0im) # Make sure that z is a complex number
    m = abs(c) + 1
    ( pochhammer(a, m) * pochhammer(b, m) / factorial(m) ) * z^m * pFq((a+m, b+m), (m+1, ), z)
end

function _d2F1_over_Gamma_c_dz(a, b, c, z)
    # Valid for nonpositive integer c.
    z *= (1 + 0im) # Make sure that z is a complex number
    m = abs(c) + 1
    # Here we explicitly compute the derivative because ForwardDiff.jl does not work very well with HypergeometricFunctions.jl
    ( pochhammer(a, m) * pochhammer(b, m) / factorial(m) ) * (m*z^(m-1) * pFq((a+m, b+m), (m+1, ), z) + z^m * (a+m)*(b+m)/(m+1) * pFq((a+m+1, b+m+1), (m+2, ), z))
end

function solve_static_Rin(s::Int, l::Int, m::Int, a)
    #=
    This is an internal function that returns the static (omega = 0) solution that satisfies
    the "purely left-going" condition at the horizon, namely the R^in solution.

    Use instead GeneralizedSasakiNakamura.Teukolsky_radial() and set omega = 0 and boundary_condition = IN
    =#
    # gamma = 0 at |a| = 1 makes kappa infinite: the exact-extremal static solutions are not implemented
    abs(a) < 1 || throw(DomainError(a, "static (omega = 0) solutions require |a| < 1."))
    gamma = sqrt(1 - a^2)
    kappa = I*a*m/gamma

    normalization_const = begin
        (-1)^(-kappa/2) * exp(I*a*m/2) * gamma^(I*a*m/(1+gamma)) * (-4*gamma^2)^(-s)
    end

    Rin(r) = begin
        x = x_from_r(a, r)
        if iszero(kappa) && isnonpositiveinteger(1-s)
            # Regularization
            normalization_const * x^(-s + kappa/2) * (1-x)^(kappa/2) * _2F1_over_Gamma_c(-l+kappa, 1+l+kappa, 1-s, x)
        else
            normalization_const * x^(-s + kappa/2) * (1-x)^(kappa/2) * pFq((-l+kappa, 1+l+kappa), (1-s+kappa,), x)
        end
    end

    dRindr(r) = begin
        x = x_from_r(a, r)
        if iszero(kappa) && isnonpositiveinteger(1-s)
            # Regularization
            normalization_const * (-1/(2*gamma)) * ( (1/2) * (1-x)^(-1+kappa/2) * x^(-1-s+kappa/2) * (2*s*(x-1) + kappa - 2*kappa*x) * _2F1_over_Gamma_c(-l+kappa, 1+l+kappa, 1-s, x) + x^(-s + kappa/2) * (1-x)^(kappa/2) * _d2F1_over_Gamma_c_dz(-l+kappa, 1+l+kappa, 1-s, x) )
        else
            normalization_const * (-1/(2*gamma)) * ( (1/2) * (1-x)^(-1+kappa/2) * x^(-1-s+kappa/2) * (2*s*(x-1) + kappa - 2*kappa*x) * pFq((-l+kappa, 1+l+kappa), (1-s+kappa,), x) + x^(-s + kappa/2) * (1-x)^(kappa/2) * pochhammer(-l+kappa, 1) * pochhammer(1+l+kappa, 1) / pochhammer(1-s+kappa, 1) * pFq((-l+kappa+1, 1+l+kappa+1), (1-s+kappa+1,), x) )
        end
    end

    return (r -> [Rin(r);  dRindr(r)])
end

function solve_static_Rup(s::Int, l::Int, m::Int, a)
    #=
    This is an internal function that returns the static (omega = 0) solution that satisfies
    the "purely right-going" condition at spatial infinity, namely the R^up solution.

    Use instead GeneralizedSasakiNakamura.Teukolsky_radial() and set omega = 0 and boundary_condition = UP
    =#
    # gamma = 0 at |a| = 1 makes kappa infinite: the exact-extremal static solutions are not implemented
    abs(a) < 1 || throw(DomainError(a, "static (omega = 0) solutions require |a| < 1."))
    gamma = sqrt(1 - a^2)
    kappa = I*a*m/gamma

    normalization_const = begin
        (-2*gamma)^(-s-l-1)
    end

    Rup(r) = begin
        x = x_from_r(a, r)
        normalization_const * x^(-s-l-1-kappa/2) * (x-1)^(kappa/2) * pFq((1+l+kappa, 1+s+l), (2+2*l,),1/x)
    end
    
    dRupdr(r) = begin
        x = x_from_r(a, r)
        normalization_const * (-1/(2*gamma)) * ( ( (1/2)*(x-1)^(-1+kappa/2) * x^(-2-l-s-kappa/2) * (-2*(1+l+s)*(x-1) + kappa) ) * pFq((1+l+kappa, 1+s+l), (2+2*l,),1/x) + x^(-s-l-1-kappa/2) * (x-1)^(kappa/2) * pochhammer(1+l+kappa, 1) * pochhammer(1+s+l, 1) / pochhammer(2+2*l, 1) * pFq((2+l+kappa, 2+s+l), (3+2*l,),1/x) * (-1/x^2) )
    end

    return (r-> [Rup(r); dRupdr(r)])
end



end
