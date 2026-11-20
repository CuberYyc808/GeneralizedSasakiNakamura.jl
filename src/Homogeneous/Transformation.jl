module Transformation

using ForwardDiff
using ..Kerr

export eta_coefficient, eta

const I = 1im # Mathematica being Mathematica

function eta_coefficient(s::Int, m::Int, a, omega, lambda, order)
    if s == 0
        if order == 0
            return 1
        elseif order == -1
            return 0
        elseif order == -2
            return 0
        elseif order == -3
            return 0
        elseif order == -4
            return 0
        else
            return 0
        end
    elseif s == +1
        if order == 0
            return -2 - lambda
        elseif order == -1
            return 2*I*a*m
        elseif order == -2
            return -3*a^2 - 2*a^2*lambda
        elseif order == -3
            return -2*a^2 + 2*I*a^3*m
        elseif order == -4
            return -a^4 - a^4*lambda
        else
            return 0
        end
    elseif s == -1
        if order == 0
            return -lambda
        elseif order == -1
            return -2*I*a*m
        elseif order == -2
            return a^2 - 2*a^2*lambda
        elseif order == -3
            return -2*a^2 - 2*I*a^3*m
        elseif order == -4
            return a^4 - a^4*lambda
        else
            return 0
        end
    elseif s == +2
        if order == 0
            return 24 + 10*lambda + lambda^2 + 12*I*omega + 12*a*m*omega - 12*a^2*omega^2
        elseif order == -1
            return -32*I*a*m - 8*I*a*m*lambda + 8*I*a^2*omega + 8*I*a^2*lambda*omega
        elseif order == -2
            return 12*a^2 - 24*I*a*m - 24*a^2*m^2 + 24*I*a^2*omega + 48*a^3*m*omega - 24*a^4*omega^2
        elseif order == -3
            return -24*a^2 + 24*I*a^3*m - 24*I*a^4*omega
        elseif order == -4
            return 12*a^4
        else
            return 0
        end
    elseif s == -2
        if order == 0
            return 2*lambda + lambda^2 - 12*I*omega + 12*a*m*omega - 12*a^2*omega^2
        elseif order == -1
            return 8*I*a*m*lambda + 24*I*a^2*omega - 8*I*a^2*lambda*omega
        elseif order == -2
            return 12*a^2 + 24*I*a*m - 24*a^2*m^2 - 24*I*a^2*omega + 48*a^3*m*omega - 24*a^4*omega^2
        elseif order == -3
            return -24*a^2 - 24*I*a^3*m + 24*I*a^4*omega
        elseif order == -4
            return 12*a^4
        else
            return 0
        end
    else
        throw(DomainError(s, "Currently only spin weight s of 0, +/-1, +/-2 are supported"))
    end
end

function eta(s::Int, m::Int, a, omega, lambda, r)
    # eta(r) = c0 + c1/r + c2/r^2 + c3/r^3 + c4/r^4
    c0 = eta_coefficient(s, m, a, omega, lambda, 0)
    c1 = eta_coefficient(s, m, a, omega, lambda, -1)
    c2 = eta_coefficient(s, m, a, omega, lambda, -2)
    c3 = eta_coefficient(s, m, a, omega, lambda, -3)
    c4 = eta_coefficient(s, m, a, omega, lambda, -4)
    return c0 + c1/r + c2/r^2 + c3/r^3 + c4/r^4
end

end