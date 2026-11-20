module Potentials

using ..Kerr
using ..Transformation

export sF, sU, sF_delta, sU_delta, sW

const I = 1im # Mathematica being Mathematica

# sF and sU in the form used away from the horizon: Delta = a^2 - 2r + r^2 written as a^2 + (-2 + r) r.
sF(s::Int, m::Int, a, omega, lambda, r) = sF_delta(s, m, a, omega, lambda, r, a^2 + (-2 + r)*r)
sU(s::Int, m::Int, a, omega, lambda, r) = sU_delta(s, m, a, omega, lambda, r, a^2 + (-2 + r)*r)


include("PotentialsDelta.jl")

end
